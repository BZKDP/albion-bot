from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import keyboard
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from detector_ai import AIDetector
from detector_local import LocalDetector
from executor import ActionExecutor
from state_machine import AlbionStateMachine

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

ALLOWED_TARGET_TYPES = ["wood", "stone", "ore", "fiber", "hide"]


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("albion_offline_bot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def make_openai_client(api_key: str, base_url: str):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class PromptPlanner:
    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self.logger = logger
        self.client = None
        self.reload_config(config)

    def reload_config(self, config: dict) -> None:
        self.config = config or {}
        planner_cfg = self.config.get("planner", {})
        self.client = None

        self.use_openai = bool(planner_cfg.get("use_openai", True))
        self.model = str(planner_cfg.get("model", "gpt-4.1-mini"))
        self.default_stop_minutes = int(planner_cfg.get("default_stop_minutes", 20))
        self.default_max_cycles = int(planner_cfg.get("default_max_cycles", 500))
        self.base_url = str(planner_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL", "")).strip()
        self.api_key_env = str(planner_cfg.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"

        if not self.use_openai:
            self.client = None
            return

        api_key = os.getenv(self.api_key_env, "").strip()
        if OpenAI is None or not api_key:
            self.client = None
            return

        self.client = make_openai_client(api_key=api_key, base_url=self.base_url)

    def build_plan(self, prompt: str) -> Dict[str, Any]:
        raw_plan = None
        if self.use_openai and self.client is not None:
            raw_plan = self._build_plan_with_openai(prompt)

        if raw_plan is None:
            raw_plan = self._build_plan_fallback(prompt)

        return self._normalize_plan(prompt, raw_plan)

    def _build_plan_with_openai(self, prompt: str) -> Optional[Dict[str, Any]]:
        schema_hint = (
            "{"
            '"objective":"string",'
            '"target_types":["wood","stone","ore","fiber","hide"],'
            '"interact_key":"string",'
            '"loop_strategy":"string",'
            '"stop_conditions":{"minutes":20,"max_cycles":500}'
            "}"
        )

        system_msg = (
            "Return ONLY one strict JSON object, no markdown. "
            "Keys must match exactly this schema: "
            f"{schema_hint}"
        )

        user_msg = (
            f"User prompt: {prompt}\n"
            f"Allowed target_types only: {', '.join(ALLOWED_TARGET_TYPES)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except Exception as exc:
            self.logger.warning("Prompt planner OpenAI call failed: %s", exc)
            return None

        raw_text = ""
        try:
            raw_content = response.choices[0].message.content
            if isinstance(raw_content, str):
                raw_text = raw_content.strip()
            elif isinstance(raw_content, list):
                chunks = []
                for item in raw_content:
                    if isinstance(item, dict):
                        text = str(item.get("text", "")).strip()
                        if text:
                            chunks.append(text)
                raw_text = "\n".join(chunks).strip()
        except Exception:
            raw_text = ""

        parsed = extract_json(raw_text) if raw_text else None
        if parsed is None:
            self.logger.warning("Prompt planner returned non-JSON; fallback parser will be used")
        return parsed

    def _build_plan_fallback(self, prompt: str) -> Dict[str, Any]:
        lower = prompt.lower()
        target_types = self._infer_targets(lower)

        minutes = self._extract_minutes(lower)
        if minutes is None:
            minutes = self.default_stop_minutes

        max_cycles = self._extract_cycles(lower)
        if max_cycles is None:
            max_cycles = self.default_max_cycles

        interact_key = "f"
        match_key = re.search(r"(?:key|button|клавиш[ауы])\s*[:=]?\s*([a-z0-9])", lower)
        if match_key:
            interact_key = match_key.group(1)

        loop_strategy = "spiral-around-start"
        if "around" in lower or "вокруг" in lower:
            loop_strategy = "circle-around-start"
        if "line" in lower or "линия" in lower:
            loop_strategy = "line-scan"

        return {
            "objective": prompt,
            "target_types": target_types,
            "interact_key": interact_key,
            "loop_strategy": loop_strategy,
            "stop_conditions": {
                "minutes": minutes,
                "max_cycles": max_cycles,
            },
        }

    def _normalize_plan(self, prompt: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        objective = str(plan.get("objective", "")).strip() if isinstance(plan, dict) else ""
        if not objective:
            objective = prompt.strip()

        raw_targets = plan.get("target_types", []) if isinstance(plan, dict) else []
        if not isinstance(raw_targets, list):
            raw_targets = []

        target_types = [str(t).lower() for t in raw_targets if str(t).lower() in ALLOWED_TARGET_TYPES]
        if not target_types:
            target_types = self._infer_targets(prompt.lower())

        interact_key = "f"
        if isinstance(plan, dict) and plan.get("interact_key"):
            interact_key = str(plan.get("interact_key")).strip().lower()[:8]
        if not interact_key:
            interact_key = "f"

        loop_strategy = "spiral-around-start"
        if isinstance(plan, dict) and plan.get("loop_strategy"):
            loop_strategy = str(plan.get("loop_strategy")).strip()[:64]

        stop_conditions = plan.get("stop_conditions", {}) if isinstance(plan, dict) else {}
        if not isinstance(stop_conditions, dict):
            stop_conditions = {}

        minutes = self._safe_int(stop_conditions.get("minutes"), self.default_stop_minutes)
        max_cycles = self._safe_int(stop_conditions.get("max_cycles"), self.default_max_cycles)

        prompt_lower = prompt.lower()
        prompt_minutes = self._extract_minutes(prompt_lower)
        if prompt_minutes is not None:
            minutes = prompt_minutes

        prompt_cycles = self._extract_cycles(prompt_lower)
        if prompt_cycles is not None:
            max_cycles = prompt_cycles

        return {
            "objective": objective,
            "target_types": target_types,
            "interact_key": interact_key,
            "loop_strategy": loop_strategy,
            "stop_conditions": {
                "minutes": max(1, minutes),
                "max_cycles": max(1, max_cycles),
            },
        }

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _extract_minutes(lower_prompt: str) -> Optional[int]:
        match = re.search(r"(\d+)\s*(?:minutes?|mins?|min|мин(?:ут(?:[аы]|у)?)?)", lower_prompt)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_cycles(lower_prompt: str) -> Optional[int]:
        match = re.search(r"(\d+)\s*(?:cycles?|цикл(?:ов|а)?)", lower_prompt)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _infer_targets(lower_prompt: str) -> list[str]:
        mapping = {
            "wood": ["wood", "tree", "timber", "birch", "древ", "берез"],
            "stone": ["stone", "rock", "кам", "булыж"],
            "ore": ["ore", "iron", "copper", "руд"],
            "fiber": ["fiber", "hemp", "cotton", "волок", "лен"],
            "hide": ["hide", "skin", "animal", "шкур", "живот"],
        }

        result = [
            target
            for target, words in mapping.items()
            if any(word in lower_prompt for word in words)
        ]
        return result or ALLOWED_TARGET_TYPES.copy()

class BotCLI:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        os.chdir(self.project_dir)
        load_dotenv(self.project_dir / ".env")

        self.console = Console()
        self.log_lines: deque[str] = deque(maxlen=16)

        self.config = self._load_config()
        log_file = self.project_dir / self.config.get("log_file", "logs/bot.log")
        self.logger = setup_logger(log_file)

        self.planner = PromptPlanner(self.config, self.logger)
        self.local_detector = LocalDetector(self.config.get("detector_local", {}), self.logger)
        self.ai_detector = AIDetector(self.config.get("detector_ai", {}), self.logger)
        self.executor = ActionExecutor(self.config, self.logger)
        self.runtime = AlbionStateMachine(
            config=self.config,
            local_detector=self.local_detector,
            ai_detector=self.ai_detector,
            executor=self.executor,
            logger=self.logger,
            log_callback=self._push_log,
        )

        self._register_hotkeys()
        self._push_log("Ready. Use start <prompt>")

    def run(self) -> None:
        try:
            while True:
                self.console.clear()
                self.console.print(self._status_panel())
                self.console.print(self._controls_panel())
                self.console.print(self._logs_panel())

                raw = Prompt.ask("Command").strip()
                if not raw:
                    continue

                cmd, _, arg = raw.partition(" ")
                cmd = cmd.lower()
                arg = arg.strip()

                if cmd == "start":
                    self._cmd_start(arg)
                elif cmd == "pause":
                    self.runtime.pause()
                elif cmd == "resume":
                    self.runtime.resume()
                elif cmd == "stop":
                    self.runtime.stop()
                elif cmd == "status":
                    self._push_log("Status refreshed")
                elif cmd == "reload-config":
                    self._cmd_reload_config()
                elif cmd in {"quit", "exit"}:
                    self.runtime.stop()
                    self.runtime.join(2.0)
                    break
                else:
                    self._push_log(f"Unknown command: {cmd}")
        finally:
            keyboard.unhook_all_hotkeys()
            self.runtime.stop()
            self.runtime.join(1.5)

    def _cmd_start(self, arg: str) -> None:
        if self.runtime.is_running():
            self._push_log("Runtime already running")
            return

        prompt = arg or Prompt.ask("Objective prompt")
        if not prompt.strip():
            self._push_log("Empty objective prompt")
            return

        plan = self.planner.build_plan(prompt)
        self._push_log(f"Plan: {json.dumps(plan, ensure_ascii=False)}")
        self.runtime.start(plan)

    def _cmd_reload_config(self) -> None:
        self.config = self._load_config()
        log_file = self.project_dir / self.config.get("log_file", "logs/bot.log")
        self.logger = setup_logger(log_file)

        self.planner.reload_config(self.config)
        self.local_detector.reload_config(self.config.get("detector_local", {}))
        self.ai_detector.reload_config(self.config.get("detector_ai", {}))
        self.executor.reload_config(self.config)
        self.runtime.reload_config(self.config)

        self._push_log("Config reloaded")

    def _load_config(self) -> Dict[str, Any]:
        primary = self.project_dir / "config.json"
        fallback = self.project_dir / "config.example.json"

        path = primary if primary.exists() else fallback
        if not path.exists():
            raise FileNotFoundError("config.json/config.example.json not found")

        return json.loads(path.read_text(encoding="utf-8-sig"))

    def _register_hotkeys(self) -> None:
        keyboard.add_hotkey("F8", self.runtime.toggle_pause)
        keyboard.add_hotkey("F9", self.runtime.stop)

    def _status_panel(self) -> Panel:
        status = self.runtime.get_status()
        elapsed = int(status.get("elapsed_seconds", 0))
        minutes, seconds = divmod(elapsed, 60)

        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", width=24)
        table.add_column(style="white")
        table.add_row("Objective", status.get("objective") or "-")
        table.add_row("FSM state", status.get("state", "-"))
        table.add_row("Running", str(status.get("is_running", False)))
        table.add_row("Paused", str(status.get("is_paused", False)))
        table.add_row("Cycles completed", str(status.get("cycles_completed", 0)))
        table.add_row("No-progress attempts", str(status.get("attempts_without_progress", 0)))
        table.add_row("Elapsed", f"{minutes:02d}:{seconds:02d}")
        table.add_row("Dry-run", str(status.get("dry_run", True)))

        return Panel(table, title="Albion Offline Bot", border_style="green")

    def _controls_panel(self) -> Panel:
        controls = Text()
        controls.append("Commands: start, pause, resume, stop, status, reload-config, exit\n", style="bold")
        controls.append("Hotkeys: F8 pause/resume, F9 stop\n")
        controls.append("Safety: dry_run=true by default; move mouse to top-left corner for failsafe")
        return Panel(controls, title="Controls", border_style="cyan")

    def _logs_panel(self) -> Panel:
        if not self.log_lines:
            body = Text("No logs yet")
        else:
            body = Text("\n".join(self.log_lines))
        return Panel(body, title="Recent Logs", border_style="magenta")

    def _push_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_lines.append(f"{timestamp} | {message}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    app = BotCLI(project_dir)
    app.run()


if __name__ == "__main__":
    main()





