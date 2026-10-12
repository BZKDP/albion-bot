from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table

from bot import PromptPlanner, setup_logger
from detector_ai import AIDetector
from detector_local import LocalDetector
from executor import ActionExecutor
from state_machine import AlbionStateMachine


def parse_region(value: str) -> Optional[dict]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        return None

    try:
        left, top, width, height = [int(item) for item in parts]
    except ValueError:
        return None

    if width <= 0 or height <= 0:
        return None

    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autostart runner for Albion offline bot")
    parser.add_argument(
        "--objective",
        default="собирать хлопок 20 минут вокруг стартовой точки",
        help="Objective prompt",
    )
    parser.add_argument("--minutes", type=int, default=None, help="Override stop_conditions.minutes")
    parser.add_argument("--max-cycles", type=int, default=None, help="Override stop_conditions.max_cycles")
    parser.add_argument("--interact-key", default=None, help="Override interact key")
    parser.add_argument(
        "--region",
        default=None,
        help="Capture region as left,top,width,height (example: 0,0,1600,900)",
    )
    parser.add_argument("--snapshot-interval", type=float, default=2.0, help="Seconds between debug frame copies")
    parser.add_argument("--status-interval", type=float, default=0.5, help="Seconds between status refresh")
    parser.add_argument("--live", action="store_true", help="Disable dry-run and send real input events")
    parser.add_argument("--force-fiber", action="store_true", default=True, help="Force target_types=['fiber'] for cotton prompt")
    parser.add_argument("--no-force-fiber", action="store_true", help="Disable force-fiber behavior")
    parser.add_argument("--hotkeys", action="store_true", help="Enable F8/F9 (optional, can create extra helper process)")
    return parser.parse_args()


def should_force_fiber(args: argparse.Namespace) -> bool:
    if args.no_force_fiber:
        return False
    return bool(args.force_fiber)


def build_table(status: dict, plan: dict, snapshot_dir: Path, debug_frame: Path, hotkeys_enabled: bool) -> Table:
    table = Table(title="Albion Auto Runner", show_header=False)
    table.add_column("key", style="cyan", width=24)
    table.add_column("value", style="white")

    elapsed = int(status.get("elapsed_seconds", 0))
    minutes, seconds = divmod(elapsed, 60)

    table.add_row("Objective", str(plan.get("objective", "-")))
    table.add_row("Target types", ", ".join(plan.get("target_types", [])) or "-")
    table.add_row("FSM state", str(status.get("state", "-")))
    table.add_row("Running", str(status.get("is_running", False)))
    table.add_row("Paused", str(status.get("is_paused", False)))
    table.add_row("Cycles", str(status.get("cycles_completed", 0)))
    table.add_row("No-progress", str(status.get("attempts_without_progress", 0)))
    table.add_row("Elapsed", f"{minutes:02d}:{seconds:02d}")
    table.add_row("Dry-run", str(status.get("dry_run", True)))
    table.add_row("Snapshots", str(snapshot_dir))
    table.add_row("Last frame", str(debug_frame))
    table.add_row("Hotkeys", "F8/F9 enabled" if hotkeys_enabled else "disabled (use Ctrl+C or stop process)")
    return table


def main() -> None:
    args = parse_args()

    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env", override=True)

    config_path = project_dir / "config.json"
    if not config_path.exists():
        config_path = project_dir / "config.example.json"

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    if args.live:
        config["dry_run"] = False

    region = parse_region(args.region) if args.region else None
    if region is not None:
        config.setdefault("screen", {})["region"] = region

    debug_frame_path = project_dir / config.get("debug_frame_path", "debug_frames/last_frame.jpg")
    debug_frame_path.parent.mkdir(parents=True, exist_ok=True)

    session_dir = project_dir / "debug_frames" / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(project_dir / config.get("log_file", "logs/bot.log"))

    planner = PromptPlanner(config, logger)
    plan = planner.build_plan(args.objective)

    objective_lower = args.objective.lower()
    if should_force_fiber(args) and ("cotton" in objective_lower or "хлоп" in objective_lower):
        plan["target_types"] = ["fiber"]

    if args.minutes is not None and args.minutes > 0:
        plan.setdefault("stop_conditions", {})["minutes"] = int(args.minutes)

    if args.max_cycles is not None and args.max_cycles > 0:
        plan.setdefault("stop_conditions", {})["max_cycles"] = int(args.max_cycles)

    if args.interact_key:
        plan["interact_key"] = str(args.interact_key).strip().lower()[:8]

    if not plan.get("interact_key"):
        plan["interact_key"] = str(config.get("executor", {}).get("interact_key", "f"))

    console = Console()

    local_detector = LocalDetector(config.get("detector_local", {}), logger)
    ai_detector = AIDetector(config.get("detector_ai", {}), logger)
    executor = ActionExecutor(config, logger)

    runtime = AlbionStateMachine(
        config=config,
        local_detector=local_detector,
        ai_detector=ai_detector,
        executor=executor,
        logger=logger,
    )

    keyboard_mod = None
    hotkeys_enabled = False
    if args.hotkeys:
        try:
            import keyboard as keyboard_mod

            keyboard_mod.add_hotkey("F8", runtime.toggle_pause)
            keyboard_mod.add_hotkey("F9", runtime.stop)
            hotkeys_enabled = True
        except Exception as exc:
            logger.warning("Hotkeys disabled: %s", exc)
            keyboard_mod = None
            hotkeys_enabled = False

    started = runtime.start(plan)
    if not started:
        console.print("Runtime is already running", style="bold red")
        if keyboard_mod is not None:
            keyboard_mod.unhook_all_hotkeys()
        return

    console.print(f"Plan: {json.dumps(plan, ensure_ascii=False)}")
    console.print(f"Snapshots dir: {session_dir}")

    last_snapshot_time = 0.0

    try:
        with Live(
            build_table(runtime.get_status(), plan, session_dir, debug_frame_path, hotkeys_enabled),
            refresh_per_second=4,
            console=console,
        ) as live:
            while runtime.is_running():
                status = runtime.get_status()
                live.update(build_table(status, plan, session_dir, debug_frame_path, hotkeys_enabled))

                now = time.time()
                if debug_frame_path.exists() and (now - last_snapshot_time) >= max(0.2, args.snapshot_interval):
                    dst = session_dir / f"frame_{int(now * 1000)}.jpg"
                    shutil.copy2(debug_frame_path, dst)
                    last_snapshot_time = now

                time.sleep(max(0.1, args.status_interval))

    except KeyboardInterrupt:
        runtime.stop()
        runtime.join(3.0)
    finally:
        if keyboard_mod is not None:
            keyboard_mod.unhook_all_hotkeys()
        runtime.stop()
        runtime.join(3.0)

    final_status = runtime.get_status()
    console.print(f"Final status: {json.dumps(final_status, ensure_ascii=False)}")
    console.print(f"Snapshots saved: {session_dir}")


if __name__ == "__main__":
    main()
