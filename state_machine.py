from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import mss
import numpy as np

from detector_ai import AIDetector
from detector_local import DetectionResult, LocalDetector
from executor import ActionExecutor


class BotState(str, Enum):
    IDLE = "IDLE"
    SEARCH = "SEARCH"
    MOVE = "MOVE"
    INTERACT = "INTERACT"
    VERIFY = "VERIFY"
    REPOSITION = "REPOSITION"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class AlbionStateMachine:
    def __init__(
        self,
        config: dict,
        local_detector: LocalDetector,
        ai_detector: AIDetector,
        executor: ActionExecutor,
        logger: Optional[logging.Logger] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.log_callback = log_callback

        self.local_detector = local_detector
        self.ai_detector = ai_detector
        self.executor = executor

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self.worker: Optional[threading.Thread] = None
        self.state: BotState = BotState.IDLE
        self.state_before_pause: BotState = BotState.SEARCH

        self.plan: Optional[Dict[str, Any]] = None
        self.started_at: float = 0.0
        self.cycles_completed: int = 0
        self.attempts_without_progress: int = 0

        self.current_target: Optional[Dict[str, Any]] = None
        self.pending_action: str = "interact"
        self.pending_key: str = ""
        self.verify_ready_at: float = 0.0
        self.active_monitor: Dict[str, int] = {"left": 0, "top": 0, "width": 0, "height": 0}

        self.reload_config(config)

    def reload_config(self, config: dict) -> None:
        self.config = config or {}
        runtime_cfg = self.config.get("runtime", {})

        self.tick_seconds = float(runtime_cfg.get("tick_seconds", 0.35))
        self.verify_delay_seconds = float(runtime_cfg.get("verify_delay_seconds", 1.4))
        self.anti_stuck_attempts = int(runtime_cfg.get("anti_stuck_attempts", 5))
        self.reposition_pixels = int(runtime_cfg.get("reposition_pixels", 120))
        self.camera_turn_key = str(runtime_cfg.get("camera_turn_key", "")).strip()

        self.debug_frame_path = Path(self.config.get("debug_frame_path", "debug_frames/last_frame.jpg"))
        self.debug_frame_path.parent.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def start(self, plan: Dict[str, Any]) -> bool:
        if self.is_running():
            self._log("Start ignored: runtime already running")
            return False

        with self.lock:
            self.plan = plan
            self.started_at = time.time()
            self.cycles_completed = 0
            self.attempts_without_progress = 0
            self.current_target = None
            self.pending_action = "interact"
            self.pending_key = ""
            self.verify_ready_at = 0.0
            self.state = BotState.SEARCH

        self.stop_event.clear()
        self.pause_event.clear()

        self.worker = threading.Thread(target=self._run_loop, name="albion-runtime", daemon=True)
        self.worker.start()
        self._log("Runtime started")
        return True

    def pause(self) -> None:
        if not self.is_running():
            return

        if self.pause_event.is_set():
            return

        self.pause_event.set()
        with self.lock:
            self.state_before_pause = self.state
            self.state = BotState.PAUSED
        self._log("Paused")

    def resume(self) -> None:
        if not self.is_running():
            return

        if not self.pause_event.is_set():
            return

        self.pause_event.clear()
        with self.lock:
            self.state = self.state_before_pause if self.state_before_pause != BotState.PAUSED else BotState.SEARCH
        self._log("Resumed")

    def toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.resume()
        else:
            self.pause()

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self._log("Stop requested")

    def join(self, timeout: float = 2.0) -> None:
        if self.worker is not None:
            self.worker.join(timeout=timeout)

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.started_at if self.started_at else 0.0
            objective = ""
            if self.plan:
                objective = str(self.plan.get("objective", ""))

            return {
                "state": self.state.value,
                "is_running": self.is_running(),
                "is_paused": self.pause_event.is_set(),
                "objective": objective,
                "cycles_completed": self.cycles_completed,
                "attempts_without_progress": self.attempts_without_progress,
                "elapsed_seconds": elapsed,
                "dry_run": bool(self.config.get("dry_run", True)),
                "plan": self.plan,
            }

    def _run_loop(self) -> None:
        try:
            with mss.mss() as sct:
                self.active_monitor = self._resolve_monitor(sct)

                while not self.stop_event.is_set():
                    if self.pause_event.is_set():
                        time.sleep(0.15)
                        continue

                    reason = self._stop_reason()
                    if reason:
                        self._log(f"Stop condition reached: {reason}")
                        self.stop_event.set()
                        break

                    frame = self._capture_frame(sct, self.active_monitor)
                    self._step(frame)
                    time.sleep(self.tick_seconds)

        except Exception as exc:
            self.logger.exception("Runtime loop crashed: %s", exc)
            self._log(f"Runtime crash: {exc}")
        finally:
            with self.lock:
                self.state = BotState.STOPPED
            self._log("Runtime stopped")

    def _step(self, frame: np.ndarray) -> None:
        state = self.state
        if state == BotState.SEARCH:
            self._handle_search(frame)
        elif state == BotState.MOVE:
            self._handle_move()
        elif state == BotState.INTERACT:
            self._handle_interact()
        elif state == BotState.VERIFY:
            self._handle_verify(frame)
        elif state == BotState.REPOSITION:
            self._handle_reposition()

    def _handle_search(self, frame: np.ndarray) -> None:
        target_types = self._plan_target_types()

        detection = self.local_detector.detect(frame, target_types)
        LocalDetector.draw_and_save_debug_frame(frame, detection, self.debug_frame_path)
        if detection:
            self.current_target = self._target_from_local(detection)
            self.pending_action = "interact"
            self.pending_key = self._current_interact_key()
            self._log(
                "Local target found: %s conf=%.2f",
                detection.target_type,
                detection.confidence,
            )
            self._set_state(BotState.MOVE)
            return

        if self.ai_detector is not None:
            ai_decision = self.ai_detector.detect(
                frame_bgr=frame,
                objective=self._plan_objective(),
                target_types=target_types,
            )
            if ai_decision and ai_decision.get("found"):
                self.current_target = self._target_from_ai(ai_decision, frame.shape[1], frame.shape[0])
                self.pending_action = str(ai_decision.get("action", "move"))
                self.pending_key = str(ai_decision.get("key", "")) or self._current_interact_key()
                self._log(
                    "AI target found: action=%s conf=%.2f reason=%s",
                    self.pending_action,
                    float(ai_decision.get("confidence", 0.0)),
                    str(ai_decision.get("reason", "")),
                )
                if self.pending_action in {"move", "interact"}:
                    self._set_state(BotState.MOVE)
                else:
                    self._set_state(BotState.REPOSITION)
                return

        self._set_state(BotState.REPOSITION)

    def _handle_move(self) -> None:
        if not self.current_target:
            self._set_state(BotState.SEARCH)
            return

        ok = self.executor.move_to(
            int(self.current_target["screen_x"]),
            int(self.current_target["screen_y"]),
        )
        self._set_state(BotState.INTERACT if ok else BotState.REPOSITION)

    def _handle_interact(self) -> None:
        action = (self.pending_action or "interact").lower()
        key = self.pending_key or self._current_interact_key()

        ok = True
        if action == "wait":
            self.executor.safe_wait(0.4)
        elif action == "move":
            ok = self.executor.click("left")
        else:
            ok_click = self.executor.click("left")
            ok_key = self.executor.press_key(key) if key else True
            ok = ok_click and ok_key

        if ok:
            self.verify_ready_at = time.time() + self.verify_delay_seconds
            self._set_state(BotState.VERIFY)
        else:
            self._set_state(BotState.REPOSITION)

    def _handle_verify(self, frame: np.ndarray) -> None:
        if time.time() < self.verify_ready_at:
            return

        target_types = self._plan_target_types()
        detection = self.local_detector.detect(frame, target_types)
        LocalDetector.draw_and_save_debug_frame(frame, detection, self.debug_frame_path)

        if detection and self.current_target:
            dx = abs(int(detection.x) - int(self.current_target["x"]))
            dy = abs(int(detection.y) - int(self.current_target["y"]))
            if dx <= 80 and dy <= 80:
                self.attempts_without_progress += 1
                self._log("No progress in VERIFY: attempts=%s", self.attempts_without_progress)
                if self.attempts_without_progress >= self.anti_stuck_attempts:
                    self._set_state(BotState.REPOSITION)
                else:
                    self._set_state(BotState.INTERACT)
                return

        self.cycles_completed += 1
        self.attempts_without_progress = 0
        self.current_target = None
        self.pending_action = "interact"
        self.pending_key = ""
        self._log("Cycle completed: %s", self.cycles_completed)
        self._set_state(BotState.SEARCH)

    def _handle_reposition(self) -> None:
        self.attempts_without_progress += 1
        distance = self.reposition_pixels
        if self.attempts_without_progress >= self.anti_stuck_attempts:
            distance *= 2
            self._log("Anti-stuck trigger: wide reposition")
            self.attempts_without_progress = 0

        self.executor.micro_reposition(distance=distance, camera_turn_key=self.camera_turn_key)
        self.current_target = None
        self.pending_action = "interact"
        self.pending_key = ""
        self._set_state(BotState.SEARCH)

    def _target_from_local(self, detection: DetectionResult) -> Dict[str, Any]:
        left = int(self.active_monitor.get("left", 0))
        top = int(self.active_monitor.get("top", 0))
        return {
            "x": int(detection.x),
            "y": int(detection.y),
            "screen_x": left + int(detection.x),
            "screen_y": top + int(detection.y),
            "confidence": float(detection.confidence),
            "target_type": detection.target_type,
        }

    def _target_from_ai(self, payload: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
        left = int(self.active_monitor.get("left", 0))
        top = int(self.active_monitor.get("top", 0))

        x = max(0, min(width - 1, int(payload.get("x", width // 2))))
        y = max(0, min(height - 1, int(payload.get("y", height // 2))))

        return {
            "x": x,
            "y": y,
            "screen_x": left + x,
            "screen_y": top + y,
            "confidence": float(payload.get("confidence", 0.0)),
            "target_type": "ai",
        }

    def _capture_frame(self, sct: mss.mss, monitor: Dict[str, int]) -> np.ndarray:
        shot = sct.grab(monitor)
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _resolve_monitor(self, sct: mss.mss) -> Dict[str, int]:
        screen_cfg = self.config.get("screen", {})
        region = screen_cfg.get("region")

        if isinstance(region, dict) and all(k in region for k in ("left", "top", "width", "height")):
            return {
                "left": int(region["left"]),
                "top": int(region["top"]),
                "width": int(region["width"]),
                "height": int(region["height"]),
            }

        monitor_index = int(screen_cfg.get("monitor", 1))
        monitor_index = max(1, min(monitor_index, len(sct.monitors) - 1))
        monitor = sct.monitors[monitor_index]
        return {
            "left": int(monitor["left"]),
            "top": int(monitor["top"]),
            "width": int(monitor["width"]),
            "height": int(monitor["height"]),
        }

    def _set_state(self, state: BotState) -> None:
        with self.lock:
            self.state = state

    def _stop_reason(self) -> str:
        if not self.plan:
            return ""

        stop_conditions = self.plan.get("stop_conditions", {})
        try:
            max_cycles = int(stop_conditions.get("max_cycles", 0))
        except (TypeError, ValueError):
            max_cycles = 0

        try:
            minutes = int(stop_conditions.get("minutes", 0))
        except (TypeError, ValueError):
            minutes = 0

        if max_cycles > 0 and self.cycles_completed >= max_cycles:
            return f"max_cycles={max_cycles}"

        if minutes > 0 and self.started_at > 0 and (time.time() - self.started_at) >= minutes * 60:
            return f"minutes={minutes}"

        return ""

    def _plan_target_types(self) -> list[str]:
        if not self.plan:
            return ["wood", "stone", "ore", "fiber", "hide"]

        types = self.plan.get("target_types", [])
        if not isinstance(types, list):
            return ["wood", "stone", "ore", "fiber", "hide"]

        normalized = [str(item).lower() for item in types if item]
        return normalized or ["wood", "stone", "ore", "fiber", "hide"]

    def _plan_objective(self) -> str:
        if not self.plan:
            return ""
        return str(self.plan.get("objective", ""))

    def _current_interact_key(self) -> str:
        if self.plan and self.plan.get("interact_key"):
            return str(self.plan.get("interact_key"))

        executor_cfg = self.config.get("executor", {})
        return str(executor_cfg.get("interact_key", "f"))

    def _log(self, message: str, *args: Any) -> None:
        if args:
            text = message % args
        else:
            text = message

        self.logger.info(text)
        if self.log_callback is not None:
            self.log_callback(text)
