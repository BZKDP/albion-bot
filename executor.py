from __future__ import annotations

import logging
import random
import time
from collections import deque
from typing import Optional, Tuple

import pydirectinput


class ActionExecutor:
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.action_timestamps: deque[float] = deque()
        pydirectinput.PAUSE = 0
        self.reload_config(config)

    def reload_config(self, config: dict) -> None:
        self.config = config or {}
        executor_cfg = self.config.get("executor", {})

        self.dry_run = bool(self.config.get("dry_run", True))
        self.interact_key = str(executor_cfg.get("interact_key", "f"))
        self.move_duration_range = self._pair(executor_cfg.get("move_duration_range", [0.18, 0.42]), (0.18, 0.42))
        self.action_delay_range = self._pair(executor_cfg.get("action_delay_range", [0.09, 0.24]), (0.09, 0.24))
        self.jitter_px = int(executor_cfg.get("jitter_px", 6))
        self.failsafe_corner_px = int(executor_cfg.get("failsafe_corner_px", 4))
        self.actions_per_minute = int(executor_cfg.get("actions_per_minute", 180))

    def move_to(self, x: int, y: int) -> bool:
        self._check_failsafe_corner()
        if not self._allow_action():
            return False

        tx = int(x + random.randint(-self.jitter_px, self.jitter_px))
        ty = int(y + random.randint(-self.jitter_px, self.jitter_px))
        duration = random.uniform(*self.move_duration_range)

        if self.dry_run:
            self.logger.info("[DRY-RUN] move_to x=%s y=%s duration=%.2f", tx, ty, duration)
        else:
            pydirectinput.moveTo(tx, ty, duration=duration)

        self._record_action()
        self._human_delay()
        return True

    def click(self, button: str = "left") -> bool:
        self._check_failsafe_corner()
        if not self._allow_action():
            return False

        if self.dry_run:
            self.logger.info("[DRY-RUN] click button=%s", button)
        else:
            pydirectinput.click(button=button)

        self._record_action()
        self._human_delay()
        return True

    def press_key(self, key: str) -> bool:
        self._check_failsafe_corner()
        if not key:
            return True
        if not self._allow_action():
            return False

        if self.dry_run:
            self.logger.info("[DRY-RUN] press_key key=%s", key)
        else:
            pydirectinput.press(key)

        self._record_action()
        self._human_delay()
        return True

    def micro_reposition(self, distance: int = 120, camera_turn_key: str = "") -> bool:
        self._check_failsafe_corner()
        if not self._allow_action():
            return False

        dx = random.randint(-distance, distance)
        dy = random.randint(-distance, distance)
        duration = random.uniform(*self.move_duration_range)

        if self.dry_run:
            self.logger.info(
                "[DRY-RUN] micro_reposition dx=%s dy=%s duration=%.2f camera_key=%s",
                dx,
                dy,
                duration,
                camera_turn_key or "-",
            )
        else:
            pydirectinput.moveRel(dx, dy, duration=duration)

        self._record_action()
        self._human_delay()

        if camera_turn_key:
            return self.press_key(camera_turn_key)
        return True

    def safe_wait(self, seconds: float) -> None:
        end = time.time() + max(0.0, seconds)
        while time.time() < end:
            self._check_failsafe_corner()
            time.sleep(min(0.1, end - time.time()))

    def _allow_action(self) -> bool:
        now = time.time()
        while self.action_timestamps and now - self.action_timestamps[0] > 60:
            self.action_timestamps.popleft()

        if len(self.action_timestamps) >= self.actions_per_minute:
            self.logger.warning("Action blocked by actions_per_minute limit")
            return False

        return True

    def _record_action(self) -> None:
        self.action_timestamps.append(time.time())

    def _human_delay(self) -> None:
        time.sleep(random.uniform(*self.action_delay_range))

    def _check_failsafe_corner(self) -> None:
        pos = pydirectinput.position()

        if hasattr(pos, "x") and hasattr(pos, "y"):
            x, y = int(pos.x), int(pos.y)
        else:
            x, y = int(pos[0]), int(pos[1])

        if x <= self.failsafe_corner_px and y <= self.failsafe_corner_px:
            raise RuntimeError("Failsafe corner reached (top-left)")

    @staticmethod
    def _pair(value, fallback: Tuple[float, float]) -> Tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return fallback

        try:
            first = float(value[0])
            second = float(value[1])
        except (TypeError, ValueError):
            return fallback

        if first <= 0 or second <= 0:
            return fallback

        if first > second:
            first, second = second, first

        return first, second
