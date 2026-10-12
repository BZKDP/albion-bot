from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class AIDetector:
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.call_timestamps: deque[float] = deque()
        self.last_call_time: float = 0.0
        self.client = None
        self.client_fingerprint: Optional[tuple[str, str]] = None
        self.reload_config(config)

    def reload_config(self, config: dict) -> None:
        self.config = config or {}
        self.client = None
        self.client_fingerprint = None

        self.enabled = bool(self.config.get("enabled", False))
        self.model = str(self.config.get("model", "gpt-4.1-mini"))
        self.cooldown_seconds = float(self.config.get("cooldown_seconds", 8))
        self.max_calls_per_minute = int(self.config.get("max_calls_per_minute", 6))
        self.timeout_seconds = float(self.config.get("timeout_seconds", 20))
        self.base_url = str(self.config.get("base_url") or os.getenv("OPENAI_BASE_URL", "")).strip()
        self.api_key_env = str(self.config.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
        self._ensure_client()

    def detect(
        self,
        frame_bgr,
        objective: str,
        target_types: list[str],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        self._ensure_client()
        if self.client is None:
            return None

        allowed, reason = self._can_call()
        if not allowed:
            self.logger.debug("AI fallback skipped: %s", reason)
            return None

        encoded = self._encode_frame(frame_bgr)
        if not encoded:
            return None

        system_prompt = (
            "You analyze one game screenshot and output ONLY strict JSON. "
            "Do not output markdown. JSON schema: "
            "{\"found\":bool,\"x\":int,\"y\":int,"
            "\"action\":\"move|interact|wait\",\"key\":\"string\","
            "\"confidence\":float,\"reason\":\"string\"}."
        )
        user_text = (
            f"Objective: {objective}\n"
            f"Target types: {', '.join(target_types)}\n"
            "If no confident target, return found=false and action=wait."
        )

        now = time.time()
        try:
            response = self.client.with_options(timeout=self.timeout_seconds).chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                            },
                        ],
                    },
                ],
            )
        except Exception as exc:
            self.logger.warning("AI fallback request failed: %s", exc)
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

        if not raw_text:
            raw_text = self._extract_text_fallback(response)

        payload = self._parse_json(raw_text)
        if payload is None:
            self.logger.warning("AI fallback returned non-JSON output")
            return None

        parsed = self._normalize_payload(payload, frame_bgr.shape[1], frame_bgr.shape[0])
        self.call_timestamps.append(now)
        self.last_call_time = now
        return parsed

    def _ensure_client(self) -> None:
        if not self.enabled:
            self.client = None
            self.client_fingerprint = None
            return

        if OpenAI is None:
            self.logger.warning("openai package is not installed; AI fallback disabled")
            self.client = None
            self.client_fingerprint = None
            return

        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            self.logger.warning("%s is missing; AI fallback disabled", self.api_key_env)
            self.client = None
            self.client_fingerprint = None
            return

        fingerprint = (api_key, self.base_url)
        if self.client is not None and self.client_fingerprint == fingerprint:
            return

        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)
        self.client_fingerprint = fingerprint

    def _can_call(self) -> tuple[bool, str]:
        now = time.time()

        while self.call_timestamps and now - self.call_timestamps[0] > 60:
            self.call_timestamps.popleft()

        if self.call_timestamps and now - self.call_timestamps[-1] < self.cooldown_seconds:
            return False, "cooldown"

        if len(self.call_timestamps) >= self.max_calls_per_minute:
            return False, "max_calls_per_minute"

        return True, "ok"

    def _encode_frame(self, frame_bgr) -> str:
        ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return ""
        return base64.b64encode(buffer).decode("ascii")

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
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

    def _normalize_payload(self, payload: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
        found = bool(payload.get("found", False))

        def clamp_int(value: Any, low: int, high: int, default: int) -> int:
            try:
                return max(low, min(high, int(value)))
            except (TypeError, ValueError):
                return default

        x = clamp_int(payload.get("x"), 0, max(0, width - 1), width // 2)
        y = clamp_int(payload.get("y"), 0, max(0, height - 1), height // 2)

        action = str(payload.get("action", "wait")).strip().lower()
        if action not in {"move", "interact", "wait"}:
            action = "wait"

        key = str(payload.get("key", "")).strip().lower()[:16]

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(payload.get("reason", "")).strip()[:280]

        return {
            "found": found,
            "x": x,
            "y": y,
            "action": action,
            "key": key,
            "confidence": confidence,
            "reason": reason,
        }

    def _extract_text_fallback(self, response: Any) -> str:
        try:
            return json.dumps(response.model_dump(), ensure_ascii=False)
        except Exception:
            return ""


