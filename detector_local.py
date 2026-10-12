from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class DetectionResult:
    target_type: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    template_name: str


class LocalDetector:
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.templates: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        self.reload_config(config)

    def reload_config(self, config: dict) -> None:
        self.config = config or {}
        self.template_dir = Path(self.config.get("template_dir", "templates"))
        self.confidence = float(self.config.get("confidence", 0.82))
        self.use_grayscale = bool(self.config.get("grayscale", True))
        self.load_templates()

    def load_templates(self) -> None:
        self.templates.clear()

        if not self.template_dir.exists():
            self.logger.warning("Template dir not found: %s", self.template_dir)
            return

        for path in sorted(self.template_dir.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue

            flag = cv2.IMREAD_GRAYSCALE if self.use_grayscale else cv2.IMREAD_COLOR
            image = cv2.imread(str(path), flag)
            if image is None:
                self.logger.warning("Failed to read template: %s", path)
                continue

            target_type = self._target_type_from_path(path)
            self.templates.setdefault(target_type, []).append((path.stem, image))

        total = sum(len(items) for items in self.templates.values())
        self.logger.info(
            "Local templates loaded: %s files across %s target groups",
            total,
            len(self.templates),
        )

    def detect(self, frame_bgr: np.ndarray, target_types: Sequence[str]) -> Optional[DetectionResult]:
        if frame_bgr is None or frame_bgr.size == 0 or not self.templates:
            return None

        if self.use_grayscale:
            frame_search = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            frame_search = frame_bgr

        pool: List[Tuple[str, str, np.ndarray]] = []
        requested = [str(t).lower() for t in target_types] if target_types else []

        for target in requested:
            for template_name, template in self.templates.get(target, []):
                pool.append((target, template_name, template))

        if not pool:
            for target, items in self.templates.items():
                for template_name, template in items:
                    pool.append((target, template_name, template))

        best: Optional[DetectionResult] = None
        fh, fw = frame_search.shape[:2]

        for target_type, template_name, template in pool:
            th, tw = template.shape[:2]
            if fh < th or fw < tw:
                continue

            result = cv2.matchTemplate(frame_search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            confidence = float(max_val)

            if confidence < self.confidence:
                continue

            if best is None or confidence > best.confidence:
                x, y = max_loc
                best = DetectionResult(
                    target_type=target_type,
                    confidence=confidence,
                    x=int(x + tw // 2),
                    y=int(y + th // 2),
                    w=int(tw),
                    h=int(th),
                    template_name=template_name,
                )

        return best

    @staticmethod
    def draw_and_save_debug_frame(
        frame_bgr: np.ndarray,
        detection: Optional[DetectionResult],
        output_path: str | Path,
    ) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return

        image = frame_bgr.copy()
        if detection is not None:
            x1 = max(0, int(detection.x - detection.w // 2))
            y1 = max(0, int(detection.y - detection.h // 2))
            x2 = min(image.shape[1] - 1, int(detection.x + detection.w // 2))
            y2 = min(image.shape[0] - 1, int(detection.y + detection.h // 2))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection.target_type} {detection.confidence:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)

    def _target_type_from_path(self, path: Path) -> str:
        stem = path.stem.lower()
        if "_" in stem:
            return stem.split("_", 1)[0]

        if path.parent != self.template_dir:
            return path.parent.name.lower()

        return "unknown"
