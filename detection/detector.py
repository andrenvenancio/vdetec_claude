"""
Detecção zero-shot de produtos em prateleiras usando YOLO-World.
Não requer treinamento — usa modelo pré-treinado com vocabulário aberto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from ultralytics import YOLOWorld

# Classes genéricas para produtos em gôndola de varejo
_DEFAULT_CLASSES = [
    "product box",
    "package",
    "bottle",
    "box",
    "food package",
    "consumer product",
]


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)
    confidence: float
    class_name: str
    crop: np.ndarray | None = field(default=None, repr=False)
    location: int | None = None


class ProductDetector:
    """
    Detector zero-shot baseado em YOLO-World.
    Baixa o modelo automaticamente na primeira execução (~100 MB).
    """

    MODEL_ID = "yolov8s-worldv2.pt"

    def __init__(
        self,
        classes: list[str] | None = None,
        conf: float = 0.05,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self._classes = classes or _DEFAULT_CLASSES
        self._conf = conf
        self._iou = iou
        self._device = device
        self._model: YOLOWorld | None = None

    def load(self) -> None:
        if self._model is None:
            self._model = YOLOWorld(self.MODEL_ID)
            self._model.set_classes(self._classes)

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.load()
        results = self._model.predict(
            image,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
            device=self._device,
        )[0]

        detections: list[Detection] = []
        if results.boxes is None:
            return detections

        h, w = image.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Descarta crops muito pequenos (ruído)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
                class_name=self._classes[int(box.cls[0])],
                crop=image[yi1:yi2, xi1:xi2].copy(),
            ))

        return sorted(detections, key=lambda d: d.confidence, reverse=True)
