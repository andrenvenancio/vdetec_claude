"""
Módulo de detecção de objetos usando YOLOv8/v11.
Responsável por localizar regiões de interesse (produtos) na imagem.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from ultralytics import YOLO

from config.settings import settings


@dataclass
class Detection:
    """Resultado bruto de uma detecção."""
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)
    confidence: float
    class_id: int
    class_name: str
    crop: np.ndarray | None = field(default=None, repr=False)


class ProductDetector:
    """
    Wrapper sobre o modelo YOLO para detecção de produtos em prateleiras.

    Uso:
        detector = ProductDetector()
        detections = detector.predict(frame)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float | None = None,
        iou: float | None = None,
        device: str | None = None,
    ) -> None:
        self._model_path = Path(model_path or settings.model_path)
        self._conf = conf or settings.detector_conf_threshold
        self._iou = iou or settings.detector_iou_threshold
        self._device = device or settings.device
        self._model: YOLO | None = None

    def load(self) -> None:
        """Carrega o modelo na memória (lazy loading)."""
        if self._model is None:
            self._model = YOLO(str(self._model_path))
            self._model.to(self._device)

    def predict(self, image: np.ndarray, extract_crops: bool = True) -> list[Detection]:
        """
        Executa detecção em uma única imagem.

        Args:
            image: Frame BGR (OpenCV) ou RGB.
            extract_crops: Se True, inclui recorte da região detectada.

        Returns:
            Lista de Detection ordenada por confiança descendente.
        """
        self.load()
        results = self._model(
            image,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        boxes = results.boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]

            crop = None
            if extract_crops:
                xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
                crop = image[yi1:yi2, xi1:xi2].copy()

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
                crop=crop,
            ))

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def predict_batch(
        self, images: Sequence[np.ndarray], extract_crops: bool = True
    ) -> list[list[Detection]]:
        """Executa detecção em lote para maior eficiência com GPU."""
        self.load()
        all_results = self._model(
            list(images),
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        batch_detections: list[list[Detection]] = []
        for results, image in zip(all_results, images):
            detections: list[Detection] = []
            boxes = results.boxes
            if boxes is None:
                batch_detections.append(detections)
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id]
                crop = None
                if extract_crops:
                    xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
                    crop = image[yi1:yi2, xi1:xi2].copy()
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    crop=crop,
                ))
            batch_detections.append(
                sorted(detections, key=lambda d: d.confidence, reverse=True)
            )
        return batch_detections
