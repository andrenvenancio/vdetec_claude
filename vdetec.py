"""
Fachada principal do sistema vdetec.

Uso mínimo:
    from vdetec import VDetec

    vd = VDetec()
    vd.load_planogram("planogram/produtos/")   # ou .json / .csv

    result = vd.identify("foto_prateleira.jpg")
    for det in result.detections:
        print(det.product_name, det.confidence, det.method)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detection.detector import ProductDetector, Detection
from recognition.pipeline import RecognitionPipeline, RecognitionResult
from planogram.loader import PlanogramLoader


# ──────────────────────────────────────────────────────────────────────────────
# Tipos de resultado
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProductMatch:
    """Um produto identificado na imagem real."""
    product_id: Optional[str]
    product_name: Optional[str]
    ean: Optional[str]
    method: str                               # barcode | ocr | clip | unknown
    confidence: float
    detector_confidence: float
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 em pixels
    crop: np.ndarray = field(repr=False)


@dataclass
class IdentificationResult:
    image_path: str
    image_shape: tuple[int, int]              # (height, width)
    detections: list[ProductMatch]
    planogram_size: int                       # produtos no índice

    @property
    def identified(self) -> list[ProductMatch]:
        return [d for d in self.detections if d.product_id]

    @property
    def unidentified(self) -> list[ProductMatch]:
        return [d for d in self.detections if not d.product_id]

    def summary(self) -> str:
        lines = [
            f"Imagem      : {self.image_path}",
            f"Detecções   : {len(self.detections)}",
            f"Identificados: {len(self.identified)}",
            f"Não ident.  : {len(self.unidentified)}",
            f"Planograma  : {self.planogram_size} produto(s)",
            "",
        ]
        for i, d in enumerate(self.detections, 1):
            nome = d.product_name or "???"
            ean = f"  EAN {d.ean}" if d.ean else ""
            lines.append(
                f"  [{i:02d}] {nome}{ean}"
                f"  | método={d.method}"
                f"  | conf={d.confidence:.2f}"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Fachada
# ──────────────────────────────────────────────────────────────────────────────

class VDetec:
    def __init__(
        self,
        detector_conf: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self._detector = ProductDetector(conf=detector_conf, device=device)
        self._pipeline: RecognitionPipeline | None = None
        self._planogram: PlanogramLoader | None = None

    # ── Carregamento do planograma ─────────────────────────────────────────

    def load_planogram(self, source: str | Path) -> "VDetec":
        """
        Carrega planograma a partir de:
          - pasta de imagens  (ex: "planogram/")
          - arquivo JSON      (ex: "planogram.json")
          - arquivo CSV       (ex: "planogram.csv")
        """
        source = Path(source)
        if source.is_dir():
            self._planogram = PlanogramLoader.from_folder(source)
        elif source.suffix.lower() == ".json":
            self._planogram = PlanogramLoader.from_json(source)
        elif source.suffix.lower() == ".csv":
            self._planogram = PlanogramLoader.from_csv(source)
        else:
            raise ValueError(f"Fonte não reconhecida: {source}")

        self._pipeline = RecognitionPipeline(self._planogram.index)
        return self

    # ── Identificação ──────────────────────────────────────────────────────

    def identify(self, image: str | Path | np.ndarray) -> IdentificationResult:
        """
        Detecta e identifica produtos em uma imagem de prateleira.

        Args:
            image: caminho para arquivo de imagem ou array BGR (OpenCV).

        Returns:
            IdentificationResult com lista de ProductMatch.
        """
        if self._pipeline is None:
            raise RuntimeError("Chame load_planogram() antes de identify().")

        if isinstance(image, (str, Path)):
            path_str = str(image)
            frame = cv2.imread(path_str)
            if frame is None:
                raise FileNotFoundError(f"Imagem não encontrada: {image}")
        else:
            path_str = "<array>"
            frame = image

        h, w = frame.shape[:2]
        detections: list[Detection] = self._detector.predict(frame)

        matches: list[ProductMatch] = []
        for det in detections:
            recog: RecognitionResult = self._pipeline.recognize(det.crop)
            matches.append(ProductMatch(
                product_id=recog.product_id,
                product_name=recog.product_name,
                ean=recog.ean,
                method=recog.method,
                confidence=recog.confidence,
                detector_confidence=det.confidence,
                bbox=det.bbox,
                crop=det.crop,
            ))

        return IdentificationResult(
            image_path=path_str,
            image_shape=(h, w),
            detections=matches,
            planogram_size=len(self._planogram),
        )

    def identify_batch(
        self, images: list[str | Path | np.ndarray]
    ) -> list[IdentificationResult]:
        return [self.identify(img) for img in images]
