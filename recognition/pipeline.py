"""
Pipeline de reconhecimento em cascata.

Dado um crop de produto detectado, tenta identificá-lo em ordem:
  1. Barcode (pyzbar)   → hit no índice por EAN
  2. OCR (EasyOCR)      → hit no índice por EAN extraído do texto
  3. CLIP similarity    → vizinho mais próximo no índice FAISS
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from recognition.barcode import BarcodeReader
from recognition.ocr import LabelOCR
from recognition.embeddings import VisualEmbedder, EmbeddingIndex


@dataclass
class RecognitionResult:
    product_id: Optional[str]
    product_name: Optional[str]
    ean: Optional[str]
    method: str       # "barcode" | "ocr" | "clip" | "unknown"
    confidence: float


class RecognitionPipeline:
    CLIP_THRESHOLD = 0.65

    def __init__(self, index: EmbeddingIndex) -> None:
        self._index = index
        self._barcode = BarcodeReader()
        self._ocr = LabelOCR()
        self._embedder = VisualEmbedder()

    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        # 1. Barcode
        bc = self._barcode.read(crop)
        if bc:
            hit = self._index.search_by_ean(bc.ean)
            if hit:
                return RecognitionResult(
                    product_id=hit.product_id,
                    product_name=hit.product_name,
                    ean=bc.ean,
                    method="barcode",
                    confidence=1.0,
                )

        # 2. OCR → EAN no texto
        ocr = self._ocr.read(crop)
        if ocr and ocr.ean:
            hit = self._index.search_by_ean(ocr.ean)
            if hit:
                return RecognitionResult(
                    product_id=hit.product_id,
                    product_name=hit.product_name,
                    ean=ocr.ean,
                    method="ocr",
                    confidence=0.95,
                )

        # 3. CLIP visual similarity
        emb = self._embedder.embed(crop)
        hits = self._index.search(emb, top_k=1)
        if hits and hits[0].score >= self.CLIP_THRESHOLD:
            best = hits[0]
            return RecognitionResult(
                product_id=best.product_id,
                product_name=best.product_name,
                ean=best.ean,
                method="clip",
                confidence=best.score,
            )

        return RecognitionResult(
            product_id=None, product_name=None, ean=None,
            method="unknown", confidence=0.0,
        )
