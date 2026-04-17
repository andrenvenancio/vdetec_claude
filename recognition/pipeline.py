"""
Pipeline de reconhecimento de produto a partir de um recorte detectado.

Estratégia em cascata (ordem de confiança):
  1. Leitura de código de barras / QR Code  → identificação exata
  2. OCR no rótulo                           → match por nome/EAN no catálogo
  3. Similaridade visual (embeddings FAISS)  → fallback visual

Retorna o produto identificado ou None se abaixo do limiar.
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
    method: str                  # "barcode" | "ocr" | "embedding" | "unknown"
    confidence: float


class RecognitionPipeline:
    """
    Orquestra as três estratégias de reconhecimento em cascata.
    """

    BARCODE_CONF = 1.0
    OCR_CONF_THRESHOLD = 0.80
    EMBEDDING_CONF_THRESHOLD = 0.70

    def __init__(
        self,
        barcode_reader: BarcodeReader | None = None,
        ocr: LabelOCR | None = None,
        embedder: VisualEmbedder | None = None,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self._barcode = barcode_reader or BarcodeReader()
        self._ocr = ocr or LabelOCR()
        self._embedder = embedder or VisualEmbedder()
        self._index = embedding_index or EmbeddingIndex()

    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        # 1. Barcode
        barcode_result = self._barcode.read(crop)
        if barcode_result:
            return RecognitionResult(
                product_id=barcode_result.product_id,
                product_name=barcode_result.product_name,
                ean=barcode_result.ean,
                method="barcode",
                confidence=self.BARCODE_CONF,
            )

        # 2. OCR
        ocr_result = self._ocr.read(crop)
        if ocr_result and ocr_result.confidence >= self.OCR_CONF_THRESHOLD:
            return RecognitionResult(
                product_id=ocr_result.product_id,
                product_name=ocr_result.product_name,
                ean=ocr_result.ean,
                method="ocr",
                confidence=ocr_result.confidence,
            )

        # 3. Embedding visual
        emb = self._embedder.embed(crop)
        emb_result = self._index.search(emb, top_k=1)
        if emb_result and emb_result[0].score >= self.EMBEDDING_CONF_THRESHOLD:
            best = emb_result[0]
            return RecognitionResult(
                product_id=best.product_id,
                product_name=best.product_name,
                ean=best.ean,
                method="embedding",
                confidence=best.score,
            )

        return RecognitionResult(
            product_id=None,
            product_name=None,
            ean=None,
            method="unknown",
            confidence=0.0,
        )
