"""OCR em rótulos de produtos usando EasyOCR."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import easyocr


@dataclass
class OCRMatch:
    raw_text: str
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    ean: Optional[str] = None
    confidence: float = 0.0


_EAN_RE = re.compile(r"\b(\d{13}|\d{8})\b")


class LabelOCR:
    """
    Extrai texto de rótulos e tenta identificar o produto via:
      - EAN presente no texto
      - Fuzzy match do nome no catálogo (delegado ao repositório)
    """

    def __init__(self, languages: list[str] | None = None) -> None:
        self._languages = languages or ["pt", "en"]
        self._reader: easyocr.Reader | None = None

    def _load(self) -> None:
        if self._reader is None:
            self._reader = easyocr.Reader(self._languages, gpu=True)

    def read(self, crop: np.ndarray) -> Optional[OCRMatch]:
        self._load()
        results = self._reader.readtext(crop, detail=1, paragraph=False)
        if not results:
            return None

        texts = []
        total_conf = 0.0
        for (_bbox, text, conf) in results:
            texts.append(text)
            total_conf += conf

        raw = " ".join(texts)
        avg_conf = total_conf / len(results) if results else 0.0

        # Tenta extrair EAN do texto
        ean_match = _EAN_RE.search(raw.replace(" ", ""))
        ean = ean_match.group(1) if ean_match else None

        return OCRMatch(
            raw_text=raw,
            ean=ean,
            confidence=avg_conf,
        )
