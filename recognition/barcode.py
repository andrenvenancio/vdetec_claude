"""Leitura de código de barras e QR Code usando pyzbar."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from pyzbar import pyzbar


@dataclass
class BarcodeMatch:
    ean: str
    product_id: Optional[str] = None
    product_name: Optional[str] = None


class BarcodeReader:
    """
    Tenta decodificar código de barras (EAN-13, EAN-8, QR Code, etc.)
    no recorte de imagem fornecido.
    """

    def read(self, crop: np.ndarray) -> Optional[BarcodeMatch]:
        # Pré-processamento: escala de cinza + nitidez
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        codes = pyzbar.decode(gray)
        if not codes:
            # Segunda tentativa com imagem redimensionada (código muito pequeno)
            enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            codes = pyzbar.decode(enlarged)

        if not codes:
            return None

        # Prioriza EAN-13 sobre outros tipos
        ean_codes = [c for c in codes if c.type in ("EAN13", "EAN8")]
        chosen = ean_codes[0] if ean_codes else codes[0]
        ean = chosen.data.decode("utf-8").strip()

        return BarcodeMatch(ean=ean)
