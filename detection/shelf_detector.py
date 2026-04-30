"""
Detector baseado na estrutura de prateleiras do planograma.

Em vez de YOLO-World (zero-shot), projeta a estrutura conhecida do planograma
(N prateleiras, proporções por Horiz_F) sobre a foto real da gôndola.
Mais robusto para gôndolas organizadas onde o layout segue o planograma.
"""
from __future__ import annotations

import cv2
import numpy as np

from detection.detector import Detection
from planogram.loader import ManifestItem, _detect_shelf_y_bands


class ShelfStructureDetector:
    """
    Detecta regiões de produto projetando a estrutura do planograma na gôndola.

    Para cada prateleira:
      1. Localiza a faixa vertical (Y) via projeção de variância.
      2. Divide a largura proporcionalmente aos facings (Horiz_F) de cada SKU.
      3. Retorna um Detection por SKU com o crop correspondente.
    """

    def __init__(
        self,
        manifest: list[ManifestItem],
        n_shelves: int,
    ) -> None:
        self._n_shelves = n_shelves
        self._items_by_shelf: dict[int, list[ManifestItem]] = {}
        for m in manifest:
            if m.shelf is not None:
                self._items_by_shelf.setdefault(m.shelf, []).append(m)
        for shelf in self._items_by_shelf:
            self._items_by_shelf[shelf].sort(key=lambda m: m.location)

    def predict(self, image: np.ndarray) -> list[Detection]:
        shelf_bands = _detect_shelf_y_bands(image, n_shelves=self._n_shelves)
        w = image.shape[1]
        detections: list[Detection] = []

        for shelf_num, (y1, y2) in enumerate(shelf_bands, 1):
            shelf_items = self._items_by_shelf.get(shelf_num, [])
            if not shelf_items:
                continue
            total_f = sum(m.horiz_f for m in shelf_items)
            x_cursor = 0
            for m in shelf_items:
                x1 = x_cursor
                x2 = x_cursor + round(m.horiz_f / total_f * w)
                x_cursor = x2
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue
                detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=1.0,
                    class_name="shelf_cell",
                    crop=crop,
                    location=m.location,
                ))

        return detections
