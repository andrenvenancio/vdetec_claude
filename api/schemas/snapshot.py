from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DetectionEventOut(BaseModel):
    id: str
    product_id: Optional[str]
    detector_confidence: float
    recognition_confidence: float
    recognition_method: Optional[str]
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 normalizados


class SnapshotResponse(BaseModel):
    id: str
    camera_id: str
    captured_at: datetime
    total_detections: int
    events: list[DetectionEventOut]
