"""
Endpoint de upload de imagem → dispara pipeline de detecção + reconhecimento.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database.session import get_db
from database.models import Camera, Snapshot, DetectionEvent
from detection.detector import ProductDetector
from recognition.pipeline import RecognitionPipeline
from api.schemas.snapshot import SnapshotResponse, DetectionEventOut

router = APIRouter()

_detector = ProductDetector()
_recognizer = RecognitionPipeline()


@router.post("/", response_model=SnapshotResponse, status_code=status.HTTP_201_CREATED)
async def upload_snapshot(
    camera_id: str = Form(...),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Recebe frame de uma câmera, persiste e processa:
    1. Salva imagem no storage
    2. Roda detector YOLO
    3. Para cada detecção, roda pipeline de reconhecimento
    4. Persiste eventos no banco
    5. Dispara verificação de alertas (tarefa Celery)
    """
    camera = await db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Salva imagem
    raw = await image.read()
    filename = f"{uuid.uuid4()}.jpg"
    dest = Path(settings.media_root) / "snapshots" / camera_id
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / filename
    filepath.write_bytes(raw)

    # Decodifica para OpenCV
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w = frame.shape[:2]

    # Persiste snapshot
    snapshot = Snapshot(
        camera_id=camera_id,
        image_path=str(filepath),
        captured_at=datetime.now(timezone.utc),
    )
    db.add(snapshot)
    await db.flush()

    # Detecção
    detections = _detector.predict(frame, extract_crops=True)

    events: list[DetectionEvent] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        recog = _recognizer.recognize(det.crop) if det.crop is not None else None

        event = DetectionEvent(
            snapshot_id=snapshot.id,
            product_id=recog.product_id if recog else None,
            bbox_x1=x1 / w,
            bbox_y1=y1 / h,
            bbox_x2=x2 / w,
            bbox_y2=y2 / h,
            detector_confidence=det.confidence,
            recognition_confidence=recog.confidence if recog else 0.0,
            recognition_method=recog.method if recog else "unknown",
        )
        db.add(event)
        events.append(event)

    snapshot.processed = True
    await db.commit()
    await db.refresh(snapshot)

    # Dispara verificação assíncrona de alertas
    from alerts.tasks import check_shelf_alerts
    check_shelf_alerts.delay(snapshot.id)

    return SnapshotResponse(
        id=snapshot.id,
        camera_id=camera_id,
        captured_at=snapshot.captured_at,
        total_detections=len(events),
        events=[
            DetectionEventOut(
                id=e.id,
                product_id=e.product_id,
                detector_confidence=e.detector_confidence,
                recognition_confidence=e.recognition_confidence,
                recognition_method=e.recognition_method,
                bbox=(e.bbox_x1, e.bbox_y1, e.bbox_x2, e.bbox_y2),
            )
            for e in events
        ],
    )
