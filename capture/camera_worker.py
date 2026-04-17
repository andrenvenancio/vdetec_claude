"""
Serviço de captura de frames de câmeras IP (RTSP).

- Roda como processo independente (container `capture`)
- Descobre câmeras ativas via API
- Captura frame a cada CAMERA_POLL_INTERVAL_SEC segundos
- Envia frame para o endpoint POST /api/v1/snapshots
"""
from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass

import cv2
import httpx
import structlog

from config.settings import settings

log = structlog.get_logger()


@dataclass
class CameraConfig:
    id: str
    name: str
    rtsp_url: str


class RTSPCapture:
    """Captura um frame de um stream RTSP com timeout."""

    def __init__(self, rtsp_url: str, timeout_sec: int = 10) -> None:
        self._url = rtsp_url
        self._timeout = timeout_sec

    def grab(self) -> bytes | None:
        cap = cv2.VideoCapture(self._url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout * 1000)

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return None

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()


class CameraWorker:
    def __init__(self, api_url: str | None = None) -> None:
        self._api_url = (api_url or settings.__dict__.get("api_url", "http://api:8000")).rstrip("/")
        self._interval = settings.camera_poll_interval_sec
        self._client = httpx.AsyncClient(timeout=30)

    async def _fetch_active_cameras(self) -> list[CameraConfig]:
        try:
            resp = await self._client.get(f"{self._api_url}/api/v1/cameras/")
            resp.raise_for_status()
            data = resp.json()
            return [
                CameraConfig(id=c["id"], name=c["name"], rtsp_url=c["rtsp_url"])
                for c in data
                if c.get("rtsp_url") and c.get("active")
            ]
        except Exception as exc:
            log.error("fetch_cameras_failed", error=str(exc))
            return []

    async def _process_camera(self, cam: CameraConfig) -> None:
        log.info("capturing", camera=cam.name)
        capture = RTSPCapture(cam.rtsp_url, timeout_sec=settings.rtsp_timeout_sec)

        # Executa captura em thread pool (cv2 é bloqueante)
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, capture.grab)

        if raw is None:
            log.warning("capture_failed", camera=cam.name)
            return

        try:
            resp = await self._client.post(
                f"{self._api_url}/api/v1/snapshots/",
                data={"camera_id": cam.id},
                files={"image": ("snapshot.jpg", raw, "image/jpeg")},
            )
            resp.raise_for_status()
            result = resp.json()
            log.info(
                "snapshot_processed",
                camera=cam.name,
                snapshot_id=result["id"],
                detections=result["total_detections"],
            )
        except Exception as exc:
            log.error("snapshot_upload_failed", camera=cam.name, error=str(exc))

    async def run(self) -> None:
        log.info("camera_worker_started", interval_sec=self._interval)
        while True:
            cameras = await self._fetch_active_cameras()
            if cameras:
                await asyncio.gather(*[self._process_camera(c) for c in cameras])
            await asyncio.sleep(self._interval)


if __name__ == "__main__":
    asyncio.run(CameraWorker().run())
