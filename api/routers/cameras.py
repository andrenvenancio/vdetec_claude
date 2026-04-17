from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from database.models import Camera

router = APIRouter()


class CameraIn(BaseModel):
    store_id: str
    name: str
    rtsp_url: Optional[str] = None
    location_description: Optional[str] = None


class CameraOut(CameraIn):
    id: str
    active: bool


@router.get("/", response_model=list[CameraOut])
async def list_cameras(store_id: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = select(Camera).where(Camera.active == True)
    if store_id:
        q = q.where(Camera.store_id == store_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
async def create_camera(payload: CameraIn, db: AsyncSession = Depends(get_db)):
    camera = Camera(**payload.model_dump())
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    return camera
