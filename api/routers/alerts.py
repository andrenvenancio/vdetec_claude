from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from database.models import StockAlert

router = APIRouter()


class AlertOut(BaseModel):
    id: str
    shelf_id: str
    product_id: Optional[str]
    alert_type: str
    message: str
    resolved: bool
    created_at: datetime


@router.get("/", response_model=list[AlertOut])
async def list_alerts(
    resolved: Optional[bool] = None,
    alert_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(StockAlert).order_by(StockAlert.created_at.desc())
    if resolved is not None:
        q = q.where(StockAlert.resolved == resolved)
    if alert_type:
        q = q.where(StockAlert.alert_type == alert_type)
    result = await db.execute(q)
    return result.scalars().all()


@router.patch("/{alert_id}/resolve", response_model=AlertOut)
async def resolve_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    alert = await db.get(StockAlert, alert_id)
    if not alert:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.resolved = True
    alert.resolved_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(alert)
    return alert
