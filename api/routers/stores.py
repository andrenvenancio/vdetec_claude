from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from database.models import Store

router = APIRouter()


class StoreIn(BaseModel):
    name: str
    cnpj: Optional[str] = None
    address: Optional[str] = None


class StoreOut(StoreIn):
    id: str
    active: bool


@router.get("/", response_model=list[StoreOut])
async def list_stores(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Store).where(Store.active == True))
    return result.scalars().all()


@router.post("/", response_model=StoreOut, status_code=status.HTTP_201_CREATED)
async def create_store(payload: StoreIn, db: AsyncSession = Depends(get_db)):
    store = Store(**payload.model_dump())
    db.add(store)
    await db.commit()
    await db.refresh(store)
    return store
