from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from database.session import get_db
from database.models import Product

router = APIRouter()


class ProductIn(BaseModel):
    ean: Optional[str] = None
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None


class ProductOut(ProductIn):
    id: str
    active: bool


@router.get("/", response_model=list[ProductOut])
async def list_products(
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Product).where(Product.active == True)
    if category:
        q = q.where(Product.category == category)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
async def create_product(payload: ProductIn, db: AsyncSession = Depends(get_db)):
    product = Product(**payload.model_dump())
    db.add(product)
    await db.commit()
    await db.refresh(product)
    return product


@router.get("/{product_id}", response_model=ProductOut)
async def get_product(product_id: str, db: AsyncSession = Depends(get_db)):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.get("/ean/{ean}", response_model=ProductOut)
async def get_product_by_ean(ean: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Product).where(Product.ean == ean))
    product = result.scalar_one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
