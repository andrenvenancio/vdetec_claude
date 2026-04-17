"""
Modelos SQLAlchemy (ORM) para o sistema vdetec.

Entidades principais:
  - Store          : farmácia / loja
  - Camera         : câmera instalada na loja
  - Shelf          : prateleira associada a uma câmera
  - Product        : catálogo de produtos
  - ShelfProduct   : produto esperado em determinada posição da prateleira (planograma)
  - Snapshot       : frame capturado por uma câmera
  - DetectionEvent : produto detectado em um snapshot
  - StockAlert     : alerta de ruptura ou divergência de planograma
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class Store(TimestampMixin, Base):
    __tablename__ = "stores"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    cnpj: Mapped[Optional[str]] = mapped_column(String(20), unique=True)
    address: Mapped[Optional[str]] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    cameras: Mapped[list["Camera"]] = relationship(back_populates="store", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera(TimestampMixin, Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    store_id: Mapped[str] = mapped_column(ForeignKey("stores.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    rtsp_url: Mapped[Optional[str]] = mapped_column(Text)   # None = upload manual
    location_description: Mapped[Optional[str]] = mapped_column(String(300))
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    store: Mapped["Store"] = relationship(back_populates="cameras")
    shelves: Mapped[list["Shelf"]] = relationship(back_populates="camera", cascade="all, delete-orphan")
    snapshots: Mapped[list["Snapshot"]] = relationship(back_populates="camera")


# ---------------------------------------------------------------------------
# Shelf / Planograma
# ---------------------------------------------------------------------------
class Shelf(TimestampMixin, Base):
    __tablename__ = "shelves"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    camera: Mapped["Camera"] = relationship(back_populates="shelves")
    planogram_items: Mapped[list["ShelfProduct"]] = relationship(back_populates="shelf", cascade="all, delete-orphan")


class ShelfProduct(TimestampMixin, Base):
    """Produto esperado em determinada prateleira (planograma)."""
    __tablename__ = "shelf_products"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    shelf_id: Mapped[str] = mapped_column(ForeignKey("shelves.id"), nullable=False)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), nullable=False)
    expected_facings: Mapped[int] = mapped_column(Integer, default=1)
    position_x: Mapped[Optional[float]] = mapped_column(Float)  # normalizado [0,1]
    position_y: Mapped[Optional[float]] = mapped_column(Float)

    shelf: Mapped["Shelf"] = relationship(back_populates="planogram_items")
    product: Mapped["Product"] = relationship()


# ---------------------------------------------------------------------------
# Product (catálogo)
# ---------------------------------------------------------------------------
class Product(TimestampMixin, Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    ean: Mapped[Optional[str]] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    brand: Mapped[Optional[str]] = mapped_column(String(100))
    category: Mapped[Optional[str]] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


# ---------------------------------------------------------------------------
# Snapshot (frame capturado)
# ---------------------------------------------------------------------------
class Snapshot(TimestampMixin, Base):
    __tablename__ = "snapshots"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)

    camera: Mapped["Camera"] = relationship(back_populates="snapshots")
    events: Mapped[list["DetectionEvent"]] = relationship(back_populates="snapshot", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# DetectionEvent (produto identificado no snapshot)
# ---------------------------------------------------------------------------
class DetectionEvent(TimestampMixin, Base):
    __tablename__ = "detection_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("snapshots.id"), nullable=False)
    product_id: Mapped[Optional[str]] = mapped_column(ForeignKey("products.id"))

    # Bounding box normalizado [0, 1]
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    detector_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    recognition_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    recognition_method: Mapped[Optional[str]] = mapped_column(String(20))  # barcode|ocr|embedding|unknown

    snapshot: Mapped["Snapshot"] = relationship(back_populates="events")
    product: Mapped[Optional["Product"]] = relationship()


# ---------------------------------------------------------------------------
# StockAlert
# ---------------------------------------------------------------------------
class StockAlert(TimestampMixin, Base):
    __tablename__ = "stock_alerts"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    shelf_id: Mapped[str] = mapped_column(ForeignKey("shelves.id"), nullable=False)
    product_id: Mapped[Optional[str]] = mapped_column(ForeignKey("products.id"))
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Tipos: "stockout" | "low_stock" | "planogram_violation" | "unidentified_product"
    message: Mapped[str] = mapped_column(Text, nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    shelf: Mapped["Shelf"] = relationship()
    product: Mapped[Optional["Product"]] = relationship()
