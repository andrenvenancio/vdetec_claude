"""
Regras de negócio para geração de alertas de prateleira.

Tipos de alerta:
  - stockout           : produto esperado no planograma não detectado
  - low_stock          : facings detectados abaixo do esperado
  - planogram_violation: produto em posição errada
  - unidentified       : produto detectado mas não reconhecido
"""
from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import select

from database.models import Shelf, ShelfProduct, DetectionEvent, StockAlert


def evaluate_planogram(
    db: Session,
    shelf: Shelf,
    events: list[DetectionEvent],
) -> list[StockAlert]:
    """
    Compara eventos detectados com o planograma da prateleira.
    Retorna lista de StockAlert a persistir (sem commit).
    """
    alerts: list[StockAlert] = []

    planogram_items: list[ShelfProduct] = db.execute(
        select(ShelfProduct).where(ShelfProduct.shelf_id == shelf.id)
    ).scalars().all()

    if not planogram_items:
        return alerts

    # Mapeia product_id → contagem de detecções
    detected_counts: dict[str, int] = {}
    for event in events:
        if event.product_id:
            detected_counts[event.product_id] = detected_counts.get(event.product_id, 0) + 1

    for item in planogram_items:
        count = detected_counts.get(item.product_id, 0)

        if count == 0:
            alerts.append(StockAlert(
                shelf_id=shelf.id,
                product_id=item.product_id,
                alert_type="stockout",
                message=(
                    f"Produto ausente na prateleira '{shelf.name}'. "
                    f"Esperado: {item.expected_facings} facing(s)."
                ),
            ))
        elif count < item.expected_facings:
            alerts.append(StockAlert(
                shelf_id=shelf.id,
                product_id=item.product_id,
                alert_type="low_stock",
                message=(
                    f"Estoque baixo na prateleira '{shelf.name}'. "
                    f"Detectado: {count} / Esperado: {item.expected_facings} facing(s)."
                ),
            ))

    # Produtos detectados mas não reconhecidos
    unidentified = [e for e in events if not e.product_id]
    if unidentified:
        alerts.append(StockAlert(
            shelf_id=shelf.id,
            product_id=None,
            alert_type="unidentified_product",
            message=(
                f"{len(unidentified)} produto(s) não identificado(s) "
                f"na prateleira '{shelf.name}'."
            ),
        ))

    return alerts
