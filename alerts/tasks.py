"""
Tarefas Celery para análise de prateleira e disparo de alertas.

Fluxo após um snapshot ser processado:
  1. Carrega eventos do snapshot
  2. Compara com planograma da prateleira
  3. Gera alertas: ruptura, baixo estoque, produto fora de posição
  4. Notifica canais configurados (e-mail, Telegram)
"""
from __future__ import annotations

from celery import Celery
from config.settings import settings

celery_app = Celery("vdetec", broker=settings.celery_broker_url)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_backend = settings.redis_url


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def check_shelf_alerts(self, snapshot_id: str) -> dict:
    """
    Verifica se há alertas a disparar com base no snapshot processado.
    Roda de forma síncrona dentro do worker Celery (sem async).
    """
    import sqlalchemy
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine, select

    # Converte URL async → sync para uso no worker síncrono
    sync_url = settings.database_url.replace("+asyncpg", "")
    engine = create_engine(sync_url, echo=False)

    with Session(engine) as db:
        from database.models import Snapshot, DetectionEvent, ShelfProduct, StockAlert, Camera, Shelf
        from alerts.rules import evaluate_planogram

        snapshot = db.get(Snapshot, snapshot_id)
        if not snapshot:
            return {"error": "snapshot not found"}

        # Descobre prateleiras associadas à câmera
        camera = db.get(Camera, snapshot.camera_id)
        shelves = db.execute(
            select(Shelf).where(Shelf.camera_id == camera.id)
        ).scalars().all()

        events = db.execute(
            select(DetectionEvent).where(DetectionEvent.snapshot_id == snapshot_id)
        ).scalars().all()

        new_alerts: list[StockAlert] = []
        for shelf in shelves:
            shelf_alerts = evaluate_planogram(db, shelf, events)
            for alert in shelf_alerts:
                db.add(alert)
                new_alerts.append(alert)

        db.commit()

    # Notificações
    if new_alerts:
        for alert in new_alerts:
            notify_alert.delay(alert.id)

    return {"snapshot_id": snapshot_id, "alerts_generated": len(new_alerts)}


@celery_app.task
def notify_alert(alert_id: str) -> None:
    """Envia notificação para os canais configurados."""
    import sqlalchemy
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine

    sync_url = settings.database_url.replace("+asyncpg", "")
    engine = create_engine(sync_url, echo=False)

    with Session(engine) as db:
        from database.models import StockAlert
        alert = db.get(StockAlert, alert_id)
        if not alert:
            return

    from alerts.channels.telegram import send_telegram
    from alerts.channels.email import send_email

    message = f"[{alert.alert_type.upper()}] {alert.message}"

    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_telegram(message)

    if settings.smtp_host:
        send_email(subject=f"vdetec Alert: {alert.alert_type}", body=message)
