"""Ponto de entrada da API FastAPI."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.routers import products, cameras, snapshots, alerts, stores

app = FastAPI(
    title="vdetec API",
    description="Identificação de produtos em prateleiras de farmácias",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stores.router,    prefix="/api/v1/stores",    tags=["Stores"])
app.include_router(cameras.router,   prefix="/api/v1/cameras",   tags=["Cameras"])
app.include_router(products.router,  prefix="/api/v1/products",  tags=["Products"])
app.include_router(snapshots.router, prefix="/api/v1/snapshots", tags=["Snapshots"])
app.include_router(alerts.router,    prefix="/api/v1/alerts",    tags=["Alerts"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
