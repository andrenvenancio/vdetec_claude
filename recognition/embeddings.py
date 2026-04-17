"""
Identificação visual por similaridade de embeddings com FAISS.

Fluxo:
  - Geração de embedding: imagem → vetor (ResNet/ViT via sentence-transformers)
  - Indexação: produto cadastrado gera embedding → armazenado no índice FAISS
  - Busca: embedding do recorte → top-k vizinhos mais próximos
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingHit:
    product_id: str
    product_name: str
    ean: Optional[str]
    score: float   # cosine similarity [0, 1]


class VisualEmbedder:
    MODEL_NAME = "clip-ViT-B-32"   # modelo CLIP multimodal via sentence-transformers

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.MODEL_NAME
        self._model: SentenceTransformer | None = None

    def _load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Retorna vetor normalizado de dimensão 512 (CLIP ViT-B/32)."""
        self._load()
        pil_img = Image.fromarray(crop[..., ::-1])  # BGR → RGB
        emb = self._model.encode(pil_img, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)


class EmbeddingIndex:
    """
    Índice FAISS (Inner-Product sobre vetores normalizados = cosine similarity).

    index.bin   → índice FAISS
    meta.pkl    → lista de dicts com product_id, product_name, ean
    """

    DEFAULT_PATH = Path("./models/faiss_index")

    def __init__(self, index_path: Path | None = None) -> None:
        self._path = index_path or self.DEFAULT_PATH
        self._index: faiss.IndexFlatIP | None = None
        self._meta: list[dict] = []

    def load(self) -> None:
        index_file = self._path / "index.bin"
        meta_file = self._path / "meta.pkl"
        if index_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                self._meta = pickle.load(f)

    def save(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._path / "index.bin"))
        with open(self._path / "meta.pkl", "wb") as f:
            pickle.dump(self._meta, f)

    def add(self, embedding: np.ndarray, product_id: str, product_name: str, ean: str | None) -> None:
        dim = embedding.shape[0]
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(embedding.reshape(1, -1))
        self._meta.append({"product_id": product_id, "product_name": product_name, "ean": ean})

    def search(self, embedding: np.ndarray, top_k: int = 3) -> list[EmbeddingHit]:
        if self._index is None or self._index.ntotal == 0:
            return []
        distances, indices = self._index.search(embedding.reshape(1, -1), top_k)
        hits: list[EmbeddingHit] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta[idx]
            hits.append(EmbeddingHit(
                product_id=meta["product_id"],
                product_name=meta["product_name"],
                ean=meta.get("ean"),
                score=float(dist),
            ))
        return hits
