"""
Identificação visual por similaridade de embeddings com FAISS.
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
    score: float


class VisualEmbedder:
    MODEL_NAME = "clip-ViT-B-32"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.MODEL_NAME
        self._model: SentenceTransformer | None = None

    def _load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        self._load()
        pil_img = Image.fromarray(crop[..., ::-1])  # BGR → RGB
        emb = self._model.encode(pil_img, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)


class EmbeddingIndex:
    """
    Índice FAISS (Inner-Product = cosine similarity sobre vetores normalizados).

    Arquivos em cache:
        index.bin  — vetores FAISS
        meta.pkl   — metadados de cada entrada (product_id, name, ean, + extras)
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._meta: list[dict] = []

    # ── Persistência ───────────────────────────────────────────────────── #

    def load(self, cache_dir: Path) -> bool:
        """Carrega índice do disco. Retorna True se bem-sucedido."""
        index_file = cache_dir / "index.bin"
        meta_file  = cache_dir / "meta.pkl"
        if not (index_file.exists() and meta_file.exists()):
            return False
        self._index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            self._meta = pickle.load(f)
        return True

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(cache_dir / "index.bin"))
        with open(cache_dir / "meta.pkl", "wb") as f:
            pickle.dump(self._meta, f)

    # ── Escrita ────────────────────────────────────────────────────────── #

    def add(
        self,
        embedding: np.ndarray,
        product_id: str,
        product_name: str,
        ean: str | None,
        **extra,
    ) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(embedding.shape[0])
        self._index.add(embedding.reshape(1, -1))
        self._meta.append({"product_id": product_id, "name": product_name, "ean": ean, **extra})

    # ── Leitura ────────────────────────────────────────────────────────── #

    def search(self, embedding: np.ndarray, top_k: int = 1) -> list[EmbeddingHit]:
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
                product_name=meta["name"],
                ean=meta.get("ean"),
                score=float(dist),
            ))
        return hits

    def search_by_ean(self, ean: str) -> EmbeddingHit | None:
        for meta in self._meta:
            if meta.get("ean") == ean:
                return EmbeddingHit(
                    product_id=meta["product_id"],
                    product_name=meta["name"],
                    ean=meta["ean"],
                    score=1.0,
                )
        return None

    @property
    def size(self) -> int:
        return len(self._meta)
