"""
Carrega o planograma e constrói o índice FAISS de embeddings CLIP.

Formatos aceitos
────────────────
1. Pasta de imagens
   Cada arquivo é um produto. O nome do arquivo vira o nome do produto.
   Suporta EAN no nome: "7891234567890_Dipirona 500mg.jpg"
   Suporta somente nome:  "Paracetamol 750mg.jpg"

2. JSON
   [
     {"name": "Dipirona 500mg", "ean": "7891234567890", "image": "imgs/dipi.jpg"},
     ...
   ]

3. CSV
   name,ean,image
   Dipirona 500mg,7891234567890,imgs/dipi.jpg

Uso rápido:
    index = PlanogramLoader.from_folder("planogram/produtos/")
    index = PlanogramLoader.from_json("planogram.json")
    index = PlanogramLoader.from_csv("planogram.csv")
"""
from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from tqdm import tqdm

from recognition.embeddings import VisualEmbedder, EmbeddingIndex

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_EAN_RE = re.compile(r"^(\d{8}|\d{13})")


@dataclass
class PlanogramItem:
    product_id: str
    name: str
    ean: str | None
    image_path: Path
    embedding: np.ndarray | None = field(default=None, repr=False)


class PlanogramLoader:
    """
    Constrói um EmbeddingIndex a partir de imagens de referência do planograma.
    """

    def __init__(self) -> None:
        self._embedder = VisualEmbedder()
        self._index = EmbeddingIndex()
        self._items: list[PlanogramItem] = []

    # ------------------------------------------------------------------ #
    # Construtores de conveniência                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_folder(cls, folder: str | Path) -> "PlanogramLoader":
        """Lê todos os arquivos de imagem de uma pasta."""
        loader = cls()
        folder = Path(folder)
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not images:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {folder}")
        items = [_parse_filename(p) for p in images]
        loader._build(items)
        return loader

    @classmethod
    def from_json(cls, json_path: str | Path) -> "PlanogramLoader":
        """Lê manifesto JSON."""
        loader = cls()
        json_path = Path(json_path)
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        base = json_path.parent
        items = [
            PlanogramItem(
                product_id=r.get("product_id") or str(uuid.uuid4()),
                name=r["name"],
                ean=r.get("ean"),
                image_path=base / r["image"],
            )
            for r in raw
        ]
        loader._build(items)
        return loader

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "PlanogramLoader":
        """Lê manifesto CSV com colunas: name, ean (opcional), image."""
        loader = cls()
        csv_path = Path(csv_path)
        base = csv_path.parent
        items = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                items.append(PlanogramItem(
                    product_id=row.get("product_id") or str(uuid.uuid4()),
                    name=row["name"],
                    ean=row.get("ean") or None,
                    image_path=base / row["image"],
                ))
        loader._build(items)
        return loader

    # ------------------------------------------------------------------ #
    # Acesso ao índice                                                     #
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> EmbeddingIndex:
        return self._index

    @property
    def items(self) -> list[PlanogramItem]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------ #
    # Construção interna                                                   #
    # ------------------------------------------------------------------ #

    def _build(self, items: list[PlanogramItem]) -> None:
        print(f"Indexando {len(items)} produto(s)...")
        for item in tqdm(items, unit="produto"):
            img = cv2.imread(str(item.image_path))
            if img is None:
                print(f"  [aviso] Imagem não encontrada: {item.image_path}")
                continue
            emb = self._embedder.embed(img)
            item.embedding = emb
            self._index.add(emb, item.product_id, item.name, item.ean)
            self._items.append(item)
        print(f"Índice pronto: {len(self._items)} produto(s) indexado(s).")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _parse_filename(path: Path) -> PlanogramItem:
    """
    Infere nome e EAN a partir do nome do arquivo.
    Exemplos:
      "7891234567890_Dipirona 500mg.jpg"  → ean=789..., name="Dipirona 500mg"
      "Paracetamol 750mg.jpg"             → ean=None,   name="Paracetamol 750mg"
    """
    stem = path.stem
    ean_match = _EAN_RE.match(stem)
    if ean_match:
        ean = ean_match.group(1)
        name = stem[len(ean):].lstrip("_- ") or stem
    else:
        ean = None
        name = stem
    return PlanogramItem(
        product_id=str(uuid.uuid4()),
        name=name,
        ean=ean,
        image_path=path,
    )
