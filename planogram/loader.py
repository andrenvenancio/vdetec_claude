"""
Carrega o planograma e constrói o índice FAISS de embeddings CLIP.

Formatos aceitos
────────────────
1. Imagem + CSV de capacidade por prateleira  ← formato adocante
   PlanogramLoader.from_image("planogram.png", "dist.csv")
   O CSV tem colunas: Prateleira, SKU_limite
   Os produtos são detectados automaticamente na imagem via YOLO-World.

2. Pasta de imagens individuais
   Cada arquivo é um produto. Nome do arquivo → nome do produto.
   Suporta EAN no prefixo: "7891234567890_Dipirona 500mg.jpg"

3. JSON  →  [{"name": ..., "ean": ..., "image": ...}]

4. CSV de manifesto  →  name, ean, image

Uso rápido:
    loader = PlanogramLoader.from_image("planogram.png", "dist.csv")
    loader = PlanogramLoader.from_folder("planogram/produtos/")
    loader = PlanogramLoader.from_json("planogram.json")
"""
from __future__ import annotations

import csv
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from recognition.embeddings import VisualEmbedder, EmbeddingIndex

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_EAN_RE = re.compile(r"^(\d{8}|\d{13})")


@dataclass
class ShelfCapacity:
    shelf: int
    sku_limit: int


@dataclass
class PlanogramItem:
    product_id: str
    name: str
    ean: str | None
    shelf: int | None                          # prateleira de origem (1-based)
    bbox: tuple[int, int, int, int] | None     # x1,y1,x2,y2 no planograma
    image_path: Path | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)


class PlanogramLoader:
    """
    Constrói um EmbeddingIndex a partir do planograma.
    """

    def __init__(self) -> None:
        self._embedder = VisualEmbedder()
        self._index = EmbeddingIndex()
        self._items: list[PlanogramItem] = []
        self._shelf_capacities: list[ShelfCapacity] = []

    # ------------------------------------------------------------------ #
    # Construtores                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_image(
        cls,
        image_path: str | Path,
        dist_csv: str | Path,
        conf: float = 0.20,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ) -> "PlanogramLoader":
        """
        Detecta produtos na imagem do planograma e indexa com CLIP.
        Na primeira execução gera e salva o índice em cache_dir.
        Nas execuções seguintes carrega do cache — sem rodar CLIP novamente.

        Args:
            image_path : imagem PNG/JPG do planograma.
            dist_csv   : CSV com colunas "Prateleira" e "SKU_limite".
            cache_dir  : pasta para salvar/carregar o índice.
                         Padrão: mesma pasta da imagem, subpasta ".vdetec_cache".
        """
        loader = cls()
        image_path = Path(image_path)
        dist_csv   = Path(dist_csv)

        cache_dir = Path(cache_dir) if cache_dir else image_path.parent / ".vdetec_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        loader._shelf_capacities = _read_dist_csv(dist_csv)

        # Tenta carregar do cache
        if loader._index.load(cache_dir):
            print(f"Índice carregado do cache: {len(loader._index._meta)} produto(s)  [{cache_dir}]")
            # Reconstrói _items a partir dos metadados do índice (sem embeddings)
            for meta in loader._index._meta:
                loader._items.append(PlanogramItem(
                    product_id=meta["product_id"],
                    name=meta["name"],
                    ean=meta.get("ean"),
                    shelf=meta.get("shelf"),
                    bbox=tuple(meta["bbox"]) if meta.get("bbox") else None,
                    image_path=image_path,
                ))
            return loader

        # Cache não existe — gera índice
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        print(f"Detectando produtos no planograma: {image_path.name}")
        detections = _detect_planogram_products(frame, conf=conf, device=device)
        print(f"  {len(detections)} produto(s) detectado(s)")

        shelf_bands = _compute_shelf_bands(frame.shape[0], loader._shelf_capacities)

        items: list[tuple[PlanogramItem, np.ndarray]] = []
        for i, (bbox, _) in enumerate(detections, 1):
            x1, y1, x2, y2 = bbox
            shelf_num = _bbox_to_shelf((y1 + y2) / 2, shelf_bands)
            crop = frame[y1:y2, x1:x2].copy()
            item = PlanogramItem(
                product_id=str(uuid.uuid4()),
                name=f"produto_{i:03d}_prat{shelf_num}",
                ean=None,
                shelf=shelf_num,
                bbox=bbox,
                image_path=image_path,
            )
            items.append((item, crop))

        loader._build_from_crops(items, extra_meta_keys=["shelf", "bbox"])
        loader._index.save(cache_dir)
        print(f"Índice salvo em cache: {cache_dir}")
        return loader

    @classmethod
    def from_folder(cls, folder: str | Path) -> "PlanogramLoader":
        loader = cls()
        folder = Path(folder)
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not images:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {folder}")
        items = [_parse_filename(p) for p in images]
        loader._build_from_files(items)
        return loader

    @classmethod
    def from_json(cls, json_path: str | Path) -> "PlanogramLoader":
        loader = cls()
        json_path = Path(json_path)
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        base = json_path.parent
        items = [
            PlanogramItem(
                product_id=r.get("product_id") or str(uuid.uuid4()),
                name=r["name"],
                ean=r.get("ean"),
                shelf=r.get("shelf"),
                bbox=None,
                image_path=base / r["image"],
            )
            for r in raw
        ]
        loader._build_from_files(items)
        return loader

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "PlanogramLoader":
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
                    shelf=int(row["shelf"]) if row.get("shelf") else None,
                    bbox=None,
                    image_path=base / row["image"],
                ))
        loader._build_from_files(items)
        return loader

    # ------------------------------------------------------------------ #
    # Acesso                                                               #
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> EmbeddingIndex:
        return self._index

    @property
    def items(self) -> list[PlanogramItem]:
        return self._items

    @property
    def shelf_capacities(self) -> list[ShelfCapacity]:
        return self._shelf_capacities

    def capacity_for_shelf(self, shelf: int) -> int | None:
        for sc in self._shelf_capacities:
            if sc.shelf == shelf:
                return sc.sku_limit
        return None

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------ #
    # Construção interna                                                   #
    # ------------------------------------------------------------------ #

    def _build_from_crops(
        self,
        items: list[tuple[PlanogramItem, np.ndarray]],
        extra_meta_keys: list[str] | None = None,
    ) -> None:
        print(f"Indexando {len(items)} produto(s) com CLIP...")
        for item, crop in tqdm(items, unit="produto"):
            emb = self._embedder.embed(crop)
            item.embedding = emb
            extra = {}
            for key in (extra_meta_keys or []):
                extra[key] = getattr(item, key, None)
            self._index.add(emb, item.product_id, item.name, item.ean, **extra)
            self._items.append(item)
        print(f"Índice pronto: {len(self._items)} produto(s).")

    def _build_from_files(self, items: list[PlanogramItem]) -> None:
        print(f"Indexando {len(items)} produto(s)...")
        for item in tqdm(items, unit="produto"):
            img = cv2.imread(str(item.image_path))
            if img is None:
                print(f"  [aviso] não encontrado: {item.image_path}")
                continue
            emb = self._embedder.embed(img)
            item.embedding = emb
            self._index.add(emb, item.product_id, item.name, item.ean)
            self._items.append(item)
        print(f"Índice pronto: {len(self._items)} produto(s).")


# ------------------------------------------------------------------ #
# Helpers internos                                                     #
# ------------------------------------------------------------------ #

def _read_dist_csv(path: Path) -> list[ShelfCapacity]:
    capacities = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shelf = int(row["Prateleira"].strip())
            limit = int(row[" SKU_limite"].strip())
            capacities.append(ShelfCapacity(shelf=shelf, sku_limit=limit))
    return capacities


def _detect_planogram_products(
    frame: np.ndarray,
    conf: float,        # não usado neste método, mantido por assinatura
    device: str,        # idem
) -> list[tuple[tuple[int, int, int, int], float]]:
    """
    Segmenta produtos de uma imagem de planograma estruturado usando
    projeção de intensidade — sem modelo de ML.

    Algoritmo:
      1. Detecta separadores horizontais (prateleiras) via projeção Y.
      2. Para cada faixa de prateleira, detecta separadores verticais
         via projeção X.
      3. Retorna bboxes de cada célula de produto.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── 1. Encontra prateleiras (faixas horizontais escuras/contrastantes) ──
    # Perfil horizontal: variância por linha
    row_var = np.var(gray.astype(np.float32), axis=1)
    # Suaviza para eliminar ruído
    kernel = np.ones(max(3, h // 80)) / max(3, h // 80)
    row_smooth = np.convolve(row_var, kernel, mode="same")

    # Linhas de prateleira = mínimos locais na variância (bordas de suporte)
    shelf_separators = _find_separators(row_smooth, n_expected=None, axis="y", size=h)

    # ── 2. Para cada faixa, encontra produtos (separadores verticais) ──
    detections: list[tuple[tuple[int, int, int, int], float]] = []
    for i, (y1, y2) in enumerate(shelf_separators):
        strip = gray[y1:y2, :]
        if strip.shape[0] < 5:
            continue
        col_var = np.var(strip.astype(np.float32), axis=0)
        k = np.ones(max(3, w // 120)) / max(3, w // 120)
        col_smooth = np.convolve(col_var, k, mode="same")
        product_cols = _find_separators(col_smooth, n_expected=None, axis="x", size=w)

        for x1, x2 in product_cols:
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            detections.append(((x1, y1, x2, y2), 1.0))

    detections.sort(key=lambda d: (d[0][1], d[0][0]))
    return detections


def _find_separators(
    profile: np.ndarray,
    n_expected: int | None,
    axis: str,
    size: int,
    min_gap: int = 10,
) -> list[tuple[int, int]]:
    """
    Dado um perfil 1-D de variância, localiza regiões de produto
    (picos de variância) e retorna seus intervalos (início, fim).
    """
    # Binariza: valores acima da mediana são "conteúdo"
    threshold = np.percentile(profile, 30)
    binary = (profile > threshold).astype(np.uint8)

    # Dilata para unir regiões próximas
    gap = max(min_gap, size // 60)
    kernel = np.ones(gap, dtype=np.uint8)
    binary = np.convolve(binary, kernel, mode="same")
    binary = (binary > 0).astype(np.uint8)

    # Encontra transições 0→1 e 1→0
    padded = np.concatenate([[0], binary, [0]])
    starts = np.where(np.diff(padded) == 1)[0]
    ends   = np.where(np.diff(padded) == -1)[0]

    segments = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_gap]
    return segments if segments else [(0, size)]


def _compute_shelf_bands(
    img_height: int,
    capacities: list[ShelfCapacity],
) -> list[tuple[float, float]]:
    n = len(capacities) if capacities else 1
    band_h = img_height / n
    return [(i * band_h, (i + 1) * band_h) for i in range(n)]


def _bbox_to_shelf(cy: float, bands: list[tuple[float, float]]) -> int:
    for i, (y_min, y_max) in enumerate(bands):
        if y_min <= cy < y_max:
            return i + 1
    return len(bands)


def _parse_filename(path: Path) -> PlanogramItem:
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
        shelf=None,
        bbox=None,
        image_path=path,
    )
