"""
Sanity check honesto: usa fotos individuais de produto (products/) como queries
contra um índice construído APENAS com os crops do planograma.

Mede CLIP cross-image: foto real do produto → crop do planograma do mesmo SKU.
Diferente do sanity_check.py original (que queria o mesmo vetor que indexou).

Uso:
    python sanity_check_products.py `
      --planogram    images/.../adocante_planogram.png `
      --dist-csv     images/.../adocante_dist.csv `
      --manifest-pdf documentos/Planograma_Adocante.pdf `
      --products     images/.../products
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from planogram.loader import PlanogramLoader
from recognition.embeddings import VisualEmbedder

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--planogram",    required=True)
    p.add_argument("--dist-csv",     required=True, dest="dist_csv")
    p.add_argument("--manifest-pdf", required=True, dest="manifest_pdf")
    p.add_argument("--products",     required=True, help="Pasta com {location}.jpg")
    p.add_argument("--top-k",        type=int, default=3)
    args = p.parse_args()

    # Cache temporário só para esse teste — não enriquece com products
    tmp_cache = Path(args.planogram).parent / ".vdetec_cache_sanity_only_planogram"
    if tmp_cache.exists():
        shutil.rmtree(tmp_cache)

    loader = PlanogramLoader.from_image_with_manifest(
        args.planogram, args.dist_csv, args.manifest_pdf,
        products_folder=None,             # apenas crops do planograma
        cache_dir=tmp_cache,
    )

    # Mapeia location -> product_id via manifest
    loc_to_pid = {m.location: next(
        (it.product_id for it in loader.items if it.ean == m.upc), None
    ) for m in loader.manifest}

    embedder = VisualEmbedder()
    products_dir = Path(args.products)

    print(f"\nQueries: fotos reais em {products_dir}")
    print(f"Indice : {len(loader.items)} crops do planograma\n")

    ok = 0
    fail = 0
    top_k_ok = 0
    rows = []

    for path in sorted(products_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_EXTS:
            continue
        try:
            loc = int(path.stem)
        except ValueError:
            continue
        expected_pid = loc_to_pid.get(loc)
        if expected_pid is None:
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue
        emb = embedder.embed(img)
        hits = loader.index.search(emb, top_k=args.top_k)

        if not hits:
            fail += 1
            continue

        best = hits[0]
        match_top1 = best.product_id == expected_pid
        match_topk = any(h.product_id == expected_pid for h in hits)

        if match_top1:
            ok += 1
        else:
            fail += 1
        if match_topk:
            top_k_ok += 1

        symbol = "OK " if match_top1 else "XX "
        rows.append((symbol, loc, best.score, best.product_name, match_topk))

    rows.sort(key=lambda r: r[1])
    for symbol, loc, score, name, in_topk in rows:
        topk_mark = "" if symbol == "OK " else (" (no top-k)" if in_topk else "")
        print(f"  {symbol} loc={loc:3d}  score={score:.3f}  -> {name[:40]}{topk_mark}")

    total = ok + fail
    print(f"\n-- Resultado ----------------------------------------")
    print(f"  Top-1 corretos    : {ok}/{total}  ({ok/total:.1%})")
    print(f"  Top-{args.top_k} corretos    : {top_k_ok}/{total}  ({top_k_ok/total:.1%})")
    print(f"  Top-1 incorretos  : {fail}/{total}")


if __name__ == "__main__":
    main()
