"""
Teste de sanidade: embeda cada crop do planograma e busca no próprio índice.
Usa só CLIP — sem barcode reader, sem EasyOCR.

Expectativa: score >= 0.99 para todos (crop idêntico ao indexado).

Uso:
    python sanity_check.py --planogram <png> --dist-csv <csv> [--save resultado.jpg]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from planogram.loader import PlanogramLoader
from recognition.embeddings import VisualEmbedder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--planogram",    required=True)
    p.add_argument("--dist-csv",     required=True, dest="dist_csv")
    p.add_argument("--manifest-pdf", default=None,  dest="manifest_pdf")
    p.add_argument("--save",         default="sanity_result.jpg")
    args = p.parse_args()

    # 1. Indexa planograma
    if args.manifest_pdf:
        loader = PlanogramLoader.from_image_with_manifest(
            args.planogram, args.dist_csv, args.manifest_pdf
        )
    else:
        loader = PlanogramLoader.from_image(args.planogram, args.dist_csv)
    embedder = VisualEmbedder()
    frame = cv2.imread(args.planogram)

    print(f"\nTestando {len(loader.items)} crop(s) contra o próprio índice...\n")

    scores: list[float] = []
    ok = 0
    fail = 0

    for item in loader.items:
        if item.bbox is None:
            continue
        x1, y1, x2, y2 = item.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = embedder.embed(crop)
        hits = loader.index.search(emb, top_k=1)

        if not hits:
            print(f"  XX [{item.name}] sem resultado no índice")
            fail += 1
            continue

        best = hits[0]
        match = best.product_id == item.product_id
        symbol = "OK" if match else "XX"
        scores.append(best.score)
        if match:
            ok += 1
        else:
            fail += 1

        print(f"  {symbol} {item.name:30s}  score={best.score:.4f}  -> {best.product_name}")

    # 2. Resumo
    print(f"\n-- Resultado ----------------------------------------")
    print(f"  Corretos  : {ok}/{len(loader.items)}")
    print(f"  Incorretos: {fail}/{len(loader.items)}")
    if scores:
        print(f"  Score mín : {min(scores):.4f}")
        print(f"  Score máx : {max(scores):.4f}")
        print(f"  Score méd : {sum(scores)/len(scores):.4f}")

    # 3. Imagem anotada
    out = frame.copy()
    for item in loader.items:
        if item.bbox is None:
            continue
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

    cv2.imwrite(args.save, out)
    print(f"\nImagem salva : {args.save}")


if __name__ == "__main__":
    main()
