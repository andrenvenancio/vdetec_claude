"""
CLI de identificação de produtos em prateleiras.

Exemplos:
  # Pasta de imagens de referência + uma foto da prateleira
  python identify.py --planogram planogram/produtos/ --image prateleira.jpg

  # JSON como planograma + múltiplas imagens
  python identify.py --planogram planogram.json --image foto1.jpg foto2.jpg

  # Salva imagem anotada em disco
  python identify.py --planogram planogram/ --image prateleira.jpg --save resultado.jpg

  # Só mostra detecções acima de 0.75 de confiança
  python identify.py --planogram planogram/ --image prateleira.jpg --min-conf 0.75
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from vdetec import VDetec, IdentificationResult, ProductMatch


# ──────────────────────────────────────────────────────────────────────────────
# Visualização
# ──────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "barcode": (0, 200, 0),    # verde
    "ocr":     (255, 165, 0),  # laranja
    "clip":    (0, 120, 255),  # azul
    "unknown": (0, 0, 200),    # vermelho
}


def annotate(frame: np.ndarray, result: IdentificationResult, min_conf: float = 0.0) -> np.ndarray:
    out = frame.copy()
    for det in result.detections:
        if det.confidence < min_conf and det.method != "barcode":
            continue
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = _COLORS.get(det.method, (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        prefix = str(det.location) if det.location is not None else (det.product_name or "???")
        label = f"{prefix} {det.confidence:.0%}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vdetec — identificação de produtos em prateleiras")
    p.add_argument("--planogram", required=True,
                   help="Imagem PNG do planograma, pasta de imagens, .json ou .csv")
    p.add_argument("--dist-csv", dest="dist_csv", metavar="CSV",
                   help="CSV de capacidade por prateleira (obrigatório quando --planogram é PNG)")
    p.add_argument("--manifest-pdf", dest="manifest_pdf", metavar="PDF", default=None,
                   help="PDF de manifesto do planograma (Location, UPC, Name, Horiz_F)")
    p.add_argument("--products-folder", dest="products_folder", metavar="DIR", default=None,
                   help="Pasta com imagens de produto nomeadas por Location (1.jpg ... 60.jpg)")
    p.add_argument("--image", nargs="+", required=True,
                   help="Uma ou mais imagens de prateleira para analisar")
    p.add_argument("--save", metavar="PATH",
                   help="Salva imagem anotada (só funciona com uma única imagem)")
    p.add_argument("--show", action="store_true",
                   help="Abre janela com imagem anotada")
    p.add_argument("--min-conf", type=float, default=0.0, dest="min_conf",
                   help="Confiança mínima de reconhecimento para exibir (0-1)")
    p.add_argument("--device", default="cpu",
                   help="Dispositivo de inferência: cpu | cuda | mps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    vd = VDetec(device=args.device)
    vd.load_planogram(
        args.planogram,
        dist_csv=args.dist_csv,
        manifest_pdf=args.manifest_pdf,
        products_folder=args.products_folder,
    )

    for img_path in args.image:
        result = vd.identify(img_path)
        print("\n" + "─" * 60)
        print(result.summary())

        if args.save or args.show:
            frame = cv2.imread(img_path)
            annotated = annotate(frame, result, min_conf=args.min_conf)

            if args.save:
                save_arg = Path(args.save)
                if len(args.image) == 1:
                    save_path = save_arg
                else:
                    folder = save_arg if save_arg.suffix == "" else save_arg.parent
                    folder.mkdir(parents=True, exist_ok=True)
                    save_path = folder / (Path(img_path).stem + "_results.jpg")
                cv2.imwrite(str(save_path), annotated)
                print(f"Imagem salva: {save_path}")

            if args.show:
                cv2.imshow(f"vdetec — {img_path}", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
