# vdetec — Identificação de Produtos em Prateleiras de Farmácias

Sistema de visão computacional que identifica produtos em gôndolas a partir de uma imagem de planograma e fotos reais de prateleiras. Usa exclusivamente redes pré-treinadas — **sem treinamento customizado**.

## Decisões de design

- **Sem fine-tuning**: YOLO-World (zero-shot) para detecção em fotos reais; projeção de variância para segmentação do planograma; CLIP para reconhecimento visual.
- **Cascata de reconhecimento**: barcode → OCR (EAN no texto) → CLIP similarity. A primeira etapa que retornar resultado encerra a busca.
- **Cache obrigatório**: o índice FAISS é salvo em `.vdetec_cache/` após a primeira indexação. Nunca re-indexar se o cache existir.
- **Não ler a pasta `images/`** sem necessidade — ela é grande. O `.claudeignore` bloqueia por padrão; `images/adocante/` está liberada para experimentos.

## Estrutura

```
vdetec.py               # Fachada principal — VDetec.load_planogram() + identify()
identify.py             # CLI com anotação visual da imagem
sanity_check.py         # Teste de sanidade: planograma vs. próprio índice (só CLIP)

planogram/
  loader.py             # PlanogramLoader — segmenta planograma e constrói índice FAISS
                        # from_image(png, dist_csv) → segmentação por projeção de variância
                        # from_folder / from_json / from_csv → imagens individuais

detection/
  detector.py           # ProductDetector — YOLO-World zero-shot para fotos reais

recognition/
  embeddings.py         # VisualEmbedder (CLIP ViT-B/32) + EmbeddingIndex (FAISS)
  pipeline.py           # RecognitionPipeline — orquestra cascata barcode→OCR→CLIP
  barcode.py            # BarcodeReader — pyzbar (EAN-13, EAN-8, QR)
  ocr.py                # LabelOCR — EasyOCR + regex para extrair EAN do texto

database/
  models.py             # ORM: Store, Camera, Shelf, ShelfProduct, Product,
                        #      Snapshot, DetectionEvent, StockAlert
  session.py            # Engine async (asyncpg) + get_db()

api/
  main.py               # FastAPI app
  routers/
    snapshots.py        # POST /snapshots/ — recebe frame, roda pipeline, persiste
    products.py         # CRUD catálogo de produtos
    cameras.py          # CRUD câmeras
    stores.py           # CRUD lojas
    alerts.py           # Listagem e resolução de alertas

alerts/
  rules.py              # evaluate_planogram() — gera StockAlert por prateleira
  tasks.py              # Celery tasks: check_shelf_alerts, notify_alert
  channels/
    telegram.py         # Notificação via Telegram Bot API
    email.py            # Notificação via SMTP

capture/
  camera_worker.py      # Worker async: captura RTSP → POST /api/v1/snapshots/

config/
  settings.py           # Pydantic-settings — lê .env
```

## Formato do planograma

O formato principal é **imagem PNG + CSV de capacidade por prateleira**:

```
adocante_planogram.png   # imagem completa da gôndola (N prateleiras visíveis)
adocante_dist.csv        # CSV: Prateleira, SKU_limite
                         # ex: 1,16 / 2,32 / 3,40 / 4,48 / 5,60
```

O `PlanogramLoader.from_image()` segmenta os produtos usando projeção de variância (OpenCV puro, sem modelo). O índice resultante é cacheado em `.vdetec_cache/` na mesma pasta.

Formatos alternativos aceitos: pasta de imagens, JSON, CSV de manifesto.

## Como rodar (uso mínimo)

```powershell
# Instalar dependências
pip install -r requirements.txt

# Sanity check (planograma vs. próprio índice)
python sanity_check.py --planogram images/.../planogram.png --dist-csv images/.../dist.csv --save out.jpg

# Identificar foto real
python identify.py --planogram images/.../planogram.png --dist-csv images/.../dist.csv --image foto.jpg --save out.jpg
```

**Nota:** comandos de múltiplas linhas no PowerShell usam backtick (`` ` ``), não barra invertida (`\`).

## Fluxo de dados (modo CLI)

```
load_planogram(png, dist_csv)
  └─ .vdetec_cache/ existe?
       ├─ Sim → carrega FAISS + meta  (< 1s)
       └─ Não → segmenta planograma por projeção
                → CLIP embed de cada crop  (~0.4s/crop no CPU)
                → salva .vdetec_cache/

identify(foto.jpg)
  └─ YOLO-World detecta regiões de produto
       └─ Para cada crop:
            1. pyzbar → EAN exato
            2. EasyOCR → EAN no texto
            3. CLIP → vizinho mais próximo no índice FAISS
```

## Dependências principais

| Pacote | Uso |
|--------|-----|
| `ultralytics` | YOLO-World (detecção zero-shot em fotos reais) |
| `sentence-transformers` | CLIP ViT-B/32 (embeddings visuais) |
| `faiss-cpu` | Índice vetorial de similaridade |
| `opencv-python-headless` | Segmentação do planograma + anotação |
| `pyzbar` | Leitura de código de barras |
| `easyocr` | OCR para extração de EAN do rótulo |

## Experimento atual

Categoria: **adoçantes** (`images/images_experiments/adocante/`)

```
planogram/
  adocante_planogram.png   # 5 prateleiras, ~31 produtos detectados
  adocante_dist.csv        # capacidades: 16 / 32 / 40 / 48 / 60 SKUs
```

## O que ainda não está pronto

- Dashboard (pasta `dashboard/` vazia)
- Migrations Alembic (`scripts/alembic_init.sh` criado, não executado)
- Indexação FAISS para os outros formatos de planograma (`from_folder`, `from_json`)
- Validação do sanity check com scores — resultado dos testes ainda pendente
- Fotos reais de prateleira para teste de identificação
