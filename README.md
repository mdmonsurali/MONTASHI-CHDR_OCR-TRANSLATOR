# MONTASHI-U_OCR-TRANSLATOR

A microservice pipeline that turns scanned/native PDFs and DOCX files into
structured, layout-preserving output: OCR → layout JSON → reconstructed DOCX,
with an optional machine-translation pass that produces a translated DOCX in
the same layout.

## Architecture

```
                 ┌─────────────┐
   Browser  ───▶ │  ui_service │  session auth, serves the frontend,
                 └──────┬──────┘  proxies to ocr_service
                        │
              ┌─────────┼─────────────┐
              ▼         ▼             ▼
      ┌───────────┐ ┌────────────┐ ┌───────────────┐
      │auth_service│ │ocr_service │ │ orchestrator_  │
      │  sessions, │ │ layout OCR │ │ service        │
      │  admin     │ │ + DOCX     │ │ upload → OCR → │
      │  users     │ │ reconstr.  │ │ GPU swap →     │
      └─────┬──────┘ └─────┬──────┘ │ translate →    │
            │              │        │ restore        │
            │              │        └───────┬────────┘
            │              │                │
            ▼              ▼                ▼
      ┌─────────┐   ┌────────────┐   ┌──────────────────┐
      │ Postgres │   │   MinIO    │   │ translator_service│
      │ metadata │   │ file bytes │   │ LLM-driven        │
      └─────────┘   └────────────┘   │ translation +      │
                                      │ DOCX reconstruction │
                                      └──────────────────────┘
```

Each service is an independent FastAPI app in its own directory, containerized
via its own `Dockerfile`, and wired together by the root
[`docker-compose.yml`](./docker-compose.yml).

| Service | Directory | Responsibility |
|---|---|---|
| **ui_service** | [`ui_service/`](ui_service/) | Serves the web frontend, session-gates every route, proxies authenticated requests to `ocr_service` with `X-User-Id` / `X-User-Role` headers. |
| **auth_service** | [`auth_service/`](auth_service/) | Login/logout, session cookies, password changes, admin user management. |
| **ocr_service** | [`ocr_service/`](ocr_service/) | Runs OCR/layout extraction on uploaded files, persists documents (UUID-keyed) to Postgres + MinIO, reconstructs DOCX from layout JSON, renders DOCX→PDF previews via LibreOffice. |
| **orchestrator_service** | [`orchestrator_service/`](orchestrator_service/) | Drives the end-to-end batch pipeline (upload → OCR → GPU swap → translate → restore) over SSE, and serves finished translation artifacts. |
| **translator_service** | [`translator_service/`](translator_service/) | Translates the OCR'd layout JSON via an LLM backend and reconstructs a translated DOCX in the original layout. |
| **reconstruction_service** | [`reconstruction_service/`](reconstruction_service/) | Shared library (not a standalone container) used by `ocr_service` / `translator_service` to turn layout JSON into OOXML: text boxes, tables (with colspan/rowspan handling), pictures, formulas. |
| **database** | [`database/`](database/) | Postgres schema (`schema.sql`) and data-model docs — see [`database/README.md`](database/README.md). |

## Data flow

1. A file is uploaded through `ui_service`, which forwards it to `ocr_service`.
2. `ocr_service` runs the OCR/layout model, stores the original file, extracted
   markdown, layout JSON, and reconstructed DOCX in MinIO, and records metadata
   in Postgres (`documents` table — see [`database/README.md`](database/README.md)).
3. For translation, `orchestrator_service` coordinates: it swaps GPU workloads
   between the OCR model and the translation LLM, calls `translator_service` to
   translate the layout JSON and reconstruct a translated DOCX, and exposes the
   result for download.
4. `reconstruction_service` is the shared code path both `ocr_service` and
   `translator_service` use to turn layout JSON back into a `.docx` — it infers
   column widths, row heights, font sizing, and merged-cell (colspan/rowspan)
   table structure from the OCR'd HTML/markdown tables and source bounding boxes.

Bytes live in MinIO under `documents/{uuid}/{source.ext,output.md,layout.json,output.docx}`
(and the analogous `translated_*` keys for translation output); Postgres only
stores metadata and MinIO object keys, never raw bytes.

## Running locally

Requires Docker and Docker Compose. Configuration is via environment
variables (see [`.env`](./.env) for the full list — GPU/VLLM settings,
service ports, Postgres/MinIO credentials).

```bash
docker compose up -d
```

Default ports (overridable via `.env`):

| Service | Port |
|---|---|
| `ui_service` | `${UI_SERVICE_PORT:-8000}` |
| `auth_service` | `${AUTH_SERVICE_PORT:-8002}` |
| OCR/VLLM backend | `${VLLM_PORT:-8888}` |
| MinIO console | `${MINIO_CONSOLE_PORT:-9001}` |

`ocr_service`, `orchestrator_service`, and `translator_service` are reached
through the internal unlimited OCR Docker network (`ocr-network`) rather than exposed
directly; `ui_service` is the public entry point.

## Key routes

Each service documents its own routes in its `main.py` module docstring —
see [`ocr_service/src/main.py`](ocr_service/src/main.py),
[`orchestrator_service/src/main.py`](orchestrator_service/src/main.py),
[`translator_service/src/main.py`](translator_service/src/main.py),
[`ui_service/src/main.py`](ui_service/src/main.py), and
[`auth_service/src/main.py`](auth_service/src/main.py) for the current list.

---

## Demo

### Administration (master account)

The **master** account is the un-deletable super-admin seeded from `MASTER_USERNAME` / `MASTER_PASSWORD` on first boot. From the Administration page it can create users and admins, rename / change role / reset password / disable / delete any account (except itself), and the Profile dropdown shows the role beside the avatar. Regular admins see only `user` rows in the same table; the master and other admins are hidden from them.

![Master account Administration page with Profile dropdown open](demo-ui/Demo-ui-(Master-Account-Administration).png)

### Single mode

Upload one file and see Markdown, DOCX preview, and JSON side-by-side.

![Single mode](demo-ui/Demo-ui-(Single).png)

### Batch mode

Drop multiple files; each row streams its own status and exposes download buttons as soon as it finishes.

![Batch mode](demo-ui/Demo-ui-(Batch).png)

### DOCX preview

Reconstructed Word document rendered inline (DOCX → PDF via LibreOffice).

![DOCX preview](demo-ui/Demo-ui-(Docx%20Preview%20ui).png)

### MinIO object store

All source uploads and generated artifacts (MD / JSON / DOCX) are persisted in MinIO under `documents/{uuid}/`. Translations land under `translations/{uuid}/` and are linked back to their source document via `translations.source_document_id` in Postgres.

![MinIO console](demo-ui/Demo-ui-(Minio%20S3%20ui).png)

### Translate mode

Upload one or more files and pick a target language from the dropdown
(English, German, Bangla, Hindi, French, Spanish, Polish, Japanese, Italian,
Korean, Chinese, Brazilian Portuguese). The orchestrator phases through
OCR → swap GPU → translate → swap back, streaming a live timeline. Picture
placement, table layout, multi-page table chaining, formula positions, and
page geometry from the original DOCX are all preserved in the translated
DOCX.

![Translate mode — English to German](demo-ui/Translation%20english%20to%20german.png)

The translated DOCX can be previewed inline (rendered to PDF by LibreOffice
in `ocr_service`). The modal header carries the target language so you can
tell multiple translations of the same document apart at a glance.

![Translated DOCX preview](demo-ui/translation%20preview.png)

### Task history

Every OCR'd document is listed under `/history`, scoped to the logged-in
user. Each row exposes inline downloads for both the original OCR artifacts
(MD, DOCX, JSON, source) and any translation that has been produced for it
(translated DOCX, translated JSON, inline preview).

![Task History page](demo-ui/Task%20History.png)


## Notes

- All inter-service calls trust `X-User-Id` / `X-User-Role` headers because
  those ports are only reachable from the internal Docker network, not
  exposed publicly.
- `reconstruction_service` has no tests directory; when changing table/layout
  reconstruction logic, verify against real OCR'd layout JSON (see
  `ocr_*.json` files at the repo root for examples) rather than relying on
  unit tests alone.
