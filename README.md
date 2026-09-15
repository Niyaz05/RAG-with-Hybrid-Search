# Hybrid Search RAG

A Hybrid Search Retrieval-Augmented Generation (RAG) system built from scratch to index, search, and answer questions over lecture slide decks using a combination of dense semantic search and sparse keyword retrieval.

The project currently implements a complete slide ingestion, adaptive chunking, and dual dense (ChromaDB + Ollama) / sparse (BM25) indexing pipeline with comprehensive unit and integration tests.

---

## Architecture Overview

```mermaid
flowchart LR
    A["Raw PDFs (data/pdfs/)"] --> B["PDF Parser (ingest/parse_pdf.py)"]
    B --> C["Parsed Pages (data/parsed_pages.json)"]
    C --> D["Adaptive Chunker (ingest/chunk.py)"]
    D --> E["Normalized Chunks (data/chunks.json)"]
    E --> F["Dual Indexer (ingest/embed_index.py)"]
    F --> G["ChromaDB Vector Store (data/chroma_db/)"]
    F --> H["BM25 Index (data/bm25_index.pkl)"]
    G -.-> I["Hybrid Retrieval & RRF (Upcoming)"]
    H -.-> I
    I -.-> J["LLM Generation & Citations (Upcoming)"]
```

---

## Dataset

The dataset contains monthly lecture slide decks for **CSET340: Advanced Computer Vision and Video Analytics** (Jan 2026 – Apr 2026), located under `data/pdfs/`:

| File | Extracted Date Range / Week Label | Description |
| :--- | :--- | :--- |
| `jan_2026.pdf` | `5-9 Jan 2026` | Course intro, fundamentals, early vision |
| `feb_2026.pdf` | `2-6 Feb 2026` | Features, geometric transformations, alignment |
| `march_2026.pdf` | `16-20 March 2026` | Motion estimation, optical flow, video analytics |
| `april_2026.pdf` | `6-10 April 2026` | Camera calibration, 3D reconstruction, stereo vision |

> [!NOTE]
> Instead of fragile, hand-maintained lookup tables, date ranges are dynamically parsed directly from each deck's cover slide text via regex (`extract_week_label`), with graceful fallback to sanitized filenames (`week_label_for`).

---

## Ingestion & Indexing Pipeline

The pipeline is designed specifically for presentation slides, respecting slide boundaries, adapting dynamically to content density, and indexing chunks across both dense and sparse representations.

### Step 1: Slide-Aware PDF Parsing (`ingest/parse_pdf.py`)

- **Per-Slide Extraction**: Uses [PyMuPDF](https://pymupdf.readthedocs.io/) (`pymupdf`) to extract selectable text page-by-page, preserving natural topic and slide boundaries.
- **Dynamic Metadata Derivation**: Automatically pulls course dates from slide 1 content.
- **Text Normalization**: 
  - Collapses runs of spaces and horizontal tabs.
  - Normalizes 3+ newlines into `\n\n` (preserving double-newlines as downstream paragraph delimiters).
  - Skips near-empty or image-only slides (< 5 characters) to prevent noise.
- **Output**: Writes `data/parsed_pages.json` with records containing `{source, week, slide, text}`.

### Step 2: Adaptive Slide Chunking (`ingest/chunk.py`)

Slides vary wildly in density—some are minimal title slides, while others contain dense mathematical explanations. The chunker adapts to both:

- **Adaptive Merging for Thin Slides (`MIN_CHARS = 120`)**:
  - Slides with fewer than 120 characters (such as title or section transition slides) are buffered and merged forward with subsequent slides from the same PDF (e.g., labeled `slides 1-2`).
  - Strict document-boundary check ensures slides are **never** merged across different PDFs.
- **Adaptive Splitting for Dense Slides (`MAX_CHARS = 1400`)**:
  - Slides exceeding 1400 characters are greedily split across `\n\n` paragraph boundaries.
  - Fallback window-slicing handles oversized paragraphs lacking newline breaks.
- **Normalized Schema**:
  ```json
  {
    "chunk_id": "c0001",
    "source": "april_2026.pdf",
    "week": "6-10 April 2026",
    "slide_label": "slides 1-2",
    "text": "..."
  }
  ```
- **Output**: Writes `data/chunks.json` and outputs summary statistics (min, max, average chunk size).

### Step 3: Dual Indexing — Dense Embeddings & Sparse BM25 (`ingest/embed_index.py`)

To achieve true hybrid search, chunks are indexed into two complementary engines:

- **Dense Semantic Embeddings (ChromaDB + Ollama)**:
  - Generates 768-dimensional semantic embeddings using local Ollama (`nomic-embed-text`).
  - Persists vectors, documents, and slide metadata (`source`, `week`, `slide_label`) in ChromaDB (`data/chroma_db/`) under the `cv_course` collection.
  - Features dependency injection (`EmbedFn`) to decouple the embedding backend and enable fast, deterministic unit testing without requiring an active Ollama instance.
- **Sparse Lexical Search (BM25Okapi)**:
  - Tokenizes slide text into normalized lowercase alphanumeric terms via regex.
  - Computes BM25 corpus statistics using `rank_bm25` for fast, exact keyword retrieval (e.g., matching acronyms, technical terms, and formulas).
  - Serializes the BM25 index and chunk references to disk (`data/bm25_index.pkl`).

---

## Project Structure

```text
RAG-with-Hybrid-Search/
├── data/
│   ├── pdfs/                     # Raw course slide PDFs
│   │   ├── jan_2026.pdf
│   │   ├── feb_2026.pdf
│   │   ├── march_2026.pdf
│   │   └── april_2026.pdf
│   ├── parsed_pages.json         # Output of parse_pdf.py
│   ├── chunks.json               # Output of chunk.py
│   ├── chroma_db/                # Persistent ChromaDB vector store (gitignored)
│   └── bm25_index.pkl            # Serialized BM25Okapi index & corpus (gitignored)
├── ingest/
│   ├── parse_pdf.py              # Step 1: PDF extraction & cleaning
│   ├── chunk.py                  # Step 2: Slide-aware adaptive chunking
│   └── embed_index.py            # Step 3: Dense (Chroma) & Sparse (BM25) indexing
├── tests/
│   ├── test_parse_pdf.py         # Unit & integration tests for parser
│   ├── test_chunk.py             # Unit tests for chunking & merge/split rules
│   └── test_embed_index.py       # Unit & integration tests for embeddings & BM25
├── requirements.txt              # Project dependencies
├── .gitignore                    # Ignored virtualenvs, local indexes & OS artifacts
├── LICENSE                       # License information
└── README.md
```

---

## Setup & Running

### 1. Environment & Dependencies

Create a virtual environment and install the required dependencies:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Ollama (Local Embeddings)

Ensure [Ollama](https://ollama.com/) is installed and the embedding model is downloaded:

```bash
# Start Ollama service (if not already running)
ollama serve

# Pull the embedding model
ollama pull nomic-embed-text
```

### 3. Run the Ingestion & Indexing Pipeline

Execute the pipeline steps in order:

```bash
# Step 1: Parse PDFs into parsed_pages.json
python3 ingest/parse_pdf.py

# Step 2: Build adaptive chunks into chunks.json
python3 ingest/chunk.py

# Step 3: Generate ChromaDB vector index and BM25 index
python3 ingest/embed_index.py
```

### 4. Run Tests

Run the full test suite via `pytest`:

```bash
pytest -v
```

Or run individual test modules:

```bash
pytest tests/test_parse_pdf.py -v
pytest tests/test_chunk.py -v
pytest tests/test_embed_index.py -v
```

#### Test Architecture
- **Unit tests**: Pure functions (`extract_week_label`, `clean_text`, `split_dense_page`, `build_chunks`, `tokenize`) tested in isolation.
- **Dependency-injected tests**: `embed_chunks()` and `build_bm25_index()` tested deterministically with a mock embedding function (no Ollama daemon required, lightning-fast in CI).
- **Integration tests**: 
  - Real PDF extraction against `data/pdfs/` (safely auto-skipped if PDFs are missing).
  - Real Ollama embedding against local daemon (safely auto-skipped if Ollama is unreachable).

---

## Next Steps

- [x] **Slide-Aware Ingestion**: Per-page PDF extraction with metadata extraction.
- [x] **Adaptive Chunking**: Dynamic merging of thin slides and paragraph-aware splitting of dense slides.
- [x] **Dense Vector Store**: ChromaDB persistence with Ollama `nomic-embed-text` embeddings.
- [x] **Sparse Keyword Index**: Lexical search with BM25Okapi serialized to disk.
- [x] **Automated Test Suite**: Multi-tier unit and integration tests covering extraction, chunking, and indexing.
- [ ] **Hybrid Search & Fusion**: Combine dense and sparse candidate lists using Reciprocal Rank Fusion (RRF).
- [ ] **LLM Generation & Slide Citation**: Ground LLM answers with precise slide citations (e.g., `"April 2026, slides 1-2"`).