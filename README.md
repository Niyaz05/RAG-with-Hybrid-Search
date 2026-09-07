# Hybrid Search RAG

A Hybrid Search Retrieval-Augmented Generation (RAG) system built from scratch to understand and implement a complete document retrieval pipeline.

Currently, the pipeline supports slide-aware PDF loading, structure-preserving layout parsing, and adaptive chunking optimized for presentation slides.

---

## Dataset

The dataset consists of monthly lecture/course slide PDFs for **CSET340: Advanced Computer Vision and Video Analytics** (Jan 2026 – Apr 2026), located under `data/raw/`:

- `Jan_2026.pdf`
- `Feb_2026.pdf`
- `March_2026.pdf`
- `April_2026.pdf`

---

## Current Progress

### Slide-Aware PDF Chunker (`retrieval/chunker.py`)
- **Block-Based Extraction**: Uses [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`) to extract text blocks in spatial reading order (top-to-bottom, left-to-right) rather than simple text streams, preserving paragraph structures.
- **Adaptive Merging**: Merges "thin" slides (slides containing very little text, below a minimum threshold of 40 words) with adjacent pages to prevent context fragmentation and ensure meaningful chunks.
- **Adaptive Splitting**: Greedily splits "dense" slides (slides containing too much text, exceeding 250 words) into smaller sub-chunks, appending alphabetical suffixes (e.g. `20a`, `20b`) to track the parent slides.
- **Structured Data Objects**: Normalizes chunks into a common dataclass representation (`Chunk`) containing:
  - `chunk_id` (e.g., `April_2026_c042`)
  - `source_pdf` (e.g., `April_2026.pdf`)
  - `slide_label` (e.g., `57-58` or `60a`)
  - `text` (cleaned textual content of the chunk)
  - `word_count` (number of words in the chunk)

---

## Project Structure

```text
hybrid_rag/
├── data/
│   └── raw/
│       ├── Jan_2026.pdf
│       ├── Feb_2026.pdf
│       ├── March_2026.pdf
│       └── April_2026.pdf
│
├── retrieval/
│   ├── __init__.py
│   └── chunker.py
│
├── testing/
│   ├── __init__.py
│   └── test_chunking.py
│
├── requirements.txt
└── README.md
```

---

## Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Tests / Demos**:
   To test the chunker on a sample PDF (`April_2026.pdf`), run the testing script:
   ```bash
   PYTHONPATH=. python testing/test_chunking.py
   ```

   Alternatively, you can run the chunker script directly, specifying an optional path to any PDF:
   ```bash
   PYTHONPATH=. python retrieval/chunker.py data/raw/March_2026.pdf
   ```

---

## Next Steps

- [ ] **Dense Embeddings**: Generate vector embeddings for the extracted text chunks.
- [ ] **BM25 Index**: Build a sparse term index to enable keyword-based search.
- [ ] **Vector Database**: Index the chunks/embeddings into a vector store.
- [ ] **Hybrid Retrieval**: Implement hybrid search, combining dense vector search and sparse BM25 search (e.g., using Reciprocal Rank Fusion).
- [ ] **LLM Integration**: Wire the retrieval results into an LLM context to answer questions about the course material.