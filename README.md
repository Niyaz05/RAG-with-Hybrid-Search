# Hybrid Search RAG

A Hybrid Search Retrieval-Augmented Generation (RAG) project built from scratch to understand and implement the complete retrieval pipeline.

## Dataset

This project uses the **Aperture Cloud Sample Corpus**, a multi-format document collection containing technical documentation in various formats, including:

- PDF (`.pdf`)
- Markdown (`.md`)
- Plain Text (`.txt`)
- HTML (`.html`)

The corpus includes documents such as onboarding guides, API references, architecture decisions, compliance policies, incident runbooks, and FAQs, making it suitable for building and evaluating a Hybrid Search RAG pipeline.

## Current Progress

- ✅ Implemented a multi-format document loader (`loader.py`)
- Supports PDF, Markdown, Text, and HTML files.
- Normalizes all documents into a common JSON format with metadata.
- Stores processed documents separately for future indexing.

## Project Structure

```text
hybrid_rag/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── loader.py
│
└── README.md
```

## Next Steps

- Text Chunking
- Dense Embeddings
- BM25 Index
- Vector Database
- Hybrid Retrieval
- LLM Integration