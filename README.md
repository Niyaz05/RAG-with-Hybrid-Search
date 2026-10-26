# Hybrid Search RAG

A Hybrid Search Retrieval-Augmented Generation (RAG) system built from scratch to index, search, and answer questions over lecture slide decks using a combination of dense semantic search, sparse keyword retrieval, Reciprocal Rank Fusion (RRF), Cross-Encoder semantic reranking, and citation-enforced LLM generation, served through an authenticated FastAPI backend with persistent per-user chat history.

The project implements a complete slide ingestion, adaptive chunking, dual dense (ChromaDB + Ollama) / sparse (BM25) indexing, a two-stage hybrid retrieval pipeline, citation-verified answer generation via a local LLM (Gemma 4 / Ollama), a retrieval evaluation harness, a JWT-secured REST API with role-based access control and SQLite-backed chat history, and comprehensive unit and integration tests throughout.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["1. Slide Ingestion & Dual Indexing"]
        A["Raw PDFs (data/pdfs/)"] --> B["PDF Parser (ingest/parse_pdf.py)"]
        B --> C["Parsed Pages (data/parsed_pages.json)"]
        C --> D["Adaptive Chunker (ingest/chunk.py)"]
        D --> E["Normalized Chunks (data/chunks.json)"]
        E --> F["Dual Indexer (ingest/embed_index.py)"]
        F --> G["ChromaDB Vector Store (data/chroma_db/)"]
        F --> H["BM25 Index (data/bm25_index.pkl)"]
    end

    subgraph Retrieval ["2. Two-Stage Hybrid Retrieval & Reranking"]
        Q["User Query"] --> S1["BM25 Search (Top 20)<br/>retrieval/search.py"]
        Q --> S2["Vector Search (Top 20)<br/>retrieval/search.py"]
        H --> S1
        G --> S2
        S1 --> RRF["Reciprocal Rank Fusion (RRF, Top 10)<br/>retrieval/fuse.py"]
        S2 --> RRF
        RRF --> CE["Cross-Encoder Reranker (Top 5)<br/>retrieval/rerank.py"]
        Q -.-> CE
    end

    subgraph Generation ["3. Citation-Enforced Answer Generation"]
        CE --> PR["Prompt Builder (generation/prompt.py)"]
        PR --> LLM["Gemma 4 via Ollama"]
        LLM --> VER["Citation Verifier<br/>(checks every cited slide against retrieved chunks)"]
        VER --> ANS["Answer + valid/invalid citations"]
    end

    subgraph Evaluation ["4. Retrieval Evaluation"]
        EQ["eval_questions.json<br/>(question + expected source/slides)"] --> EV["run_eval.py<br/>recall@k per method"]
        S1 -.-> EV
        S2 -.-> EV
        CE -.-> EV
    end

    subgraph Backend ["5. FastAPI Backend (Auth + Chat History)"]
        CL["Client"] -->|"POST /auth/login"| AUTH["JWT Auth<br/>backend/auth.py"]
        CL -->|"POST /chat + Bearer token"| CHAT["Chat Route<br/>backend/routes/chat_routes.py"]
        AUTH --> CHAT
        CHAT -->|"answer_fn (injected)"| Q
        ANS --> CHAT
        CHAT --> DB[("SQLite (data/app.db)<br/>users, chat_sessions, messages")]
    end
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

> [!WARNING]
> `april_2026.pdf` actually spans **two** course weeks (6-10 April and 13-17 April), each with its own cover slide. `extract_week_label` currently reads only page 1's date range and applies it to the whole file, so slides from the second week are mislabeled as `6-10 April 2026`. Tracked as a known issue, to be fixed before the ingestion layer is considered final.

---

## Ingestion & Indexing Pipeline

*(Steps 1-3 unchanged from the ingestion stage — see inline docstrings in `ingest/` for full detail.)*

### Step 1: Slide-Aware PDF Parsing (`ingest/parse_pdf.py`)
Per-slide text extraction via PyMuPDF, dynamic week-label derivation from cover-slide text, whitespace normalization, near-empty slide skipping. Outputs `data/parsed_pages.json`.

### Step 2: Adaptive Slide Chunking (`ingest/chunk.py`)
Merges thin slides (< `MIN_CHARS = 120`) forward into the next slide from the same PDF; splits dense slides (> `MAX_CHARS = 1400`) on paragraph boundaries, with a hard character-split fallback for oversized single paragraphs. Outputs `data/chunks.json`.

### Step 3: Dual Indexing — Dense Embeddings & Sparse BM25 (`ingest/embed_index.py`)
Builds a ChromaDB vector index (768-dim `nomic-embed-text` embeddings via Ollama) and a parallel BM25Okapi keyword index, both keyed by the same `chunk_id`. Uses dependency injection (`EmbedFn`) so indexing logic is unit-testable without a live Ollama instance.

---

## Hybrid Retrieval & Reranking Pipeline

*(Steps 4-7 unchanged — full detail in `retrieval/` docstrings.)*

Pure lexical search (BM25) fails on synonyms/conceptual phrasing; pure vector search often misses precise keywords, acronyms (`CLAHE`), or notation (`fx`, `fy`). The two-stage pipeline (BM25 top-20 + vector top-20 → RRF fusion → top-10 → cross-encoder rerank → top-5) bridges this gap.

### Step 4: First-Stage Candidate Retrieval (`retrieval/search.py`)
### Step 5: Reciprocal Rank Fusion (`retrieval/fuse.py`)

$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Fuses on rank position (not raw score) so BM25's unbounded scores and vector search's bounded cosine similarities never need to be reconciled directly.

### Step 6: Cross-Encoder Semantic Reranking (`retrieval/rerank.py`)
`cross-encoder/ms-marco-MiniLM-L-6-v2` jointly scores `(query, document)` pairs — full cross-attention, too expensive to run on the whole corpus, cheap enough on the fused top-10.

### Step 7: End-to-End Hybrid Orchestrator & Diagnostics (`retrieval/hybrid_retrieve.py`)
`retrieve()` chains all of the above; `compare_methods()` prints BM25-only / vector-only / fused+reranked side by side for manual inspection.

---

## Citation-Enforced Answer Generation (`generation/prompt.py`)

Retrieval alone doesn't answer a question — a local LLM (**Gemma 4**, via Ollama) synthesizes the final answer from the top-5 reranked chunks, under an explicit grounding contract, with the output **verified in code**, not just requested in the prompt.

### Prompting Rules
The system prompt requires the model to:
1. Use only the provided slide excerpts — no outside knowledge.
2. Cite every claim in the exact format `[source_file.pdf, slide N]`, copied verbatim from the excerpt header.
3. Reply with a fixed refusal string if the excerpts don't cover the question.

### Citation Verification (`verify_citations`)
Prompting reduces hallucinated citations but cannot guarantee their absence — a fluent model can still fabricate a plausible-looking `[source, slide]` tag. So every citation the model outputs is parsed via regex and checked against the chunks *actually retrieved* for that query:
- **Valid**: source file matches a retrieved chunk AND the cited slide number overlaps that chunk's `slide_label` (handles merged ranges like `slides 1-2` correctly via set intersection, not string equality).
- **Invalid**: anything else — flagged, not silently trusted.

If retrieval returns zero chunks, the system refuses **without calling the model at all** — cheaper, and impossible to hallucinate from no context.

### Example (real output, Gemma 4 via Ollama)

```
Q: Why do we need two different focal lengths fx and fy?

A: You use different focal lengths ($F_x$ and $F_y$) in order to accommodate
non-equal pixel density in the x and y directions [april_2026.pdf, slide 7].
This is necessary if the pixels are rectangular [april_2026.pdf, slide 7].

valid:   [('april_2026.pdf', 'slide 7'), ('april_2026.pdf', 'slide 7')]
invalid: []
```

Both citations verified against the actually-retrieved chunk — zero invalid citations on this query.

---

## Retrieval Evaluation Harness (`eval/run_eval.py`)

Citation verification checks *generation* faithfulness; it says nothing about whether retrieval found the *right* slide in the first place. The eval harness measures that separately via **recall@k**: for each question in `eval/eval_questions.json` (with a hand-labeled expected source PDF + slide numbers), is a chunk from the expected slide anywhere in the top-k results?

```bash
python3 eval/run_eval.py
```

Reports recall@5 for BM25-only, vector-only, and hybrid+rerank side by side, so a pipeline change (chunk size, RRF's `k`, rerank cutoff) can be checked for regressions before being trusted.

> [!WARNING]
> **Current result on the initial 10-question set: all three methods score 100% recall@5.** This does **not** mean hybrid search is proven superior — it means the current eval set is too easy to discriminate between methods (every question's phrasing is close enough to the source slide's wording that any single method finds it). The set needs harder, more paraphrased questions — and coverage across all four PDFs, not just April — before recall@5 numbers are meaningful for comparing methods or reporting externally. Tracked as a follow-up before these numbers go anywhere near a resume.

`eval_questions.json` format:
```json
{
  "question": "Why do we use different focal lengths fx and fy?",
  "expected_source": "april_2026.pdf",
  "expected_slides": [7]
}
```

---

## Backend API: Authentication & Chat History (`backend/`)

The retrieval and generation pipeline is exposed through a FastAPI backend so that identity, access control, and conversation history live behind a real API rather than inside a script. Clients (a CLI, a Streamlit app, anything that speaks HTTP) never touch Chroma, BM25, or Ollama directly.

### Data Model (`backend/db.py`, SQLModel + SQLite)

```text
users (id, email UNIQUE, hashed_password, role, created_at)
   └── chat_sessions (id, user_id → users.id, title, created_at)
          └── messages (id, session_id → chat_sessions.id, role, content,
                        citations_json, created_at)
```

- **`role`** is `student` or `admin`. Registration **always** creates a `student`; promotion to `admin` is a deliberate manual step, so nobody can grant themselves access to destructive operations at signup.
- **Citations are stored per message** as a JSON list of `[source, slide_label]` pairs, so a past answer can be revisited along with exactly which slides backed it.
- SQLite keeps setup at zero. Migrating to Postgres is a connection-string change, since the table definitions are identical.

### Authentication (`backend/auth.py`)

- **Password hashing**: bcrypt via `passlib` — one-way, salted per password (hashing the same password twice yields different strings, both of which verify), deliberately slow to resist brute force.
- **JWT access tokens** (`python-jose`, HS256): stateless — the server verifies the signature instead of looking up a session. Tokens carry the user's email (`sub`) and role, and expire after 24 hours.
- **`get_current_user`**: a FastAPI dependency that validates the bearer token and loads the user. Any route that declares it becomes "login required".
- **`require_admin`**: stacks on top of `get_current_user` and rejects non-admins with `403`. Reserved for the document-ingestion endpoint, since re-indexing is expensive and destructive.
- The signing key is read from the `JWT_SECRET_KEY` environment variable. The built-in fallback is for local development only.

### Endpoints

| Method | Path | Auth | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/auth/register` | none | Create a `student` account |
| `POST` | `/auth/login` | none | Exchange email + password for a JWT (OAuth2 password form) |
| `POST` | `/chat` | user | Ask a question; returns the cited answer, persists both turns to a session |
| `GET` | `/sessions` | user | List the current user's chat sessions, newest first |
| `GET` | `/sessions/{id}` | user | Full message history of one session, with citations |

`POST /chat` accepts `{"message": "...", "session_id": 3}`. Omit `session_id` to start a new session; include it to continue one. Interactive docs are auto-generated at `/docs`.

### Access-Control Details

- A user can only read or continue **their own** sessions. Requesting another user's session returns `404`, not `403`: a `403` would confirm the session exists, a `404` reveals nothing.
- Unauthenticated requests to any protected route return `401`.

### Testability: dependency injection through FastAPI

The real answer function (`retrieve()` + `generate_answer()`) needs Ollama, so `/chat` receives it as an injected dependency (`get_answer_fn`). Tests replace it with a fake through `app.dependency_overrides`, and replace the database session with an isolated in-memory SQLite engine. As a result the whole API — registration, login, token validation, chat persistence, cross-user isolation — is tested without Ollama and without touching the real `data/app.db`.

> [!NOTE]
> **bcrypt pin.** `passlib` 1.7.4 (last released 2020) probes an internal `bcrypt.__about__` attribute that newer `bcrypt` releases removed. The mismatch makes every password check fail with a misleading "password cannot be longer than 72 bytes" error. `requirements.txt` therefore pins `bcrypt==4.0.1`.

> [!NOTE]
> **Python version.** The code is written to run on Python 3.9 (`Optional[...]` rather than `X | None`). Python 3.10+ is recommended.

### Known Limitations

- **JWTs cannot be revoked individually** before they expire; that would need a token blocklist.
- **`get_answer_fn` reloads the BM25 and Chroma indexes on every `/chat` request.** It should be cached so the indexes load once per server process.
- The index is **shared** across all users; there is no per-user document access control yet.

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
│   ├── bm25_index.pkl            # Serialized BM25Okapi index & corpus (gitignored)
│   └── app.db                    # SQLite: users, sessions, messages (gitignored)
├── ingest/
│   ├── parse_pdf.py              # Step 1: PDF extraction & cleaning
│   ├── chunk.py                  # Step 2: Slide-aware adaptive chunking
│   └── embed_index.py            # Step 3: Dense (Chroma) & Sparse (BM25) indexing
├── retrieval/
│   ├── search.py                 # Step 4: Parallel BM25 & vector search
│   ├── fuse.py                   # Step 5: Reciprocal Rank Fusion (RRF)
│   ├── rerank.py                 # Step 6: Cross-Encoder semantic reranker
│   └── hybrid_retrieve.py        # Step 7: End-to-end pipeline & comparison runner
├── generation/
│   └── prompt.py                 # Citation-enforced prompt building, generation, verification
├── eval/
│   ├── eval_questions.json       # Hand-labeled question -> expected source/slides
│   └── run_eval.py               # recall@k evaluation across BM25/vector/hybrid
├── backend/
│   ├── main.py                   # FastAPI app + startup (creates tables)
│   ├── db.py                     # SQLModel tables, engine, session dependency
│   ├── auth.py                   # Password hashing, JWT, get_current_user, require_admin
│   └── routes/
│       ├── __init__.py
│       ├── auth_routes.py        # POST /auth/register, /auth/login
│       └── chat_routes.py        # POST /chat, GET /sessions, GET /sessions/{id}
├── tests/
│   ├── test_parse_pdf.py         # Unit & integration tests for parser
│   ├── test_chunk.py             # Unit tests for chunking & merge/split rules
│   ├── test_embed_index.py       # Unit & integration tests for embeddings & BM25
│   ├── test_search.py            # Unit tests for BM25 and vector search
│   ├── test_fuse.py              # Unit tests for RRF algorithm & properties
│   ├── test_rerank.py            # Unit & integration tests for cross-encoder reranker
│   ├── test_prompt.py            # Unit tests for prompt building & citation verification
│   ├── test_eval.py              # Unit tests for the recall@k metric itself
│   ├── test_auth.py              # Unit tests for password hashing & JWT handling
│   └── test_api.py               # API integration tests (auth, chat, sessions, isolation)
├── requirements.txt              # Project dependencies
├── .gitignore                    # Ignored virtualenvs, local indexes, app.db & OS artifacts
├── LICENSE                       # License information
└── README.md
```

---

## Setup & Running

### 1. Environment & Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Ollama (Local Embeddings + Generation)

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull gemma4
```

### 3. Run the Ingestion & Indexing Pipeline

```bash
python3 ingest/parse_pdf.py
python3 ingest/chunk.py
python3 ingest/embed_index.py
```

### 4. Run Hybrid Retrieval (diagnostics)

```bash
python3 retrieval/hybrid_retrieve.py
```

### 5. Generate a Citation-Verified Answer

```bash
python3 -c "
import sys; sys.path.insert(0,'generation'); sys.path.insert(0,'retrieval'); sys.path.insert(0,'ingest')
from prompt import generate_answer, ollama_chat
from search import load_bm25_index, load_vector_index
from hybrid_retrieve import retrieve, build_chunk_lookup
bm25, chunks = load_bm25_index(); col = load_vector_index(); lk = build_chunk_lookup(chunks)
q = 'Why do we need two different focal lengths fx and fy?'
r = generate_answer(q, retrieve(q, bm25, chunks, col, lk), ollama_chat)
print(r['answer']); print('valid:', r['citations_valid']); print('invalid:', r['citations_invalid'])
"
```

### 6. Run the Retrieval Evaluation Harness

```bash
python3 eval/run_eval.py
```

### 7. Start the API Server

```bash
export JWT_SECRET_KEY="replace-with-a-long-random-string"
cd backend
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API docs. Register, click **Authorize**, log in, then try `POST /chat`. This path needs Ollama running, because `/chat` calls the real retrieval and generation pipeline.

To promote an account to admin (required later for document upload), edit the database directly:

```bash
sqlite3 data/app.db "UPDATE user SET role='admin' WHERE email='you@example.com';"
```

### 8. Run Tests

```bash
pytest -v
```

Or individually:

```bash
pytest tests/test_parse_pdf.py -v
pytest tests/test_chunk.py -v
pytest tests/test_embed_index.py -v
pytest tests/test_search.py -v
pytest tests/test_fuse.py -v
pytest tests/test_rerank.py -v
pytest tests/test_prompt.py -v
pytest tests/test_eval.py -v
pytest tests/test_auth.py -v
pytest tests/test_api.py -v
```

#### Test Architecture (72 tests)
- **Pure Unit Tests**: Fast, deterministic logic testing without network, disk, or model dependencies — text cleaning/date regex, adaptive chunk merge/split, tokenization, RRF properties, citation regex parsing, slide-range overlap checks, the recall@k metric itself, bcrypt hashing (salting, verification), and JWT creation, tamper detection and expiry.
- **Dependency-Injected Tests**: `embed_chunks()`/`search_vector()` with a fake embedding function + ephemeral ChromaDB; `rerank()` with a fake word-overlap scorer; `generate_answer()` with a fake `chat_fn`; and the whole FastAPI app with an in-memory SQLite database plus a fake `answer_fn` via `dependency_overrides`.
- **API Integration Tests** (`test_api.py`): registration defaults and duplicate rejection, login success/failure, `401` on protected routes without a token, session creation and reuse, ordered message history with citations, and cross-user session isolation.
- **Graceful Integration Tests**: real PDF parsing, real Ollama embedding, real cross-encoder scoring — each auto-skipped if its dependency (PDFs on disk / Ollama daemon / model download) isn't available.

---

## Next Steps

- [x] **Slide-Aware Ingestion**: Per-page PDF extraction with metadata extraction.
- [x] **Adaptive Chunking**: Dynamic merging of thin slides and paragraph-aware splitting of dense slides.
- [x] **Dense Vector Store**: ChromaDB persistence with Ollama `nomic-embed-text` embeddings.
- [x] **Sparse Keyword Index**: Lexical search with BM25Okapi serialized to disk.
- [x] **Two-Stage Hybrid Search**: Reciprocal Rank Fusion (RRF) combining dense and sparse candidate pools.
- [x] **Cross-Encoder Reranking**: Re-scoring top candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- [x] **Citation-Enforced Generation**: Gemma 4 (Ollama) answers grounded in retrieved slides, with code-level citation verification against retrieved chunk IDs.
- [x] **Evaluation Harness**: recall@k across BM25-only, vector-only, and hybrid+rerank.
- [x] **Auth & Backend (FastAPI)**: JWT-based auth with student/admin roles, bcrypt password hashing, `require_admin` dependency.
- [x] **Chat History**: Per-user persisted sessions and messages (SQLite), with per-message citations and cross-user isolation.
- [x] **Automated Test Suite**: 72 unit and integration tests across all pipeline and API modules.
- [ ] **Cache index loading**: load BM25/Chroma once per server process instead of on every `/chat` request.
- [ ] **Expand eval set**: grow past 10 questions, cover all four PDFs, include paraphrased (non-exact-wording) questions so recall@5 can actually discriminate between methods.
- [ ] **Fix April week-label bug**: `extract_week_label` needs to detect and label the second cover slide (13-17 April) separately.
- [ ] **Document Ingestion API**: Admin-gated (`require_admin`) endpoint to upload and re-index new PDFs.
- [ ] **Frontend**: Streamlit client (login, chat, history, admin upload) as a pure API consumer.