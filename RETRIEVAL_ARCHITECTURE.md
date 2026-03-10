# Retrieval Architecture — Graph RAG Lumen

## Overview

Every user query goes through a multi-stage pipeline before the LLM generates an answer.
The graph is never bypassed — for eurocode questions, the LLM always reads from retrieved pages, never from its training data.

The architecture uses **three parallel retrieval paths** (graph traversal, dense vector, sparse BM25) merged via Reciprocal Rank Fusion, followed by **cross-encoder reranking** and **graph enrichment**.

---

## Pipeline Diagram

```
User Question
     │
     ▼
┌─────────────┐
│  1. Route   │  greeting / offtopic / eurocode
└──────┬──────┘
       │ (eurocode path)
       ▼
┌─────────────┐
│ 2. Translate│  English → German  (mistral-small3.2:24b)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 3. Keywords │  Extract 3-6 key technical terms  (mistral-small3.2:24b)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  4. 3-Path Hybrid Search (Neo4j)                      │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Path 1:     │  │  Path 2:     │  │  Path 3:     │ │
│  │  Top-Down    │  │  Dense       │  │  Sparse      │ │
│  │  Graph       │  │  Vector      │  │  BM25        │ │
│  │  Traversal   │  │  (page +     │  │  (exact      │ │
│  │  (rich       │  │   section    │  │   Eurocode   │ │
│  │   summaries) │  │   embeddings)│  │   terms)     │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │         │
│         └────────────┬────┘─────────────────┘         │
│                      ▼                                │
│               N-way RRF Fusion                        │
│             (rewards pages in multiple paths)         │
└──────────────────┬───────────────────────────────────-┘
                   │  ~15 candidate pages
                   ▼
┌──────────────────────────┐
│  5. Cross-Encoder Rerank │  BAAI/bge-reranker-v2-m3
│     scores (query, page) │  drops irrelevant pages
└──────────┬───────────────┘
           │  6-8 pages
           ▼
┌──────────────────────────┐
│  6. Graph Enrichment     │  NEXT_PAGE adjacency
│     (post-rerank)        │  HAS_FIGURE, HAS_FORMULA
└──────────┬───────────────┘
           │
           ▼
┌─────────────────┐
│ 7. Build Context│  Format pages + replace images with [IMG-N]
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  8. Answer LLM  │  Stream answer (TABLES + FORMULAS in REMINDER)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  9. Post-proc   │  Restore [IMG-N] → markdown, fix LaTeX, append figures
└─────────────────┘
```

---

## Detailed Step Descriptions

### Step 1 — Router

- Model: `mistral-small3.2:24b`
- Classifies the message into exactly one category:
  - **greeting** — hellos, farewells, thanks, capability questions, small-talk → friendly reply, no graph search
  - **offtopic** — astronomy, cooking, history, geography, politics, sports, entertainment, general science → instant rejection
  - **eurocode** — structural engineering, Eurocodes, loads, materials, bridges, etc. → full pipeline
- Default on failure: `eurocode` (always searches rather than incorrectly rejecting)

### Step 2 — Translation (German)

- Model: `mistral-small3.2:24b` with domain-specific system prompt
- Translates English queries to German using standard Eurocode terminology
- If already German, returned unchanged
- Fallback on failure: original question used as-is

### Step 3 — Keyword Extraction

- Model: `mistral-small3.2:24b` with few-shot examples
- Extracts 3-6 key technical terms from the query
- Strips: question words, articles, prepositions, generic verbs
- Keeps: Eurocode terms, load types, norm numbers, numeric values, abbreviations
- Fallback on failure: static German stopword filter
- Keywords used exclusively for BM25 fulltext search; vector search uses the full German query

### Step 4 — 3-Path Hybrid Search

All three paths share a single embedding call — no redundant computation.

**Path 1: Top-Down Graph Traversal** (`_topdown_search`)

Uses rich chapter and document summaries to navigate the hierarchy:

1. Vector search on `document_embedding_index` → top 3 documents (configurable via `TOPDOWN_MAX_DOCS`)
2. Vector search on `chapter_embedding_index` filtered to those documents → top 5 chapters (configurable via `TOPDOWN_MAX_CHAPTERS`)
3. Fetch all pages from those chapters via `CONTAINS_PAGE` relationship

Rich summaries contain:
- All section titles with first paragraphs
- Formula symbols (λ, γ, σ, ψ, ...)
- Table captions
- LLM-generated 150-word description

Graceful degradation: returns `[]` if summary indexes don't exist yet.

**Path 2: Dense Vector** (`_dense_vector_search`)

Dual-level vector search for maximum recall:

- Sub-path A: `page_embedding_index` — direct page-level similarity (cosine, threshold ≥ 0.5)
- Sub-path B: `section_embedding_index` — section-level similarity resolved to pages via `HAS_SECTION`
- Results merged with 2-way RRF before joining the main fusion

**Path 3: Sparse BM25** (`_fulltext_search_pages`)

Exact term matching — critical for Eurocode references:

- Index: `page_fulltext` (Lucene BM25)
- Input: LLM-extracted keywords (or stopword-filtered fallback)
- Excellent for: section numbers ("3.2.1"), load combinations ("ψ₀ · Qk"), norm references ("EN 1993-1-1")
- Lucene special chars sanitized before query

**N-way RRF Fusion** (`_rrf_merge_multi`)

- Score formula: `RRF(page) = Σ 1/(k + rank_i + 1)` across all paths where page appears
- `k = 60` (configurable via `HYBRID_RRF_K`)
- Pages appearing in multiple paths rank higher — the key insight that improves accuracy
- Deduplication by `(page_number, document)` key
- Output: ~15 candidate pages (configurable via `HYBRID_CANDIDATES_PER_PATH`)

### Step 5 — Cross-Encoder Reranking

- Model: `BAAI/bge-reranker-v2-m3` (multilingual, supports German)
- Runs on CPU (GPU reserved for Ollama)
- Each page split into overlapping chunks (4000 chars, 400 overlap)
- All `(query, chunk)` pairs scored in one batched inference call
- **Max chunk score** used as page score — ensures relevant content anywhere in the page is captured
- Filters: top `RERANKER_TOP_K` (default 8) pages with score ≥ `RERANKER_THRESHOLD` (default -5.0)
- Result: 6-8 high-confidence pages

### Step 6 — Graph Enrichment (Post-Rerank)

After reranking selects the best pages, the graph expands context for completeness:

- **NEXT_PAGE adjacency**: Fetches the next page for top results (tables often span pages). Max `ENRICHMENT_MAX_ADJACENT` (default 2) extra pages.
- **HAS_FIGURE**: Collects figures from sections on retrieved pages. Max `ENRICHMENT_MAX_FIGURES` (default 10).
- **HAS_FORMULA**: Collects formulas from sections on retrieved pages. Max `ENRICHMENT_MAX_FORMULAS` (default 10).

This is enrichment, not retrieval — it adds context without changing the ranking.

### Step 7 — Build Context

- Pages formatted as numbered blocks: `[N] Seite X | Chapter (Dokument: ...)\n<content>`
- **Image placeholder system**:
  - OCR-embedded image markdown (`![caption](/api/images/...)`) replaced with `[IMG-N]` tokens
  - Mapping stored: `{IMG-1: "![caption](/api/images/...)", ...}`
  - Prevents LLM from seeing/corrupting full image URLs
- **Extra figures**: stored separately (not injected into LLM context)

### Step 8 — Answer LLM

- Model: `mistral-small3.2:24b`, streamed token-by-token
- System prompt (`ANSWER_SYSTEM`) enforces:
  - Copy formulas **verbatim** from context — no rewriting
  - Copy tables as markdown — no summarization
  - Reply in user's language
  - If answer is not in context: output **only** `"Diese Information ist im bereitgestellten Kontext nicht vorhanden."`

### Step 9 — Post-processing

- Restore `[IMG-N]` → real image markdown
- Fix bare LaTeX (`\[...\]` → `$$...$$`, etc.)
- Append extra figures section

---

## Embeddings

### Auto-Detected Dimensions

Embedding dimensions are auto-detected at runtime by probing the configured model. No hardcoded dimension values — works with any embedding model (bge-m3 = 1024-dim, nomic = 768-dim, etc.).

### Chunked Full-Text Embedding

All node types are embedded using chunked embedding (`_embed_chunked`) — **no text truncation**:

1. Text split into overlapping chunks (configurable via `EMBEDDING_CHUNK_SIZE`, `EMBEDDING_CHUNK_OVERLAP`)
2. Each chunk embedded independently
3. Combined using strategy:
   - **mean**: equal-weight average (pages, tables, figures, formulas, documents)
   - **weighted**: first chunk gets 2x weight (sections, chapters — title/intro most important)

### Embedded Node Types (8 vector indexes)

| Node | Index | Strategy | Source Text |
|------|-------|----------|-------------|
| Document | `document_embedding_index` | mean | Rich summary (all chapter summaries concatenated) |
| Chapter | `chapter_embedding_index` | weighted | Rich summary (sections, formulas, tables, LLM description) |
| Section | `section_embedding_index` | weighted | title + full_text |
| Page | `page_embedding_index` | mean | Full OCR content |
| Table | `table_embedding_index` | mean | section_title + caption + content + annotation |
| Figure | `figure_embedding_index` | mean | section_title + caption + description + annotation |
| Formula | `formula_embedding_index` | mean | section_title + unicode/latex |
| Concept | `concept_embedding_index` | mean | name + description |

---

## Rich Chapter & Document Summaries

Generated at build time (not query time) via `POST /api/graph/generate-summaries`.

### Chapter Summary Structure

```
Kapitel {number}: {title}

Abschnitte:
- {section_title}: {first_paragraph}
- {section_title}: {first_paragraph}
...

Formelsymbole: λ, γ, σ, ψ, ...

Tabellenüberschriften:
- {caption_1}
- {caption_2}

Zusammenfassung:
{LLM-generated 150-word description based on full section texts}
```

### Document Summary

Concatenation of all chapter summaries with document metadata header. No additional LLM call.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `TOPDOWN_MAX_DOCS` | 3 | Max documents in graph traversal path |
| `TOPDOWN_MAX_CHAPTERS` | 5 | Max chapters in graph traversal path |
| `HYBRID_RRF_K` | 60 | RRF smoothing parameter |
| `HYBRID_CANDIDATES_PER_PATH` | 15 | Candidates per search path before fusion |
| `RERANKER_ENABLED` | true | Enable/disable cross-encoder reranking |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Cross-encoder model |
| `RERANKER_TOP_K` | 8 | Max pages after reranking |
| `RERANKER_THRESHOLD` | -5.0 | Score cutoff for reranker |
| `ENRICHMENT_MAX_ADJACENT` | 2 | Max adjacent pages added |
| `ENRICHMENT_MAX_FIGURES` | 10 | Max figures from graph enrichment |
| `ENRICHMENT_MAX_FORMULAS` | 10 | Max formulas from graph enrichment |
| `EMBEDDING_CHUNK_SIZE` | 6000 | Chars per embedding chunk |
| `EMBEDDING_CHUNK_OVERLAP` | 500 | Overlap between embedding chunks |
| `SUMMARY_MAX_WORDS` | 150 | LLM summary length |

---

## Data Flow in Neo4j

```
Document  (summary_text, embedding)
  └─► Chapter (HAS_CHAPTER)  (summary_text, embedding)
        └─► Page (CONTAINS_PAGE)  ◄── what we retrieve
              ├─► Page (NEXT_PAGE) ◄── enrichment: adjacent pages
              └─► Section (HAS_SECTION)  (embedding)
                    ├─► Figure (HAS_FIGURE)  ◄── enrichment: figures
                    └─► Formula (HAS_FORMULA) ◄── enrichment: formulas

Page.content = full OCR text (formulas + markdown tables + inline images)
Page.embedding = chunked full-text embedding (auto-detected dimensions)
```

Indexes:
- 8 vector indexes (auto-detected dimensions, cosine similarity)
- 6 fulltext indexes (Lucene BM25)
- 5 property indexes + 9 unique constraints

---

## Build Pipeline

```
rebuild-graph.sh pipeline:
  1. Clear graph (DETACH DELETE all nodes)
  2. Drop all vector indexes
  3. Build graph from JSON (ingest all documents)
  4. Generate summaries (chapter + document)
  5. Generate embeddings (all 8 node types)
```

API endpoints:
- `POST /api/graph/build` — ingest JSON files
- `POST /api/graph/generate-summaries` — create rich summaries
- `POST /api/graph/generate-embeddings` — embed all nodes
- `POST /api/graph/compute-similarity` — SEMANTICALLY_SIMILAR edges
