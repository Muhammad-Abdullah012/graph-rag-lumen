# Retrieval Architecture — Graph RAG Lumen

## Overview

Every user query goes through a multi-stage pipeline before the LLM generates an answer.
The graph is never bypassed — for eurocode questions, the LLM always reads from retrieved pages, never from its training data.

The architecture uses **four parallel retrieval paths** (intent-targeted, graph traversal, dense vector, sparse BM25) plus an optional **SEMANTICALLY_SIMILAR expansion**, all merged via Reciprocal Rank Fusion, followed by **cross-encoder reranking** and **graph enrichment**.

---

## Pipeline Diagram

```
User Question
     │
     ▼
┌─────────────┐
│  1. Route   │  greeting / question / offtopic
└──────┬──────┘
       │ (question path)
       ▼
┌─────────────┐
│ 2. Translate│  English → German  (mistral-small3.2:24b)
└──────┬──────┘
       │
       ▼
┌───────────────────────────────────────────────────────────────────┐
│  3. 4-Path Hybrid Search (Neo4j)                                   │
│                                                                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐      │
│  │  Path 0:  │  │  Path 1:  │  │  Path 2:  │  │  Path 3:  │      │
│  │  Intent   │  │  Top-Down │  │  Dense    │  │  Sparse   │      │
│  │ Targeted  │  │  Graph    │  │  Vector   │  │  BM25     │      │
│  │ (2× RRF   │  │ Traversal │  │ (page +   │  │ (German   │      │
│  │  weight)  │  │  (rich    │  │  section  │  │  terms)   │      │
│  │           │  │ summaries)│  │ embeddings│  │           │      │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘      │
│        └──────────────┴──────────────┴──────────────┘             │
│                              ▼                                     │
│                  Preliminary RRF merge (top 3)                     │
│                              │                                     │
│              ┌───────────────┘                                     │
│              ▼                                                     │
│  ┌─────────────────────────┐                                       │
│  │  Path 4 (conditional):  │  SEMANTICALLY_SIMILAR edges from      │
│  │  Cross-doc expansion    │  top preliminary sections             │
│  └───────────┬─────────────┘                                       │
│              │ (if SEMANTIC_SIMILAR_ENABLED and results found)     │
│              ▼                                                     │
│         Final N-way RRF Fusion                                     │
│       (rewards pages in multiple paths)                            │
└──────────────────────┬────────────────────────────────────────────┘
                       │  ~20 candidate pages
                       ▼
┌──────────────────────────┐
│  4. Cross-Encoder Rerank │  BAAI/bge-reranker-v2-m3
│     scores (query, page) │  drops irrelevant pages
└──────────┬───────────────┘
           │  up to 12 pages
           ▼
┌──────────────────────────┐
│  5. Graph Enrichment     │  NEXT_PAGE adjacency
│     (post-rerank)        │  HAS_FIGURE, HAS_FORMULA
└──────────┬───────────────┘
           │
           ▼
┌─────────────────┐
│ 6. Budget Trim  │  Drop lowest-ranked pages exceeding TOTAL_CONTEXT_BUDGET
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 7. Build Context│  Format pages + replace images with [IMG-N]
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│  8. Hallucination    │  Skip LLM if context < HALLUCINATION_MIN_CONTEXT_CHARS
│     Gate             │  → instant refusal message
└──────┬───────────────┘
       │
       ▼
┌─────────────────┐
│  9. Answer LLM  │  Stream answer (TABLES + FORMULAS in REMINDER)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  10. Post-proc  │  Restore [IMG-N] → markdown, fix LaTeX, append figures
└─────────────────┘
```

---

## Detailed Step Descriptions

### Step 1 — Router

- Model: `mistral-small3.2:24b`
- Classifies the message into exactly one category:
  - **greeting** — hellos, farewells, thanks, capability questions, small-talk → friendly reply, no graph search
  - **offtopic** — astronomy, cooking, history, geography, politics, sports, entertainment, general science → instant rejection
  - **question** — structural engineering, Eurocodes, loads, materials, bridges, etc. → full pipeline
- Default on failure: `question` (always searches rather than incorrectly rejecting)

### Step 2 — Translation (German)

- Model: `mistral-small3.2:24b` with domain-specific system prompt
- Translates English queries to German using standard Eurocode terminology
- If already German, returned unchanged
- Fallback on failure: original question used as-is
- The German translation is passed exclusively to the BM25 path — all vector paths embed the **original** query (bge-m3 is multilingual)

### Step 3 — 4-Path Hybrid Search

All vector paths share a single embedding call — no redundant computation. The original (untranslated) query is embedded; only BM25 uses the German translation.

**Path 0: Intent-Targeted Search** (`_intent_targeted_search`)

Detects explicit structured references in the query and runs exact-match Cypher lookups:

- Regex patterns detect: table references ("Tabelle 3.1"), section numbers ("3.2.1"), figure references ("Bild 4.5"), norm identifiers ("EN 1993-1-1")
- When a norm reference is detected, lookups are scoped to that document
- Returns pages containing the matched Table / Section / Figure nodes
- Results are added **twice** to the RRF input list, giving them **double weight** in fusion
- Toggleable via `QUERY_INTENT_ENABLED`; returns `[]` when disabled

**Path 1: Top-Down Graph Traversal** (`_topdown_search`)

Uses rich chapter and document summaries to navigate the hierarchy:

1. Vector search on `document_embedding_index` → top `TOPDOWN_MAX_DOCS` documents (default 5)
2. Vector search on `chapter_embedding_index` filtered to those documents → top `TOPDOWN_MAX_CHAPTERS` chapters (default 8), excluding chapter types in `SEARCH_EXCLUDED_CHAPTER_TYPES`
3. Fetch all pages from those chapters via `CONTAINS_PAGE` relationship

Rich summaries contain:
- All section titles with first paragraphs
- Formula symbols (λ, γ, σ, ψ, ...)
- Table captions
- LLM-generated description

Graceful degradation: if top-down yields fewer than `TOPDOWN_FALLBACK_MIN` pages, falls back to a broader chapter vector search. Returns `[]` if summary indexes don't exist yet.

**Path 2: Dense Vector** (`_dense_vector_search`)

Dual-level vector search for maximum recall:

- Sub-path A: `page_embedding_index` — direct page-level similarity (cosine, threshold ≥ `VECTOR_PAGE_THRESHOLD`, default 0.45)
- Sub-path B: `section_embedding_index` — section-level similarity resolved to pages via `HAS_SECTION`
- Results merged with 2-way RRF before joining the main fusion

**Path 3: Sparse BM25** (`_fulltext_search_pages`)

Exact term matching — critical for Eurocode references:

- Index: `page_fulltext` (Lucene BM25)
- Input: German translation of the query (`bm25_query` parameter)
- Excellent for: section numbers ("3.2.1"), load combinations ("ψ₀ · Qk"), norm references ("EN 1993-1-1")
- Lucene special chars sanitized; terms shorter than `BM25_FUZZY_MIN_LENGTH` are not fuzzified

**Path 4 (conditional): SEMANTICALLY_SIMILAR Expansion** (`_semantic_similar_expansion`)

Cross-document coverage via pre-computed similarity edges:

1. A **preliminary RRF merge** of the top 3 pages from paths 0–3 is computed first
2. Section IDs from those top pages are collected
3. `SEMANTICALLY_SIMILAR` edges (score ≥ `SEMANTIC_SIMILAR_MIN_SCORE`, default 0.80) are followed to find conceptually related sections in other documents
4. Pages containing those related sections are added as Path 4
- Only runs if `SEMANTIC_SIMILAR_ENABLED` is true and preliminary results contain section IDs

**Final N-way RRF Fusion** (`_rrf_merge_multi`)

- Score formula: `RRF(page) = Σ 1/(k + rank_i + 1)` across all paths where page appears
- `k = 60` (configurable via `HYBRID_RRF_K`)
- Path 0 (intent) appears twice → effectively 2× weight
- Pages appearing in multiple paths rank higher
- Deduplication by `(page_number, document)` key
- Output: ~20 candidate pages (configurable via `HYBRID_CANDIDATES_PER_PATH`)

### Step 4 — Cross-Encoder Reranking

- Model: `BAAI/bge-reranker-v2-m3` (multilingual, supports German)
- Runs on CPU (GPU reserved for Ollama)
- Each page split into overlapping chunks (`RERANKER_CHUNK_SIZE` = 4000 chars, `RERANKER_CHUNK_OVERLAP` = 400)
- All `(query, chunk)` pairs scored in one batched inference call
- **Max chunk score** used as page score — ensures relevant content anywhere in the page is captured
- Filters: top `RERANKER_TOP_K` (default 12) pages with score ≥ `RERANKER_THRESHOLD` (default 0.0)
- Result: up to 12 high-confidence pages

### Step 5 — Graph Enrichment (Post-Rerank)

After reranking selects the best pages, the graph expands context for completeness:

- **NEXT_PAGE adjacency**: Fetches the next page for top results (tables often span pages). Max `ENRICHMENT_MAX_ADJACENT` (default 4) extra pages.
- **HAS_FIGURE**: Collects figures from sections on retrieved pages. Max `ENRICHMENT_MAX_FIGURES` (default 10).
- **HAS_FORMULA**: Collects formulas from sections on retrieved pages. Max `ENRICHMENT_MAX_FORMULAS` (default 10).

This is enrichment, not retrieval — it adds context without changing the ranking.

### Step 6 — Context Budget Trim

- Iterates pages in ranked order, accumulating character counts
- Stops adding pages once the total exceeds `TOTAL_CONTEXT_BUDGET` (default 80 000 chars)
- The first page is always included regardless of size
- Ensures the LLM prompt stays within a predictable token budget

### Step 7 — Build Context

- Pages formatted as numbered blocks: `[N] Seite X | Chapter (Dokument: ...)\n<content>`
- Relevance score appended to header if available: `[Relevanz: 95.3]`
- **Image placeholder system**:
  - OCR-embedded image markdown (`![caption](/api/images/...)`) replaced with `[IMG-N]` tokens
  - Mapping stored: `{IMG-1: "![caption](/api/images/...)", ...}`
  - Prevents LLM from seeing/corrupting full image URLs
- **Extra figures**: stored separately, appended after the answer
- **Extra formulas**: appended as a `RELATED FORMULAS:` section at the end of the context

### Step 8 — Hallucination Gate

- If context starts with the no-results marker or is shorter than `HALLUCINATION_MIN_CONTEXT_CHARS` (default 200 chars), the LLM call is **skipped entirely**
- Returns a fixed refusal string immediately, avoiding hallucinated answers on empty context

### Step 9 — Answer LLM

- Model: `mistral-small3.2:24b`, streamed token-by-token
- System prompt (`ANSWER_SYSTEM`) enforces:
  - Scan all context pages before answering (explicit two-step: extract then answer)
  - Copy formulas **verbatim** from context — no rewriting
  - Copy tables as markdown — no summarization
  - Reply in user's language
  - If answer is not in context: output the language-appropriate not-found message

### Step 10 — Post-processing

- Restore `[IMG-N]` → real image markdown
- Fix bare LaTeX (`\[...\]` → `$$...$$`, `\(...\)` → `$$...$$`, unwrapped LaTeX lines → `$$...$$`)
- Append extra figures section (`---\n**Abbildungen:**\n...`)

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
{LLM-generated description based on full section texts}
```

### Document Summary

Concatenation of all chapter summaries with document metadata header. No additional LLM call.

---

## Configuration (Environment Variables)

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `QUERY_INTENT_ENABLED` | true | Enable Path 0 (intent-targeted exact lookup) |
| `TOPDOWN_MAX_DOCS` | 5 | Max documents in graph traversal path |
| `TOPDOWN_MAX_CHAPTERS` | 8 | Max chapters in graph traversal path |
| `TOPDOWN_FALLBACK_MIN` | 3 | Min top-down results before triggering broader fallback |
| `SEARCH_EXCLUDED_CHAPTER_TYPES` | table_of_contents,other | Chapter types skipped in top-down path |
| `VECTOR_PAGE_THRESHOLD` | 0.45 | Cosine similarity cutoff for page vector search |
| `VECTOR_DOC_THRESHOLD` | 0.35 | Cosine similarity cutoff for document vector search |
| `HYBRID_RRF_K` | 60 | RRF smoothing parameter |
| `HYBRID_CANDIDATES_PER_PATH` | 20 | Candidates per search path before fusion |
| `BM25_FUZZY_MIN_LENGTH` | 8 | Min term length for BM25 fuzzy matching |
| `SEMANTIC_SIMILAR_ENABLED` | true | Enable Path 4 (cross-doc SEMANTICALLY_SIMILAR expansion) |
| `SEMANTIC_SIMILAR_MIN_SCORE` | 0.80 | Min edge score for SEMANTICALLY_SIMILAR traversal |

### Reranking

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | true | Enable/disable cross-encoder reranking |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Cross-encoder model |
| `RERANKER_TOP_K` | 12 | Max pages after reranking |
| `RERANKER_THRESHOLD` | 0.0 | Score cutoff for reranker |
| `RERANKER_CHUNK_SIZE` | 4000 | Chars per reranker chunk |
| `RERANKER_CHUNK_OVERLAP` | 400 | Overlap between reranker chunks |

### Enrichment & Context

| Variable | Default | Description |
|----------|---------|-------------|
| `ENRICHMENT_MAX_ADJACENT` | 4 | Max adjacent pages added |
| `ENRICHMENT_MAX_FIGURES` | 10 | Max figures from graph enrichment |
| `ENRICHMENT_MAX_FORMULAS` | 10 | Max formulas from graph enrichment |
| `TOTAL_CONTEXT_BUDGET` | 80000 | Max total chars of page content sent to LLM |
| `HALLUCINATION_MIN_CONTEXT_CHARS` | 200 | Min context chars required to call the LLM |
| `PAGE_CONTENT_LIMIT` | 50000 | Max chars returned per page from Neo4j |
| `SECTION_CONTENT_LIMIT` | 50000 | Max chars returned per section from Neo4j |

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_CHUNK_SIZE` | 6000 | Chars per embedding chunk |
| `EMBEDDING_CHUNK_OVERLAP` | 500 | Overlap between embedding chunks |

### Summaries

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARY_MAX_WORDS` | 500 | LLM summary length target |
| `SUMMARY_CHAPTER_TYPES` | main_chapter,appendix | Chapter types included in summary generation |
| `SUMMARY_LLM_INPUT_LIMIT` | 24000 | Max chars of section text fed to summary LLM |

---

## Data Flow in Neo4j

```
Document  (summary_text, embedding)
  └─► Chapter (HAS_CHAPTER)  (summary_text, embedding)
        └─► Page (CONTAINS_PAGE)  ◄── what we retrieve
              ├─► Page (NEXT_PAGE) ◄── enrichment: adjacent pages
              └─► Section (HAS_SECTION)  (embedding)
                    ├─► Figure (HAS_FIGURE)  ◄── enrichment: figures
                    ├─► Formula (HAS_FORMULA) ◄── enrichment: formulas
                    ├─► Table (HAS_TABLE) ◄── intent lookup
                    └─► Section (SEMANTICALLY_SIMILAR) ◄── Path 4 expansion

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
