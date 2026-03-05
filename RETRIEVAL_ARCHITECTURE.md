# Retrieval Architecture — Graph RAG Lumen

## Overview

Every user query goes through a 7-step pipeline before the LLM generates an answer.
The graph is never bypassed — for eurocode questions, the LLM always reads from retrieved pages, never from its training data.

---

## Step-by-Step Pipeline

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
┌──────────────────────────────────┐
│  4. Hybrid Retrieval (Neo4j)     │
│                                  │
│  ┌──────────────┐  ┌──────────┐  │
│  │ Vector Search│  │  BM25    │  │
│  │ (embeddings) │  │ Fulltext │  │
│  └──────┬───────┘  └────┬─────┘  │
│         │               │        │
│         └──────┬────────┘        │
│                ▼                 │
│         RRF Merge + Dedup        │
│                ▼                 │
│       Adjacent Pages (+2)        │
└──────────────┬───────────────────┘
               │  up to 10 pages
               ▼
┌─────────────────┐
│ 5. Build Context│  Format pages + replace images with [IMG-N]
│                 │  Collect extra figures separately
└──────┬──────────┘
       │  ~80 000 chars
       ▼
┌─────────────────┐
│  6. Answer LLM  │  Stream answer (TABLES + FORMULAS in REMINDER)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  7. Post-proc   │  Restore [IMG-N] → markdown, fix LaTeX, append Abbildungen
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

### Step 4 — Hybrid Retrieval

**4a. Vector Search** (`_vector_search_pages`)
- Embedding model: `nomic-embed-text-v2-moe` (768 dimensions)
- Index: `page_embedding_index` (cosine similarity)
- Threshold: `score >= 0.5`
- Fetches `limit × 2 = 16` candidates
- Each Page node stores its full OCR text (formulas, tables, images inline)

**4b. BM25 Fulltext Search** (`_fulltext_search_pages`)
- Index: `page_fulltext` (Lucene BM25)
- Input: LLM-extracted keywords (or stopword-filtered fallback)
- Lucene special chars are sanitized before the query
- Fetches `limit × 2 = 16` candidates

**4c. Reciprocal Rank Fusion** (`_rrf_merge`)
- Merges the two ranked lists into one
- Score formula: `RRF(page) = 1/(k + rank_vector + 1) + 1/(k + rank_fulltext + 1)` with `k=60`
- Pages appearing in both lists rank higher than pages in only one list
- Deduplication by `(page_number, document)` — prevents the same page appearing twice via different chapter paths
- Final output: top `limit = 8` pages

**4d. Adjacent Pages** (`_fetch_adjacent_pages`)
- For the top 3 RRF results, fetches their `NEXT_PAGE` neighbour
- Adds up to 2 extra pages (if not already in the result set)
- Purpose: tables and continuation text often land on the page immediately after the referencing page
- Total: up to 10 pages sent to context

### Step 5 — Build Context

- Pages formatted as numbered blocks: `[N] Seite X | Chapter (Dokument: ...)\n<content>`
- **Image placeholder system**:
  - OCR-embedded image markdown (`![caption](/api/images/...)`) is replaced with short `[IMG-N]` tokens in page text
  - A mapping is stored: `{IMG-1: "![caption](/api/images/...)", ...}`
  - This prevents the LLM from seeing/copying/corrupting full image URLs
- **Extra figures**:
  - Section IDs from all retrieved pages are collected
  - `get_figures_for_sections(section_ids[:20], limit=10)` fetches Figure nodes linked to those sections
  - These figures are stored separately in `extra_figures` list (NOT injected into context)
  - Prevents the LLM from explaining/repeating the placeholder system
- Each page content capped at 8 000 chars
- Total context can reach ~80 000 chars

### Step 6 — Answer LLM

- Model: `mistral-small3.2:24b`, streamed token-by-token
- System prompt (`ANSWER_SYSTEM`) enforces:
  - Copy formulas **verbatim** from context — no rewriting
  - Copy tables as markdown — no summarization
  - Reply in user's language
  - If answer is not in context: output **only** the sentence `"Diese Information ist im bereitgestellten Kontext nicht vorhanden."` — nothing else
- REMINDER in user prompt covers only **TABLES** and **FORMULAS** (no image instructions — prevents LLM from explaining the placeholder system)

### Step 7 — Post-processing

**7a. Restore image placeholders**
- Replace `[IMG-N]` tokens in the LLM answer with real image markdown: `![caption](/api/images/...)`
- Done after LLM generation so the LLM never sees the full URLs

**7b. Fix bare LaTeX**
- `\[...\]` → `$$...$$`
- `\(...\)` → `$$...$$`
- Lines with LaTeX tokens but no `$` → wrap in `$$...$$`

**7c. Append extra figures**
- If `extra_figures` is non-empty:
  - Append `\n\n---\n**Abbildungen:**\n`
  - Append one `![caption](path)` line per figure
- Done entirely outside the LLM (no risk of hallucination or repetition)

---

## Current Limitations & Accuracy Issues

### 1. Missing Documents (Highest Impact)
**Problem:** If the relevant Eurocode part is not uploaded, the answer is always wrong.
The retrieval can only return pages that exist in the graph.

**Fix:** Upload all relevant Eurocode parts. There is no workaround in retrieval.

---

### 2. Embedding Quality for Technical Text
**Problem:** `nomic-embed-text-v2-moe` is a general-purpose model. It does not
understand Eurocode-specific terminology deeply.

**Symptoms:** Score threshold at 0.5 still allows semantically incorrect pages through.

**Possible fix:**
- Fine-tune the embedding model on Eurocode text (complex, requires labeled data)
- Use a stronger embedding model (e.g. `mxbai-embed-large`, `bge-m3`) — swap via `OLLAMA_EMBEDDING_MODEL`
- Re-embed all pages after switching model (requires graph rebuild)

---

### 3. BM25 Vocabulary Mismatch
**Problem:** German compound words behave differently in Lucene BM25.
`"Straßenbrücke"` does not match `"Brücke"` — they are different tokens.

**Symptoms:** A query using one German compound word misses pages that use a
different but synonymous compound.

**Possible fix:**
- Use a German language analyzer in Neo4j fulltext index (stemming/decomposition)
- Currently Neo4j uses the default Lucene analyzer; switching to `german` analyzer
  would require dropping and recreating the `page_fulltext` index

---

### 4. LLM Instruction Following (Hallucination)
**Problem:** `mistral-small3.2:24b` sometimes ignores the "not found" rule and generates
an answer from its training data instead.

**Symptoms:** Model answers with general Eurocode knowledge when the specific value is not in the retrieved pages.

**Mitigation:** Off-topic queries are blocked at the router before any graph search.

**Possible fix:**
- Use Claude API (`claude-sonnet-4-6`) for the answer step only, keeping local embeddings and retrieval
- Add cross-encoder re-ranker to prevent low-quality pages from being used
- Post-process: check if answer contains the "not found" phrase and if not, re-run with a stricter prompt

---

### 5. Re-ranking is Order-Only (No Score Cutoff After Merge)
**Problem:** After RRF merge, all 8 pages are passed to the LLM regardless of
their actual relevance. The 8th page might be very weakly related.

**Symptoms:** LLM context contains irrelevant pages, which can confuse the model
or cause it to quote an unrelated passage.

**Possible fix:**
- Add a cross-encoder re-ranker (e.g. `BAAI/bge-reranker-v2-m3`) as a 4th step
  after RRF, which scores each (query, page) pair and can drop low-scoring pages
- This is the biggest accuracy improvement that does not require new documents

---

### 6. Table Data Depends on Re-processing
**Problem:** Documents uploaded before `table_format="markdown"` was set store only
`[tbl-N.html]` placeholder text — no actual table content in the graph.

**Status:** Current OCR pipeline uses `table_format="markdown"`, producing pipe-delimited tables.

**Fix:** Re-upload all documents so Mistral OCR re-runs with correct table format
and table content is embedded into the page text.

---

### 7. Single-Page Granularity
**Problem:** Retrieval is at the Page level. If the answer spans two pages and
the second page scores low, it may be missed.

**Mitigation:** Adjacent page fetching for top 3 results partially addresses this.

**Possible fix:** Overlapping page windows (include page N-1 and N+1 for all top 8 results,
not just top 3). Increases context size but improves recall for multi-page answers.

---

### 8. Image URL Hallucination
**Problem:** Images are stored as hashed filenames (`/api/images/33a378d1_...`).
If the LLM sees a raw image URL in context, it may corrupt the hash during generation.

**Status:** `[IMG-N]` placeholder system (Step 5) prevents this by replacing all image
markdown before the LLM sees it. Images are restored after generation, and extra figures
are appended directly in post-processing without LLM involvement.

**Result:** LLM never sees or copies image URLs, eliminating hallucination risk for this channel.

---

## Improvement Roadmap (Priority Order)

| Priority | Action | Effort | Impact | Status |
|---|---|---|---|---|
| 1 | Upload all missing Eurocode PDFs | Low | Very high | Ongoing |
| 2 | Re-process all existing documents (table fix) | Low | High | Done (`table_format=markdown`) |
| 3 | Cross-encoder re-ranker after RRF | Medium | High | Pending |
| 4 | Switch to better embedding model (`bge-m3`) | Medium | Medium | Pending |
| 5 | German analyzer in Neo4j fulltext index | Medium | Medium | Pending |
| 6 | Use Claude API for the answer step only | Low (cost) | High (hallucination fix) | Pending |
| 7 | Expand adjacent page window (N-1, N+1 for all top 8) | Low | Medium | Pending |

---

## Data Flow in Neo4j

```
Document
  └─► Chapter (HAS_CHAPTER)
        └─► Page (CONTAINS_PAGE)  ◄── what we retrieve
              ├─► Page (NEXT_PAGE) ◄── adjacent page fetch
              └─► Section (HAS_SECTION)
                    └─► Figure (HAS_FIGURE) ◄── extra_figures fetch

Page.content = full OCR text (formulas + markdown tables + inline images)
Page.embedding = nomic-embed-text-v2-moe 768-dim vector
```

Indexes:
- `page_embedding_index` — vector index on `Page.embedding`
- `page_fulltext` — Lucene BM25 index on `Page.content`
