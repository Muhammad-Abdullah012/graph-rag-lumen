"""Graph Querier – Query the Neo4j Graph-RAG knowledge graph.

Supports
========
- Structural navigation  (Document → Chapter → Section)
- Full-text search        (Section, Concept, Table, Formula)
- Semantic (vector) search on Section and Concept embeddings
- Concept-aware retrieval (MENTIONS, RELATED_TO)
- Cross-document linking  (SEMANTICALLY_SIMILAR)
- Graph statistics
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from backend.app.modules.database import get_neo4j_connection

logger = logging.getLogger(__name__)


class GraphQuerier:
    """Query the Graph-RAG knowledge graph stored in Neo4j."""

    def __init__(self):
        self.db = get_neo4j_connection()

    # Common German function words that add noise to BM25 fulltext search.
    # Technical terms, section numbers, load names etc. are kept.
    _DE_STOPWORDS: frozenset = frozenset({
        "welche", "welcher", "welches", "welchen", "welchem",
        "sind", "sein", "ist", "war", "wird", "wird", "werden",
        "für", "bei", "einer", "einem", "eines", "eine", "ein",
        "zu", "zum", "zur", "nach", "aus", "mit", "von", "vom",
        "in", "im", "an", "am", "auf", "über", "unter", "durch",
        "die", "der", "das", "dem", "den", "des",
        "und", "oder", "aber", "auch", "noch", "nicht", "kein", "keine",
        "wie", "was", "wann", "wo", "warum", "welche",
        "können", "dürfen", "müssen", "sollen", "muss", "soll", "darf",
        "hat", "haben", "hatte", "hatten",
        "sich", "es", "sie", "er", "wir", "ich", "ihr",
        "sowie", "um", "als", "dass", "ob", "wenn", "weil",
        "berücksichtigen", "berücksichtigt", "verwenden", "verwendet",
        "bestimmen", "bestimmt", "ermitteln", "ermittelt",
        "angeben", "angegeben", "anwenden", "angewendet",
        "geben", "gibt", "nehmen", "ansetzen",
    })

    @staticmethod
    def _sanitize_lucene(query: str) -> str:
        """Strip Lucene fulltext query special characters.

        Characters like + - & | ! ( ) { } [ ] ^ " ~ * ? : \\ / are Lucene
        operators that cause ParseException when present in natural-language
        queries (e.g. "Gewölbe- und Betonbrücken" or "Φ2 und Φ3:").
        We simply remove them so the index receives plain word tokens.
        """
        cleaned = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', ' ', query)
        return ' '.join(cleaned.split()) or '*'

    @classmethod
    def _to_keywords(cls, query: str) -> str:
        """Extract key technical terms from a natural-language query for BM25 search.

        Strips German stopwords and short filler words so that fulltext search
        matches on specific technical terms (e.g. "Kombinationswert Temperatur
        Straßenbrücke") rather than common sentence words that appear on hundreds
        of pages.  Falls back to the sanitized query if nothing survives.
        """
        sanitized = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', ' ', query)
        words = sanitized.split()
        keywords = [
            w for w in words
            if w.lower().rstrip(".,?!:") not in cls._DE_STOPWORDS
            and (
                len(w) >= 4                          # normal words
                or re.match(r'^\d[\d.,]*$', w)       # numbers: 71, 8.3, 1.5
                or (len(w) >= 2 and w[0].isupper())  # abbrevs / variables: EC, ψ, G
            )
        ]
        return ' '.join(keywords) if keywords else sanitized

    # ================================================================== #
    #  1. STRUCTURAL NAVIGATION
    # ================================================================== #

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with chapter and section counts."""
        return self.db.execute_query(
            """MATCH (d:Document)
               OPTIONAL MATCH (d)-[:HAS_CHAPTER]->(ch:Chapter)
               WITH d, count(DISTINCT ch) AS ch_count
               OPTIONAL MATCH (d)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s:Section)
               RETURN d.id AS id,
                      d.filename AS filename,
                      d.document_type AS document_type,
                      d.eurocode_part AS eurocode_part,
                      d.language AS language,
                      ch_count AS chapters,
                      count(DISTINCT s) AS sections
               ORDER BY d.filename""",
        )

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return a single Document by its id."""
        results = self.db.execute_query(
            """MATCH (d:Document {id: $id})
               RETURN d {.*} AS doc""",
            {"id": doc_id},
        )
        return results[0] if results else None

    def list_chapters(self, document_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """List chapters, optionally filtered by document keyword."""
        if document_keyword:
            return self.db.execute_query(
                """MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)
                   WHERE toLower(d.filename) CONTAINS toLower($kw)
                      OR toLower(d.id) CONTAINS toLower($kw)
                   RETURN ch.id AS id, ch.number AS number, ch.title AS title,
                          ch.chapter_type AS chapter_type, d.filename AS document
                   ORDER BY ch.number""",
                {"kw": document_keyword},
            )
        return self.db.execute_query(
            """MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)
               RETURN ch.id AS id, ch.number AS number, ch.title AS title,
                      ch.chapter_type AS chapter_type, d.filename AS document
               ORDER BY d.filename, ch.number""",
        )

    def get_chapter_sections(self, chapter_id: str) -> List[Dict[str, Any]]:
        """Get all sections under a chapter (via pages)."""
        return self.db.execute_query(
            """MATCH (ch:Chapter {id: $id})-[:HAS_SECTION]->(s:Section)
               RETURN DISTINCT s.id AS id, s.number AS number, s.title AS title,
                      s.level AS level, s.start_page AS start_page,
                      s.end_page AS end_page,
                      s.content_preview AS preview
               ORDER BY s.start_page, s.number""",
            {"id": chapter_id},
        )

    def list_sections(self, document_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sections, optionally filtered by document keyword."""
        if document_keyword:
            return self.db.execute_query(
                """MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s:Section)
                   WHERE toLower(d.filename) CONTAINS toLower($kw)
                   RETURN DISTINCT s.title AS section, d.filename AS document
                   ORDER BY s.title
                   UNION
                   MATCH (s:Section)
                   WHERE toLower(s.title) CONTAINS toLower($kw)
                   OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Chapter)<-[:HAS_CHAPTER]-(d:Document)
                   RETURN DISTINCT s.title AS section, d.filename AS document
                   ORDER BY s.title
                   LIMIT 50""",
                {"kw": document_keyword},
            )
        return self.db.execute_query(
            """MATCH (s:Section)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN DISTINCT s.title AS section, d.filename AS document
               ORDER BY d.filename, s.title
               LIMIT 200""",
        )

    def get_section_detail(self, section_id: str) -> Optional[Dict[str, Any]]:
        """Get full section detail including tables, figures, formulas, subsections."""
        results = self.db.execute_query(
            """MATCH (s:Section {id: $id})
               OPTIONAL MATCH (s)-[:HAS_TABLE]->(t:Table)
               OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
               OPTIONAL MATCH (s)-[:HAS_FORMULA]->(frm:Formula)
               OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(sub:Section)
               OPTIONAL MATCH (s)-[:MENTIONS]->(c:Concept)
               RETURN s.id AS id, s.title AS title, s.number AS number,
                      s.level AS level, s.full_text AS full_text,
                      s.content_preview AS preview,
                      s.start_page AS start_page, s.end_page AS end_page,
                      collect(DISTINCT {id: t.id, number: t.number,
                              caption: t.caption, content: t.content}) AS tables,
                      collect(DISTINCT {id: f.id, number: f.number,
                              caption: f.caption, description: f.description}) AS figures,
                      collect(DISTINCT {id: frm.id, latex: frm.latex,
                              formula: coalesce(frm.unicode, frm.latex)}) AS formulas,
                      collect(DISTINCT {id: sub.id, title: sub.title,
                              number: sub.number}) AS subsections,
                      collect(DISTINCT {name: c.name,
                              description: c.description}) AS concepts""",
            {"id": section_id},
        )
        return results[0] if results else None

    # ================================================================== #
    #  2. SECTION SEARCH (full-text + keyword)
    # ================================================================== #

    def search_sections(self, keyword: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Search sections by title / content via full-text index, fallback CONTAINS.

        Returns section content (4000 chars) plus all associated formula LaTeX
        so the LLM sees the exact formulas belonging to each section.
        """
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('section_fulltext', $query)
                   YIELD node, score
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(node)
                   OPTIONAL MATCH (node)-[:HAS_FORMULA]->(frm:Formula)
                   RETURN node.id AS id, node.title AS title,
                          substring(node.full_text, 0, 4000) AS content,
                          node.start_page AS page,
                          d.filename AS document, score,
                          collect(DISTINCT frm.latex) AS formula_latex
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": self._sanitize_lucene(keyword), "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (s:Section)
               WHERE toLower(s.title) CONTAINS toLower($kw)
                  OR toLower(s.full_text) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               OPTIONAL MATCH (s)-[:HAS_FORMULA]->(frm:Formula)
               RETURN s.id AS id, s.title AS title,
                      substring(s.full_text, 0, 4000) AS content,
                      s.start_page AS page,
                      d.filename AS document, 0.5 AS score,
                      collect(DISTINCT frm.latex) AS formula_latex
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    # ================================================================== #
    #  3. CONCEPT QUERIES
    # ================================================================== #

    def lookup_concept(self, name: str) -> List[Dict[str, Any]]:
        """Look up a concept by exact normalized name."""
        return self.db.execute_query(
            """MATCH (c:Concept)
               WHERE c.normalized_name = toLower($name) OR c.name = $name
               OPTIONAL MATCH (c)<-[m:MENTIONS]-(s:Section)
               OPTIONAL MATCH (c)-[r:RELATED_TO]-(rel:Concept)
               RETURN c.name AS concept, c.description AS description,
                      collect(DISTINCT {section: s.title,
                              confidence: m.confidence}) AS mentioned_in,
                      collect(DISTINCT rel.name) AS related_concepts""",
            {"name": name},
        )

    def search_concepts(self, keyword: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Search concepts by keyword.

        Strategy order:
          1. Vector search on concept_embedding_index  (semantic)
          2. Full-text search on concept_fulltext index
          3. CONTAINS fallback
        """
        # Strategy 1: vector search
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'concept_embedding_index', $limit, $embedding
                       ) YIELD node, score
                       RETURN node.name AS concept, node.description AS description,
                              node.normalized_name AS normalized_name, score
                       ORDER BY score DESC""",
                    {"limit": limit, "embedding": embedding},
                )
                if results:
                    return results
        except Exception:
            pass

        # Strategy 2: full-text search
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('concept_fulltext', $query)
                   YIELD node, score
                   RETURN node.name AS concept, node.description AS description,
                          node.normalized_name AS normalized_name, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": self._sanitize_lucene(keyword), "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        # Strategy 3: CONTAINS fallback
        return self.db.execute_query(
            """MATCH (c:Concept)
               WHERE toLower(c.name) CONTAINS toLower($kw)
                  OR toLower(c.description) CONTAINS toLower($kw)
               RETURN c.name AS concept, c.description AS description,
                      c.normalized_name AS normalized_name, 0.5 AS score
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def get_concept_sections(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get all sections that MENTION a given concept."""
        return self.db.execute_query(
            """MATCH (c:Concept)<-[m:MENTIONS]-(s:Section)
               WHERE c.normalized_name = toLower($name) OR c.name = $name
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               OPTIONAL MATCH (s)-[:HAS_FORMULA]->(frm:Formula)
               RETURN s.id AS id, s.title AS title,
                      s.title AS section,
                      substring(s.full_text, 0, 4000) AS content,
                      s.content_preview AS preview,
                      m.confidence AS confidence, d.filename AS document,
                      s.start_page AS page,
                      collect(DISTINCT frm.latex) AS formula_latex
               ORDER BY m.confidence DESC
               LIMIT 20""",
            {"name": concept_name},
        )

    def get_related_concepts(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get concepts related to the given concept (co-occurrence)."""
        return self.db.execute_query(
            """MATCH (c:Concept)-[r:RELATED_TO]-(rel:Concept)
               WHERE c.normalized_name = toLower($name) OR c.name = $name
               RETURN rel.name AS concept, rel.description AS description,
                      r.weight AS weight
               ORDER BY r.weight DESC""",
            {"name": concept_name},
        )

    # ================================================================== #
    #  4. TABLE / FIGURE / FORMULA QUERIES
    # ================================================================== #

    def search_tables(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search tables by caption or content."""
        try:
            from backend.app.modules.ollama_client import get_ollama_client

            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes('table_embedding_index', $limit, $embedding)
                       YIELD node, score
                       MATCH (s:Section)-[:HAS_TABLE]->(node)
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                       RETURN node.caption AS caption, node.content AS content,
                              node.annotation AS annotation, node.number AS number,
                              s.title AS section, d.filename AS document, score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    {"embedding": embedding, "limit": limit},
                )
                if results:
                    return results
        except Exception:
            pass

        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('table_fulltext', $query)
                   YIELD node, score
                   MATCH (s:Section)-[:HAS_TABLE]->(node)
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                   RETURN node.caption AS caption, node.content AS content,
                          node.number AS number, s.title AS section,
                          d.filename AS document, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": self._sanitize_lucene(keyword), "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_TABLE]->(t:Table)
               WHERE toLower(t.caption) CONTAINS toLower($kw)
                  OR toLower(t.content) CONTAINS toLower($kw)
                  OR toLower(coalesce(t.section_title, '')) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN t.caption AS caption, t.content AS content,
                      t.number AS number, s.title AS section,
                      d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def search_figures(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search figures by caption / description. Returns image_path for rendering."""
        try:
            from backend.app.modules.ollama_client import get_ollama_client

            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes('figure_embedding_index', $limit, $embedding)
                       YIELD node, score
                       MATCH (s:Section)-[:HAS_FIGURE]->(node)
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                       RETURN node.caption AS caption, node.description AS description,
                              node.annotation AS annotation, node.number AS number,
                              node.image_type AS image_type,
                              coalesce(node.image_path, '') AS image_path,
                              s.title AS section, d.filename AS document, score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    {"embedding": embedding, "limit": limit},
                )
                if results:
                    return results
        except Exception:
            pass

        # Strategy 2: full-text index on figure_fulltext (caption + description + annotation)
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('figure_fulltext', $query)
                   YIELD node, score
                   MATCH (s:Section)-[:HAS_FIGURE]->(node)
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                   RETURN node.caption AS caption, node.description AS description,
                          node.annotation AS annotation, node.number AS number,
                          node.image_type AS image_type,
                          coalesce(node.image_path, '') AS image_path,
                          s.title AS section, d.filename AS document, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": self._sanitize_lucene(keyword), "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        # Strategy 3: CONTAINS fallback
        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FIGURE]->(f:Figure)
               WHERE toLower(f.caption) CONTAINS toLower($kw)
                  OR toLower(f.description) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN f.caption AS caption, f.description AS description,
                      f.number AS number, f.image_type AS image_type,
                      coalesce(f.image_path, '') AS image_path,
                      s.title AS section, d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def search_formulas(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search formulas by LaTeX content.

        Strategy order:
          1. Vector search on formula_embedding_index (semantic — handles natural-language queries)
          2. Full-text search on formula_fulltext index  (keyword/LaTeX fragments)
          3. CONTAINS fallback
        """
        # Strategy 1: vector search (handles "formula for load combination" type queries)
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes('formula_embedding_index', $limit, $embedding)
                       YIELD node, score
                       WITH node, max(score) AS score
                       MATCH (s:Section)-[:HAS_FORMULA]->(node)
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                       RETURN node.latex AS latex,
                              coalesce(node.unicode, node.latex) AS formula,
                              node.id AS id,
                              s.title AS section, d.filename AS document, score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    {"embedding": embedding, "limit": limit},
                )
                if results:
                    return results
        except Exception:
            pass

        # Strategy 2: full-text search
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('formula_fulltext', $query)
                   YIELD node, score
                   WITH node, max(score) AS score
                   MATCH (s:Section)-[:HAS_FORMULA]->(node)
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                   RETURN node.latex AS latex,
                          coalesce(node.unicode, node.latex) AS formula,
                          node.id AS id,
                          s.title AS section, d.filename AS document, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": self._sanitize_lucene(keyword), "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula)
               WHERE toLower(f.latex) CONTAINS toLower($kw)
                  OR toLower(f.unicode) CONTAINS toLower($kw)
                  OR toLower(coalesce(f.section_title, '')) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               WITH f, s, d
               ORDER BY f.id
               RETURN f.latex AS latex,
                      coalesce(f.unicode, f.latex) AS formula,
                      f.id AS id,
                      s.title AS section, d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def list_formulas(self) -> List[Dict[str, Any]]:
        """List all formulas."""
        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN f.latex AS latex,
                      coalesce(f.unicode, f.latex) AS formula,
                      f.id AS id,
                      s.title AS section, d.filename AS document
               ORDER BY s.title
               LIMIT 100""",
        )

    # ================================================================== #
    #  5. SEMANTIC (VECTOR) SEARCH
    # ================================================================== #

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over Section embeddings using the vector index.
        Falls back to full-text search if vector index unavailable.
        Includes associated formula LaTeX for each section.
        """
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(query)

            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'section_embedding_index', $top_k, $embedding
                       ) YIELD node, score
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(node)
                       OPTIONAL MATCH (node)-[:HAS_FORMULA]->(frm:Formula)
                       RETURN node.id AS id, node.title AS title,
                              substring(node.full_text, 0, 4000) AS content,
                              node.start_page AS page,
                              d.filename AS document,
                              score,
                              collect(DISTINCT frm.latex) AS formula_latex
                       ORDER BY score DESC""",
                    {"top_k": top_k, "embedding": embedding},
                )
                if results:
                    return results
        except Exception as e:
            logger.warning("Vector search failed, falling back: %s", e)

        # Fallback: full-text
        return self.search_sections(query, limit=top_k)

    def semantic_concept_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over Concept embeddings."""
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(query)

            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'concept_embedding_index', $top_k, $embedding
                       ) YIELD node, score
                       RETURN node.name AS concept,
                              node.description AS description,
                              score
                       ORDER BY score DESC""",
                    {"top_k": top_k, "embedding": embedding},
                )
                if results:
                    return results
        except Exception as e:
            logger.warning("Concept vector search failed: %s", e)

        return self.search_concepts(query, limit=top_k)

    # ================================================================== #
    #  6. CROSS-DOCUMENT QUERIES
    # ================================================================== #

    def get_similar_sections(self, section_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sections semantically similar to a given section (cross-document)."""
        return self.db.execute_query(
            """MATCH (s:Section {id: $id})-[r:SEMANTICALLY_SIMILAR]-(similar:Section)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(similar)
               RETURN similar.title AS title,
                      similar.content_preview AS preview,
                      d.filename AS document,
                      r.score AS score,
                      similar.start_page AS page
               ORDER BY r.score DESC
               LIMIT $limit""",
            {"id": section_id, "limit": limit},
        )

    def get_figures_for_sections(
        self, section_ids: List[str], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get Figure nodes linked to the given sections via HAS_FIGURE.

        This is the primary figure retrieval path: find relevant sections
        first (semantic/fulltext), then pull their attached images.
        """
        if not section_ids:
            return []
        return self.db.execute_query(
            """UNWIND $ids AS sid
               MATCH (s:Section {id: sid})-[:HAS_FIGURE]->(f:Figure)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN f.caption     AS caption,
                      f.description AS description,
                      f.annotation  AS annotation,
                      f.number      AS number,
                      f.image_type  AS image_type,
                      coalesce(f.image_path, '') AS image_path,
                      s.title       AS section,
                      d.filename    AS document
               LIMIT $limit""",
            {"ids": section_ids, "limit": limit},
        )

    # ================================================================== #
    #  6b. PAGE SEARCH
    # ================================================================== #

    def _vector_search_pages(
        self, embedding: list, limit: int
    ) -> List[Dict[str, Any]]:
        """Vector similarity search on page embeddings.

        Uses a WITH aggregation step to collapse the OPTIONAL MATCH fan-out
        (a page linked to multiple chapters produces multiple rows without it).
        Only returns pages with cosine similarity >= 0.5 to filter low-relevance noise.
        Fetches limit*2 candidates so the RRF merge has enough to work with.

        For orphan pages (pure table pages with no CONTAINS_PAGE link), falls
        back to the neighbouring page's chapter/document via NEXT_PAGE.
        """
        return self.db.execute_query(
            """CALL db.index.vector.queryNodes('page_embedding_index', $limit, $embedding)
               YIELD node, score
               WHERE node.content IS NOT NULL AND trim(node.content) <> ''
                 AND score >= 0.5
               OPTIONAL MATCH (ch:Chapter)-[:CONTAINS_PAGE]->(node)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(ch)
               OPTIONAL MATCH (node)-[:HAS_SECTION]->(s:Section)
               WITH node, score,
                    collect(DISTINCT ch.title)[0]    AS direct_ch,
                    collect(DISTINCT d.filename)[0]  AS direct_doc,
                    collect(DISTINCT s.title)         AS section_titles,
                    collect(DISTINCT s.id)            AS section_ids
               // Variable-length fallback for orphan pages (up to 10 hops back)
               CALL {
                 WITH node, direct_doc
                 WITH node WHERE direct_doc IS NULL
                 MATCH (linked:Page)-[:NEXT_PAGE*1..10]->(node)
                 WHERE EXISTS { MATCH (:Chapter)-[:CONTAINS_PAGE]->(linked) }
                 WITH linked ORDER BY linked.page_number DESC LIMIT 1
                 MATCH (ch_fb:Chapter)-[:CONTAINS_PAGE]->(linked)
                 MATCH (d_fb:Document)-[:HAS_CHAPTER]->(ch_fb)
                 RETURN collect(DISTINCT ch_fb.title)[0] AS fb_ch,
                        collect(DISTINCT d_fb.filename)[0] AS fb_doc
                 UNION
                 WITH node, direct_doc
                 WITH node WHERE direct_doc IS NOT NULL
                 RETURN null AS fb_ch, null AS fb_doc
               }
               RETURN node.id         AS page_id,
                      node.page_number AS page_number,
                      substring(node.content, 0, 8000) AS content,
                      node.header      AS header,
                      coalesce(direct_ch, fb_ch)   AS chapter,
                      coalesce(direct_doc, fb_doc) AS document,
                      score,
                      section_titles, section_ids
               ORDER BY score DESC""",
            {"limit": limit * 2, "embedding": embedding},
        ) or []

    def _fulltext_search_pages(
        self, query: str, limit: int, keywords: str = ""
    ) -> List[Dict[str, Any]]:
        """BM25 fulltext search on page content.

        Same WITH-aggregation fix as _vector_search_pages to prevent duplicate
        rows for pages that belong to multiple chapters.  Same NEXT_PAGE
        fallback for orphan table pages.

        Uses *keywords* (LLM-extracted) when provided; falls back to the
        static stopword filter (_to_keywords) when the LLM call failed.
        """
        search_terms = keywords if keywords else self._to_keywords(query)
        return self.db.execute_query(
            """CALL db.index.fulltext.queryNodes('page_fulltext', $query)
               YIELD node, score
               WHERE node.content IS NOT NULL AND trim(node.content) <> ''
               OPTIONAL MATCH (ch:Chapter)-[:CONTAINS_PAGE]->(node)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(ch)
               OPTIONAL MATCH (node)-[:HAS_SECTION]->(s:Section)
               WITH node, score,
                    collect(DISTINCT ch.title)[0]    AS direct_ch,
                    collect(DISTINCT d.filename)[0]  AS direct_doc,
                    collect(DISTINCT s.title)         AS section_titles,
                    collect(DISTINCT s.id)            AS section_ids
               // Variable-length fallback for orphan pages (up to 10 hops back)
               CALL {
                 WITH node, direct_doc
                 WITH node WHERE direct_doc IS NULL
                 MATCH (linked:Page)-[:NEXT_PAGE*1..10]->(node)
                 WHERE EXISTS { MATCH (:Chapter)-[:CONTAINS_PAGE]->(linked) }
                 WITH linked ORDER BY linked.page_number DESC LIMIT 1
                 MATCH (ch_fb:Chapter)-[:CONTAINS_PAGE]->(linked)
                 MATCH (d_fb:Document)-[:HAS_CHAPTER]->(ch_fb)
                 RETURN collect(DISTINCT ch_fb.title)[0] AS fb_ch,
                        collect(DISTINCT d_fb.filename)[0] AS fb_doc
                 UNION
                 WITH node, direct_doc
                 WITH node WHERE direct_doc IS NOT NULL
                 RETURN null AS fb_ch, null AS fb_doc
               }
               RETURN node.id         AS page_id,
                      node.page_number AS page_number,
                      substring(node.content, 0, 8000) AS content,
                      node.header      AS header,
                      coalesce(direct_ch, fb_ch)   AS chapter,
                      coalesce(direct_doc, fb_doc) AS document,
                      score,
                      section_titles, section_ids
               ORDER BY score DESC LIMIT $limit""",
            {"query": self._sanitize_lucene(search_terms), "limit": limit * 2},
        ) or []

    @staticmethod
    def _rrf_merge(
        list_a: List[Dict[str, Any]],
        list_b: List[Dict[str, Any]],
        limit: int,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion: merge two ranked page lists into one.

        Pages that appear in both vector and fulltext results receive a higher
        combined RRF score, naturally surfacing the most relevant pages.
        Deduplication is by (page_number, document) key.
        """
        def _key(p: Dict[str, Any]) -> tuple:
            # Normalise None → "" so the same page always gets the same key
            # regardless of whether the OPTIONAL MATCH resolved the document.
            return (p.get("page_number"), p.get("document") or "")

        scores: Dict[tuple, Dict[str, Any]] = {}
        for rank, page in enumerate(list_a):
            key = _key(page)
            if key in scores:
                scores[key]["rrf"] += 1.0 / (k + rank + 1)
            scores[key] = {"page": page, "rrf": 1.0 / (k + rank + 1)}
        for rank, page in enumerate(list_b):
            key = _key(page)
            if key in scores:
                scores[key]["rrf"] += 1.0 / (k + rank + 1)
            else:
                scores[key] = {"page": page, "rrf": 1.0 / (k + rank + 1)}
        sorted_items = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
        return [item["page"] for item in sorted_items[:limit]]

    def _fetch_adjacent_pages(
        self,
        top_pages: List[Dict[str, Any]],
        existing_keys: set,
        max_extra: int = 2,
    ) -> List[Dict[str, Any]]:
        """Fetch the NEXT_PAGE neighbour for the top 3 results.

        Tables extracted with table_format='markdown' often land on the page
        immediately after the text that references them.  Fetching the next
        page ensures those table pages are included even when their embedding
        score is lower than the referencing page.

        Uses the NEXT_PAGE relationship (built per-document during ingestion)
        so adjacency is always within the same document.
        """
        if not top_pages:
            return []
        page_ids = [p["page_id"] for p in top_pages[:3] if p.get("page_id")]
        if not page_ids:
            return []

        results = self.db.execute_query(
            """UNWIND $ids AS pid
               MATCH (p:Page {id: pid})-[:NEXT_PAGE]->(nxt:Page)
               WHERE nxt.content IS NOT NULL AND trim(nxt.content) <> ''
               OPTIONAL MATCH (ch:Chapter)-[:CONTAINS_PAGE]->(nxt)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(ch)
               WITH p, nxt,
                    collect(DISTINCT ch.title)[0]    AS direct_ch,
                    collect(DISTINCT d.filename)[0]  AS direct_doc
               // Variable-length fallback for orphan adjacent pages
               CALL {
                 WITH nxt, direct_doc, p
                 WITH nxt, p WHERE direct_doc IS NULL
                 // First try: traverse NEXT_PAGE backwards from nxt
                 OPTIONAL MATCH (linked:Page)-[:NEXT_PAGE*1..10]->(nxt)
                 WHERE EXISTS { MATCH (:Chapter)-[:CONTAINS_PAGE]->(linked) }
                 WITH nxt, p, linked ORDER BY linked.page_number DESC LIMIT 1
                 // Second try: use the source page's chapter (same document)
                 OPTIONAL MATCH (ch_src:Chapter)-[:CONTAINS_PAGE]->(p)
                 OPTIONAL MATCH (d_src:Document)-[:HAS_CHAPTER]->(ch_src)
                 OPTIONAL MATCH (ch_fb:Chapter)-[:CONTAINS_PAGE]->(linked)
                 OPTIONAL MATCH (d_fb:Document)-[:HAS_CHAPTER]->(ch_fb)
                 RETURN coalesce(collect(DISTINCT ch_fb.title)[0],
                                 collect(DISTINCT ch_src.title)[0]) AS fb_ch,
                        coalesce(collect(DISTINCT d_fb.filename)[0],
                                 collect(DISTINCT d_src.filename)[0]) AS fb_doc
                 UNION
                 WITH nxt, direct_doc, p
                 WITH nxt WHERE direct_doc IS NOT NULL
                 RETURN null AS fb_ch, null AS fb_doc
               }
               RETURN nxt.id          AS page_id,
                      nxt.page_number  AS page_number,
                      substring(nxt.content, 0, 8000) AS content,
                      nxt.header       AS header,
                      coalesce(direct_ch, fb_ch)   AS chapter,
                      coalesce(direct_doc, fb_doc) AS document,
                      0.0              AS score,
                      []               AS section_titles,
                      []               AS section_ids""",
            {"ids": page_ids},
        ) or []

        unique: List[Dict[str, Any]] = []
        for page in results:
            key = (page.get("page_number"), page.get("document") or "")
            if key not in existing_keys:
                existing_keys.add(key)
                unique.append(page)
            if len(unique) >= max_extra:
                break
        return unique

    def search_pages(
        self, query: str, limit: int = 5, keywords: str = ""
    ) -> List[Dict[str, Any]]:
        """Hybrid page search: vector + fulltext merged with RRF, with adjacent pages.

        Strategy:
        1. Run vector similarity search (score >= 0.5 threshold).
        2. Run BM25 fulltext search using *keywords* (LLM-extracted) when
           provided, or the static stopword filter as fallback.
        3. Merge both with Reciprocal Rank Fusion — pages in both lists score
           higher; deduplication by (page_number, document) eliminates the
           fan-out bug where one page appeared multiple times via different
           chapter matches.
        4. Fetch the NEXT_PAGE neighbour for the top-3 results (up to 2 extra)
           to catch table pages that follow the referencing text page.
        """
        # Step 1: embed query
        embedding = None
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(query)
        except Exception as e:
            logger.debug("Embedding failed: %s", e)

        # Step 2: run both searches
        vector_results: List[Dict[str, Any]] = []
        if embedding:
            try:
                vector_results = self._vector_search_pages(embedding, limit)
            except Exception as e:
                logger.debug("Vector page search failed: %s", e)

        fulltext_results: List[Dict[str, Any]] = []
        try:
            fulltext_results = self._fulltext_search_pages(query, limit, keywords=keywords)
        except Exception as e:
            logger.debug("Fulltext page search failed: %s", e)

        # Step 3: merge with RRF (handles deduplication)
        merged = self._rrf_merge(vector_results, fulltext_results, limit)
        if not merged:
            return []

        # Step 4: fetch adjacent pages for top 3 results
        existing_keys = {
            (p.get("page_number"), p.get("document") or "") for p in merged
        }
        adjacent = self._fetch_adjacent_pages(merged, existing_keys)
        if adjacent:
            merged.extend(adjacent)

        # Final deduplication safety net — catches any residual duplicates
        # that slip through if document resolves inconsistently across queries.
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for p in merged:
            key = (p.get("page_number"), p.get("document") or "")
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def get_sections_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full section content for a list of section IDs.

        Used to materialise the sections that belong to the top-scoring pages
        found during the page-first semantic search phase.
        """
        if not ids:
            return []
        return self.db.execute_query(
            """UNWIND $ids AS sid
               MATCH (s:Section {id: sid})
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               OPTIONAL MATCH (s)-[:HAS_FORMULA]->(frm:Formula)
               RETURN s.id AS id, s.title AS title,
                      substring(s.full_text, 0, 4000) AS content,
                      s.start_page AS page,
                      d.filename AS document,
                      1.0 AS score,
                      collect(DISTINCT frm.latex) AS formula_latex""",
            {"ids": ids},
        )

    def get_tables_for_sections(
        self, section_ids: List[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get Table nodes attached to the given sections.

        Used after page-first retrieval to narrow tables to only those that
        belong to the already-identified relevant sections.
        """
        if not section_ids:
            return []
        return self.db.execute_query(
            """UNWIND $ids AS sid
               MATCH (s:Section {id: sid})-[:HAS_TABLE]->(t:Table)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN t.caption AS caption, t.content AS content,
                      t.number AS number, s.title AS section,
                      d.filename AS document, 1.0 AS score
               LIMIT $limit""",
            {"ids": section_ids, "limit": limit},
        )

    def get_formulas_for_sections(
        self, section_ids: List[str], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get Formula nodes attached to the given sections.

        Used after page-first retrieval to narrow formulas to only those that
        belong to the already-identified relevant sections.
        """
        if not section_ids:
            return []
        return self.db.execute_query(
            """UNWIND $ids AS sid
               MATCH (s:Section {id: sid})-[:HAS_FORMULA]->(f:Formula)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN f.latex AS latex,
                      coalesce(f.unicode, f.latex) AS formula,
                      f.id AS id,
                      s.title AS section, d.filename AS document, 1.0 AS score
               LIMIT $limit""",
            {"ids": section_ids, "limit": limit},
        )

    # ================================================================== #
    #  7. GENERAL / BROAD SEARCH (combines everything)
    # ================================================================== #

    def general_search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Broad search across every node type in the graph.

        Search order (page-first narrowing strategy):
          1. Semantic search on Pages  → identifies the most relevant pages
          2. Extract section IDs from those pages
          3. Fetch full Section content for those IDs  (high-quality, grounded)
          4. Fetch Tables / Formulas / Figures from the same section IDs
          5. Concept search + expansion (adds any concept-mentioned sections)
          6. Chapter matching
          Fallback: if no sections surface from pages, falls back to a direct
          semantic section search so the system is never empty-handed.

        Returns a dict keyed by category: pages, semantic, tables, formulas,
        figures, concepts, chapters.
        """
        results: Dict[str, List] = {}

        # ── Step 1: Semantic page search ─────────────────────────────────
        # Pages are the entry point — they carry full-page context and link
        # directly to their child sections via HAS_SECTION.
        pages = self.search_pages(query, limit=8)
        if pages:
            results["pages"] = pages

        # ── Step 2: Collect section IDs from top pages ───────────────────
        section_ids: List[str] = []
        section_page_scores: Dict[str, float] = {}   # sid → real page score
        seen_ids: set = set()
        for p in pages:
            page_score = float(p.get("score") or 0.5)
            for sid in (p.get("section_ids") or []):
                if sid and sid not in seen_ids:
                    section_ids.append(sid)
                    section_page_scores[sid] = page_score
                    seen_ids.add(sid)

        # ── Step 3: Fetch sections linked to those pages ─────────────────
        if section_ids:
            sections = self.get_sections_by_ids(section_ids)
            if sections:
                # Replace the hardcoded 1.0 score with the actual page vector
                # score so sections from highly-relevant pages rank higher than
                # sections from tangentially-matching pages.
                for s in sections:
                    sid = s.get("id", "")
                    if sid in section_page_scores:
                        s["score"] = section_page_scores[sid]
                results["semantic"] = sections

        # Fallback: if pages yielded no section links, do a direct vector
        # search on sections so context is never empty.
        if not results.get("semantic"):
            try:
                fallback_secs = self.semantic_search(query, top_k=8)
                if fallback_secs:
                    results["semantic"] = fallback_secs
                    section_ids = [s["id"] for s in fallback_secs if s.get("id")]
            except Exception as e:
                logger.debug("Fallback semantic section search failed: %s", e)

        # ── Step 4: Tables, Formulas, Figures — narrowed to page sections ─
        if section_ids:
            tables = self.get_tables_for_sections(section_ids, limit=5)
            if tables:
                results["tables"] = tables

            formulas = self.get_formulas_for_sections(section_ids, limit=10)
            if formulas:
                results["formulas"] = formulas

            figures = self.get_figures_for_sections(section_ids[:10], limit=10)
            if figures:
                results["figures"] = figures

        # ── Step 5: Concept search + section expansion ───────────────────
        concepts = self.search_concepts(query, limit=10)
        if concepts:
            results["concepts"] = concepts
            existing_ids = set(section_ids)
            for c in concepts[:3]:
                cname = c.get("concept", "")
                if cname:
                    csecs = self.get_concept_sections(cname)
                    if csecs:
                        for cs in csecs:
                            cid = cs.get("id", "")
                            if cid and cid not in existing_ids:
                                results.setdefault("semantic", []).append(cs)
                                existing_ids.add(cid)

        # ── Step 6: Chapters ─────────────────────────────────────────────
        chapters = self.list_chapters(query)
        if chapters:
            results["chapters"] = chapters[:10]

        return results

    # ================================================================== #
    #  8. GRAPH STATISTICS
    # ================================================================== #

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return counts of each node type plus relationship counts."""
        result = self.db.execute_query(
            """OPTIONAL MATCH (doc:Document) WITH count(doc) AS documents
               OPTIONAL MATCH (v:Volume) WITH documents, count(v) AS volumes
               OPTIONAL MATCH (ch:Chapter) WITH documents, volumes, count(ch) AS chapters
               OPTIONAL MATCH (p:Page) WITH documents, volumes, chapters, count(p) AS pages
               OPTIONAL MATCH (s:Section) WITH documents, volumes, chapters, pages, count(s) AS sections
               OPTIONAL MATCH (t:Table) WITH documents, volumes, chapters, pages, sections, count(t) AS tables
               OPTIONAL MATCH (f:Figure) WITH documents, volumes, chapters, pages, sections, tables, count(f) AS figures
               OPTIONAL MATCH (fm:Formula) WITH documents, volumes, chapters, pages, sections, tables, figures, count(fm) AS formulas
               OPTIONAL MATCH (c:Concept) WITH documents, volumes, chapters, pages, sections, tables, figures, formulas, count(c) AS concepts
               RETURN documents, volumes, chapters, pages, sections,
                      tables, figures, formulas, concepts""",
        )
        stats = result[0] if result else {}

        # Relationship counts
        try:
            rel_result = self.db.execute_query(
                """OPTIONAL MATCH ()-[m:MENTIONS]->() WITH count(m) AS mentions
                   OPTIONAL MATCH ()-[r:RELATED_TO]->() WITH mentions, count(r) AS related_to
                   OPTIONAL MATCH ()-[ss:SEMANTICALLY_SIMILAR]->() WITH mentions, related_to, count(ss) AS semantic_links
                   RETURN mentions, related_to, semantic_links""",
            )
            if rel_result:
                stats.update(rel_result[0])
        except Exception:
            pass

        return stats


# ===================================================================== #
#  Singleton
# ===================================================================== #

_querier: Optional[GraphQuerier] = None


def get_graph_querier() -> GraphQuerier:
    global _querier
    if _querier is None:
        _querier = GraphQuerier()
    return _querier
