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
from config.settings import settings

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

    # ISO standard naming pattern for Eurocode / DIN / ISO norm references.
    # Used to protect references like "EN 1993-1-1" from being mangled by
    # Lucene escaping — they become phrase queries instead.
    _NORM_REF_RE: re.Pattern = re.compile(
        r'\b(EN|EC|DIN|ISO|prEN)\s*\d{4}(?:-\d+)*\b', re.IGNORECASE
    )

    @classmethod
    def _escape_lucene(cls, query: str, fuzzy: bool = False) -> str:
        """Escape Lucene fulltext query special characters.

        Norm references (EN 1993-1-1, DIN 1045-1, etc.) are converted to
        Lucene phrase queries so BM25 matches them as a unit rather than
        splitting on hyphens.  All other special characters are escaped with
        backslash to avoid ParseException.

        When *fuzzy* is True, individual tokens longer than
        ``settings.bm25_fuzzy_min_length`` characters get a ``~1`` edit-
        distance suffix appended.  This handles German morphological
        inflections (e.g. dative "Kopfbolzendübeln" vs nominative
        "Kopfbolzendübel" in the index) without a stemmer dependency.
        """
        # 1. Extract and protect norm references as Lucene phrase queries
        protected_phrases: list[str] = []
        def _protect_ref(m: re.Match) -> str:
            # Replace hyphens with spaces inside the phrase so Lucene matches
            phrase = m.group(0).replace("-", " ")
            protected_phrases.append(f'"{phrase}"')
            return ""  # Remove from remaining text

        remaining = cls._NORM_REF_RE.sub(_protect_ref, query)

        # 2. Escape Lucene specials in remaining text
        specials = r'([+\-&|!(){}\[\]^"~*?:\\/])'
        escaped = re.sub(specials, r'\\\1', remaining)
        escaped = escaped.strip()

        # 3. Optionally apply fuzzy matching to long individual tokens
        if fuzzy and escaped:
            min_len = settings.bm25_fuzzy_min_length
            tokens = escaped.split()
            tokens = [
                f"{t}~1" if len(t) >= min_len and not t.startswith('"') else t
                for t in tokens
            ]
            escaped = " ".join(tokens)

        # 4. Combine: protected phrase queries + escaped remainder
        parts = protected_phrases + ([escaped] if escaped else [])
        return ' '.join(parts) or '*'

    # Short domain terms that must never be filtered by the stopword/length
    # filter in _to_keywords.  These are standard abbreviations from Eurocode,
    # DIN, and structural engineering notation.
    _DOMAIN_TERMS: frozenset = frozenset({
        "EN", "EC", "DIN", "NA", "NDP", "NCI", "ULS", "SLS",
        "Ed", "Rd", "Ek", "Rk", "fy", "fu", "fck", "fcd",
    })

    # Eurocode abbreviation → full-form synonyms used in indexed text.
    # Applied in BM25 queries so abbreviated queries find indexed content.
    # Keys are uppercase for case-insensitive lookup.
    _EUROCODE_SYNONYMS: Dict[str, List[str]] = {
        "EC1": ["EN 1991", "Eurocode 1"],
        "EC2": ["EN 1992", "Eurocode 2"],
        "EC3": ["EN 1993", "Eurocode 3"],
        "EC4": ["EN 1994", "Eurocode 4"],
        "EC5": ["EN 1995", "Eurocode 5"],
        "EC6": ["EN 1996", "Eurocode 6"],
        "EC7": ["EN 1997", "Eurocode 7"],
        "EC8": ["EN 1998", "Eurocode 8"],
        "EC9": ["EN 1999", "Eurocode 9"],
        "ULS": ["Grenzzustand", "Tragfähigkeit", "Grenzzustand der Tragfähigkeit"],
        "SLS": ["Gebrauchstauglichkeit", "Grenzzustand der Gebrauchstauglichkeit"],
        "NDP": ["Nationaler Anhang", "national bestimmte Parameter"],
        "NCI": ["Nationaler Anhang"],
    }

    @classmethod
    def _expand_synonyms(cls, keywords: str) -> str:
        """Expand Eurocode abbreviations with their full forms for BM25 search.

        Takes a space-separated keyword string and appends synonyms for any
        recognised abbreviations.  The synonyms are added as extra OR terms so
        pages using full form names are also retrieved.
        """
        words = keywords.split()
        extras: List[str] = []
        for w in words:
            syns = cls._EUROCODE_SYNONYMS.get(w.upper(), [])
            extras.extend(syns)
        if extras:
            return keywords + " " + " ".join(extras)
        return keywords

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

        Returns section content (configurable via SECTION_CONTENT_LIMIT) plus
        all associated formula LaTeX so the LLM sees the exact formulas.
        """
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('section_fulltext', $query)
                   YIELD node, score
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(node)
                   OPTIONAL MATCH (node)-[:HAS_FORMULA]->(frm:Formula)
                   RETURN node.id AS id, node.title AS title,
                          substring(node.full_text, 0, $content_limit) AS content,
                          node.start_page AS page,
                          d.filename AS document, score,
                          collect(DISTINCT frm.latex) AS formula_latex
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": self._escape_lucene(keyword), "limit": limit, "content_limit": settings.section_content_limit},
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
                      substring(s.full_text, 0, $content_limit) AS content,
                      s.start_page AS page,
                      d.filename AS document, 0.5 AS score,
                      collect(DISTINCT frm.latex) AS formula_latex
               LIMIT $limit""",
            {"kw": keyword, "limit": limit, "content_limit": settings.section_content_limit},
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
                {"query": self._escape_lucene(keyword), "limit": limit},
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
                      substring(s.full_text, 0, $content_limit) AS content,
                      s.content_preview AS preview,
                      m.confidence AS confidence, d.filename AS document,
                      s.start_page AS page,
                      collect(DISTINCT frm.latex) AS formula_latex
               ORDER BY m.confidence DESC
               LIMIT 20""",
            {"name": concept_name, "content_limit": settings.section_content_limit},
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
                {"query": self._escape_lucene(keyword), "limit": limit},
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
                {"query": self._escape_lucene(keyword), "limit": limit},
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
                {"query": self._escape_lucene(keyword), "limit": limit},
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
                              substring(node.full_text, 0, $content_limit) AS content,
                              node.start_page AS page,
                              d.filename AS document,
                              score,
                              collect(DISTINCT frm.latex) AS formula_latex
                       ORDER BY score DESC""",
                    {"top_k": top_k, "embedding": embedding, "content_limit": settings.section_content_limit},
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
                 AND score >= $vector_threshold
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
                      substring(node.content, 0, $content_limit) AS content,
                      node.header      AS header,
                      coalesce(direct_ch, fb_ch)   AS chapter,
                      coalesce(direct_doc, fb_doc) AS document,
                      score,
                      section_titles, section_ids
               ORDER BY score DESC""",
            {"limit": limit * 2, "embedding": embedding, "content_limit": settings.page_content_limit, "vector_threshold": settings.vector_page_threshold},
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
        search_terms = self._expand_synonyms(search_terms)
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
                      substring(node.content, 0, $content_limit) AS content,
                      node.header      AS header,
                      coalesce(direct_ch, fb_ch)   AS chapter,
                      coalesce(direct_doc, fb_doc) AS document,
                      score,
                      section_titles, section_ids
               ORDER BY score DESC LIMIT $limit""",
            {"query": self._escape_lucene(search_terms, fuzzy=True), "limit": limit * 2, "content_limit": settings.page_content_limit},
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
            else:
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
        page_ids = [p["page_id"] for p in top_pages[:settings.enrichment_max_adjacent] if p.get("page_id")]
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
                      substring(nxt.content, 0, $content_limit) AS content,
                      nxt.header       AS header,
                      coalesce(direct_ch, fb_ch)   AS chapter,
                      coalesce(direct_doc, fb_doc) AS document,
                      0.0              AS score,
                      []               AS section_titles,
                      []               AS section_ids""",
            {"ids": page_ids, "content_limit": settings.page_content_limit},
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

    # ----------------------------------------------------------------- #
    #  3-Path Hybrid Search
    # ----------------------------------------------------------------- #

    def _topdown_search(
        self, embedding: list, limit: int
    ) -> List[Dict[str, Any]]:
        """Path 1: Graph traversal via rich document/chapter summaries.

        1. Score Documents by vector similarity on ``document_embedding_index``.
        2. Score Chapters within those docs on ``chapter_embedding_index``.
        3. Fetch all pages from the top chapters via ``CONTAINS_PAGE``.

        Returns ``[]`` gracefully if the indexes don't exist yet.
        """
        from config.settings import settings

        # Step 1: top documents
        try:
            doc_rows = self.db.execute_query(
                """CALL db.index.vector.queryNodes(
                       'document_embedding_index', $max_docs, $embedding
                   ) YIELD node, score
                   WHERE score >= $vector_threshold
                   RETURN node.id AS doc_id, node.filename AS filename, score
                   ORDER BY score DESC""",
                {"max_docs": settings.topdown_max_docs, "embedding": embedding, "vector_threshold": settings.vector_doc_threshold},
            ) or []
        except Exception as e:
            logger.debug("Top-down document search unavailable: %s", e)
            return []

        if not doc_rows:
            return []

        doc_ids = [r["doc_id"] for r in doc_rows]

        # Step 2: top sections within those documents.
        # Sections are more granular than chapters — their focused embeddings discriminate
        # specific technical content far better than averaged chapter-level embeddings.
        # Query per document so that large documents (many sections) cannot monopolise
        # the global ANN result pool and starve smaller but more relevant documents.
        sections_per_doc = max(settings.topdown_max_chapters, 1)
        section_rows: list = []
        for doc_id in doc_ids:
            try:
                rows = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'section_embedding_index', $candidates, $embedding
                       ) YIELD node, score
                       WHERE score >= $vector_threshold
                       MATCH (p:Page)-[:HAS_SECTION]->(node)
                       MATCH (d:Document {id: $doc_id})-[:HAS_CHAPTER]->(ch:Chapter)-[:CONTAINS_PAGE]->(p)
                       RETURN DISTINCT node.id AS section_id, score
                       ORDER BY score DESC
                       LIMIT $max_sections""",
                    {
                        "candidates": sections_per_doc * 20,
                        "embedding": embedding,
                        "doc_id": doc_id,
                        "max_sections": sections_per_doc,
                        "vector_threshold": settings.vector_doc_threshold,
                    },
                ) or []
                section_rows.extend(rows)
            except Exception as e:
                logger.debug("Top-down section search failed for doc %s: %s", doc_id, e)
        section_rows.sort(key=lambda r: r["score"], reverse=True)
        section_rows = section_rows[: settings.topdown_max_chapters * len(doc_ids)]

        if not section_rows:
            return []

        section_ids = [r["section_id"] for r in section_rows]

        # Step 3: fetch the pages that contain those sections.
        # Group by page to collapse duplicates from the multi-chapter mapping issue.
        pages_raw = self.db.execute_query(
            """UNWIND $section_ids AS sid
               MATCH (p:Page)-[:HAS_SECTION]->(s:Section {id: sid})
               WHERE p.content IS NOT NULL AND trim(p.content) <> ''
               MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)-[:CONTAINS_PAGE]->(p)
               WITH p, d, collect(DISTINCT ch.title)[0] AS chapter
               OPTIONAL MATCH (p)-[:HAS_SECTION]->(s_all:Section)
               RETURN p.id          AS page_id,
                      p.page_number  AS page_number,
                      substring(p.content, 0, $content_limit) AS content,
                      p.header       AS header,
                      chapter,
                      d.filename     AS document,
                      0.0            AS score,
                      collect(DISTINCT s_all.title) AS section_titles,
                      collect(DISTINCT s_all.id)    AS section_ids""",
            {"section_ids": section_ids, "content_limit": settings.page_content_limit},
        ) or []

        # Deduplicate pages (same page may be reached via multiple sections)
        seen: set = set()
        unique_pages = []
        for p in pages_raw:
            key = (p.get("page_id") or p.get("page_number"), p.get("document"))
            if key not in seen:
                seen.add(key)
                unique_pages.append(p)

        logger.debug("Top-down search: %d docs → %d sections → %d pages",
                      len(doc_ids), len(section_ids), len(unique_pages))
        return unique_pages[:limit]

    def _dense_vector_search(
        self, embedding: list, limit: int
    ) -> List[Dict[str, Any]]:
        """Path 2: Dense vector search on page + section embeddings.

        Runs vector search on both ``page_embedding_index`` and
        ``section_embedding_index``, resolves section hits to their pages,
        and merges with 2-way RRF.
        """
        # Sub-path A: page-level vector search (existing)
        page_results: List[Dict[str, Any]] = []
        try:
            page_results = self._vector_search_pages(embedding, limit)
        except Exception as e:
            logger.debug("Dense page vector search failed: %s", e)

        # Sub-path B: section-level vector search → resolve to pages.
        # For each matched section, select the single most content-rich page
        # (by content length) while excluding table-of-contents chapter pages.
        # This prevents TOC entries from crowding out the actual content pages
        # when multiple pages share the same section (e.g. pg=5 TOC vs pg=82
        # formula page for "C.2 Ermüdungsfestigkeit").
        section_page_results: List[Dict[str, Any]] = []
        try:
            section_page_results = self.db.execute_query(
                """CALL db.index.vector.queryNodes(
                       'section_embedding_index', $limit, $embedding
                   ) YIELD node AS section, score
                   WHERE score >= $vector_threshold
                   MATCH (p:Page)-[:HAS_SECTION]->(section)
                   WHERE p.content IS NOT NULL AND trim(p.content) <> ''
                   MATCH (ch:Chapter)-[:CONTAINS_PAGE]->(p)
                   WHERE ch.chapter_type <> 'table_of_contents'
                   MATCH (d:Document)-[:HAS_CHAPTER]->(ch)
                   WITH section, score, p, ch.title AS ch_title, d.filename AS doc_filename
                   ORDER BY score DESC, size(p.content) DESC
                   WITH section.id AS section_key, score,
                        collect(p)[0]          AS best_page,
                        collect(ch_title)[0]   AS chapter,
                        collect(doc_filename)[0] AS document,
                        section.title          AS section_title
                   RETURN best_page.id         AS page_id,
                          best_page.page_number AS page_number,
                          substring(best_page.content, 0, $content_limit) AS content,
                          best_page.header      AS header,
                          chapter, document, score,
                          [section_title]       AS section_titles,
                          [section_key]         AS section_ids
                   ORDER BY score DESC""",
                {"limit": limit * 2, "embedding": embedding, "content_limit": settings.page_content_limit, "vector_threshold": settings.vector_page_threshold},
            ) or []
        except Exception as e:
            logger.debug("Dense section vector search failed: %s", e)

        # Merge page-vector and section-vector results
        if page_results and section_page_results:
            return self._rrf_merge(page_results, section_page_results, limit)
        return page_results or section_page_results

    @staticmethod
    def _rrf_merge_multi(
        ranked_lists: List[List[Dict[str, Any]]],
        limit: int,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """N-way Reciprocal Rank Fusion.

        Generalises ``_rrf_merge`` to merge any number of ranked page lists.
        Pages appearing in multiple lists naturally receive higher combined
        scores.  Deduplication is by ``(page_number, document)`` key.
        """
        def _key(p: Dict[str, Any]) -> tuple:
            return (p.get("page_number"), p.get("document") or "")

        scores: Dict[tuple, Dict[str, Any]] = {}
        for ranked_list in ranked_lists:
            for rank, page in enumerate(ranked_list):
                key = _key(page)
                rrf_score = 1.0 / (k + rank + 1)
                if key in scores:
                    scores[key]["rrf"] += rrf_score
                else:
                    scores[key] = {"page": page, "rrf": rrf_score}

        sorted_items = sorted(
            scores.values(), key=lambda x: x["rrf"], reverse=True
        )
        return [item["page"] for item in sorted_items[:limit]]

    # Regex patterns for structured reference detection in queries.
    # These are standard Eurocode / DIN numbering conventions.
    _TABLE_REF_RE: re.Pattern = re.compile(
        r'[Tt]abelle?\s+(\d+[\.\d]*)', re.IGNORECASE
    )
    _SECTION_REF_RE: re.Pattern = re.compile(
        r'(?:Abschnitt|Section|Kapitel)\s+(\d+[\.\d]*)', re.IGNORECASE
    )
    _FIGURE_REF_RE: re.Pattern = re.compile(
        r'(?:Bild|Abbildung|Figure|Fig\.?)\s+(\d+[\.\d]*)', re.IGNORECASE
    )

    def _intent_targeted_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Detect structured references in query and run targeted lookups.

        When the query contains explicit references like "Tabelle 3.1" or
        "EN 1993-1-1", runs exact-match Cypher queries against table/section/
        figure numbers and document names.  Returns pages containing those
        elements, ranked by match specificity.
        """
        if not settings.query_intent_enabled:
            return []

        pages: List[Dict[str, Any]] = []

        # Detect norm reference to scope the search
        norm_match = self._NORM_REF_RE.search(query)
        doc_filter = ""
        doc_params: Dict[str, Any] = {}
        if norm_match:
            norm_ref = norm_match.group(0).replace("-", " ").strip()
            doc_filter = "AND d.filename CONTAINS $norm_ref"
            doc_params["norm_ref"] = norm_ref

        # Targeted table lookup
        table_match = self._TABLE_REF_RE.search(query)
        if table_match:
            table_num = table_match.group(1)
            try:
                rows = self.db.execute_query(
                    f"""MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)
                              -[:HAS_SECTION]->(s:Section)-[:HAS_TABLE]->(t:Table)
                        WHERE t.number CONTAINS $table_num {doc_filter}
                        MATCH (s)-[:CONTAINS_PAGE]->(p:Page)
                        RETURN DISTINCT p.page_number AS page_number,
                               p.content AS content,
                               ch.title AS chapter,
                               d.filename AS document
                        LIMIT $limit""",
                    {"table_num": table_num, "limit": limit, **doc_params},
                ) or []
                pages.extend(rows)
            except Exception as e:
                logger.debug("Intent table lookup failed: %s", e)

        # Targeted section lookup
        section_match = self._SECTION_REF_RE.search(query)
        if section_match:
            section_num = section_match.group(1)
            try:
                rows = self.db.execute_query(
                    f"""MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)
                              -[:HAS_SECTION]->(s:Section)
                        WHERE s.number STARTS WITH $section_num {doc_filter}
                        MATCH (s)-[:CONTAINS_PAGE]->(p:Page)
                        RETURN DISTINCT p.page_number AS page_number,
                               p.content AS content,
                               ch.title AS chapter,
                               d.filename AS document
                        LIMIT $limit""",
                    {"section_num": section_num, "limit": limit, **doc_params},
                ) or []
                pages.extend(rows)
            except Exception as e:
                logger.debug("Intent section lookup failed: %s", e)

        # Targeted figure lookup
        figure_match = self._FIGURE_REF_RE.search(query)
        if figure_match:
            figure_num = figure_match.group(1)
            try:
                rows = self.db.execute_query(
                    f"""MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)
                              -[:HAS_SECTION]->(s:Section)-[:HAS_FIGURE]->(f:Figure)
                        WHERE f.number CONTAINS $figure_num {doc_filter}
                        MATCH (s)-[:CONTAINS_PAGE]->(p:Page)
                        RETURN DISTINCT p.page_number AS page_number,
                               p.content AS content,
                               ch.title AS chapter,
                               d.filename AS document
                        LIMIT $limit""",
                    {"figure_num": figure_num, "limit": limit, **doc_params},
                ) or []
                pages.extend(rows)
            except Exception as e:
                logger.debug("Intent figure lookup failed: %s", e)

        if pages:
            logger.info(
                "Intent detection found %d targeted results (table=%s, section=%s, figure=%s, norm=%s)",
                len(pages),
                table_match.group(0) if table_match else None,
                section_match.group(0) if section_match else None,
                figure_match.group(0) if figure_match else None,
                norm_match.group(0) if norm_match else None,
            )
        return pages

    def _semantic_similar_expansion(
        self, pages: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """Follow SEMANTICALLY_SIMILAR edges from the top-result sections.

        Collects unique section IDs from the supplied pages (top results after
        initial RRF merge), queries for sections linked by a SEMANTICALLY_SIMILAR
        edge with score >= ``settings.semantic_similar_min_score``, and returns
        the pages containing those similar sections.  This adds cross-document
        coverage so, for example, an EC3 section that conceptually references
        a topic explained in EC8 still surfaces the EC8 pages.
        """
        section_ids: List[str] = []
        seen: set = set()
        for p in pages:
            for sid in (p.get("section_ids") or []):
                if sid and sid not in seen:
                    section_ids.append(sid)
                    seen.add(sid)

        if not section_ids:
            return []

        try:
            return self.db.execute_query(
                """UNWIND $sids AS sid
                   MATCH (s:Section {id: sid})-[r:SEMANTICALLY_SIMILAR]-(sim:Section)
                   WHERE r.score >= $min_score
                   MATCH (p:Page)-[:HAS_SECTION]->(sim)
                   WHERE p.content IS NOT NULL AND trim(p.content) <> ''
                   OPTIONAL MATCH (ch:Chapter)-[:CONTAINS_PAGE]->(p)
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(ch)
                   WITH p,
                        max(r.score)                     AS score,
                        collect(DISTINCT ch.title)[0]   AS chapter,
                        collect(DISTINCT d.filename)[0] AS document,
                        collect(DISTINCT sim.title)      AS section_titles,
                        collect(DISTINCT sim.id)         AS section_ids
                   RETURN p.id          AS page_id,
                          p.page_number  AS page_number,
                          substring(p.content, 0, $content_limit) AS content,
                          p.header       AS header,
                          chapter, document,
                          score,
                          section_titles, section_ids
                   ORDER BY score DESC
                   LIMIT $limit""",
                {
                    "sids": section_ids,
                    "min_score": settings.semantic_similar_min_score,
                    "content_limit": settings.page_content_limit,
                    "limit": limit,
                },
            ) or []
        except Exception as e:
            logger.debug("Semantic similar expansion failed: %s", e)
            return []

    def search_hybrid(
        self,
        query: str,
        limit: int = 8,
        bm25_query: str = "",
        _debug: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: intent detection + graph traversal + dense vector + sparse BM25.

        *query* is embedded with bge-m3 (multilingual) for paths 1-2 so both
        English and German queries work without translation.

        *bm25_query* overrides the BM25 (path 3) input — pass a German translation
        so Lucene term matching works against the German-indexed text.  Falls back
        to *query* when not provided.

        *_debug* is an optional dict populated with per-path result counts and
        compact page summaries; written to debug_raw.jsonl by the agent layer.
        """
        from config.settings import settings

        # Embed the original query — bge-m3 is multilingual so no translation needed.
        embedding = None
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(query)
        except Exception as e:
            logger.debug("Embedding failed: %s", e)

        candidates = settings.hybrid_candidates_per_path

        # BM25 operates on German text; use the German translation when provided.
        _bm25_input = bm25_query or query

        # Path 0: Intent-based targeted lookup (doubled weight in RRF)
        path_intent: List[Dict[str, Any]] = []
        try:
            path_intent = self._intent_targeted_search(query, candidates)
        except Exception as e:
            logger.debug("Intent path failed: %s", e)

        # Path 1: Graph traversal (top-down via rich summaries)
        path_topdown: List[Dict[str, Any]] = []
        if embedding:
            try:
                path_topdown = self._topdown_search(embedding, candidates)
            except Exception as e:
                logger.debug("Top-down path failed: %s", e)

        # Path 2: Dense vector (page + section embeddings)
        path_vector: List[Dict[str, Any]] = []
        if embedding:
            try:
                path_vector = self._dense_vector_search(embedding, candidates)
            except Exception as e:
                logger.debug("Dense vector path failed: %s", e)

        # Path 3: Sparse BM25 — uses German query for accurate term matching
        path_bm25: List[Dict[str, Any]] = []
        try:
            path_bm25 = self._fulltext_search_pages(_bm25_input, candidates)
        except Exception as e:
            logger.debug("BM25 path failed: %s", e)

        # Collect non-empty paths for RRF.
        # Intent results are added twice to double their RRF weight,
        # making targeted matches rank above generic vector hits.
        active_paths = []
        if path_intent:
            active_paths.append(path_intent)
            active_paths.append(path_intent)  # double weight
        active_paths.extend(p for p in [path_topdown, path_vector, path_bm25] if p)

        if not active_paths:
            if _debug is not None:
                _debug.update({"bm25_input_query": _bm25_input, "bm25_keywords": "",
                               "path_intent": {"n": 0}, "path_topdown": {"n": 0},
                               "path_vector": {"n": 0}, "path_bm25": {"n": 0},
                               "path_similar": {"n": 0}, "merged": {"n": 0}})
            return []

        # Preliminary merge to discover top sections for semantic expansion.
        preliminary = self._rrf_merge_multi(active_paths, limit=3, k=settings.hybrid_rrf_k)

        # Path 4: SEMANTICALLY_SIMILAR expansion from top preliminary sections.
        path_similar: List[Dict[str, Any]] = []
        if settings.semantic_similar_enabled and preliminary:
            path_similar = self._semantic_similar_expansion(preliminary, candidates)
            if path_similar:
                active_paths.append(path_similar)

        if len(active_paths) == 1:
            merged = active_paths[0][:limit]
        else:
            merged = self._rrf_merge_multi(
                active_paths, limit=limit, k=settings.hybrid_rrf_k
            )

        logger.info(
            "Hybrid search: intent=%d, topdown=%d, vector=%d, bm25=%d, similar=%d → merged=%d",
            len(path_intent), len(path_topdown), len(path_vector),
            len(path_bm25), len(path_similar), len(merged),
        )

        # Populate debug dict with per-path summaries for observability.
        if _debug is not None:
            def _ps(pages: List[Dict]) -> List[Dict]:
                return [
                    {
                        "pg": p.get("page_number"),
                        "doc": (p.get("document") or "")[-50:],
                        "ch": (p.get("chapter") or "")[:40],
                        "score": round(float(p.get("score") or 0), 3),
                    }
                    for p in pages
                ]
            _debug["bm25_input_query"] = _bm25_input
            _debug["bm25_keywords"] = self._expand_synonyms(self._to_keywords(_bm25_input))
            _debug["path_intent"]  = {"n": len(path_intent),  "pages": _ps(path_intent)}
            _debug["path_topdown"] = {"n": len(path_topdown), "pages": _ps(path_topdown)}
            _debug["path_vector"]  = {"n": len(path_vector),  "pages": _ps(path_vector)}
            _debug["path_bm25"]    = {"n": len(path_bm25),    "pages": _ps(path_bm25)}
            _debug["path_similar"] = {"n": len(path_similar), "pages": _ps(path_similar)}
            _debug["merged"]       = {"n": len(merged),       "pages": _ps(merged)}

        return merged

    # ----------------------------------------------------------------- #
    #  Post-rerank graph enrichment
    # ----------------------------------------------------------------- #

    def enrich_with_graph(
        self,
        pages: List[Dict[str, Any]],
    ) -> tuple:
        """Expand reranked pages with graph context for completeness.

        Called *after* the cross-encoder reranker has selected the final
        6-8 pages.  Uses the graph to add:

        1. **NEXT_PAGE** adjacency — catches tables on the following page.
        2. **HAS_FIGURE** — figures from sections on retrieved pages.
        3. **HAS_FORMULA** — formulas from sections on retrieved pages.

        Returns
        -------
        (enriched_pages, extra_figures, extra_formulas)
        """
        from config.settings import settings

        # 1. Adjacent pages
        existing_keys = {
            (p.get("page_number"), p.get("document") or "") for p in pages
        }
        adjacent = self._fetch_adjacent_pages(
            pages, existing_keys, max_extra=settings.enrichment_max_adjacent
        )
        enriched_pages = list(pages)
        if adjacent:
            enriched_pages.extend(adjacent)

        # Collect section IDs from all pages (including adjacent)
        section_ids: List[str] = []
        seen_sids: set = set()
        for p in enriched_pages:
            for sid in (p.get("section_ids") or []):
                if sid and sid not in seen_sids:
                    section_ids.append(sid)
                    seen_sids.add(sid)

        # 2. Figures
        extra_figures: List[Dict[str, Any]] = []
        if section_ids:
            try:
                extra_figures = self.get_figures_for_sections(
                    section_ids[:20], limit=settings.enrichment_max_figures
                )
            except Exception as e:
                logger.debug("Figure enrichment failed: %s", e)

        # 3. Formulas
        extra_formulas: List[Dict[str, Any]] = []
        if section_ids:
            try:
                extra_formulas = self.db.execute_query(
                    """UNWIND $sids AS sid
                       MATCH (s:Section {id: sid})-[:HAS_FORMULA]->(f:Formula)
                       RETURN f.latex      AS latex,
                              f.unicode    AS unicode,
                              s.title      AS section,
                              f.section_title AS section_title
                       LIMIT $limit""",
                    {
                        "sids": section_ids[:20],
                        "limit": settings.enrichment_max_formulas,
                    },
                ) or []
            except Exception as e:
                logger.debug("Formula enrichment failed: %s", e)

        return enriched_pages, extra_figures, extra_formulas

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
                      substring(s.full_text, 0, $content_limit) AS content,
                      s.start_page AS page,
                      d.filename AS document,
                      1.0 AS score,
                      collect(DISTINCT frm.latex) AS formula_latex""",
            {"ids": ids, "content_limit": settings.section_content_limit},
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
