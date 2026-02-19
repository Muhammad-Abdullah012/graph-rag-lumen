"""LaTeX → Unicode formula converter.

Converts LaTeX math expressions to readable Unicode text for storage
in the knowledge graph as ``Formula.unicode`` alongside the raw
``Formula.latex`` source.

**Two-stage pipeline:**
1. **pylatexenc**: Parse LaTeX semantically, extract text, normalize \mathrm{}, \text{}
2. **flatlatex**: Convert remaining LaTeX symbols (Greek, operators) to Unicode
3. **Fallback**: Basic regex-based cleanup if both fail

This combination handles Eurocode patterns better than either library alone:
- Extracts plain text from \\mathrm{E} → E (instead of mangled output)
- Converts Greek letters γ, σ, etc. to Unicode
- Converts structural math: √, fractions, superscripts
- Handles operators: ·, ×, ±, ≤, ≥, etc.

Examples
--------
>>> latex_to_unicode(r"d_{\\mathrm{E}}")
'd_E'

>>> latex_to_unicode(r"\\gamma_G \\cdot G_k + \\gamma_Q \\cdot Q_k")
'γ_G · G_k + γ_Q · Q_k'

>>> latex_to_unicode(r"\\sigma = \\frac{F}{A}")
'σ = F/A'

>>> latex_to_unicode(r"m_{\\mathrm{a}} = \\rho \\pi (a_y^2 \\cos^2 \\theta + a_x^2 \\sin^2 \\theta)")
'm_a = ρ π (a_y² cos² θ + a_x² sin² θ)'
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Pre-processing: normalise LaTeX before flatlatex sees it
# ---------------------------------------------------------------------------

# \mathrm{X} → X   (flatlatex doesn't handle \mathrm)
_RE_MATHRM = re.compile(r"\\mathrm\{([^}]*)\}")
# \text{X} → X
_RE_TEXT = re.compile(r"\\text\{([^}]*)\}")
# \textbf{X} → X
_RE_TEXTBF = re.compile(r"\\textbf\{([^}]*)\}")
# \textit{X} → X
_RE_TEXTIT = re.compile(r"\\textit\{([^}]*)\}")
# \operatorname{X} → X
_RE_OPNAME = re.compile(r"\\operatorname\{([^}]*)\}")
# \left and \right (sizing) → plain delimiters
_RE_LEFT = re.compile(r"\\left\s*([(\[{|.])")
_RE_RIGHT = re.compile(r"\\right\s*([)\]}|.])")
# \, \; \! \quad \qquad → spaces
_RE_SPACES = re.compile(r"\\[,;!]|\\quad|\\qquad")
# \begin{...} ... \end{...} → strip environments
_RE_ENV = re.compile(r"\\(?:begin|end)\{[^}]*\}")
# && alignment markers
_RE_ALIGN = re.compile(r"&&?")
# \\ line breaks in align environments
_RE_LINEBREAK = re.compile(r"\\\\")


def _preprocess(latex: str) -> str:
    """Normalise LaTeX to something flatlatex handles well."""
    s = latex.strip()
    # Strip dollar signs / display math delimiters
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()

    s = _RE_MATHRM.sub(r"\1", s)
    s = _RE_TEXT.sub(r"\1", s)
    s = _RE_TEXTBF.sub(r"\1", s)
    s = _RE_TEXTIT.sub(r"\1", s)
    s = _RE_OPNAME.sub(r"\1", s)
    s = _RE_LEFT.sub(r"\1", s)
    s = _RE_RIGHT.sub(r"\1", s)
    s = _RE_SPACES.sub(" ", s)
    s = _RE_ENV.sub("", s)
    s = _RE_LINEBREAK.sub("; ", s)
    s = _RE_ALIGN.sub(" ", s)

    return s.strip()


# ---------------------------------------------------------------------------
#  Post-processing: clean up flatlatex artefacts
# ---------------------------------------------------------------------------

# flatlatex wraps subscripts in [brackets] when it can't find a Unicode subscript
_RE_BRACKET_SUB = re.compile(r"\[([^\]]{1,20})\]")


def _postprocess(text: str) -> str:
    """Clean up flatlatex output artefacts."""
    s = text

    # Replace [X] bracket subscripts with _X or _{X}
    def _sub_replace(m: re.Match) -> str:
        inner = m.group(1)
        # If it looks like a leftover \mathrm, just return the content
        if inner.startswith("\\mathrm"):
            return inner.replace("\\mathrm", "")
        # Single character subscript → use Unicode subscript notation
        if len(inner) == 1:
            return f"_{inner}"
        return f"_{{{inner}}}"

    s = _RE_BRACKET_SUB.sub(_sub_replace, s)

    # Remove any remaining backslash commands that weren't converted
    s = re.sub(r"\\(?:mathrm|text|textbf|operatorname)\s*", "", s)

    # Collapse multiple spaces
    s = re.sub(r"  +", " ", s)

    return s.strip()


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

# Lazy-loaded converter instance
_converter = None


def _get_converter():
    """Create/return a singleton flatlatex converter."""
    global _converter
    if _converter is None:
        import flatlatex
        _converter = flatlatex.converter()
    return _converter


def latex_to_unicode(latex: str) -> str:
    """Convert a LaTeX math expression to readable Unicode text.

    Two-stage pipeline:
    1. Use pylatexenc to parse & extract text (handles \\mathrm{}, \\text{})
    2. Use flatlatex to convert remaining LaTeX symbols
    3. Fallback to basic regex if both fail

    Always returns a string. On failure, returns the original LaTeX
    with dollar signs and basic commands stripped.

    Parameters
    ----------
    latex : str
        Raw LaTeX math expression (with or without $ delimiters).

    Returns
    -------
    str
        Unicode representation of the formula.
    """
    if not latex or not latex.strip():
        return latex or ""

    # Stage 1: Try pylatexenc (parses LaTeX, extracts text)
    try:
        result = _pylatexenc_convert(latex)
        if result and result.strip() and result != latex.strip():
            return result
    except Exception as e:
        logger.debug("pylatexenc conversion failed for %r: %s", latex[:80], e)

    # Stage 2: Fallback to flatlatex + pre/post-processing
    preprocessed = _preprocess(latex)
    if not preprocessed:
        return latex.strip()

    try:
        conv = _get_converter()
        result = conv.convert(preprocessed)
        return _postprocess(result)
    except Exception as e:
        logger.debug("flatlatex conversion failed for %r: %s", latex[:80], e)
        # Stage 3: Basic fallback without external converters
        return _fallback_convert(preprocessed)


def _pylatexenc_convert(latex: str) -> Optional[str]:
    """Use pylatexenc to parse LaTeX and extract readable text.

    Handles \\mathrm{}, \\text{}, and other semantic elements better
    than flatlatex alone.
    """
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        from pylatexenc.parser import LatexParser
        
        # Parse the LaTeX
        parser = LatexParser()
        latex_clean = latex.strip()
        # Remove outer $ or $$ delimiters
        if latex_clean.startswith("$$") and latex_clean.endswith("$$"):
            latex_clean = latex_clean[2:-2].strip()
        elif latex_clean.startswith("$") and latex_clean.endswith("$"):
            latex_clean = latex_clean[1:-1].strip()
        
        nodes = parser.parse(latex_clean)
        # Extract text (LatexNodes2Text handles \mathrm properly)
        converter = LatexNodes2Text()
        text = converter.latex_to_text(nodes)
        
        if text and text.strip():
            # Now apply Unicode conversions for Greek letters & operators
            return _apply_unicode_symbolics(text)
    except ImportError:
        logger.debug("pylatexenc not available, skipping")
    except Exception as e:
        logger.debug("pylatexenc parse error: %s", e)
    
    return None


def _apply_unicode_symbolics(text: str) -> str:
    """Apply Unicode transformations to parsed text.
    
    Converts:
    - Greek letters: \\alpha → α, etc.
    - Operators: \\cdot → ·, etc.
    - Powers: ^2 → ², etc.
    """
    s = text.strip()
    
    # Greek letters (uppercase and lowercase)
    _greek = {
        "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
        "\\epsilon": "ε", "\\zeta": "ζ", "\\eta": "η", "\\theta": "θ",
        "\\iota": "ι", "\\kappa": "κ", "\\lambda": "λ", "\\mu": "μ",
        "\\nu": "ν", "\\xi": "ξ", "\\pi": "π", "\\rho": "ρ",
        "\\sigma": "σ", "\\tau": "τ", "\\upsilon": "υ", "\\phi": "φ",
        "\\chi": "χ", "\\psi": "ψ", "\\omega": "ω",
        "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
        "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ", "\\Phi": "Φ",
        "\\Psi": "Ψ", "\\Omega": "Ω",
    }
    for latex_name, unicode_char in _greek.items():
        s = s.replace(latex_name, unicode_char)
    
    # Operators
    _operators = {
        "\\cdot": "·", "\\times": "×", "\\pm": "±",
        "\\leq": "≤", "\\geq": "≥", "\\neq": "≠",
        "\\approx": "≈", "\\infty": "∞",
        "\\sum": "Σ", "\\prod": "Π", "\\int": "∫",
        "\\partial": "∂", "\\nabla": "∇",
    }
    for latex_op, unicode_op in _operators.items():
        s = s.replace(latex_op, unicode_op)
    
    # Remove any remaining backslash commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    
    # Collapse extra spaces
    s = re.sub(r"  +", " ", s)
    
    return s.strip()


def _fallback_convert(s: str) -> str:
    """Minimal LaTeX→Unicode without flatlatex (used as fallback)."""
    # Greek letters
    _greek = {
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
        "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
        "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
        "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
        "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
        "chi": "χ", "psi": "ψ", "omega": "ω",
        "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
        "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
        "Psi": "Ψ", "Omega": "Ω",
    }
    for name, char in _greek.items():
        s = s.replace(f"\\{name}", char)

    # Common operators
    s = s.replace("\\cdot", "·")
    s = s.replace("\\times", "×")
    s = s.replace("\\pm", "±")
    s = s.replace("\\leq", "≤")
    s = s.replace("\\geq", "≥")
    s = s.replace("\\neq", "≠")
    s = s.replace("\\approx", "≈")
    s = s.replace("\\infty", "∞")
    s = s.replace("\\sum", "Σ")
    s = s.replace("\\prod", "Π")
    s = s.replace("\\int", "∫")
    s = s.replace("\\partial", "∂")
    s = s.replace("\\nabla", "∇")

    # \sqrt{X} → √(X)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", s)
    # \frac{a}{b} → a/b
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)

    # Strip remaining backslash commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Strip braces
    s = s.replace("{", "").replace("}", "")
    # Collapse spaces
    s = re.sub(r"  +", " ", s)

    return s.strip()
