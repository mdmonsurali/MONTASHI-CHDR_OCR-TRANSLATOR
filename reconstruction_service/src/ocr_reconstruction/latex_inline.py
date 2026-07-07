"""General inline-LaTeX handling for flowing text.

OCR/VLM output frequently wraps *inline* math — units, statistics, simple
sub/superscripts — in ``$...$`` (or ``\\(...\\)``) even inside otherwise plain
CJK prose, e.g. ``用 $3 \\mathrm{r}/\\mathrm{min}$ 的角速度`` or
``$0.25 \\pm 0.10 \\text{ Nm}$`` or ``$Q_{10}=2$``. Standalone ``$$...$$``
display equations are handled separately (see ``formula.py``); this module is
only for the inline spans that live inside a ``Text``/``Caption``/table-cell
string, which must stay *text* (selectable, reflowable) rather than becoming an
image.

The conversion is intentionally **document-agnostic**: it maps LaTeX commands
to Unicode (operators, Greek, spacing, font-style wrappers) and renders
``_{...}`` / ``^{...}`` as real Word sub/superscript runs. There is no
per-document vocabulary — any inline math the OCR emits is normalised the same
way.

Public API:
    strip_inline_math_to_plain(s)  -> str
        Best-effort plain-Unicode rendering (no run structure). Handy for
        measurement/wrapping, where we only need the visible glyph count.

    build_inline_runs(text, base_rpr, make_run) -> str
        Split ``text`` on inline-math delimiters and emit a sequence of
        ``<w:r>`` fragments, toggling vertAlign for sub/superscripts. The
        caller supplies ``base_rpr`` (the shared run-properties XML) and a
        ``make_run(rpr_extra, chunk)`` factory so this module stays free of
        OOXML-escaping / namespace concerns.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple


# Command → Unicode tables. These are standard LaTeX, not document-specific.
_SYMBOLS = {
    r"\pm": "\u00b1", r"\mp": "\u2213", r"\times": "\u00d7", r"\div": "\u00f7",
    r"\cdot": "\u00b7", r"\ast": "*", r"\star": "\u22c6",
    r"\leq": "\u2264", r"\le": "\u2264", r"\geq": "\u2265", r"\ge": "\u2265",
    r"\neq": "\u2260", r"\ne": "\u2260", r"\approx": "\u2248",
    r"\equiv": "\u2261", r"\sim": "\u223c", r"\propto": "\u221d",
    r"\ll": "\u226a", r"\gg": "\u226b",
    r"\rightarrow": "\u2192", r"\to": "\u2192", r"\leftarrow": "\u2190",
    r"\Rightarrow": "\u21d2", r"\Leftarrow": "\u21d0",
    r"\leftrightarrow": "\u2194", r"\Leftrightarrow": "\u21d4",
    r"\infty": "\u221e", r"\partial": "\u2202", r"\nabla": "\u2207",
    r"\degree": "\u00b0", r"\circ": "\u00b0", r"\prime": "\u2032",
    r"\pm ": "\u00b1", r"\sqrt": "\u221a", r"\sum": "\u2211",
    r"\prod": "\u220f", r"\int": "\u222b", r"\angle": "\u2220",
    r"\perp": "\u22a5", r"\parallel": "\u2225", r"\cdots": "\u22ef",
    r"\ldots": "\u2026", r"\dots": "\u2026", r"\vdots": "\u22ee",
    r"\pm\pm": "\u00b1",
    # Greek (lower)
    r"\alpha": "\u03b1", r"\beta": "\u03b2", r"\gamma": "\u03b3",
    r"\delta": "\u03b4", r"\epsilon": "\u03b5", r"\varepsilon": "\u03b5",
    r"\zeta": "\u03b6", r"\eta": "\u03b7", r"\theta": "\u03b8",
    r"\iota": "\u03b9", r"\kappa": "\u03ba", r"\lambda": "\u03bb",
    r"\mu": "\u03bc", r"\nu": "\u03bd", r"\xi": "\u03be", r"\pi": "\u03c0",
    r"\rho": "\u03c1", r"\sigma": "\u03c3", r"\tau": "\u03c4",
    r"\upsilon": "\u03c5", r"\phi": "\u03c6", r"\varphi": "\u03c6",
    r"\chi": "\u03c7", r"\psi": "\u03c8", r"\omega": "\u03c9",
    # Greek (upper)
    r"\Gamma": "\u0393", r"\Delta": "\u0394", r"\Theta": "\u0398",
    r"\Lambda": "\u039b", r"\Xi": "\u039e", r"\Pi": "\u03a0",
    r"\Sigma": "\u03a3", r"\Phi": "\u03a6", r"\Psi": "\u03a8",
    r"\Omega": "\u03a9",
    r"\%": "%", r"\&": "&", r"\#": "#", r"\_": "_", r"\{": "{", r"\}": "}",
    r"\$": "$",
}

# Whitespace-producing commands and the LaTeX active tilde (nbsp).
_SPACES = (r"\quad", r"\qquad", r"\,", r"\;", r"\:", r"\!", r"\ ", r"\>")

# Font-style / grouping wrappers whose *argument* is kept verbatim (as text).
_UNWRAP = (
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb",
    "mathfrak", "text", "textbf", "textit", "textrm", "textsf", "texttt",
    "operatorname", "boldsymbol", "bm", "hbox", "mbox", "rm", "bf", "it",
)

# Unicode super/subscript glyph maps — used for the common single-token cases
# (digits, +, -, =, parens, a few letters) so short scripts read inline
# without needing a separate run. Anything not covered falls back to a real
# Word vertAlign run, which handles arbitrary content.
_SUP = {
    "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
    "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
    "+": "\u207a", "-": "\u207b", "=": "\u207c", "(": "\u207d", ")": "\u207e",
    "n": "\u207f", "i": "\u2071",
}
_SUB = {
    "0": "\u2080", "1": "\u2081", "2": "\u2082", "3": "\u2083", "4": "\u2084",
    "5": "\u2085", "6": "\u2086", "7": "\u2087", "8": "\u2088", "9": "\u2089",
    "+": "\u208a", "-": "\u208b", "=": "\u208c", "(": "\u208d", ")": "\u208e",
}

# Inline-math delimiters: $...$ and \(...\). $$...$$ is intentionally NOT
# matched here — those are display formulas routed to formula.py.
_INLINE_RE = re.compile(r"\$(?!\$)(.+?)(?<!\$)\$|\\\((.+?)\\\)", re.DOTALL)


def _take_group(s: str, i: int) -> Tuple[str, int]:
    """If s[i] == '{', return (inner, index_after_close) with brace matching.
    Otherwise return (single_char, i+1). Assumes i is at the argument start."""
    if i >= len(s):
        return "", i
    if s[i] != "{":
        return s[i], i + 1
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)   # unbalanced — take the rest


def _convert_symbols(s: str) -> str:
    """Replace LaTeX commands with Unicode, unwrap font-style groups, drop
    spacing commands. Operates on a math-mode string with no $ delimiters and
    no top-level sub/superscripts (those are handled by the tokenizer)."""
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "~":                       # LaTeX nbsp
            out.append(" ")
            i += 1
            continue
        if ch != "\\":
            # Drop math-mode braces used purely for grouping.
            if ch in "{}":
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        # ch == '\\' : read a command name (letters) or a single symbol.
        m = re.match(r"\\([A-Za-z]+)", s[i:])
        if not m:
            # Escaped punctuation like \% \& \_ \{ \} \$ \, etc.
            two = s[i:i + 2]
            if two in _SYMBOLS:
                out.append(_SYMBOLS[two])
            elif two in (r"\,", r"\;", r"\:", r"\!", r"\ ", r"\>"):
                out.append(" ")
            else:
                out.append(two[1:])          # keep the char, drop backslash
            i += 2
            continue
        cmd = m.group(1)
        token = "\\" + cmd
        end = i + len(token)
        if cmd in _UNWRAP:
            arg, end = _take_group(s, end)
            out.append(_convert_symbols(arg))   # keep argument as plain text
        elif token in _SYMBOLS:
            out.append(_SYMBOLS[token])
        elif token in _SPACES:
            out.append(" ")
        else:
            # Unknown command: drop the backslash, keep the name (best effort).
            out.append(cmd)
        i = end
    # Collapse the runs of spaces the drops may have introduced.
    return re.sub(r"[ \t]{2,}", " ", "".join(out))


def _script_to_unicode(inner: str, table) -> Optional[str]:
    """If every char of the (already symbol-converted) script maps to a
    Unicode super/subscript glyph, return the mapped string; else None."""
    if not inner:
        return None
    mapped = []
    for ch in inner:
        if ch in table:
            mapped.append(table[ch])
        else:
            return None
    return "".join(mapped)


# A math segment is a (text, vert) pair. vert is None / "superscript" /
# "subscript". Consecutive baseline segments are merged by the caller.
Segment = Tuple[str, Optional[str]]


def _tokenize_math(body: str) -> List[Segment]:
    """Turn one inline-math body (no $ delimiters) into baseline/super/sub
    segments, converting commands to Unicode along the way."""
    segs: List[Segment] = []
    buf: List[str] = []

    def flush():
        if buf:
            segs.append((_convert_symbols("".join(buf)), None))
            buf.clear()

    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch in ("^", "_"):
            arg, j = _take_group(body, i + 1)
            conv = _convert_symbols(arg)
            table = _SUP if ch == "^" else _SUB
            uni = _script_to_unicode(conv, table)
            if uni is not None:
                buf.append(uni)                       # inline Unicode glyphs
            else:
                flush()
                segs.append((conv, "superscript" if ch == "^" else "subscript"))
            i = j
        else:
            buf.append(ch)
            i += 1
    flush()
    # Drop empty baseline segments produced by adjacent scripts.
    return [(t, v) for (t, v) in segs if t != "" or v is not None]


def strip_inline_math_to_plain(s: str) -> str:
    """Best-effort plain-text rendering of a string that may contain inline
    math. Used for width measurement / wrapping (glyph count only)."""
    if not s or ("$" not in s and "\\(" not in s):
        return s

    def repl(m: re.Match) -> str:
        body = m.group(1) if m.group(1) is not None else m.group(2)
        return "".join(t for (t, _v) in _tokenize_math(body))

    return _INLINE_RE.sub(repl, s)


def build_inline_runs(
    text: str,
    base_rpr: str,
    make_run: Callable[[str, str], str],
) -> str:
    """Emit ``<w:r>`` fragments for ``text``, converting inline ``$...$`` math.

    ``base_rpr``  : the inner XML of the shared ``<w:rPr>`` (without the
                    surrounding tag) applied to every run.
    ``make_run``  : ``make_run(rpr_extra, chunk)`` builds one ``<w:r>`` whose
                    run-properties are ``base_rpr + rpr_extra`` and whose body
                    is the (already newline-split) ``chunk``. Keeping this in
                    the caller lets ooxml.py own escaping and ``<w:br/>``.

    Returns the concatenated run XML. Baseline text keeps ``rpr_extra=""``;
    super/subscript spans add the corresponding ``<w:vertAlign/>``.
    """
    parts: List[str] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            parts.append(make_run("", text[pos:m.start()]))
        body = m.group(1) if m.group(1) is not None else m.group(2)
        for seg_text, vert in _tokenize_math(body):
            if seg_text == "":
                continue
            extra = ""
            if vert == "superscript":
                extra = '<w:vertAlign w:val="superscript"/>'
            elif vert == "subscript":
                extra = '<w:vertAlign w:val="subscript"/>'
            parts.append(make_run(extra, seg_text))
        pos = m.end()
    if pos < len(text):
        parts.append(make_run("", text[pos:]))
    return "".join(parts)
