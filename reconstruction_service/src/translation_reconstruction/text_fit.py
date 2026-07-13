"""Text measurement + binary-search font fit. CJK-aware wrapping that matches
how Word lays text out closely enough for fitting.

Font discovery
--------------
Measurement quality depends on loading a scalable TrueType face. The document
is RENDERED in Calibri (see UNIFIED_FONT in json_to_docx.py), so for the
fitter's width predictions to match what Word actually draws we measure with a
Calibri-metric-compatible face when one is available:

    Carlito  ── metric-compatible open clone of Calibri (best match)
    Calibri  ── the real thing, if MS fonts are installed
    then wider generic fallbacks (DejaVu / Liberation / FreeSans / Arial)

Discovery order, first hit wins:
  1. Explicit override via env vars DOCX_MEASURE_FONT / DOCX_MEASURE_FONT_BOLD
  2. A broad list of known absolute paths across Linux/macOS/Windows
  3. `fc-match` (fontconfig), if present, to resolve a family by name
  4. matplotlib's bundled DejaVu as a last structured fallback

If NOTHING scalable is found, `_FONT_PATH` stays None and the fitter degrades
to "render at base size, unwrapped" — which silently clips translated text.
That used to fail invisibly; we now emit a one-time warning so a missing-font
environment is diagnosable instead of looking like a layout bug.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import warnings
from typing import Dict, List, Optional, Tuple

from PIL import ImageDraw, ImageFont


# Regular-weight candidates, BEST METRIC MATCH FIRST. Carlito is the
# metric-compatible Calibri clone; measuring with it (rather than the ~5-8%
# wider DejaVu) stops the fitter from over-shrinking and over-wrapping.
_FONT_FALLBACK_CANDIDATES = (
    # Carlito (Calibri clone) — Debian/Ubuntu, Fedora
    "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
    "/usr/share/fonts/crosextra-carlito/Carlito-Regular.ttf",
    "/usr/share/fonts/google-crosextra-carlito/Carlito-Regular.ttf",
    # Real Calibri, if MS core fonts are present
    "/usr/share/fonts/truetype/msttcorefonts/Calibri.ttf",
    "/usr/share/fonts/microsoft/Calibri.ttf",
    "C:\\Windows\\Fonts\\calibri.ttf",
    "/Library/Fonts/Calibri.ttf",
    # Generic wide fallbacks (correct shape, looser metrics)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
)
_FONT_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/crosextra-carlito/Carlito-Bold.ttf",
    "/usr/share/fonts/google-crosextra-carlito/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Calibri_Bold.ttf",
    "/usr/share/fonts/microsoft/Calibrib.ttf",
    "C:\\Windows\\Fonts\\calibrib.ttf",
    "/Library/Fonts/Calibri Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
)

# Family names to try via fontconfig (`fc-match`) when no absolute path hits.
_FC_FAMILIES_REGULAR = ("Carlito", "Calibri", "Liberation Sans", "DejaVu Sans", "Arial")
_FC_FAMILIES_BOLD = (
    "Carlito:bold", "Calibri:bold", "Liberation Sans:bold",
    "DejaVu Sans:bold", "Arial:bold",
)


def _resolve_font_path(candidates) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_via_fontconfig(families) -> Optional[str]:
    """Use `fc-match` to resolve a family name to a concrete file path.
    Returns None when fontconfig is unavailable or matches nothing usable."""
    fc = shutil.which("fc-match")
    if not fc:
        return None
    for fam in families:
        try:
            out = subprocess.run(
                [fc, "-f", "%{file}", fam],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            continue
        path = (out.stdout or "").strip()
        if path and os.path.exists(path) and path.lower().endswith((".ttf", ".otf")):
            return path
    return None


def _resolve_via_matplotlib(bold: bool) -> Optional[str]:
    """Last structured fallback: matplotlib ships DejaVu Sans inside its own
    package, so if it's importable we can always get a real scalable face."""
    try:
        from matplotlib import font_manager
        from matplotlib.font_manager import FontProperties
        prop = FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal"
        )
        path = font_manager.findfont(prop, fallback_to_default=True)
        if path and os.path.exists(path):
            return path
    except Exception:
        return None
    return None


def _discover_font(env_var: str, abs_candidates, fc_families, bold: bool) -> Optional[str]:
    # 1) explicit override
    override = os.environ.get(env_var)
    if override and os.path.exists(override):
        return override
    # 2) known absolute paths (metric-best first)
    p = _resolve_font_path(abs_candidates)
    if p:
        return p
    # 3) fontconfig by family name
    p = _resolve_via_fontconfig(fc_families)
    if p:
        return p
    # 4) matplotlib's bundled DejaVu
    return _resolve_via_matplotlib(bold)


_FONT_PATH = _discover_font(
    "DOCX_MEASURE_FONT", _FONT_FALLBACK_CANDIDATES, _FC_FAMILIES_REGULAR, bold=False,
)
_FONT_PATH_BOLD = _discover_font(
    "DOCX_MEASURE_FONT_BOLD", _FONT_BOLD_CANDIDATES, _FC_FAMILIES_BOLD, bold=True,
) or _FONT_PATH

if _FONT_PATH is None:
    # Loud, one-time warning. Without a measurement font the fitter cannot
    # shrink or wrap text, so every translated box that needs >1 line will
    # clip. This makes that failure visible instead of silent.
    warnings.warn(
        "text_fit: no scalable measurement font found — text fitting and "
        "reflow are DISABLED, translated text will clip. Install a font "
        "(e.g. `apt-get install fonts-crosextra-carlito fonts-dejavu-core`) "
        "or set DOCX_MEASURE_FONT to a .ttf path. Searched known paths, "
        "fontconfig, and matplotlib.",
        RuntimeWarning,
        stacklevel=2,
    )
    print("[text_fit] WARNING: no measurement font; text will clip.", file=sys.stderr)

_MEASURE_DRAW = ImageDraw.Draw(
    __import__("PIL.Image", fromlist=["new"]).new("RGB", (1, 1))
)
_FONT_CACHE: Dict[Tuple[str, int], "ImageFont.FreeTypeFont"] = {}


def get_font(size_px: int, bold: bool = False) -> Optional["ImageFont.FreeTypeFont"]:
    path = _FONT_PATH_BOLD if bold else _FONT_PATH
    if not path or size_px <= 0:
        return None
    key = (path, size_px)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(path, size_px)
        except Exception:
            return None
        _FONT_CACHE[key] = f
    return f


# Anchored text boxes are emitted with lIns=rIns=0 (see
# ooxml.build_anchored_textbox_xml → bodyPr), so the usable text width of such a
# box IS its full geometric width. Callers that render into those boxes pass this
# as a FIXED per-box edge pad (Word's line-fill rounding only), instead of the
# proportional `* width_safety` haircut — which, on a wide box, discarded more
# than a whole word and wrapped text to a new line where Word keeps it on one.
# Keep in sync with ooxml.py bodyPr insets: if those become non-zero, set this to
# (lIns + rIns) in points.
_TEXTBOX_EDGE_PAD_PT = 2.0


_CJK_RE = re.compile(
    r"[\u1100-\u11ff\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3130-\u318f"
    r"\u3400-\u4dbf\u4e00-\u9fff\ua000-\ua4cf\uac00-\ud7af\uf900-\ufaff"
    r"\ufe30-\ufe4f\uff00-\uffef]"
)


def has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))


def is_cjk_char(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def measure_width_px(text: str, font, size_px: int) -> float:
    if not text:
        return 0.0
    total = 0.0
    run = ""
    for ch in text:
        if is_cjk_char(ch):
            if run:
                b = _MEASURE_DRAW.textbbox((0, 0), run, font=font)
                total += (b[2] - b[0])
                run = ""
            total += size_px
        else:
            run += ch
    if run:
        b = _MEASURE_DRAW.textbbox((0, 0), run, font=font)
        total += (b[2] - b[0])
    return total


def _segments(paragraph: str) -> List[str]:
    toks: List[str] = []
    buf = ""
    for ch in paragraph:
        if is_cjk_char(ch):
            if buf:
                toks.append(buf)
                buf = ""
            toks.append(ch)
        elif ch == " ":
            buf += " "
            toks.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        toks.append(buf)
    return toks


def wrap_to_width(text: str, font, max_w_px: float, size_px: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        cur = ""
        for tok in _segments(paragraph):
            candidate = cur + tok
            if not cur.strip() or measure_width_px(candidate, font, size_px) <= max_w_px:
                cur = candidate
            else:
                lines.append(cur.rstrip())
                cur = "" if tok.strip() == "" else tok
        if cur.strip():
            lines.append(cur.rstrip())
    return lines or [""]


def _wrapped_lines_if_fit(
    text: str,
    size_px: int,
    box_w_px: float,
    box_h_px: float,
    width_safety: float = 0.93,
    height_safety: float = 1.10,
    bold: bool = False,
    edge_pad_px: Optional[float] = None,
) -> Optional[List[str]]:
    font = get_font(size_px, bold=bold)
    if font is None:
        return None
    # `edge_pad_px` (fixed reserve) models a real zero-inset text box; when not
    # supplied, fall back to the legacy proportional haircut so every existing
    # caller (e.g. the table path) is byte-for-byte unchanged.
    wrap_w = (
        max(1.0, box_w_px - edge_pad_px) if edge_pad_px is not None
        else box_w_px * width_safety
    )
    lines = wrap_to_width(text, font, wrap_w, size_px)
    for line in lines:
        if measure_width_px(line, font, size_px) > box_w_px:
            return None
    asc, desc = font.getmetrics()
    natural_h = asc + desc
    if has_cjk(text):
        natural_h = max(natural_h, size_px * 1.2)
    line_h = natural_h * height_safety
    if line_h * len(lines) > box_h_px:
        return None
    return lines


_ELLIPSIS = "…"


def _truncate_to_fit(
    text: str,
    size_px: int,
    box_w_px: float,
    box_h_px: float,
    bold: bool = False,
) -> List[str]:
    font = get_font(size_px, bold=bold)
    if font is None:
        return [text]

    safe_w = box_w_px * 0.93
    lines = wrap_to_width(text, font, safe_w, size_px)
    if _fits_box(lines, font, size_px, box_w_px, box_h_px):
        return lines

    lo, hi = 0, len(text)
    best_lines: List[str] = [_ELLIPSIS]
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + _ELLIPSIS
        cand_lines = wrap_to_width(candidate, font, safe_w, size_px)
        if _fits_box(cand_lines, font, size_px, box_w_px, box_h_px):
            best_lines = cand_lines
            lo = mid + 1
        else:
            hi = mid - 1
    return best_lines


def _fits_box(
    lines: List[str],
    font,
    size_px: int,
    box_w_px: float,
    box_h_px: float,
    width_safety: float = 0.93,
    height_safety: float = 1.10,
) -> bool:
    safe_w = box_w_px * width_safety
    for line in lines:
        if measure_width_px(line, font, size_px) > safe_w:
            return False
    asc, desc = font.getmetrics()
    natural_h = asc + desc
    if any(has_cjk(line) for line in lines):
        natural_h = max(natural_h, size_px * 1.2)
    return natural_h * height_safety * len(lines) <= box_h_px


def fit_multiline(
    text: str,
    box_w_pt: float,
    box_h_pt: float,
    max_size_pt: float = 11.0,
    min_size_pt: float = 6.0,  # Increased from 1.0 to enforce a readable floor layout
    dpi: int = 72,
    bold: bool = False,
    edge_pad_pt: float = 0.0,
) -> Tuple[float, Optional[List[str]]]:
    if not text.strip() or box_w_pt <= 0 or box_h_pt <= 0 or _FONT_PATH is None:
        return max_size_pt, None

    pt_to_px = dpi / 72.0
    box_w_px = box_w_pt * pt_to_px
    box_h_px = box_h_pt * pt_to_px
    # A positive `edge_pad_pt` switches wrapping to the FIXED zero-inset text-box
    # model (usable width = full width − pad); the default 0.0 keeps the legacy
    # proportional haircut, so all existing callers are unchanged.
    edge_pad_px = (edge_pad_pt * pt_to_px) if edge_pad_pt > 0 else None
    hi = max(min_size_pt, max_size_pt)
    lo = min_size_pt

    hi_lines = _wrapped_lines_if_fit(
        text, int(round(hi * pt_to_px)), box_w_px, box_h_px, bold=bold,
        edge_pad_px=edge_pad_px,
    )
    if hi_lines is not None:
        return float(hi), hi_lines

    best_size = float(min_size_pt)
    best_lines: Optional[List[str]] = None
    while lo <= hi:
        mid = (lo + hi) / 2.0
        size_px = max(1, int(round(mid * pt_to_px)))
        lines = _wrapped_lines_if_fit(
            text, size_px, box_w_px, box_h_px, bold=bold,
            edge_pad_px=edge_pad_px,
        )
        if lines is not None:
            best_size, best_lines = mid, lines
            lo = mid + 0.5
        else:
            hi = mid - 0.5

    # Fallback path: the text can't fit the box even at the readable floor size.
    # Return `lines=None` to SIGNAL this to the caller instead of silently
    # truncating to an ellipsis (which loses the content). Callers respond by
    # growing the box / row so the full text stays visible — see
    # `text_entry.render_text_entry` (body_auto_fit) and `table.render_table`.
    # Truncation-to-ellipsis is still available on demand via `fit_or_truncate`.
    if best_lines is None:
        return float(min_size_pt), None

    return max(float(min_size_pt), best_size), best_lines


def fit_or_truncate(
    text: str,
    box_w_pt: float,
    box_h_pt: float,
    max_size_pt: float = 11.0,
    min_size_pt: float = 6.0,
    dpi: int = 72,
    bold: bool = False,
) -> Tuple[float, List[str]]:
    """Like `fit_multiline` but ALWAYS returns renderable lines: if the text
    can't fit even at the floor size, it is truncated with an ellipsis rather
    than reported as unfittable. Use only where growing the box is not an
    option and clipping is the lesser evil."""
    size, lines = fit_multiline(
        text, box_w_pt, box_h_pt, max_size_pt=max_size_pt,
        min_size_pt=min_size_pt, dpi=dpi, bold=bold,
    )
    if lines is not None:
        return size, lines
    if not text.strip() or box_w_pt <= 0 or box_h_pt <= 0 or _FONT_PATH is None:
        return max_size_pt, [text]
    pt_to_px = dpi / 72.0
    floor_px = max(1, int(round(min_size_pt * pt_to_px)))
    truncated = _truncate_to_fit(
        text, floor_px, box_w_pt * pt_to_px, box_h_pt * pt_to_px, bold=bold,
    )
    return float(min_size_pt), truncated
