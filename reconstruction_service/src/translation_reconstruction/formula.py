"""Formula rendering: LaTeX → PNG via pdflatex when available, matplotlib
mathtext as a fallback. Emits a positioned picture or, when image generation
fails, a centred text box with the raw LaTeX in Cambria Math."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from io import BytesIO
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 18

# Make regular text content (i.e. anything outside `$...$` math segments) fall
# back to a multi-script font when the default serif font has no glyph for a
# codepoint. matplotlib walks this list per-glyph and picks the first font
# that contains the requested character. Pure math segments (rendered via
# matplotlib's mathtext at `mathtext.fontset = "cm"`) keep their Computer
# Modern glyphs for letters, digits, operators, sub/superscripts.
#
# Important: matplotlib only registers the FIRST face inside a TTC (TrueType
# Collection) file. Noto Sans/Serif CJK ships as a single TTC whose first
# face is `JP`, so the registered names are `Noto Sans CJK JP` and
# `Noto Serif CJK JP` even though the file covers SC / TC / JP / KR / Han
# ideographs. Listing `Noto Sans CJK SC` here does nothing.
#
# Order: CJK first (covers Han ideographs used in Chinese, Japanese, Korean
# OCR output), then DejaVu Serif for Latin / Greek / common symbols.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Noto Serif CJK JP",   # covers Han / Hiragana / Katakana / Hangul
    "Noto Sans CJK JP",    # same coverage, sans-serif fallback
    "DejaVu Serif",        # Latin / Greek / common symbols
]
plt.rcParams["axes.unicode_minus"] = False

# Best-effort: rebuild matplotlib's font manager if it hasn't picked up the
# newly-installed Noto CJK TTC yet. Safe no-op when the cache is already
# fresh; only runs at module import time so the cost is one-shot.
try:
    from matplotlib import font_manager as _font_manager
    if not any("CJK" in (f.name or "") for f in _font_manager.fontManager.ttflist):
        _font_manager._load_fontmanager(try_read_cache=False)
except Exception:
    pass


def create_formula_image(
    formula_text: str,
    fontsize: int = 18,
    bold: bool = False,
    target_width_pt: Optional[float] = None,
) -> Optional[BytesIO]:
    """Render the formula to a PNG tightly cropped to the glyph extent.

    The caller is responsible for placing the PNG inside its bbox at the
    PNG's own aspect ratio (see `render_formula`). That keeps the formula
    text at its rendered size and avoids surrounding it with the big
    whitespace canvas that a bbox-aspect figsize would produce.

    When ``bold=True``: pdflatex output uses ``\\mathversion{bold}``;
    matplotlib fallback uses ``fontweight="bold"`` and the bolder
    ``dejavuserif`` math font.

    ``target_width_pt``: when provided (matplotlib fallback only), the
    figure width is sized to this value in points. That makes matplotlib's
    line-wrapping produce a PNG whose aspect ratio matches the destination
    bbox, so mixed-content formulas (Chinese prose + inline math) don't
    render as one 50:1-wide strip that then shrinks to unreadable in the
    bbox scaling step in ``render_formula``.
    """
    clean = formula_text.strip()
    clean = re.sub(r"\$+", "", clean)
    clean = clean.replace("\\n", " ").replace("\n", " ").strip()
    # Unlimited-OCR sometimes emits LaTeX with whitespace between a command
    # and its `{...}` arguments (e.g. `\frac { 1 } { 2 }`). pdflatex accepts
    # this, but matplotlib's mathtext does not — it raises
    # `Expected \frac{num}{den}, found '{'`. Collapse the spaces so both
    # renderers can parse the same input.
    clean = re.sub(r"(\\[A-Za-z]+)\s+\{", r"\1{", clean)
    # Same for the space between two adjacent `{...}` arguments:
    # `\frac{a} {b}` → `\frac{a}{b}`.
    clean = re.sub(r"\}\s+\{", "}{", clean)
    # Fill empty `{}` so commands like `\frac{}{x}` get a placeholder. Empty
    # args raise the same mathtext ParseSyntaxException as missing braces.
    clean = re.sub(r"\{\s*\}", r"{\\,}", clean)
    if not clean:
        return None

    # Try pdflatex first when ImageMagick is permitted to read PDFs AND the
    # input is pure Latin/ASCII math. Some hosts ship `policy.xml` with
    # `coder PDF rights="none"` (Ubuntu / Debian default), which makes
    # `convert pdf png` fail. In that case we fall through to matplotlib.
    #
    # Skip pdflatex entirely when the input contains any non-ASCII glyph
    # (CJK, accented Latin, Greek, etc.): the default LaTeX preamble has no
    # CJK font, so pdflatex silently drops those characters and still emits
    # a PDF containing only the Latin/math atoms. That produced the "Chinese
    # text disappeared, only numbers survived" bug. matplotlib's mathtext
    # path handles mixed scripts via `_build_renderable_string` + per-glyph
    # font fallback to Noto CJK, so it's the right renderer for those cases.
    has_non_ascii = any(ord(ch) > 127 for ch in clean)
    if not has_non_ascii:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = os.path.join(tmpdir, "formula.tex")
                # \mathversion{bold} bolds every glyph in math mode globally
                # (including \frac, \sqrt, etc.).
                math_version = "\\mathversion{bold}" if bold else ""
                font_pt = max(6, int(fontsize))
                line_pt = int(round(font_pt * 1.2))
                with open(tex_file, "w") as f:
                    f.write(
                        "\\documentclass[border=0pt,varwidth]{standalone}\n"
                        "\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
                        f"\\begin{{document}}\n"
                        f"\\fontsize{{{font_pt}}}{{{line_pt}}}\\selectfont\n"
                        f"{math_version}${clean}$\n"
                        f"\\end{{document}}"
                    )
                latex_result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode",
                     "-output-directory", tmpdir, tex_file],
                    capture_output=True, timeout=10,
                )
                pdf_file = os.path.join(tmpdir, "formula.pdf")
                png_file = os.path.join(tmpdir, "formula.png")
                # Only accept the PDF when pdflatex reported success. On
                # failure pdflatex often still emits a partial PDF that
                # renders with missing/broken glyphs, so trust rc, not the
                # file's existence.
                if latex_result.returncode == 0 and os.path.exists(pdf_file):
                    result = subprocess.run(
                        ["convert", "-density", "300", "-quality", "100",
                         pdf_file, png_file],
                        capture_output=True, timeout=10,
                    )
                    if result.returncode == 0 and os.path.exists(png_file):
                        buf = BytesIO()
                        with open(png_file, "rb") as f:
                            buf.write(f.read())
                        buf.seek(0)
                        return buf
        except Exception:
            pass

    # Matplotlib fallback. We split the LaTeX into math segments and
    # `\text{...}` text segments, then build a string where math is wrapped
    # in `$...$` (rendered via matplotlib's mathtext / Computer Modern) and
    # text is left bare (rendered via the regular text path, which uses
    # `font.serif` and supports per-glyph font fallback to Noto CJK for any
    # non-Latin script the OCR emits).
    #
    # Wrapping the whole formula in `$...$` like before forces every glyph
    # — including CJK characters in `\text{...}` — through mathtext's
    # `rm` font (Computer Modern), which doesn't ship CJK glyphs and renders
    # tofu boxes. Splitting first preserves math layout AND renders text in
    # the right script.
    try:
        # See `_wrap_parts` for the wrapping rationale: keeps mixed CJK+math
        # paragraphs from rendering as one 50:1-wide strip.
        parts = _build_renderable_parts(clean)
        rendered = _wrap_parts(parts, fontsize, target_width_pt)
        if target_width_pt and target_width_pt > 0:
            fig_w = max(2.0, target_width_pt / 72.0)
        else:
            fig_w = 6.0
        fig, ax = plt.subplots(figsize=(fig_w, 3.0))
        fig.patch.set_facecolor("white")
        ax.text(
            0.5, 0.5, rendered, fontsize=fontsize,
            ha="center", va="center", transform=ax.transAxes,
            family="serif",
            fontweight=("bold" if bold else "normal"),
            math_fontfamily=("dejavuserif" if bold else "cm"),
        )
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none",
                    pad_inches=0.02, format="png")
        plt.close(fig)
        buf.seek(0)
        return _trim_to_ink(buf)
    except Exception as e:
        print(f"Error creating formula image: {e}")
        if "fig" in locals():
            plt.close(fig)
        return None


# Match `\text{ ... }` with no nested braces (sufficient for the LaTeX subset
# the OCR model produces in practice — `\text{...}` is always single-level).
_TEXT_RE = re.compile(r"\\text\s*\{([^}]*)\}")


def _is_text_command_at_brace_depth_zero(latex: str, start: int) -> bool:
    """True if `latex[start:]` begins a `\\text{...}` AT brace depth zero
    (i.e. not nested inside another command's argument group).

    A nested `\\text` (inside `\\frac{...}` etc.) cannot be hoisted out
    because doing so would leave an empty `\\frac{}` that mathtext rejects."""
    depth = 0
    for ch in latex[:start]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    return depth == 0


def _build_renderable_parts(latex: str) -> list[str]:
    """Split a LaTeX formula into an ordered list of matplotlib-renderable
    fragments. Text fragments are bare (rendered via the regular text path,
    with per-glyph font fallback so CJK / accented chars work); math
    fragments are wrapped in `$...$` (rendered via mathtext).

    Behaviour:
      - Top-level `\\text{加速老化因子}` → bare `加速老化因子` fragment.
      - `\\text{...}` *inside* another command's brace group (e.g. inside
        `\\frac{...}`) is rewritten in place to `\\mathrm{...}` because
        mathtext can't parse `\\text` in math mode but accepts `\\mathrm`.
        That loses CJK fallback for those nested labels but the formula
        still renders. (CJK *outside* nested groups stays high-fidelity.)
      - Everything else stays as math and is wrapped in `$...$`.
    """
    # First pass: rewrite any nested `\text{...}` → `\mathrm{...}`.
    rewritten_parts: list[str] = []
    pos = 0
    for m in _TEXT_RE.finditer(latex):
        rewritten_parts.append(latex[pos:m.start()])
        if _is_text_command_at_brace_depth_zero(latex, m.start()):
            rewritten_parts.append(m.group(0))
        else:
            rewritten_parts.append(f"\\mathrm{{{m.group(1)}}}")
        pos = m.end()
    rewritten_parts.append(latex[pos:])
    latex = "".join(rewritten_parts)

    # Second pass: hoist remaining (top-level) \text{...} chunks.
    parts: list[str] = []
    pos = 0
    for m in _TEXT_RE.finditer(latex):
        math_chunk = latex[pos:m.start()]
        if math_chunk.strip():
            parts.append(f"${math_chunk}$")
        text_chunk = m.group(1)
        if text_chunk:
            parts.append(text_chunk)
        pos = m.end()
    tail = latex[pos:]
    if tail.strip():
        parts.append(f"${tail}$")
    return parts if parts else [latex]


def _build_renderable_string(latex: str) -> str:
    """Concatenated form of `_build_renderable_parts`."""
    return "".join(_build_renderable_parts(latex))


# Rough width in points of one "average" glyph at fontsize 1pt. Used to
# estimate line width for the wrap heuristic. See ocr_reconstruction/formula.py
# for calibration notes.
_APPROX_CHAR_WIDTH_PER_PT = 0.55


def _wrap_parts(parts: list[str], fontsize: int, target_width_pt: Optional[float]) -> str:
    """Wrap fragment list into multiple lines so no line exceeds
    `target_width_pt`. Wrapping happens only at fragment boundaries — never
    inside a math atom or `\\text{...}` body. No-op when target_width_pt
    is None."""
    if not target_width_pt or target_width_pt <= 0:
        return "".join(parts)
    max_line_chars = max(4, int(target_width_pt / (fontsize * _APPROX_CHAR_WIDTH_PER_PT)))
    lines: list[str] = []
    current = ""
    for part in parts:
        visible = part[1:-1] if part.startswith("$") and part.endswith("$") else part
        if current and len(current) + len(visible) > max_line_chars:
            lines.append(current)
            current = part
        else:
            current += part
    if current:
        lines.append(current)
    return "\n".join(lines)


# Render resolution used for matplotlib output. The pixel→point mapping
# `pt = px / DPI * 72` means the caller can predict the final glyph height in
# points from the trimmed PNG height. Higher DPI gives sharper glyphs in Word.
DPI = 300


def _trim_to_ink(buf: BytesIO) -> BytesIO:
    """Crop a PNG to its non-white bounding box (whitespace removal).

    Matplotlib's `bbox_inches='tight'` leaves axes padding around the glyphs.
    Trimming to the ink bbox makes the returned PNG dimensions equal the
    glyph dimensions, so the caller can size the placed extent to match the
    target text size precisely.
    """
    from PIL import Image
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # `getbbox` on the inverted-greyscale image finds the bbox of non-zero
    # (= non-white) pixels. Threshold at 240 to ignore JPEG-ish white noise.
    gray = img.convert("L")
    mask = gray.point(lambda v: 0 if v >= 240 else 255)
    bbox = mask.getbbox()
    if not bbox:
        # Empty image — return the original to avoid losing data.
        buf.seek(0)
        return buf
    cropped = img.crop(bbox)
    out = BytesIO()
    cropped.save(out, format="PNG")
    out.seek(0)
    return out


def render_formula(ctx, entry: Dict) -> None:
    """Emit a Formula entry. Renders as a positioned picture when image
    generation succeeds, otherwise a centred text box with the raw LaTeX in
    Cambria Math.

    Sizing rule:
      - Render at `fontsize = entry.style.size` (the OCR-detected point size,
        which matches surrounding body text by construction of the OCR
        heuristic). The matplotlib PNG is then trimmed to its ink bbox by
        `create_formula_image`.
      - The placed extent in EMU uses `placed_h_pt = png_h_px / DPI * 72`,
        which equals the natural rendering height of the (trimmed) glyphs at
        the requested fontsize. Word draws the PNG at exactly that size, so
        the visible glyphs read at `entry.style.size` pt — the same point
        size as normal text on the page.
      - The placed rectangle is clipped to the bbox dimensions (so a
        formula whose OCR-detected size would overshoot the bbox shrinks
        proportionally) and centred inside the bbox.
    """
    text = (entry.get("text") or "").strip()
    bbox = entry.get("bbox")
    if not text or not bbox or len(bbox) != 4:
        return
    from .geometry import bbox_px_to_emu, EMU_PER_PT
    from .ooxml import (
        build_anchored_picture_xml, build_anchored_textbox_xml,
        build_paragraph_xml, build_run_xml, add_image_relationship,
    )
    bbox_x_emu, bbox_y_emu, bbox_w_emu, bbox_h_emu = bbox_px_to_emu(
        bbox, ctx.zoom, ctx.page_w_pt, ctx.page_h_pt,
    )

    # Use the OCR-detected font size — that's the size of the surrounding
    # body text the formula sits in. Fall back to 11pt (the doc's Normal
    # size) if the OCR style is missing or unusable.
    style = entry.get("style") or {}
    try:
        target_pt = float(style.get("size") or 11.0)
    except (TypeError, ValueError):
        target_pt = 11.0

    # Operator-configurable size boost + bold. Defaults give a clearly
    # larger and bolder formula than the surrounding text. Override per
    # deployment via env.
    try:
        size_multiplier = float(os.environ.get("FORMULA_SIZE_MULTIPLIER", "1.6"))
    except (TypeError, ValueError):
        size_multiplier = 1.6
    bold = os.environ.get("FORMULA_BOLD", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    target_pt *= max(0.5, size_multiplier)
    target_pt = max(6.0, min(72.0, target_pt))   # safety clamp

    # Pass the destination bbox width so the matplotlib fallback wraps
    # mixed-content formulas (CJK prose + inline math) to roughly match the
    # bbox aspect ratio.
    target_width_pt = bbox_w_emu / EMU_PER_PT if bbox_w_emu > 0 else None
    img_buf = create_formula_image(
        text, fontsize=int(round(target_pt)), bold=bold,
        target_width_pt=target_width_pt,
    )
    if img_buf is None:
        fallback_style = {
            **style, "font": "Cambria Math",
            "size": target_pt,
            "bold": bold or bool(style.get("bold")),
        }
        runs_xml = build_run_xml(re.sub(r"\$+", "", text), fallback_style)
        para_xml = build_paragraph_xml(runs_xml, alignment="center")
        ctx.xml_chunks.append(
            build_anchored_textbox_xml(
                bbox_x_emu, bbox_y_emu, bbox_w_emu, bbox_h_emu,
                para_xml, ctx._next_id(),
            )
        )
        return

    # Measure the (trimmed) PNG. After trimming, png height equals the ink
    # height; converting px → pt at DPI gives the natural rendering size in
    # points.
    from PIL import Image as _PILImage
    img_buf.seek(0)
    png_w_px, png_h_px = _PILImage.open(img_buf).size
    img_buf.seek(0)

    if png_w_px <= 0 or png_h_px <= 0 or bbox_w_emu <= 0 or bbox_h_emu <= 0:
        placed_x = bbox_x_emu
        placed_y = bbox_y_emu
        placed_w = bbox_w_emu
        placed_h = bbox_h_emu
    else:
        natural_w_pt = png_w_px / DPI * 72.0
        natural_h_pt = png_h_px / DPI * 72.0
        natural_w_emu = int(round(natural_w_pt * EMU_PER_PT))
        natural_h_emu = int(round(natural_h_pt * EMU_PER_PT))

        # If the natural rendering overshoots the bbox in either axis, scale
        # down uniformly so the formula stays inside its source bbox.
        scale_w = bbox_w_emu / natural_w_emu if natural_w_emu else 1.0
        scale_h = bbox_h_emu / natural_h_emu if natural_h_emu else 1.0
        scale = min(1.0, scale_w, scale_h)
        placed_w = max(1, int(round(natural_w_emu * scale)))
        placed_h = max(1, int(round(natural_h_emu * scale)))

        # Centre inside the bbox so the glyphs sit where the OCR detected
        # them (matches the surrounding text baseline closely enough).
        placed_x = bbox_x_emu + (bbox_w_emu - placed_w) // 2
        placed_y = bbox_y_emu + (bbox_h_emu - placed_h) // 2

    rel_id = add_image_relationship(ctx.doc, img_buf, "png")
    ctx.xml_chunks.append(
        build_anchored_picture_xml(
            placed_x, placed_y, placed_w, placed_h,
            rel_id, ctx._next_id(), pic_name="Formula"
        )
    )
