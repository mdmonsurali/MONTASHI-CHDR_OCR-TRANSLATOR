"""vLLM client + Unlimited-OCR document-parsing call.

Talks to the upstream `vllm/vllm-openai:unlimited-ocr` server over the
OpenAI-compatible API. Unlimited-OCR emits a custom format:

    <|det|>category [x1, y1, x2, y2]<|/det|>{text for that region}

…repeated, in reading order. Bboxes are normalized to a 1000×1000 canvas
(empirically: max coords ≤ 1000 regardless of input image size). We rescale
to original-image pixels here so downstream font_attribution,
picture_recovery, layoutjson2md, and json_to_docx (which expect pixel
bboxes, like DotsOCR produced) keep working.
"""

import base64
import logging
import os
import re

from openai import AsyncOpenAI, OpenAI
from PIL import Image

log = logging.getLogger("ocr_service")

VLLM_PORT = os.getenv("VLLM_PORT", "8888")
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "3600"))

# Served-model name baked into vllm/vllm-openai:unlimited-ocr.
MODEL_NAME = os.getenv("OCR_MODEL_NAME", "Unlimited-OCR")

# Upstream HF Space classifies the two configs as:
#   gundam = base_size=1024, image_size=640,  crop_mode=True  → fast / lower-res
#   base   = base_size=1024, image_size=1024, crop_mode=False → accurate / higher-res
# Default to `base` for accuracy on multi-page documents; override to
# `gundam` via env if you need lower latency / less VRAM per request.
IMAGE_MODE = os.getenv("OCR_IMAGE_MODE", "base")

# Unlimited-OCR's canonical prompt (see baidu/Unlimited-OCR README).
PARSE_PROMPT = "<image>document parsing."

# Normalized bbox canvas size used by the model.
BBOX_CANVAS = 1000

# Map Unlimited-OCR's lowercase tag names to the Title-cased categories
# the downstream renderer (layoutjson2md / json_to_docx) was built for.
# Underscore variants come from Unlimited-OCR's nested markers
# (image_caption, page_number, page_footer, page_header, ...).
# All keys are lowercase; lookup downcases the raw tag before matching.
_CATEGORY_MAP = {
    "title":           "Title",
    "text":            "Text",
    "aside_text":      "Text",          # margin/sidenote text — render like body
    "image":           "Picture",
    "picture":         "Picture",
    "figure":          "Picture",
    "chart":           "Picture",       # charts ship as raster images
    "table":           "Table",
    "list":            "List-item",
    "list-item":       "List-item",
    "list_item":       "List-item",
    "formula":         "Formula",
    "equation":        "Formula",       # Unlimited-OCR uses Equation for display math
    "caption":         "Caption",
    "image_caption":   "Caption",
    "table_caption":   "Caption",
    "figure_caption":  "Caption",
    "footnote":        "Footnote",
    "page_footnote":   "Footnote",
    "footer":          "Page-footer",
    "page-footer":     "Page-footer",
    "page_footer":     "Page-footer",
    "header":          "Page-header",
    "page-header":     "Page-header",
    "page_header":     "Page-header",
    "page_number":     "Page-footer",   # Unlimited-OCR emits page-num markers; render as page-footer
    "section":         "Section-header",
    "section-header":  "Section-header",
    "section_header":  "Section-header",
}

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=VLLM_TIMEOUT)
    return _client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=VLLM_TIMEOUT)
    return _async_client


# <|det|>category [x1, y1, x2, y2]<|/det|>
# Category names can contain letters, underscores, and hyphens
# (e.g. `image_caption`, `page_number`, `section-header`).
_DET_RE = re.compile(
    r"<\|det\|>\s*([A-Za-z_\-]+)\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*<\|/det\|>"
)


def _rescale(bbox: list[int], img_w: int, img_h: int) -> list[int]:
    """Map a normalized-canvas bbox (0..1000) to original-image pixels."""
    x1, y1, x2, y2 = bbox
    sx = img_w / BBOX_CANVAS
    sy = img_h / BBOX_CANVAS
    return [
        max(0, min(img_w, round(x1 * sx))),
        max(0, min(img_h, round(y1 * sy))),
        max(0, min(img_w, round(x2 * sx))),
        max(0, min(img_h, round(y2 * sy))),
    ]


# Match an inline LaTeX block: `\( ... \)` or `\[ ... \]`. Non-greedy so
# multiple math spans in one paragraph are captured individually.
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)


def _normalize_math(text: str) -> str:
    """Rewrite `\\( ... \\)` and `\\[ ... \\]` to `$ ... $` so markdown
    renderers (KaTeX, MathJax) display inline math. Leaves non-math text
    untouched. Strips one leading space inside the math so `$ x $` formats
    cleanly."""
    def _sub(m: re.Match) -> str:
        body = (m.group(1) or m.group(2) or "").strip()
        return f"${body}$"
    return _INLINE_MATH_RE.sub(_sub, text)


def _is_mostly_math(text: str) -> bool:
    """True if a Text entry is dominated by a single `\\( ... \\)` math span
    (>= 60% of the non-whitespace characters live inside math delimiters).
    Such an entry is really a displayed formula and is better rendered as
    a `Formula` entry so the matplotlib PNG path fires in the DOCX writer.
    """
    spans = list(_INLINE_MATH_RE.finditer(text))
    if not spans:
        return False
    math_chars = sum(len((m.group(1) or m.group(2) or "")) for m in spans)
    total_chars = len(re.sub(r"\s+", "", text))
    if total_chars == 0:
        return False
    return math_chars / total_chars >= 0.6


def _strip_math_delims(text: str) -> str:
    """For an entry being promoted to Formula, drop the outer `\\( \\)` /
    `\\[ \\]` math delimiters so the matplotlib mathtext renderer gets bare
    LaTeX (it cannot parse `\\(...\\)` / `\\[...\\]` wrappers).

    Everything *inside* the wrappers is preserved character-for-character —
    `\\widetilde`, `\\frac`, `\\mathrm`, sub/superscripts, braces, etc.

    For entries that mix math spans with prose, the prose between math
    spans is wrapped in `\\text{...}` so the matplotlib renderer routes it
    through its regular text path (which supports CJK + accents); see
    `_build_renderable_string` in reconstruction_service/formula.py.
    """
    spans = list(_INLINE_MATH_RE.finditer(text))
    if not spans:
        return text.strip()
    parts: list[str] = []
    pos = 0
    for m in spans:
        prose = text[pos:m.start()].strip()
        if prose:
            parts.append(f"\\text{{{prose}}}")
        body = ((m.group(1) or m.group(2)) or "").strip()
        if body:
            parts.append(body)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        parts.append(f"\\text{{{tail}}}")
    return " ".join(parts)


def _post_process_math(entries: list[dict]) -> list[dict]:
    """Normalize inline math + promote math-dominant Text entries to Formula.

    Tables keep their HTML structure but inline `\\(...\\)` inside cells is
    rewritten to `$...$` so a markdown-aware renderer can display the math.

    Entries the model already tagged as Formula (via `formula` or `equation`)
    often carry `\\[ ... \\]` display-math delimiters in their text; we strip
    those so the matplotlib renderer gets raw LaTeX.
    """
    for entry in entries:
        cat = entry.get("category")
        text = entry.get("text", "")
        if not text:
            continue
        if cat == "Formula":
            # Strip outer math delimiters if the model wrapped the LaTeX.
            stripped = _strip_math_delims(text)
            if stripped:
                entry["text"] = stripped
        elif cat == "Text" and _is_mostly_math(text):
            entry["category"] = "Formula"
            entry["text"] = _strip_math_delims(text)
        elif cat in ("Text", "Table", "Caption", "Title",
                     "Page-header", "Page-footer", "List-item",
                     "Footnote", "Section-header"):
            entry["text"] = _normalize_math(text)
    return entries


def _parse_response(content: str, img_w: int, img_h: int) -> list:
    """Convert Unlimited-OCR's `<|det|>` stream into a list of layout entries.

    Each entry: {"bbox": [x1,y1,x2,y2], "category": "Text"|..., "text": "..."}
    with bbox in original-image pixels.

    The model emits a flat stream of `<|det|>cat [bbox]<|/det|>{text}` blocks
    in reading order. Some blocks (e.g. `image`) contain nested `<|det|>`
    children like `image_caption` or `page_number`. We treat every marker as
    its own entry — for nested children that means the parent's "text" is
    truncated at the first nested marker, and the child entry follows
    immediately after with its own bbox + category + text.

    A post-pass normalizes inline LaTeX (`\\(...\\)` → `$...$`) so KaTeX-
    style markdown renders correctly, and promotes Text entries that are
    almost entirely math to `Formula` so the matplotlib formula renderer
    in json_to_docx fires.
    """
    matches = list(_DET_RE.finditer(content))
    if not matches:
        log.warning("[ocr] no <|det|> markers in response (len=%d, head=%r)",
                    len(content), content[:120])
        return []

    entries: list[dict] = []
    for i, m in enumerate(matches):
        raw_cat = m.group(1).lower()
        norm_bbox = [int(m.group(2)), int(m.group(3)),
                     int(m.group(4)), int(m.group(5))]
        bbox = _rescale(norm_bbox, img_w, img_h)
        # Text for this entry runs until the next <|det|> marker.
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        category = _CATEGORY_MAP.get(raw_cat, raw_cat.title())
        entry: dict = {"bbox": bbox, "category": category}
        if category != "Picture":
            entry["text"] = text
        entries.append(entry)

    return _post_process_math(entries)


def _build_messages(image_path: str) -> list:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": PARSE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ],
    }]


def _extra_body() -> dict:
    return {
        "images_config": {"image_mode": IMAGE_MODE},
        "skip_special_tokens": False,
    }


def _image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.width, img.height


def process_image(image_path: str, vllm_url: str | None = None) -> list:
    """Sync layout extraction. vllm_url ignored (kept for backwards compat)."""
    img_w, img_h = _image_size(image_path)
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=_build_messages(image_path),
        temperature=0.0,
        max_tokens=int(os.getenv("OCR_MAX_TOKENS", "30000")),
        extra_body=_extra_body(),
    )
    return _parse_response(response.choices[0].message.content, img_w, img_h)


async def process_image_async(image_path: str) -> list:
    """Async layout extraction. Lets the caller run many pages concurrently."""
    img_w, img_h = _image_size(image_path)
    response = await get_async_client().chat.completions.create(
        model=MODEL_NAME,
        messages=_build_messages(image_path),
        temperature=0.0,
        max_tokens=int(os.getenv("OCR_MAX_TOKENS", "30000")),
        extra_body=_extra_body(),
    )
    return _parse_response(response.choices[0].message.content, img_w, img_h)
