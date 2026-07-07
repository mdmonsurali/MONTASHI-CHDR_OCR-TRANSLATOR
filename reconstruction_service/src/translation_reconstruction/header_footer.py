"""Page-header / Page-footer routing into the section's actual header/footer
container instead of as floating body shapes."""
from __future__ import annotations

from typing import Dict, List


def apply_text_to_header_footer(
    section,
    entries: List[Dict],
    kind: str,
    page_w_pt: float,
    zoom: float,
) -> None:
    """Write the page's Page-header / Page-footer entries into the section's
    actual header or footer container. Each entry becomes a run inside the
    first paragraph (or one paragraph per entry if there are several).

    Header/footer containers autofit (no fixed-height clip), but we still
    zero before/after spacing for tighter vertical fidelity."""
    from .geometry import alignment_for_bbox
    from .ooxml import build_run_xml, parse_xml_fragment
    from .text_fit import fit_multiline

    if not entries:
        return
    container = section.header if kind == "Page-header" else section.footer
    # Reuse the default first paragraph for the first entry; add fresh
    # paragraphs for any additional entries on the same page.
    base_para = container.paragraphs[0]
    base_para.clear()
    for i, entry in enumerate(entries):
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        para = base_para if i == 0 else container.add_paragraph()
        alignment = alignment_for_bbox(entry.get("bbox"), page_w_pt, zoom)
        para_ppr = (
            f'<w:pPr><w:spacing w:before="0" w:after="0"/>'
            f'<w:jc w:val="{alignment}"/></w:pPr>'
        )

        style = dict(entry.get("style") or {})
        bbox = entry.get("bbox")
        if bbox and len(bbox) == 4:
            box_w_pt = max(1.0, (bbox[2] - bbox[0]) / zoom)
            box_h_pt = max(1.0, (bbox[3] - bbox[1]) / zoom)
            declared_size_pt = float(style.get("size") or 11.0)
            fitted, _lines = fit_multiline(
                text, box_w_pt, box_h_pt, max_size_pt=declared_size_pt,
            )
            if fitted < declared_size_pt:
                style["size"] = fitted
        runs_xml = build_run_xml(text, style)
        full_para_xml = (
            f'<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f'{para_ppr}{runs_xml}</w:p>'
        )
        new_p = parse_xml_fragment(full_para_xml)
        # Swap the paragraph element in place so docx keeps its handle.
        para._p.getparent().replace(para._p, new_p)
        para._p = new_p
