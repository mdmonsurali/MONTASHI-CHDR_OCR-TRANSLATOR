"""Picture handling: dedup, cropping from source pages, and standalone
floating-picture rendering. In-cell picture placement lives in `table.py`
since it has to interact with the cell grid."""
from __future__ import annotations

import uuid
from io import BytesIO
from typing import Dict


# Chandra layout labels that carry a raster crop.
PICTURE_LABELS = {"Image", "Figure", "Picture"}


def _is_picture(entry) -> bool:
    return entry.get("category") in PICTURE_LABELS


def deduplicate_pictures(layout_result):
    """Remove duplicate Picture entries with identical bboxes."""
    seen_bboxes = set()
    deduped = []
    for entry in layout_result:
        if _is_picture(entry):
            bbox_tuple = tuple(entry.get("bbox") or [])
            if bbox_tuple not in seen_bboxes:
                seen_bboxes.add(bbox_tuple)
                deduped.append(entry)
        else:
            deduped.append(entry)
    return deduped


def process_pictures(full_result):
    """Crop Picture regions from each page's original image and attach the
    crop + a stable id/filename to the entry. MinIO upload happens upstream.
    """
    updated_results = []

    for page_idx, page_result in enumerate(full_result, start=1):
        original_img = page_result["original_image"]
        layout_result = page_result["layout_result"]
        markdown_content = page_result.get("markdown_content", "")

        layout_result = deduplicate_pictures(layout_result)

        new_layout_result = []
        img_replacements = []
        picture_count = 0

        for entry in layout_result:
            if _is_picture(entry):
                picture_count += 1
                bbox = entry["bbox"]
                cropped = original_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                short = uuid.uuid4().hex[:6]
                img_id = f"page{page_idx}_pic{picture_count}_{short}"
                image_filename = f"{img_id}.png"

                # Preserve the original Chandra label (Image/Figure/Diagram) so
                # the downstream dispatch and picture-asset upload still match.
                new_entry = {
                    **entry,
                    "image_obj": cropped,
                    "id": img_id,
                    "image_filename": image_filename,
                }
                new_layout_result.append(new_entry)

                img_replacements.append(
                    f"![Picture {img_id}](images/{image_filename})"
                )
            else:
                new_layout_result.append(entry)

        for replacement in img_replacements:
            markdown_content = markdown_content.replace(
                "![Image](Image detected)", replacement, 1
            )

        updated_page_result = {
            **page_result,
            "layout_result": new_layout_result,
            "markdown_content": markdown_content,
        }
        updated_results.append(updated_page_result)

    return updated_results


def render_standalone_picture(ctx, entry: Dict) -> None:
    """Emit a Picture entry as a floating page-anchored shape. Used for any
    picture that ISN'T contained in a table (table-contained ones go inline
    inside the matching cell in table.py)."""
    bbox = entry.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    img_obj = entry.get("image_obj")
    if img_obj is None:
        return
    from .geometry import bbox_px_to_emu
    from .ooxml import (
        build_anchored_picture_xml, add_image_relationship,
    )
    x, y, w, h = bbox_px_to_emu(
        bbox, ctx.zoom, ctx.page_w_pt, ctx.page_h_pt,
    )
    buf = BytesIO()
    img_obj.save(buf, format="PNG")
    buf.seek(0)
    rel_id = add_image_relationship(ctx.doc, buf, "png")
    ctx.xml_chunks.append(
        build_anchored_picture_xml(
            x, y, w, h, rel_id, ctx._next_id(),
            pic_name=entry.get("id") or "Picture",
        )
    )
