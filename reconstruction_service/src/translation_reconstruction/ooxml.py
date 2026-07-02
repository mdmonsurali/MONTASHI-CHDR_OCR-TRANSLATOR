"""Low-level OOXML string builders shared by every renderer.

Holds the namespace constants and the small helpers that turn (text, style,
geometry) tuples into the XML fragments python-docx expects to find inside a
<w:r>: floating textboxes, floating pictures, inline pictures, run/paragraph
properties, XML escapes, image relationships.
"""
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional

from docx import Document


# OOXML namespace URIs we need beyond what python-docx already registers.
NS_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_WPS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def attrs_for_anchor() -> str:
    """Inline xmlns attrs used on the root element when we hand-roll XML."""
    return (
        f' xmlns:wp="{NS_WP}"'
        f' xmlns:a="{NS_A}"'
        f' xmlns:pic="{NS_PIC}"'
        f' xmlns:r="{NS_R}"'
        f' xmlns:w="{NS_W}"'
        f' xmlns:wps="{NS_WPS}"'
        f' xmlns:mc="{NS_MC}"'
    )


_XML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}


def xml_escape(text: str) -> str:
    if not text:
        return ""
    out = []
    for ch in text:
        if ch in _XML_ESCAPES:
            out.append(_XML_ESCAPES[ch])
        elif ord(ch) < 0x20 and ch not in ("\n", "\t"):
            continue
        else:
            out.append(ch)
    return "".join(out)


_WHITE_THRESHOLD = 240


def rgb_hex(color) -> str:
    if not color:
        return "000000"
    try:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
    except (TypeError, ValueError, IndexError):
        return "000000"
    # Source PDFs sometimes carry white / near-white text designed for a
    # dark background. The reconstructed DOCX renders on a white page, so
    # rewrite anything near-white to black to keep it readable.
    if r >= _WHITE_THRESHOLD and g >= _WHITE_THRESHOLD and b >= _WHITE_THRESHOLD:
        return "000000"
    return f"{r:02X}{g:02X}{b:02X}"


def half_points(size_pt) -> int:
    try:
        v = float(size_pt)
    except (TypeError, ValueError):
        v = 11.0
    if v <= 0:
        v = 11.0
    return int(round(v * 2))


def build_run_xml(text: str, style: Optional[Dict]) -> str:
    """A single <w:r> with rPr applied. Multiline text becomes multiple <w:t>
    runs separated by <w:br/>.
    """
    style = style or {}
    font = xml_escape(style.get("font") or "Calibri")
    size_hp = half_points(style.get("size") or 11)
    bold = bool(style.get("bold"))
    italic = bool(style.get("italic"))
    color_hex = rgb_hex(style.get("color"))

    rpr_parts = [
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>',
        f'<w:sz w:val="{size_hp}"/>',
        f'<w:szCs w:val="{size_hp}"/>',
        f'<w:color w:val="{color_hex}"/>',
    ]
    if bold:
        rpr_parts.append("<w:b/>")
        rpr_parts.append("<w:bCs/>")
    if italic:
        rpr_parts.append("<w:i/>")
        rpr_parts.append("<w:iCs/>")
    rpr = "<w:rPr>" + "".join(rpr_parts) + "</w:rPr>"

    # Preserve spaces and convert newlines to soft line breaks.
    text = text or ""
    body_parts: List[str] = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i > 0:
            body_parts.append("<w:br/>")
        body_parts.append(
            f'<w:t xml:space="preserve">{xml_escape(line)}</w:t>'
        )
    return f"<w:r>{rpr}{''.join(body_parts)}</w:r>"


def build_paragraph_xml(
    runs_xml: str,
    alignment: Optional[str] = None,
    line_pt: Optional[float] = None,
) -> str:
    """Build a <w:p>. Always zeroes before/after spacing so Word's default
    Normal spacing (1.15x line, 10pt after) can't inflate the block height
    and push content past a fixed-height text box. When `line_pt` is given,
    pins an EXACT line height (twentieths of a point) so N lines occupy
    exactly the box height — measurement equals render, no vertical clip.
    """
    spacing = '<w:spacing w:before="0" w:after="0"'
    if line_pt and line_pt > 0:
        spacing += f' w:line="{max(1, int(round(line_pt * 20)))}" w:lineRule="exact"'
    spacing += "/>"
    jc = f'<w:jc w:val="{alignment}"/>' if alignment else ""
    return f"<w:p><w:pPr>{spacing}{jc}</w:pPr>{runs_xml}</w:p>"


def build_anchored_textbox_xml(
    x_emu: int,
    y_emu: int,
    w_emu: int,
    h_emu: int,
    inner_body_xml: str,
    docpr_id: int,
    body_auto_fit: bool = False,
) -> str:
    """Floating text box anchored at (x,y) on the page, holding arbitrary
    `<w:p>...` content. Returns an `<mc:AlternateContent>` block to drop
    inside a `<w:r>`.

    When `body_auto_fit=True`, the shape uses `<a:spAutoFit/>` so Word grows
    the shape to fit content that overflows the bbox-derived `cy`.
    Otherwise `<a:noAutofit/>` is used so the exact-fit path gets the precise
    box height it measured.

    CRITICAL FIX: Forces allowOverlap="0" to stop boundary collision overlaps.
    """
    name = f"TxtBox{docpr_id}"
    autofit_xml = "<a:spAutoFit/>" if body_auto_fit else "<a:noAutofit/>"
    return (
        f'<mc:AlternateContent{attrs_for_anchor()}>'
        f'<mc:Choice Requires="wps">'
        f'<w:drawing>'
        f'<wp:anchor distT="0" distB="0" distL="0" distR="0"'
        f' simplePos="0" relativeHeight="{docpr_id}" behindDoc="0"'
        f' locked="0" layoutInCell="1" allowOverlap="0">'
        f'<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{x_emu}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{y_emu}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{w_emu}" cy="{h_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:wrapNone/>'
        f'<wp:docPr id="{docpr_id}" name="{name}"/>'
        f'<wp:cNvGraphicFramePr/>'
        f'<a:graphic>'
        f'<a:graphicData uri="{NS_WPS}">'
        f'<wps:wsp>'
        f'<wps:cNvSpPr txBox="1"/>'
        f'<wps:spPr><a:xfrm>'
        f'<a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/>'
        f'</a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:noFill/>'
        f'<a:ln><a:noFill/></a:ln>'
        f'</wps:spPr>'
        f'<wps:txbx><w:txbxContent>{inner_body_xml}</w:txbxContent></wps:txbx>'
        f'<wps:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"'
        f' anchor="t" anchorCtr="0">{autofit_xml}</wps:bodyPr>'
        f'</wps:wsp>'
        f'</a:graphicData></a:graphic>'
        f'</wp:anchor>'
        f'</w:drawing>'
        f'</mc:Choice>'
        f'<mc:Fallback><w:pict/></mc:Fallback>'
        f'</mc:AlternateContent>'
    )


def build_anchored_picture_xml(
    x_emu: int,
    y_emu: int,
    w_emu: int,
    h_emu: int,
    rel_id: str,
    docpr_id: int,
    pic_name: str,
) -> str:
    """Floating picture anchored at (x,y) on the page.
    
    CRITICAL FIX: Forces allowOverlap="0" to stop boundary collision overlaps.
    """
    return (
        f'<mc:AlternateContent{attrs_for_anchor()}>'
        f'<mc:Choice Requires="wps">'
        f'<w:drawing>'
        f'<wp:anchor distT="0" distB="0" distL="0" distR="0"'
        f' simplePos="0" relativeHeight="{docpr_id}" behindDoc="0"'
        f' locked="0" layoutInCell="1" allowOverlap="0">'
        f'<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{x_emu}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{y_emu}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{w_emu}" cy="{h_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:wrapNone/>'
        f'<wp:docPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic>'
        f'<a:graphicData uri="{NS_PIC}">'
        f'<pic:pic>'
        f'<pic:nvPicPr>'
        f'<pic:cNvPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<pic:cNvPicPr/>'
        f'</pic:nvPicPr>'
        f'<pic:blipFill>'
        f'<a:blip r:embed="{rel_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'</pic:spPr>'
        f'</pic:pic>'
        f'</a:graphicData></a:graphic>'
        f'</wp:anchor>'
        f'</w:drawing>'
        f'</mc:Choice>'
        f'<mc:Fallback><w:pict/></mc:Fallback>'
        f'</mc:AlternateContent>'
    )


def build_inline_picture_xml(
    w_emu: int,
    h_emu: int,
    rel_id: str,
    docpr_id: int,
    pic_name: str,
) -> str:
    """Inline picture (flows with paragraph text) — used inside table cells so
    the image sits in the cell instead of floating beside it. Geometry is the
    picture's intrinsic w/h; the cell sizes itself to host it.
    """
    return (
        f'<w:r xmlns:w="{NS_W}">'
        f'<w:drawing xmlns:wp="{NS_WP}" xmlns:a="{NS_A}" xmlns:pic="{NS_PIC}" xmlns:r="{NS_R}">'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{w_emu}" cy="{h_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic>'
        f'<a:graphicData uri="{NS_PIC}">'
        f'<pic:pic>'
        f'<pic:nvPicPr>'
        f'<pic:cNvPr id="{docpr_id}" name="{xml_escape(pic_name)}"/>'
        f'<pic:cNvPicPr/>'
        f'</pic:nvPicPr>'
        f'<pic:blipFill>'
        f'<a:blip r:embed="{rel_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'</pic:spPr>'
        f'</pic:pic>'
        f'</a:graphicData></a:graphic>'
        f'</wp:inline>'
        f'</w:drawing>'
        f'</w:r>'
    )


def parse_xml_fragment(xml_str: str):
    """Parse a fragment that already declares its namespaces inline."""
    from lxml import etree
    return etree.fromstring(xml_str)


def add_image_relationship(doc: Document, img_buffer: BytesIO, ext: str) -> str:
    """Add an image part to the document and return its relationship id."""
    img_buffer.seek(0)
    rel_id, _image = doc.part.get_or_add_image(img_buffer)
    return rel_id