"""QA: report any overlapping floating boxes in a reconstructed .docx.
Usage: python check_overlaps.py output.docx
General (not tied to any document): unzips the docx, reads every anchored box's
position + size, and reports 2D overlaps per page. Exit code 1 if any overlap.
"""
import sys, zipfile
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
TOL = 9144  # ~0.01 inch in EMU

def boxes_by_page(xml):
    root = etree.fromstring(xml)
    pages, cur = [], []
    for el in root.iter():
        if el.tag == WP + "anchor":
            ph, pv, ext = (el.find(WP + "positionH"), el.find(WP + "positionV"),
                           el.find(WP + "extent"))
            if None in (ph, pv, ext):
                continue
            xo, yo = ph.find(WP + "posOffset"), pv.find(WP + "posOffset")
            if xo is None or yo is None:
                continue
            cur.append((int(xo.text), int(yo.text),
                        int(ext.get("cx")), int(ext.get("cy"))))
        elif el.tag == W + "sectPr" and cur:
            pages.append(cur); cur = []
    if cur:
        pages.append(cur)
    return pages

def overlap(a, b):
    ax, ay, acx, acy = a; bx, by, bcx, bcy = b
    return (min(ax+acx, bx+bcx) - max(ax, bx) > TOL and
            min(ay+acy, by+bcy) - max(ay, by) > TOL)

def main(path):
    xml = zipfile.ZipFile(path).read("word/document.xml")
    pages = boxes_by_page(xml)
    total = 0
    for pi, bs in enumerate(pages, 1):
        n = sum(overlap(bs[i], bs[j])
                for i in range(len(bs)) for j in range(i+1, len(bs)))
        if n:
            print(f"  page {pi}: {n} overlapping pair(s)")
            total += n
    print(f"{len(pages)} pages, {total} overlapping pair(s)")
    return 1 if total else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
