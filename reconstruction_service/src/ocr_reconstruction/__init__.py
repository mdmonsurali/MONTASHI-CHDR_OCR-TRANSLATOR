"""Reconstruction package: turn OCR layout JSON into a DOCX whose page
geometry and per-entry positions match the original document.

Public API:
    json_to_docx(layout_results, output_path)
    process_pictures(full_result)
"""
from .json_to_docx import json_to_docx
from .picture import process_pictures

__all__ = ["json_to_docx", "process_pictures"]
