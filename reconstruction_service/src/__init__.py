"""Reconstruction service package containing two parallel reconstruction
pipelines:

- `ocr_reconstruction`: used by ocr_service to turn raw OCR layout JSON
  into a DOCX.
- `translation_reconstruction`: used by translator_service to turn
  translated layout JSON into a DOCX.

The two share the same starting code but are kept as independent packages
so each service can evolve its reconstruction logic without affecting the
other.
"""
