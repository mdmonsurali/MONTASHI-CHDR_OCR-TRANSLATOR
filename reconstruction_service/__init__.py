"""Reconstruction service: two parallel packages, one per consumer.

- `ocr_reconstruction` — imported by ocr_service.
- `translation_reconstruction` — imported by translator_service.

Each ships an identical copy of the renderer modules (geometry, text_fit,
ooxml, table, formula, picture, etc.) so the two services can evolve their
reconstruction independently.
"""
