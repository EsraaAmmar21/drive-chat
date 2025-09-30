from __future__ import annotations
from typing import Tuple
from src.parsers.local_pdf import extract_text_from_pdf_bytes

def parse_pdf(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (text, method_used).
    Local first: PyMuPDF + (optional) OCR fallback via Tesseract.
    """
    try:
        text, method = extract_text_from_pdf_bytes(
            pdf_bytes,
            max_pages=None,            # parse full PDFs
            enable_ocr_fallback=False,  # set False if you don't have Tesseract installed
            ocr_min_chars=25,
        )
        return text, method
    except Exception as e:
        return "", f"local_pdf_error:{type(e).__name__}"
