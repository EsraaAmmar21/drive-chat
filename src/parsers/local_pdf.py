from __future__ import annotations
import io
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image

# OCR is optional; we import lazily and guard its absence
try:
    import pytesseract  # type: ignore
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False


def _extract_text_pymupdf(doc: fitz.Document, max_pages: Optional[int]) -> Tuple[str, List[int]]:
    """
    Return (joined_text, empty_page_idxs) using PyMuPDF text extraction.
    """
    texts: List[str] = []
    empties: List[int] = []
    n_pages = len(doc)
    end = n_pages if max_pages is None else min(n_pages, max_pages)
    for i in range(end):
        page = doc.load_page(i)
        t = page.get_text("text")  # 'text' preserves line breaks reasonably
        if t and t.strip():
            texts.append(t)
        else:
            empties.append(i)
    return ("\n".join(texts), empties)


def _ocr_page(doc: fitz.Document, page_idx: int, dpi: int = 180) -> str:
    """
    Render a page to an image and OCR it via Tesseract. Returns extracted text or "".
    """
    if not _HAS_TESS:
        return ""
    page = doc.load_page(page_idx)
    # render page as RGB image
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    max_pages: Optional[int] = None,
    enable_ocr_fallback: bool = True,
    ocr_min_chars: int = 25,
) -> Tuple[str, str]:
    """
    Robust local PDF extraction.

    Steps:
      1) Use PyMuPDF to extract text for each page.
      2) If a page is text-empty and OCR is enabled, run OCR for that page only.
    Returns: (text, method_used) where method_used ∈ {"pymupdf", "pymupdf+ocr", "pymupdf_no_text"}
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        # First pass: pure text extraction
        base_text, empty_pages = _extract_text_pymupdf(doc, max_pages)

        if not enable_ocr_fallback or not empty_pages or not _HAS_TESS:
            method = "pymupdf" if base_text.strip() else "pymupdf_no_text"
            return base_text.strip(), method

        # OCR only on pages that had no text
        ocr_texts: List[str] = []
        for idx in empty_pages:
            t = _ocr_page(doc, idx)
            if len(t.strip()) >= ocr_min_chars:
                ocr_texts.append(t)

        if ocr_texts:
            method = "pymupdf+ocr"
            return ("\n".join([base_text] + ocr_texts)).strip(), method
        else:
            method = "pymupdf" if base_text.strip() else "pymupdf_no_text"
            return base_text.strip(), method
