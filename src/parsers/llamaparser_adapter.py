# src/parsers/llamaparser_adapter.py
from __future__ import annotations
import os
import uuid
import tempfile
import time
from typing import List, Tuple

from src.config import LLAMA_CLOUD_API_KEY, USE_LLAMAPARSE

RETRY_BACKOFF_SEC = [1.0, 3.0]

def llamaparse_bytes(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Parse a PDF (bytes) via LlamaParse and return (text, method_used).
    Windows-safe: write to a temp *path*, close it, then parse.
    """
    if not USE_LLAMAPARSE:
        return "", "llamaparse_disabled"
    if not LLAMA_CLOUD_API_KEY:
        return "", "llamaparse_no_api_key"

    try:
        from llama_parse import LlamaParse  # correct class
    except Exception as e:
        return "", f"llamaparse_import_error:{type(e).__name__}"

    last_err = None
    for attempt, delay in enumerate([0.0] + RETRY_BACKOFF_SEC, start=1):
        if delay:
            time.sleep(delay)

        tmp_path = os.path.join(tempfile.gettempdir(), f"llp_{uuid.uuid4().hex}.pdf")
        try:
            # 1) Write and CLOSE the file (Windows needs this)
            with open(tmp_path, "wb") as f:
                f.write(pdf_bytes)

            # 2) Parse by file path
            parser = LlamaParse(
                api_key=LLAMA_CLOUD_API_KEY,
                result_type="markdown",   # often cleaner than plain text
                num_workers=1,
                verbose=False,
                language="en",
            )
            docs: List = parser.load_data(tmp_path)

            parts: List[str] = []
            for d in docs:
                text = getattr(d, "text", None)
                if text is None and hasattr(d, "get_content"):
                    text = d.get_content()  # type: ignore[attr-defined]
                if text:
                    parts.append(text)

            return ("\n".join(parts).strip(), "llamaparse")

        except Exception as e:
            last_err = e
        finally:
            # 3) Always try to clean up
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return "", f"llamaparse_error:{type(last_err).__name__ if last_err else 'Unknown'}"
