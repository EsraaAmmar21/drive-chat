# src/rag/index.py
from __future__ import annotations
from typing import Dict, List, Tuple
import hashlib

import numpy as np
import faiss

from transformers import AutoTokenizer
from src.parsers.parser_chain import parse_pdf
from src.drive.drive_search import _get_embedder  # singleton embedder

# Use the same tokenizer family you used elsewhere for token-aware chunking
_TOKENIZER = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", use_fast=True
)

def _chunk_text_tokenaware(text: str, max_tokens: int = 500, stride: int = 50) -> List[Tuple[int,int,str]]:
    """Return [(start_char, end_char, chunk_text), ...] using sliding token windows."""
    if not text: return []
    enc = _TOKENIZER(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_overflowing_tokens=True,
        max_length=max_tokens,
        stride=stride,
        truncation=True,
    )
    chunks: List[Tuple[int,int,str]] = []
    if hasattr(enc, "encodings") and enc.encodings:
        for enc_i in enc.encodings:
            offsets = getattr(enc_i, "offsets", None)
            if not offsets: continue
            s, e = offsets[0][0], offsets[-1][1]
            if e > s:
                chunks.append((s, e, text[s:e]))
    else:
        offsets = enc.get("offset_mapping")
        if offsets:
            s, e = offsets[0][0], offsets[-1][1]
            if e > s:
                chunks = [(s, e, text[s:e])]
    return chunks

def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def parse_and_chunk(
    downloaded_files: List[Dict],
    max_chars_per_doc: int = 300_000,
    max_tokens: int = 500,
    stride: int = 50,
) -> List[Dict]:
    """
    For each downloaded file (with pdf_bytes), parse locally and produce chunk dicts:
    {text, file_name, file_id, chunk_index}
    De-duplicates identical chunk texts across files.
    """
    all_chunks: List[Dict] = []
    seen_hash = set()
    for f in downloaded_files:
        pdf_bytes = f.get("pdf_bytes")
        if not pdf_bytes: continue

        text, method = parse_pdf(pdf_bytes)
        print(f"[parse] {f.get('file_name','?')} -> method={method}, text_chars={len(text) if text else 0}")
        if not text: continue

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc]

        spans = _chunk_text_tokenaware(text, max_tokens=max_tokens, stride=stride)
        print(f"[chunk] {f.get('file_name','?')} -> chunks={len(spans)}")
        if not spans: continue

        for idx, (_, _, chunk_text) in enumerate(spans):
            chunk_text_norm = " ".join(chunk_text.split())  # whitespace normalize
            h = _sha1_text(chunk_text_norm)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            all_chunks.append({
                "text": chunk_text_norm,
                "file_name": f.get("file_name",""),
                "file_id": f.get("file_id",""),
                "chunk_index": idx,
            })

    return all_chunks

def build_faiss_index(chunks: List[Dict]):
    """
    Build an IP (cosine) FAISS index with normalized embeddings.
    Returns: (index, id2meta)
    """
    if not chunks:
        return None, []

    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    vecs = embedder.encode(texts, normalize_embeddings=True)  # [N, D] normalized
    vecs = np.asarray(vecs, dtype="float32")

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product == cosine on normalized vectors
    index.add(vecs)

    return index, chunks  # id2meta is just the chunks list aligned by row

def search_faiss(
    query: str,
    index,
    id2meta: List[Dict],
    top_k: int = 8,
) -> List[Dict]:
    """
    Retrieve top_k chunks for the query. Returns chunk dicts augmented with 'score'.
    """
    if index is None or not id2meta:
        return []

    embedder = _get_embedder()
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")  # [1, D]
    scores, idxs = index.search(qv, top_k)  # [1, K]
    out: List[Dict] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(id2meta):
            continue
        meta = dict(id2meta[idx])  # copy
        meta["score"] = float(score)
        out.append(meta)
    return out
