# src/retriever/collect.py
from __future__ import annotations
from typing import List, Dict, Tuple

import numpy as np
from transformers import AutoTokenizer

from src.config import TARGET_CONTEXT_CHUNKS, MIN_SIM_THRESHOLD, PARSE_BUDGET_FILES
from src.parsers.parser_chain import parse_pdf
from src.drive.drive_search import _get_embedder  # reuse the same embedder

# Load a tokenizer compatible with the sentence-transformers model (fast variant)
_TOKENIZER = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", use_fast=True
)

def _chunk_text_tokenaware(
    text: str,
    max_tokens: int = 500,
    stride: int = 50,
) -> List[Tuple[int, int, str]]:
    """
    Token-aware sliding window over `text`.
    Returns list of (char_start, char_end, chunk_text).
    """
    if not text:
        return []

    enc = _TOKENIZER(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_overflowing_tokens=True,
        max_length=max_tokens,
        stride=stride,
        truncation=True,
    )

    chunks: List[Tuple[int, int, str]] = []
    # Fast tokenizers expose per-overflow encodings with offsets
    if hasattr(enc, "encodings") and enc.encodings:  # ★
        for enc_i in enc.encodings:
            offsets = getattr(enc_i, "offsets", None)
            if not offsets:
                continue
            start_char = offsets[0][0]
            end_char = offsets[-1][1]
            if end_char <= start_char:
                continue
            chunk = text[start_char:end_char]
            chunks.append((start_char, end_char, chunk))
    else:
        # Fallback for older tokenizers that return offset_mapping in the dict
        offsets = enc.get("offset_mapping")
        if offsets:
            start_char = offsets[0][0]
            end_char = offsets[-1][1]
            if end_char > start_char:
                chunks = [(start_char, end_char, text[start_char:end_char])]

    return chunks

def _cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N, D], b: [D] (assumes normalized vectors)
    return a @ b

def collect_relevant_context(
    user_query: str,
    downloaded_files: List[Dict],
    parse_budget_files: int = PARSE_BUDGET_FILES,
    target_chunks: int = TARGET_CONTEXT_CHUNKS,
    min_sim_threshold: float = MIN_SIM_THRESHOLD,
    max_chunks_per_file: int = 300,   # ★ safety cap
) -> List[Dict]:
    """
    For each downloaded PDF (bytes), parse via LlamaParse-first,
    chunk token-aware (500 / 50), embed in batches, score vs query,
    and stop early once we have enough good chunks.

    Returns chunks:
      [{ "text", "file_name", "file_id", "chunk_index", "score" }]
    """
    if not downloaded_files:
        return []

    embedder = _get_embedder()
    qv = embedder.encode([user_query], normalize_embeddings=True)[0]  # [D]

    collected: List[Tuple[float, Dict]] = []
    files_parsed = 0

    for f in downloaded_files:
        if files_parsed >= parse_budget_files:
            break

        pdf_bytes = f.get("pdf_bytes")
        if not pdf_bytes:
            continue

        text, method = parse_pdf(pdf_bytes)
        # DEBUG: show parse outcome ★
        print(f"[parse] {f.get('file_name','?')} -> method={method}, text_chars={len(text) if text else 0}")

        if not text:
            continue

        # Optional: cap extremely large docs to keep things snappy
        if len(text) > 300_000:
            text = text[:300_000]

        spans = _chunk_text_tokenaware(text, max_tokens=500, stride=50)
        # DEBUG: show chunk count ★
        print(f"[chunk] {f.get('file_name','?')} -> chunks={len(spans)}")

        if not spans:
            files_parsed += 1
            continue

        # Normalize whitespace for nicer previews (no effect on embeddings) ★
        def _norm(s: str) -> str:
            return " ".join(s.split())

        # Safety cap per file ★
        if len(spans) > max_chunks_per_file:
            spans = spans[:max_chunks_per_file]

        # Embed chunks in reasonable batches
        BATCH = 64
        chunk_texts: List[str] = [_norm(c) for _, _, c in spans]
        sims_all: List[float] = []
        for i in range(0, len(chunk_texts), BATCH):
            batch = chunk_texts[i:i + BATCH]
            vecs = embedder.encode(batch, normalize_embeddings=True)  # [B, D]
            sims = _cosine_batch(np.asarray(vecs), qv)                # [B]
            sims_all.extend([float(s) for s in sims])

        # DEBUG: show best similarity ★
        if sims_all:
            print(f"[score] {f.get('file_name','?')} -> max_sim={max(sims_all):.3f}, threshold={min_sim_threshold:.2f}")

        # Keep only relevant chunks from this file
        for idx, (sim, chunk_text) in enumerate(zip(sims_all, chunk_texts)):
            if sim >= min_sim_threshold:
                collected.append((
                    sim,
                    {
                        "text": chunk_text,
                        "file_name": f.get("file_name", ""),
                        "file_id": f.get("file_id", ""),
                        "chunk_index": idx,
                        "score": sim,
                    },
                ))

        files_parsed += 1

        # Early stop: if we already have enough globally, break
        if len(collected) >= target_chunks:
            break

    # Sort globally by similarity and return the top-N
    collected.sort(key=lambda x: x[0], reverse=True)
    top = [item for _, item in collected[:target_chunks]]
    return top
