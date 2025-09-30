# src/rag/prompt.py
from __future__ import annotations
from typing import List, Dict

def _truncate(s: str, max_chars: int) -> str:
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars - 1] + "…")

def _wants_summary(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ("summary", "summarize", "overview", "profile"))

def build_grounded_prompt(
    user_query: str,
    chunks: List[Dict],
    max_chunk_chars: int = 900,
) -> str:
    lines = []
    lines.append("[System Instruction]:")
    lines.append("You are an assistant. Use ONLY the provided context to answer.")
    lines.append('If the answer is not in the context, reply exactly:')
    lines.append('"I don’t know based on the provided documents."')
    lines.append("")

    if _wants_summary(user_query):
        lines.append("Task: Write a concise resume summary from the context below.")
        lines.append("Guidelines: 3–6 bullet points or a short paragraph.")
        lines.append("Include role/title, core skills/tools, notable achievements, and domain focus.")
        lines.append("Do not invent facts; only use details present in the context.")
        lines.append("")
    else:
        lines.append("When the question asks for entities like skills, tools, dates, or definitions,")
        lines.append("extract them directly from the context and present them as a short, clear list or sentence.")
        lines.append("Do not invent facts. If only partial information exists, return only what is present.")
        lines.append("")

    lines.append("[User Question]:")
    lines.append(user_query.strip())
    lines.append("")
    lines.append("[Retrieved Context with File Names]:")

    for c in chunks[:6]:  # keep focused
        fn = c.get("file_name", "")
        text = _truncate((c.get("text") or "").replace("\n", " "), max_chunk_chars)
        lines.append(f"File: {fn} | Chunk: {text}")
    lines.append("")
    return "\n".join(lines)
