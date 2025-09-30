# src/agent/agent.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from textwrap import shorten

from googleapiclient.discovery import Resource

from src.drive import search_by_title_and_download
from src.rag.index import parse_and_chunk, build_faiss_index, search_faiss
from src.rag.prompt import build_grounded_prompt
from src.llm.hf_client import generate_hf_llama

class DriveRAGAgent:
    """
    Wrapper agent that orchestrates:
      1) search titles in Drive → rank → download bytes
      2) parse locally → token-aware chunking
      3) build FAISS → retrieve top-k chunks
      4) build grounded prompt → call LLaMA via HF Inference API
    """

    def __init__(
        self,
        drive_service: Resource,
        max_candidates: int = 500,
        top_n_rank: int = 30,
        max_files_download: int = 3,
        faiss_top_k: int = 8,
    ) -> None:
        self.svc = drive_service
        self.max_candidates = max_candidates
        self.top_n_rank = top_n_rank
        self.max_files_download = max_files_download
        self.faiss_top_k = faiss_top_k

    def retrieve(self, query: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Returns: (sidebar_items, downloaded_files, retrieved_chunks)
        - sidebar_items: ranked by filename
        - downloaded_files: bytes for parsing
        - retrieved_chunks: top chunks via FAISS (each has text, file_name, file_id, chunk_index, score)
        """
        # 1) Drive search → download bytes
        sidebar, downloaded = search_by_title_and_download(
            self.svc,
            user_query=query,
            max_candidates=self.max_candidates,
            top_n_rank=self.top_n_rank,
            max_files_download=self.max_files_download,
        )

        if not downloaded:
            return sidebar, [], []

        # 2) Parse → chunk
        chunks = parse_and_chunk(downloaded)
        if not chunks:
            return sidebar, downloaded, []

        # 3) Build FAISS → retrieve top-k
        index, id2meta = build_faiss_index(chunks)
        top_chunks = search_faiss(query, index, id2meta, top_k=self.faiss_top_k)
        return sidebar, downloaded, top_chunks

    def answer(self, query: str) -> Tuple[str, List[str], List[Dict]]:
        """
        Returns: (answer_text, source_file_names, retrieved_chunks)
        - If no chunks, returns the mandated fallback.
        """
        _, _, top_chunks = self.retrieve(query)
        if not top_chunks:
            return "I don’t know based on the provided documents.", [], []

        prompt = build_grounded_prompt(query, top_chunks)
        try:
            answer = generate_hf_llama(prompt).strip()
        except Exception:
            answer = ""

        if not answer:
            answer = "I don’t know based on the provided documents."

        sources = sorted({c.get("file_name", "") for c in top_chunks if c.get("file_name")})
        return answer, sources, top_chunks
    
    def answer_from_chunks(self, query: str, top_chunks: List[Dict]) -> Tuple[str, List[str]]:
        if not top_chunks:
            return "I don’t know based on the provided documents.", []
        prompt = build_grounded_prompt(query, top_chunks)
        try:
            answer = generate_hf_llama(prompt).strip()
        except Exception:
            answer = ""
        if not answer:
            answer = "I don’t know based on the provided documents."
        sources = sorted({c.get("file_name", "") for c in top_chunks if c.get("file_name")})
        return answer, sources

