# src/agent/simple_agent.py
# src/agent/simple_agent.py
from __future__ import annotations
from typing import List, Dict, Tuple

from googleapiclient.discovery import Resource

from src.drive import search_by_title_and_download
from src.retriever.collect import collect_relevant_context
from src.llm.hf_client import LLMClient
# from src.rag.prompt import build_grounded_prompt  # not needed since LLMClient builds prompt
from src.config import PARSE_BUDGET_FILES, TARGET_CONTEXT_CHUNKS, MIN_SIM_THRESHOLD


class SimpleDriveAgent:
    """
    Wrapper/orchestrator:
      1) search Drive (filename semantics)
      2) download raw PDF bytes
      3) parse + chunk + rank vs query
      4) ask LLM with the top chunks
    Shapes match src/main.py expectations.
    """

    def __init__(
        self,
        drive_service: Resource,
        *,
        max_files_download: int = 3,
        top_k_chunks: int = TARGET_CONTEXT_CHUNKS,
        parse_budget_files: int = PARSE_BUDGET_FILES,
        min_sim_threshold: float = MIN_SIM_THRESHOLD,
        llm: LLMClient | None = None,
    ) -> None:
        self.svc = drive_service
        self.max_files_download = max_files_download
        self.top_k_chunks = top_k_chunks
        self.parse_budget_files = parse_budget_files
        self.min_sim_threshold = min_sim_threshold
        self.llm = llm or LLMClient()

    def retrieve(self, user_query: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Returns:
          sidebar:    [{id,name,url,modifiedTime,size,title_score}]
          downloaded: [{file_id,file_name,url,modifiedTime,size,title_score,pdf_bytes}]
          top_chunks: [{text,file_name,file_id,chunk_index,score}]
        """
        sidebar, downloaded = search_by_title_and_download(
            self.svc,
            user_query,
            max_candidates=300,
            top_n_rank=30,
            max_files_download=self.max_files_download,
        )

        top_chunks = collect_relevant_context(
            user_query,
            downloaded_files=downloaded,
            parse_budget_files=self.parse_budget_files,
            target_chunks=self.top_k_chunks,
            min_sim_threshold=self.min_sim_threshold,
        )

        return sidebar, downloaded, top_chunks

    def answer_from_chunks(
        self,
        user_query: str,
        top_chunks: List[Dict],
    ) -> Tuple[str, List[str]]:
        """
        Ask the LLM using the top chunks.
        Returns:
          answer:  str
          sources: ["<filename — drive link>", ...]
        """
        contexts = [c.get("text", "") for c in top_chunks if c.get("text")]

        if not contexts:
            return (
                "I couldn't find relevant text in your Drive for this query. "
                "Try different keywords or add more files.",
                [],
            )

        answer = self.llm.answer(user_query, contexts)

        # Build Drive source links
        sources: List[str] = []
        seen = set()
        for c in top_chunks:
            fid = c.get("file_id")
            fname = c.get("file_name") or "file"
            if not fid or fid in seen:
                continue
            seen.add(fid)
            sources.append(f"{fname} — https://drive.google.com/file/d/{fid}/view")

        return answer, sources
