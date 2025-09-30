# search on related pdfs for user query
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from googleapiclient.discovery import Resource

from .drive_download import download_pdf_bytes

# ---------------------------
# Lightweight NLP (English)
# ---------------------------
_STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","by","about","from",
    "is","are","was","were","be","being","been","at","as","it","its","that","this",
    "these","those","into","over","under","than","then","but","if","so","not","no",
    "can","could","should","would","may","might","will","shall","do","does","did",
    "how","what","why","when","where","which","who","whom","you","your","yours"
}

def _normalize_query(user_query: str, max_terms: int = 8) -> str:
    # keep quoted phrases; lightly clean the rest
    phrases = re.findall(r'"([^"]+)"', user_query)
    q = re.sub(r'"[^"]+"', " ", user_query)
    q = q.replace("-", " ")
    toks = re.findall(r"[A-Za-z0-9_]+", q.lower())
    core = []
    seen = set()
    for t in toks:
        if len(t) < 3 or t in _STOP: 
            continue
        if t not in seen:
            core.append(t)
            seen.add(t)
        if len(core) >= max_terms:
            break
    return " ".join(phrases + core) if (phrases or core) else user_query.strip()

# ---------------------------
# Local embedder (public model)
# ---------------------------
_EMBEDDER = None
def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Public English model; no HF token needed.
    If you set HF_TOKEN/HUGGINGFACE_HUB_TOKEN for private models, we’ll use it.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        kwargs = {}
        tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if tok:
            kwargs["use_auth_token"] = tok
        _EMBEDDER = SentenceTransformer(model_name, **kwargs)
    return _EMBEDDER

# ---------------------------
# Drive listing (metadata only)
# ---------------------------
def list_all_pdfs(
    service: Resource,
    page_size: int = 200,
    max_total: int = 1000,
) -> List[Dict]:
    """
    List PDFs with minimal metadata (no bytes). Paginates.
    Returns: [{id,name,url,modifiedTime,size}]
    """
    q = "mimeType='application/pdf' and trashed=false"
    fields = "nextPageToken, files(id,name,webViewLink,modifiedTime,size,mimeType)"
    results: List[Dict] = []
    page_token = None

    while True:
        resp = (service.files()
                .list(q=q,
                      pageSize=page_size,
                      pageToken=page_token,
                      fields=fields,
                      includeItemsFromAllDrives=True,
                      supportsAllDrives=True,
                      orderBy="modifiedTime desc")
                .execute())
        items = resp.get("files", [])
        for f in items:
            results.append({
                "id": f["id"],
                "name": f.get("name", ""),
                "url": f.get("webViewLink"),
                "modifiedTime": f.get("modifiedTime"),
                "size": int(f["size"]) if "size" in f else None,
            })
        page_token = resp.get("nextPageToken")
        if not page_token or len(results) >= max_total:
            break

    # de-dup by id (shouldn’t be necessary but safe)
    uniq = {r["id"]: r for r in results}
    return list(uniq.values())

# ---------------------------
# Title-level semantic index
# ---------------------------
def _embed_titles(files: List[Dict]) -> np.ndarray:
    emb = _get_embedder()
    titles = [(f.get("name") or "").strip() for f in files]
    if not titles:
        return np.zeros((0, 384), dtype="float32")
    vecs = emb.encode(titles, normalize_embeddings=True)
    return np.asarray(vecs)

def rank_files_by_title(
    files: List[Dict],
    user_query: str,
    top_n: int = 25,
) -> List[Dict]:
    """
    Rank PDFs by semantic similarity between filename and user query.
    Returns same dicts with an added 'title_score'.
    """
    if not files:
        return []
    q = _normalize_query(user_query)
    emb = _get_embedder()
    title_mat = _embed_titles(files)
    qv = emb.encode([q], normalize_embeddings=True)[0]
    scores = title_mat @ qv  # cosine (vectors normalized)
    idxs = np.argsort(-scores)[:top_n]
    out: List[Dict] = []
    for i in idxs:
        f = dict(files[i])
        f["title_score"] = float(scores[i])
        out.append(f)
    return out

# ---------------------------
# Download top-M files (bytes only)
# ---------------------------
def download_top_files(
    service: Resource,
    ranked_files: List[Dict],
    max_files: int = 5,
) -> List[Dict]:
    """
    Download raw bytes of the top ranked PDFs. No parsing here.
    Returns: [{file_id,file_name,url,modifiedTime,size,pdf_bytes}]
    """
    picked = ranked_files[:max_files]
    out: List[Dict] = []
    for f in picked:
        try:
            pdf = download_pdf_bytes(service, f["id"])
            out.append({
                "file_id": f["id"],
                "file_name": f.get("name", ""),
                "url": f.get("url"),
                "modifiedTime": f.get("modifiedTime"),
                "size": f.get("size"),
                "title_score": f.get("title_score", 0.0),
                "pdf_bytes": pdf,  # hand off to your Llama parser later
            })
        except Exception:
            # Skip failures but continue
            continue
    return out

# ---------------------------
# One-call entry point for wrapper
# ---------------------------
def search_by_title_and_download(
    service: Resource,
    user_query: str,
    max_candidates: int = 300,
    top_n_rank: int = 30,
    max_files_download: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    1) List all PDFs (metadata only).
    2) Rank by filename similarity to user_query.
    3) Download bytes of top files (no parsing).
    Returns:
      sidebar_items: ranked metadata [{id,name,url,modifiedTime,size,title_score}]
      downloaded:     [{file_id,file_name,url,modifiedTime,size,title_score,pdf_bytes}]
    """
    all_pdfs = list_all_pdfs(service, page_size=200, max_total=max_candidates)
    if not all_pdfs:
        return [], []

    ranked = rank_files_by_title(all_pdfs, user_query, top_n=top_n_rank)

    # Sidebar-friendly items (no bytes)
    sidebar_items = [{
        "id": f["id"],
        "name": f["name"],
        "url": f["url"],
        "modifiedTime": f["modifiedTime"],
        "size": f["size"],
        "title_score": f["title_score"],
    } for f in ranked]

    downloaded = download_top_files(service, ranked, max_files=max_files_download)
    return sidebar_items, downloaded


































# import concurrent.futures as cf
# import os
# import re
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# # You can keep Resource for hints, but it's not a formal type. Using Any is also fine.
# from googleapiclient.discovery import Resource

# from .drive_download import download_pdf_bytes, extract_pdf_text

# # --- Optional WordNet expansion (graceful fallback) ---
# _HAS_WORDNET = False
# try:
#     import nltk  # type: ignore
#     from nltk.corpus import wordnet as wn  # type: ignore
#     try:
#         wn.synsets("test")
#         _HAS_WORDNET = True
#     except Exception:
#         _HAS_WORDNET = False
# except Exception:
#     _HAS_WORDNET = False

# # --- Lazy singleton embedder for semantic scoring ---
# _EMBEDDER = None  # ✅ add this

# def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#     """
#     Uses a public Sentence-Transformers model; no HF token required.
#     If you do need a token for a private model, set HF_TOKEN/HUGGINGFACE_HUB_TOKEN.
#     """
#     global _EMBEDDER
#     if _EMBEDDER is None:
#         from sentence_transformers import SentenceTransformer  # lazy import
#         kwargs = {}
#         hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
#         if hf_token:
#             kwargs["use_auth_token"] = hf_token
#         _EMBEDDER = SentenceTransformer(model_name, **kwargs)
#     return _EMBEDDER

# # --- Minimal stoplist (semantic rerank does the heavy lifting) ---
# _STOP = {
#     "the","a","an","and","or","of","to","in","on","for","with","by","about","from",
#     "is","are","was","were","be","being","been","at","as","it","its","that","this",
#     "these","those","into","over","under","than","then","but","if","so","not","no",
#     "can","could","should","would","may","might","will","shall","do","does","did",
#     "how","what","why","when","where","which","who","whom","you","your","yours",
# }

# # ---------------------------
# # Query parsing & expansion
# # ---------------------------
# def analyze_user_query(
#     user_query: str,
#     max_terms: int = 6,
#     max_synonyms_per_term: int = 2,
# ) -> Dict[str, List[str] | str]:
#     """
#     Produce:
#       - phrases: quoted spans
#       - core_terms: main keywords (>=3 chars, not stopwords)
#       - expanded_terms: conservative WordNet synonyms (if available)
#       - normalized_query: cleaned string used for embeddings
#     """
#     phrases = re.findall(r'"([^"]+)"', user_query)
#     q_wo_quotes = re.sub(r'"[^"]+"', " ", user_query)
#     # normalize hyphens so "fine-tuning" splits into two tokens
#     q_wo_quotes = q_wo_quotes.replace("-", " ")
#     tokens = re.findall(r"[A-Za-z0-9_]+", q_wo_quotes.lower())

#     core_terms: List[str] = []
#     seen = set()
#     for t in tokens:
#         if len(t) < 3 or t in _STOP:
#             continue
#         if t not in seen:
#             core_terms.append(t)
#             seen.add(t)
#         if len(core_terms) >= max_terms:
#             break

#     expanded_terms: List[str] = []
#     if _HAS_WORDNET and core_terms:
#         for term in core_terms[:3]:
#             syns = set()
#             for s in wn.synsets(term):
#                 for lemma in s.lemmas():
#                     name = lemma.name().replace("_", " ").lower()
#                     if name != term and len(name) >= 3 and name not in _STOP:
#                         syns.add(name)
#                     if len(syns) >= max_synonyms_per_term:
#                         break
#                 if len(syns) >= max_synonyms_per_term:
#                     break
#             expanded_terms.extend(sorted(syns))

#     # ✅ Always return, even if WordNet is unavailable
#     normalized_query = " ".join(phrases + core_terms) if (phrases or core_terms) else user_query.strip()
#     return {
#         "phrases": phrases,
#         "core_terms": core_terms,
#         "expanded_terms": expanded_terms,
#         "normalized_query": normalized_query,
#     }

# # ---------------------------
# # Drive search helpers
# # ---------------------------
# def _escape_q_val(s: str) -> str:
#     return s.replace("'", "\\'")

# def _name_clause(terms: List[str], joiner: str) -> Optional[str]:
#     if not terms:
#         return None
#     return "(" + f" {joiner} ".join([f"name contains '{_escape_q_val(t)}'" for t in terms]) + ")"

# def _fulltext_clause(terms: List[str], joiner: str) -> Optional[str]:
#     if not terms:
#         return None
#     return "(" + f" {joiner} ".join([f"fullText contains '{_escape_q_val(t)}'" for t in terms]) + ")"

# def _combine_clauses(*clauses: Optional[str]) -> str:
#     present = [c for c in clauses if c]
#     if not present:
#         return ""
#     return "(" + " and ".join(present) + ")"

# def _build_drive_q(core_terms: List[str], expanded_terms: List[str], phrases: List[str], precise: bool) -> str:
#     base = "mimeType='application/pdf' and trashed=false"
#     joiner_main = "and" if precise else "or"
#     main_terms = core_terms if (precise or not expanded_terms) else (core_terms + expanded_terms)

#     name_part = _name_clause(main_terms, joiner_main)
#     fulltext_part = _fulltext_clause(main_terms, joiner_main)

#     phrase_clauses = []
#     for p in phrases:
#         phrase_clauses.append(_combine_clauses(
#             f"name contains '{_escape_q_val(p)}'",
#             f"fullText contains '{_escape_q_val(p)}'"
#         ))

#     filter_part = _combine_clauses(name_part, fulltext_part, *phrase_clauses)
#     return f"{base} and {filter_part}" if filter_part else base

# # ---------------------------
# # Public: search PDFs (no download)
# # ---------------------------
# def search_pdfs(
#     service: Resource,
#     user_query: str,
#     page_size: int = 100,
#     max_candidates: int = 100,
# ) -> List[Dict]:
#     """
#     Two-pass search:
#       - Pass A (precise): AND on core terms (name + fullText)
#       - Pass B (recall):  OR on (core + expanded) terms
#     Returns: {id, name, url, modifiedTime, size}
#     """
#     plan = analyze_user_query(user_query)
#     results: List[Dict] = []

#     for precise in (True, False):
#         q = _build_drive_q(plan["core_terms"], plan["expanded_terms"], plan["phrases"], precise=precise)
#         resp = (
#             service.files()
#             .list(
#                 q=q,
#                 pageSize=page_size,
#                 fields="files(id,name,webViewLink,modifiedTime,size,mimeType),nextPageToken",
#                 includeItemsFromAllDrives=True,    # ✅ correct casing
#                 supportsAllDrives=True,
#                 orderBy="modifiedTime desc",        # ✅ correct spelling
#             )
#             .execute()
#         )
#         items = resp.get("files", [])
#         for f in items:
#             results.append({
#                 "id": f["id"],
#                 "name": f.get("name", ""),
#                 "url": f.get("webViewLink"),
#                 "modifiedTime": f.get("modifiedTime"),
#                 "size": int(f["size"]) if "size" in f else None,  # ✅ correct key
#             })

#         # ✅ de-dup and cap AFTER the loop
#         uniq = {r["id"]: r for r in results}
#         results = list(uniq.values())
#         if len(results) >= max_candidates:
#             results = results[:max_candidates]
#             break

#     return results

# # ---------------------------
# # Peek & semantic re-rank
# # ---------------------------
# def _keyword_hits(text: str, tokens: List[str]) -> int:
#     if not text or not tokens:
#         return 0
#     t = text.lower()
#     # ✅ correct counting
#     return sum(t.count(tok.lower()) for tok in tokens)

# def _semantic_score(query: str, doc_text: str) -> float:
#     if not doc_text.strip():
#         return 0.0
#     emb = _get_embedder()
#     qv = emb.encode([query], normalize_embeddings=True)
#     dv = emb.encode([doc_text], normalize_embeddings=True)
#     # cosine (vectors normalized)
#     return float(np.dot(qv[0], dv[0]))

# def _peek_one(
#     service: Resource,
#     file_dict: Dict,
#     pages_to_peek: int,
#     normalized_query: str,
#     core_terms: List[str],
#     semantic_weight: float,
#     lexical_weight: float,
# ) -> Dict:
#     try:
#         pdf_bytes = download_pdf_bytes(service, file_dict["id"])

#         # First peek (fast)
#         peek_text = extract_pdf_text(pdf_bytes, max_pages=pages_to_peek)

#         # If first pages are empty/cover, try a deeper single retry
#         if len(peek_text) < 50 and pages_to_peek < 12:
#             peek_text = extract_pdf_text(pdf_bytes, max_pages=12)

#         snippet = (peek_text[:400] + "…") if len(peek_text) > 400 else peek_text
#         peek_chars = len(peek_text)

#         # Semantic (content) score
#         sem = _semantic_score(normalized_query, peek_text) if peek_text else 0.0

#         # Lexical bump if any core term appears in CONTENT
#         lex_content = 1.0 if _keyword_hits(peek_text, core_terms) > 0 else 0.0

#         # Title bonus (helps scanned PDFs with no extractable text)
#         title_hits = _keyword_hits(file_dict.get("name", ""), core_terms)
#         title_bonus = 1.0 if title_hits > 0 else 0.0

#         score = (semantic_weight * sem) + (lexical_weight * max(lex_content, title_bonus))

#         return {
#             **file_dict,
#             "semantic_score": round(score, 6),
#             "snippet": snippet,
#             "peek_chars": peek_chars,
#             "title_hits": title_hits,
#         }
#     except Exception:
#         return {**file_dict, "semantic_score": 0.0, "snippet": "", "peek_chars": 0, "title_hits": 0}



# def peek_and_rank(
#     service: Resource,
#     candidates: List[Dict],
#     plan: Dict[str, List[str] | str],
#     pages_to_peek: int = 5,
#     top_n_peek: int = 25,
#     semantic_weight: float = 0.8,
#     lexical_weight: float = 0.2,
#     max_workers: int = 3,
# ) -> List[Dict]:
#     if not candidates:
#         return []
#     subset = candidates[:top_n_peek]
#     results: List[Dict] = []
#     with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
#         futures = [
#             ex.submit(
#                 _peek_one,
#                 service,
#                 f,
#                 pages_to_peek,
#                 plan["normalized_query"],  # type: ignore[index]
#                 plan["core_terms"],        # type: ignore[index]
#                 semantic_weight,
#                 lexical_weight,
#             )
#             for f in subset
#         ]
#         for fut in cf.as_completed(futures):
#             results.append(fut.result())
#     results.sort(key=lambda x: x.get("semantic_score", 0.0), reverse=True)
#     return results

# # ---------------------------
# # Load top-K full documents (for the RAG pipeline)
# # ---------------------------
# def load_documents(
#     service: Resource,
#     ranked_sidebar_items: List[Dict],
#     top_k_load: int = 5,
#     max_workers: int = 3,
# ) -> List[Dict]:
#     """
#     Download and extract FULL text for the top-K ranked files.
#     Returns: [{file_id, file_name, text}]
#     """
#     top = ranked_sidebar_items[:top_k_load]

#     def _load_one(f: Dict) -> Optional[Dict]:
#         try:
#             pdf_bytes = download_pdf_bytes(service, f["id"])
#             txt = extract_pdf_text(pdf_bytes, max_pages=None)
#             return {"file_id": f["id"], "file_name": f["name"], "text": txt}
#         except Exception:
#             return None

#     docs: List[Dict] = []
#     with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
#         futures = [ex.submit(_load_one, f) for f in top]
#         for fut in cf.as_completed(futures):
#             res = fut.result()
#             if res and res.get("text"):
#                 docs.append(res)
#     return docs

# # ---------------------------
# # One-call entrypoint for the wrapper agent
# # ---------------------------
# def search_pdfs_and_load(
#     service: Resource,
#     user_query: str,
#     page_size: int = 100,
#     max_candidates: int = 100,
#     pages_to_peek: int = 5,
#     top_n_peek: int = 25,
#     top_k_load: int = 5,
#     semantic_weight: float = 0.8,
#     lexical_weight: float = 0.2,
#     max_workers: int = 3,
# ) -> Tuple[List[Dict], List[Dict]]:
#     """
#     1) search_pdfs -> candidate PDFs (name + fullText)
#     2) peek_and_rank -> semantic screen of first few pages
#     3) load_documents -> full text for top-K winners
#     """
#     plan = analyze_user_query(user_query)
#     candidates = search_pdfs(service, user_query, page_size=page_size, max_candidates=max_candidates)
#     if not candidates:
#         return [], []

#     sidebar_items = peek_and_rank(
#         service,
#         candidates,
#         plan,
#         pages_to_peek=pages_to_peek,
#         top_n_peek=top_n_peek,
#         semantic_weight=semantic_weight,
#         lexical_weight=lexical_weight,
#         max_workers=max_workers,
#     )
#     # Keep items that have some semantic match OR a filename match (for scanned PDFs)
#     MIN_SEM = 0.12
#     sidebar_items = [
#         s for s in sidebar_items
#         if s.get("semantic_score", 0.0) >= MIN_SEM or s.get("title_hits", 0) > 0
#     ]

#     loaded_docs = load_documents(
#         service,
#         sidebar_items,
#         top_k_load=top_k_load,
#         max_workers=max_workers,
#     )

#     return sidebar_items, loaded_docs
