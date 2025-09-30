# src/main.py
from datetime import datetime
from textwrap import shorten

from src.auth.authentication import authenticate_drive
from src.agent.simple_agent import SimpleDriveAgent

def fmt_bytes(n):
    if n is None: return "-"
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def fmt_dt(iso):
    if not iso: return "-"
    try: return datetime.fromisoformat(iso.replace("Z","+00:00")).strftime("%Y-%m-%d")
    except: return iso

def print_section(title):
    print(f"\n{title}\n{'─'*len(title)}")

def print_table(rows, headers):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i,h in enumerate(headers)] if rows else [len(h) for h in headers]
    def line(cols): return "  ".join(str(c).ljust(w) for c,w in zip(cols, widths))
    print(line(headers)); print(line(["-"*w for w in widths]))
    for r in rows: print(line(r))

def main():
    svc = authenticate_drive()
    agent = SimpleDriveAgent(svc, max_files_download=3, top_k_chunks=6)

    q = input("🔎 Ask something (e.g., 'summary Esraa resume', 'list Esraa skills'): ").strip()
    if not q: print("No query."); return

    sidebar, downloaded, top_chunks = agent.retrieve(q)

    print_section("Top matches by filename")
    rows = [[
        f"{s.get('title_score',0):.3f}",
        shorten(s['name'], 50, placeholder="…"),
        fmt_dt(s.get("modifiedTime")),
        fmt_bytes(s.get("size") or 0),
        s.get("url") or "-"
    ] for s in sidebar[:10]]
    print_table(rows, headers=["score","file name","modified","size","link"])

    print_section("Retrieved chunks (simple)")
    rows = [[
        f"{c.get('score',0):.3f}",
        shorten(c["file_name"], 36, placeholder="…"),
        c["chunk_index"],
        shorten(c["text"].replace("\n"," "), 80, placeholder="…"),
    ] for c in top_chunks]
    print_table(rows, headers=["sim","file","chunk#","preview"])

    print_section("Model answer (grounded)")
    ans, sources = agent.answer_from_chunks(q, top_chunks)
    print(ans)

    print_section("Sources")
    for s in sources: print(f"- {s}")

if __name__ == "__main__":
    main()




























# from auth.authentication import authenticate_drive
# from drive import search_pdfs_and_load

# def main():
#     service = authenticate_drive()  # OAuth only; no listing side-effects

#     # Ask once for a query to test
#     user_q = input("🔎 Ask something (e.g., 'what is fine tuning in transformers?'): ").strip()
#     if not user_q:
#         print("No query entered. Exiting.")
#         return

#     # Drive search → peek → semantic re-rank → load top-K
#     sidebar_items, loaded_docs = search_pdfs_and_load(
#         service,
#         user_query=user_q,
#         page_size=100,      # Drive API page size
#         max_candidates=100, # hard cap of candidates across passes
#         pages_to_peek=4,    # peek first N pages per candidate
#         top_n_peek=20,      # how many candidates to semantically screen
#         top_k_load=5,       # fully download best K
#         semantic_weight=0.85,
#         lexical_weight=0.15,
#         max_workers=3,      # small pool to avoid rate limits
#     )

#     # Sidebar preview
#     if not sidebar_items:
#         print("No candidate PDFs found.")
#     else:
#         print("\n=== Sidebar candidates (top 10) ===")
#         for s in sidebar_items[:10]:
#             print(f"{s.get('semantic_score',0):.3f} | {s['name']} -> {s.get('url')}")

#     # Loaded docs preview

#     print(f"\nLoaded {len(loaded_docs)} PDFs for downstream RAG.")
#     for d in loaded_docs:
#         # build a short, single-line preview safely (no backslashes in f-string expr)
#         snippet = (d["text"][:200] + "…") if len(d["text"]) > 200 else d["text"]
#         preview = snippet.replace("\n", " ")[:120]
#         print(f"- {d['file_name']} | chars={len(d['text'])} | preview: {preview}")

# if __name__ == "__main__":
#     main()
