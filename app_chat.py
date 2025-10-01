# app_chat.py
import os, time
import streamlit as st
from urllib.parse import urlencode

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from src.agent.simple_agent import SimpleDriveAgent
from src.llm.hf_client import LLMClient

# --- allow credentials.json to come from a secret (Streamlit/Spaces/Render) ---
CREDS_JSON_ENV = os.getenv("GOOGLE_CLIENT_SECRETS_JSON")  # full JSON string
CREDS_PATH = os.getenv("GOOGLE_CLIENT_SECRETS", "credentials.json")
if CREDS_JSON_ENV and not os.path.exists(CREDS_PATH):
    with open(CREDS_PATH, "w", encoding="utf-8") as f:
        f.write(CREDS_JSON_ENV)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CLIENT_SECRETS", "credentials.json")
# IMPORTANT: set this to your deployed Streamlit URL in secrets, e.g. https://<you>-<app>.streamlit.app
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")

SESSION_KEY = "drive_creds"
TTL_SECONDS = 6 * 60 * 60

st.set_page_config(page_title="Drive Chat", page_icon="📂", layout="centered")
st.title("📂 Drive Chat (Streamlit)")
st.caption("Grounded answers from your Google Drive PDFs (read-only).")

# -----------------------
# Session helpers
# -----------------------
def _session_get():
    sess = st.session_state.get(SESSION_KEY)
    if not sess:
        return None
    if time.time() - sess.get("ts", 0) > TTL_SECONDS:
        del st.session_state[SESSION_KEY]
        return None
    sess["ts"] = int(time.time())
    return sess

def _session_save(creds_dict: dict):
    st.session_state[SESSION_KEY] = {"creds": creds_dict, "ts": int(time.time())}

def _require_creds() -> Credentials | None:
    sess = _session_get()
    if not sess:
        return None
    cdict = sess["creds"]
    creds = Credentials.from_authorized_user_info(cdict, SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request as GRequest
        creds.refresh(GRequest())
        _session_save({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        })
    return creds

# -----------------------
# OAuth callback handling (?code=...)
# -----------------------
params = st.query_params
code = params.get("code")
if code and not _session_get():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    # Compose the full redirect-back URL with the code and pass to fetch_token
    redirect_back = f"{REDIRECT_URI}?{urlencode({'code': code})}"
    flow.fetch_token(authorization_response=redirect_back)
    creds = flow.credentials
    _session_save({
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    })
    # Clean the URL to remove ?code=...
    st.query_params.clear()
    st.rerun()

creds = _require_creds()

with st.sidebar:
    st.header("Settings")
    max_files_download = st.number_input("Max files to download", 1, 10, 3)
    top_k_chunks = st.number_input("Top chunks for LLM", 1, 20, 8)
    show_debug = st.checkbox("Show debug tables", True)
    if st.button("Sign out", use_container_width=True):
        if SESSION_KEY in st.session_state:
            del st.session_state[SESSION_KEY]
        st.rerun()

if creds is None:
    st.info("Sign in with Google Drive to start.")
    if st.button("🔐 Sign in with Google Drive", type="primary"):
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI
        )
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        st.link_button("Open Google consent", auth_url, use_container_width=True)
    st.stop()

st.success("Signed in ✔️")

# --- Chat history ---
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in turn["sources"]:
                    st.write(f"- {s}")

prompt = st.chat_input("Ask (e.g., 'start chating with your pdf')")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Drive → parsing → retrieving → asking LLM…"):
            svc = build("drive", "v3", credentials=creds)
            agent = SimpleDriveAgent(
                svc,
                max_files_download=int(max_files_download),
                top_k_chunks=int(top_k_chunks),
                llm=LLMClient(),
            )
            sidebar, downloaded, top_chunks = agent.retrieve(prompt)
            answer, sources = agent.answer_from_chunks(prompt, top_chunks)

        st.markdown(answer or "_(no answer)_")
        if sources:
            with st.expander("Sources", expanded=True):
                for s in sources:
                    st.write(f"- {s}")

        if show_debug:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top filename matches")
                if sidebar:
                    st.dataframe([{
                        "score": round(x.get("title_score", 0), 3),
                        "file": x.get("name", ""),
                        "modified": x.get("modifiedTime", "-"),
                        "size": x.get("size", "-"),
                        "link": x.get("url", "-"),
                    } for x in sidebar[:10]], use_container_width=True)
                else:
                    st.write("_No PDFs found in Drive._")
            with col2:
                st.subheader("Retrieved chunks")
                if top_chunks:
                    st.dataframe([{
                        "sim": round(c.get("score", 0), 3),
                        "file": c.get("file_name", ""),
                        "chunk#": c.get("chunk_index", 0),
                        "preview": (c.get("text", "")[:160] + "…") if c.get("text") else "",
                    } for c in top_chunks], use_container_width=True)
                else:
                    st.write("_No relevant chunks passed threshold._")

        st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
