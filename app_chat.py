# app_chat.py
import os, time, json
import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from src.agent.simple_agent import SimpleDriveAgent
from src.llm.hf_client import LLMClient  # keeps your current LLM client

st.set_page_config(page_title="Drive QA", page_icon="📂", layout="wide")

# ---- Config from secrets/env
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRETS_JSON = os.getenv("GOOGLE_CLIENT_SECRETS_JSON", "")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "")
assert CLIENT_SECRETS_JSON and REDIRECT_URI, "Missing GOOGLE_CLIENT_SECRETS_JSON or GOOGLE_REDIRECT_URI"

# ---- Session helpers
def get_creds() -> Credentials | None:
    c = st.session_state.get("creds_dict")
    return Credentials.from_authorized_user_info(c, SCOPES) if c else None

def set_creds(creds: Credentials):
    st.session_state["creds_dict"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

def drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds)

# ---- Build OAuth URL
def get_auth_url():
    flow = Flow.from_client_config(json.loads(CLIENT_SECRETS_JSON), scopes=SCOPES, redirect_uri=REDIRECT_URI)
    auth_url, _state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = _state
    return auth_url

# ---- Process callback ?code=...
def maybe_finish_oauth():
    params = st.query_params  # Streamlit >=1.31
    code = params.get("code", [None])[0] if isinstance(params.get("code"), list) else params.get("code")
    state = params.get("state", [None])[0] if isinstance(params.get("state"), list) else params.get("state")
    if not code:
        return
    # finalize
    flow = Flow.from_client_config(json.loads(CLIENT_SECRETS_JSON), scopes=SCOPES, redirect_uri=REDIRECT_URI)
    flow.fetch_token(authorization_response=st.experimental_get_query_params().get("code") and st.experimental_get_query_params())  # backward compat, ignored if None
    # The safer way:
    flow.fetch_token(code=code)
    set_creds(flow.credentials)
    # Clear code from URL
    st.query_params.clear()

# ---- UI
st.title("📂 Drive Chat (grounded)")

with st.sidebar:
    st.header("Auth")
    if st.button("Sign out", use_container_width=True):
        for k in ("creds_dict", "oauth_state"): st.session_state.pop(k, None)
        st.rerun()

    with st.expander("Settings", expanded=False):
        max_files_download = st.number_input("Max files to download", 1, 10, 3, key="mx")
        top_k_chunks = st.number_input("Top chunks to feed LLM", 1, 20, 8, key="tk")
        show_debug = st.checkbox("Show debug tables", value=True, key="dbg")

# Try to finish OAuth if we have ?code= in the URL
maybe_finish_oauth()

creds = get_creds()
if not creds:
    st.info("Sign in to your Google Drive to begin.")
    if st.button("🔐 Sign in with Google", type="primary"):
        st.stop()  # ensures the next rerun opens the URL immediately
    # Print link for the user to click
    st.markdown(f"[Continue → Google]({get_auth_url()})")
    st.stop()

# Build Drive service and agent
svc = drive_service(creds)
agent = SimpleDriveAgent(svc, max_files_download=st.session_state.get("mx", 3), top_k_chunks=st.session_state.get("tk", 8), llm=LLMClient())

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.chat_input("Ask about your Drive… e.g., 'start chating with your pdfs'")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]: st.write(f"- {s}")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching Drive → parsing → retrieving → asking LLM…"):
            t0 = time.time()
            sidebar, downloaded, top_chunks = agent.retrieve(prompt)
            answer, sources = agent.answer_from_chunks(prompt, top_chunks)
            dt = time.time() - t0
        st.markdown(answer or "_(no answer)_")
        if sources:
            with st.expander("Sources", expanded=True):
                for s in sources: st.write(f"- {s}")
        if show_debug:
            st.caption(f"⚙️ {dt:.1f}s | files scanned: {len(sidebar)} | chunks used: {len(top_chunks)}")
    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
