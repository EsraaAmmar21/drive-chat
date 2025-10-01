# app_chat.py
# app_chat.py
import os
import time
import json
import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from src.agent.simple_agent import SimpleDriveAgent
from src.llm.hf_client import LLMClient

st.set_page_config(page_title="Drive QA", page_icon="📂", layout="wide")

# ---------------------------------------------------------------------
# Config (must be provided via environment or Streamlit secrets)
#   GOOGLE_CLIENT_SECRETS_JSON : the full OAuth client JSON (as string)
#   GOOGLE_REDIRECT_URI        : your deployed app URL (exact, no trailing path)
# ---------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRETS_JSON = os.getenv("GOOGLE_CLIENT_SECRETS_JSON", "")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "")

if not CLIENT_SECRETS_JSON or not REDIRECT_URI:
    st.error(
        "Missing GOOGLE_CLIENT_SECRETS_JSON or GOOGLE_REDIRECT_URI. "
        "Set them in Streamlit Cloud -> Settings -> Secrets (or environment)."
    )
    st.stop()


# ---------- Helpers
def _flow() -> Flow:
    """Create a new OAuth flow pre-configured for our app."""
    return Flow.from_client_config(
        json.loads(CLIENT_SECRETS_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )


def get_creds() -> Credentials | None:
    """Return cached creds from the session, refreshing if needed."""
    cdict = st.session_state.get("creds_dict")
    if not cdict:
        return None
    creds = Credentials.from_authorized_user_info(cdict, SCOPES)
    # Let google-auth handle refresh lazily when used; we keep it simple here.
    return creds


def set_creds(creds: Credentials) -> None:
    """Persist credentials in the session."""
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


def get_auth_url() -> str:
    """Start OAuth and return the Google consent URL."""
    flow = _flow()
    url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = state
    return url


def maybe_finish_oauth() -> None:
    """
    If the URL contains ?code=..., complete the OAuth code exchange
    and store credentials in the session, then clear query params.
    """
    params = st.query_params
    code = params.get("code")
    if isinstance(code, list):
        code = code[0]

    state = params.get("state")
    if isinstance(state, list):
        state = state[0]

    if not code:
        return

    # Optional: verify state (if both present)
    if "oauth_state" in st.session_state and state and state != st.session_state["oauth_state"]:
        st.warning("OAuth state mismatch; ignoring response.")
        return

    flow = _flow()
    flow.fetch_token(code=code)
    set_creds(flow.credentials)

    # Clear the URL params to avoid re-triggering
    try:
        st.query_params.clear()
    except Exception:
        pass


# ---------- UI
st.title("📂 Drive Chat (grounded)")

with st.sidebar:
    st.header("Auth")
    if st.button("Sign out", use_container_width=True):
        for k in ("creds_dict", "oauth_state"):
            st.session_state.pop(k, None)
        st.rerun()

    with st.expander("Settings", expanded=False):
        st.number_input("Max files to download", 1, 10, 3, key="mx")
        st.number_input("Top chunks to feed LLM", 1, 20, 8, key="tk")
        st.checkbox("Show debug tables", value=True, key="dbg")

# Try to finish OAuth if we have ?code= in the URL
maybe_finish_oauth()

creds = get_creds()
if not creds:
    st.info("Sign in to your Google Drive to begin.")
    auth_url = get_auth_url()
    st.markdown(f"[🔐 Sign in with Google]({auth_url})")
    st.stop()

# Build Drive service and agent
svc = drive_service(creds)
agent = SimpleDriveAgent(
    svc,
    max_files_download=st.session_state.get("mx", 3),
    top_k_chunks=st.session_state.get("tk", 8),
    llm=LLMClient(),
)

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.chat_input("Ask about your Drive… e.g., 'start chatting with your pdfs'")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.write(f"- {s}")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Drive → parsing → retrieving → asking LLM…"):
            t0 = time.time()
            sidebar, downloaded, top_chunks = agent.retrieve(prompt)
            answer, sources = agent.answer_from_chunks(prompt, top_chunks)
            dt = time.time() - t0

        st.markdown(answer or "_(no answer)_")
        if sources:
            with st.expander("Sources", expanded=True):
                for s in sources:
                    st.write(f"- {s}")

        if st.session_state.get("dbg", True):
            st.caption(
                f"⚙️ Completed in {dt:.1f}s | files scanned: {len(sidebar)} | chunks used: {len(top_chunks)}"
            )

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
