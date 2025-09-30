# server.py
import os, secrets, time
from typing import Dict, Optional, List
from fastapi import FastAPI, Request, HTTPException, Depends, Response, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from src.agent.simple_agent import SimpleDriveAgent
from src.llm.hf_client import LLMClient

# ---------- Config ----------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CLIENT_SECRETS", "credentials.json")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/oauth2/callback")

SESSION_COOKIE = "sid"
SESSION_TTL_SECONDS = 60 * 60 * 6  # 6 hours

# ---------- Poor-man session store (for demo); use Redis in production ----------
_sessions: Dict[str, Dict] = {}  # sid -> {"creds": {...}, "ts": int}

def _new_sid() -> str:
    return secrets.token_urlsafe(24)

def _get_session(request: Request) -> Optional[Dict]:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid: return None
    sess = _sessions.get(sid)
    if not sess: return None
    # TTL
    if time.time() - sess.get("ts", 0) > SESSION_TTL_SECONDS:
        _sessions.pop(sid, None)
        return None
    sess["ts"] = int(time.time())
    return {"sid": sid, **sess}

def _save_session(response: Response, creds_dict: dict):
    sid = _new_sid()
    _sessions[sid] = {"creds": creds_dict, "ts": int(time.time())}
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, secure=True, samesite="lax")

def _require_creds(request: Request) -> Credentials:
    sess = _get_session(request)
    if not sess: raise HTTPException(401, "Not signed in.")
    cdict = sess["creds"]
    creds = Credentials.from_authorized_user_info(cdict, SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request as GRequest
        creds.refresh(GRequest())
        # persist refreshed token
        sess["creds"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        }
    return creds

def _drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds)

# ---------- App ----------
app = FastAPI(title="Drive Chat (Multi-user)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- UI (super-simple chat page) ----------
CHAT_HTML = """
<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drive Chat</title>
<style>
body{font-family:Inter,system-ui,Arial;padding:24px;max-width:900px;margin:0 auto;background:#0b0d12;color:#e6e9ef}
.card{background:#141821;border:1px solid #212636;border-radius:14px;padding:16px;margin:12px 0}
.input{width:100%;padding:12px;border-radius:12px;border:1px solid #2a3142;background:#0f131a;color:#e6e9ef}
.btn{padding:10px 14px;border-radius:10px;border:1px solid #2a3142;background:#1c2330;color:#e6e9ef;cursor:pointer}
.row{display:flex;gap:8px;align-items:center}
.small{color:#9aa4b2;font-size:12px}
a{color:#7cc0ff}
</style>
</head><body>
<h2>📂 Drive Chat (grounded)</h2>
<div id="auth"></div>
<div id="chat" style="display:none">
  <div class="card">
    <form id="askform" onsubmit="return false;">
      <div class="row">
        <input id="q" class="input" placeholder="Ask (e.g., summary Esraa resume)" />
        <button class="btn" onclick="ask()">Ask</button>
        <button class="btn" onclick="logout()">Sign out</button>
      </div>
      <div class="small">Answers are grounded in your Google Drive PDFs (read-only).</div>
    </form>
  </div>
  <div id="out"></div>
</div>
<script>
async function check() {
  const r = await fetch('/me'); 
  if (r.ok) {
    document.getElementById('auth').innerHTML = '';
    document.getElementById('chat').style.display = 'block';
  } else {
    const a = await (await fetch('/auth/start')).json();
    document.getElementById('auth').innerHTML =
      `<div class="card"><a class="btn" href="${a.url}">🔐 Sign in with Google Drive</a></div>`;
    document.getElementById('chat').style.display = 'none';
  }
}
async function ask() {
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  const out = document.getElementById('out');
  out.insertAdjacentHTML('afterbegin', `<div class="card"><div><b>You:</b> ${q}</div></div>`);
  const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:new URLSearchParams({query:q})});
  if (!r.ok) {
    out.insertAdjacentHTML('afterbegin', `<div class="card">Error: ${r.status}</div>`);
    return;
  }
  const data = await r.json();
  const sources = (data.sources||[]).map(s=>`<div class="small">• <a href="${s.split(' — ').pop()}" target="_blank">${s}</a></div>`).join('');
  out.insertAdjacentHTML('afterbegin', `<div class="card"><div><b>Assistant:</b> ${data.answer||''}</div>${sources}</div>`);
}
async function logout(){
  await fetch('/logout', {method:'POST'});
  location.reload();
}
check();
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(CHAT_HTML)

# ---------- OAuth flow ----------
@app.get("/auth/start")
def auth_start():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    # stash state in a short-lived cookie? for demo we’ll trust HF state
    return JSONResponse({"url": auth_url})

@app.get("/oauth2/callback")
def auth_callback(request: Request):
    # Finalize the OAuth flow
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(authorization_response=str(request.url))
    creds: Credentials = flow.credentials
    creds_dict = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    resp = RedirectResponse(url="/")
    _save_session(resp, creds_dict)
    return resp

@app.get("/me")
def me(request: Request):
    sess = _get_session(request)
    if not sess:
        raise HTTPException(401, "Not signed in")
    return {"ok": True}

@app.post("/logout")
def logout(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid: _sessions.pop(sid, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp

# ---------- Chat endpoint ----------
@app.post("/ask")
def ask(request: Request, query: str = Form(...)):
    creds = _require_creds(request)
    svc = _drive_service(creds)
    agent = SimpleDriveAgent(svc, max_files_download=3, top_k_chunks=8, llm=LLMClient())
    sidebar, downloaded, top = agent.retrieve(query)
    answer, sources = agent.answer_from_chunks(query, top)
    return {"answer": answer, "sources": sources, "sidebar": sidebar[:5]}
