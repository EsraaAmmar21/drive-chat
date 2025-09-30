# src/llm/hf_client.py

# src/llm/hf_client.py
# src/llm/hf_client.py  (replace the answer() method)

# src/llm/hf_client.py
# src/llm/hf_client.py
from typing import List
from huggingface_hub import InferenceClient
from requests import HTTPError

from src.config import HF_TOKEN, HF_LLM_MODEL, HF_TEMPERATURE, HF_MAX_NEW_TOKENS

_ALIAS = {
    "meta-llama/meta-llama-3-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3.1-8b-i": "meta-llama/Llama-3.1-8B-Instruct",
}
PUBLIC_FALLBACK = "HuggingFaceH4/zephyr-7b-beta"

class LLMClient:
    def __init__(self):
        model = (HF_LLM_MODEL or "").strip()
        self.model = _ALIAS.get(model.lower(), model) or "meta-llama/Llama-3.1-8B-Instruct"
        self.token = (HF_TOKEN or "").strip()
        self.temperature = float(HF_TEMPERATURE)
        self.max_new_tokens = int(HF_MAX_NEW_TOKENS)

        # Force default HF Serverless provider; ignore env provider routes.
        self.client = InferenceClient(model=self.model, token=self.token, provider=None)

    def _chat(self, model_id: str, system: str, user: str, use_token: bool = True) -> str:
        client = self.client if (use_token and model_id == self.model) else InferenceClient(
            model=model_id, token=(self.token if use_token else None), provider=None
        )
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        msg = resp.choices[0].message
        # message.content may be a string or list of content parts
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content).strip()

    def _text_gen(self, model_id: str, prompt: str, use_token: bool = True) -> str:
        client = self.client if (use_token and model_id == self.model) else InferenceClient(
            model=model_id, token=(self.token if use_token else None), provider=None
        )
        out = client.text_generation(
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            repetition_penalty=1.05,
        )
        return out.strip()

    def answer(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "I couldn't find relevant text in your Drive for this query."

        ctx = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts) if c)
        system = ("You are a helpful assistant. Answer ONLY using the provided context. "
                  "If the answer isn't in the context, say you don't know.")
        user = (f"Question:\n{question}\n\nUse ONLY this context:\n{ctx}\n\n"
                "Answer concisely:")

        # 1) Prefer chat (works with providers that only expose 'conversational')
        try:
            return self._chat(self.model, system, user, use_token=True)
        except Exception as e1:
            # 2) Try canonical Llama 3.1 with chat
            if self.model.lower() != "meta-llama/llama-3.1-8b-instruct":
                try:
                    return self._chat("meta-llama/Llama-3.1-8B-Instruct", system, user, use_token=True)
                except Exception:
                    pass
            # 3) Try text-generation on configured model
            try:
                prompt = f"{system}\n\n{user}\n"
                return self._text_gen(self.model, prompt, use_token=True)
            except Exception:
                # 4) Public fallback (no token) via chat
                try:
                    return self._chat(PUBLIC_FALLBACK, system, user, use_token=False)
                except Exception as e4:
                    # 5) Last resort: public fallback via text_generation
                    try:
                        prompt = f"{system}\n\n{user}\n"
                        return self._text_gen(PUBLIC_FALLBACK, prompt, use_token=False)
                    except Exception as e5:
                        return f"(HF request failed: {e1} / fallbacks: {e4} ; {e5})"































# from __future__ import annotations
# import json
# import os
# import time
# from typing import Optional

# import requests

# from src.config import HF_TOKEN, HF_LLM_MODEL, HF_TEMPERATURE, HF_MAX_NEW_TOKENS

# _HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}"

# def _extract_text_from_hf_payload(data) -> str:
#     """
#     Try hard to pull generated text from various HF payload shapes.
#     """
#     # Common: list of objects with "generated_text"
#     if isinstance(data, list):
#         for item in data:
#             if isinstance(item, dict):
#                 if "generated_text" in item and isinstance(item["generated_text"], str):
#                     return item["generated_text"]
#                 # some pipelines may put it under 'summary_text' or 'translation_text'
#                 for k in ("text", "summary_text", "translation_text"):
#                     if k in item and isinstance(item[k], str):
#                         return item[k]

#     # Dict shapes
#     if isinstance(data, dict):
#         # TGI-like: {'generated_text': '...'}
#         if "generated_text" in data and isinstance(data["generated_text"], str):
#             return data["generated_text"]
#         # OpenAI-like: {'choices': [{'text': '...'}]}
#         choices = data.get("choices")
#         if isinstance(choices, list) and choices and isinstance(choices[0], dict):
#             txt = choices[0].get("text")
#             if isinstance(txt, str):
#                 return txt
#         # Fallback keys
#         for k in ("text", "answer", "output"):
#             v = data.get(k)
#             if isinstance(v, str):
#                 return v

#     # Last resort: stringify the payload
#     return str(data)

# def generate_hf_llama(
#     prompt: str,
#     temperature: float = HF_TEMPERATURE,
#     max_new_tokens: int = HF_MAX_NEW_TOKENS,
#     timeout: int = 60,
#     top_p: Optional[float] = None,
#     repetition_penalty: Optional[float] = None,
#     trim_prompt_chars: int = 20000,  # guard against very long prompts
# ) -> str:
#     """
#     Call Hugging Face Inference API for text generation with LLaMA 3.
#     Returns generated text or raises RuntimeError with a concise message.
#     """
#     if not HF_TOKEN:
#         raise RuntimeError("Missing HF_TOKEN in environment (.env).")

#     if trim_prompt_chars and len(prompt) > trim_prompt_chars:
#         prompt = prompt[:trim_prompt_chars - 1] + "…"

#     headers = {
#         "Authorization": f"Bearer {HF_TOKEN}",
#         "Content-Type": "application/json",
#     }
#     params = {
#         "temperature": float(temperature),
#         "max_new_tokens": int(max_new_tokens),
#         "return_full_text": False,
#         "do_sample": temperature > 0.0,
#     }
#     if top_p is not None:
#         params["top_p"] = float(top_p)
#     if repetition_penalty is not None:
#         params["repetition_penalty"] = float(repetition_penalty)

#     payload = {"inputs": prompt, "parameters": params}

#     # Retry/backoff; respect 'estimated_time' if model is loading
#     backoffs = [0, 1.5, 3.0, 5.0]
#     last_err: Optional[str] = None

#     for delay in backoffs:
#         if delay:
#             time.sleep(delay)
#         try:
#             resp = requests.post(_HF_API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
#         except requests.RequestException as e:
#             last_err = f"Network error: {e}"
#             continue

#         # Model still loading?
#         if resp.status_code in (503, 429):
#             try:
#                 info = resp.json()
#             except Exception:
#                 info = {}
#             # HF often returns {'estimated_time': seconds, 'error': 'Model xyz is currently loading'}
#             est = info.get("estimated_time")
#             last_err = f"HF API busy ({resp.status_code})."
#             if isinstance(est, (int, float)) and est > 0 and est < 30:
#                 time.sleep(float(est))
#                 # loop and retry
#                 continue
#             # no useful estimate; just try next backoff
#             continue

#         if resp.status_code == 401:
#             raise RuntimeError("HF API unauthorized (401). Check HF_TOKEN or model access.")

#         if resp.status_code != 200:
#             # Surface any returned JSON error if available
#             try:
#                 err = resp.json()
#                 if isinstance(err, dict) and "error" in err:
#                     raise RuntimeError(f"HF API error {resp.status_code}: {err['error']}")
#             except Exception:
#                 pass
#             raise RuntimeError(f"HF API error {resp.status_code}: {resp.text[:200]}")

#         # OK
#         try:
#             data = resp.json()
#         except Exception as e:
#             raise RuntimeError(f"HF API returned non-JSON: {e}")

#         # If there's an explicit 'error' field even with 200
#         if isinstance(data, dict) and "error" in data and data["error"]:
#             last_err = f"HF API error: {data['error']}"
#             continue

#         text = _extract_text_from_hf_payload(data).strip()
#         return text

#     raise RuntimeError(last_err or "HF API failed without a clear error.")

