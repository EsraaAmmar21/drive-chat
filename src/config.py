import os
from dotenv import load_dotenv
load_dotenv()  # reads .env from the current working dir (root)
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
USE_LLAMAPARSE = os.getenv("USE_LLAMAPARSE", "false").lower() == "true"
PARSE_BUDGET_FILES = int(os.getenv("PARSE_BUDGET_FILES", "5"))
TARGET_CONTEXT_CHUNKS = int(os.getenv("TARGET_CONTEXT_CHUNKS", "8"))
MIN_SIM_THRESHOLD = float(os.getenv("MIN_SIM_THRESHOLD", "0.35"))