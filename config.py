"""Global configuration for Ola Support Agent.

Ground Rule 3: Must be imported FIRST by every entry-point script to disable
CrewAI telemetry and OpenTelemetry SDK before any other library imports.
"""

import os
from pathlib import Path

# --- Must run before any crewai import ---
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Runtime flags
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"
MOCK_LLM_DEBUG = os.getenv("MOCK_LLM_DEBUG", "false").lower() == "true"

# Paths
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Knowledge base & Embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_FIXED = "ola_kb_fixed"
COLLECTION_SENTENCE = "ola_kb_sentence"
DEPLOYED_COLLECTION = "ola_kb_sentence"  # Task 5 winner: Prec 0.533, Rec 0.767, Mean 0.650 vs fixed (0.500 / 0.733 / 0.617)

# Chunking & Retrieval parameters
FIXED_CHUNK_SIZE = 160      # characters
FIXED_CHUNK_OVERLAP = 40    # characters
TOP_K = 3
SIMILARITY_THRESHOLDS = {
    "ola_kb_fixed": 0.43,
    "ola_kb_sentence": 0.41,
}
# midpoints of measured gaps: fixed (max OOS 0.3149, min IN 0.5475); sentence (max OOS 0.2406, min IN 0.5814)
SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLDS["ola_kb_sentence"]  # threshold for the deployed collection
SECOND_SENTENCE_MARGIN = 0.12  # max similarity drop allowed for 2nd sentence in extractive mock generation

# Escalation parameters & weights
ESCALATION_WEIGHT_FLAG = 0.6
ESCALATION_WEIGHT_AGE = 0.4
ESCALATION_THRESHOLD = 0.32  # 0.4 * P80(24.2 days) / 30 = 0.3227 -> 0.32 (Phase 1 data)
ACTIVITY_FACTORS = {
    "Open": 1.0,
    "In Progress": 1.0,
    "Escalated": 1.0,
    "Resolved": 0.4,
    "Closed": 0.0,
}

# Budget & Governance caps (Phase 8 calibration: policy=2727, ticket=2504, mixed=4586 -> 1.5*4586=6879 -> 7000)
MAX_INPUT_TOKENS = 256
MAX_REQUEST_TOKENS = 7000
COST_PER_1K_TOKENS_INR = 0.05  # notional, for cost reporting only
CACHE_MAX_ENTRIES = 256

# Fallback refusal
FALLBACK_ANSWER = "I don't know — that isn't covered by Ola's support policy knowledge base."
