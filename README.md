Completed track: **Ola (Business Operations / Customer Support)**

---

# Ola Domain Support Agent

An enterprise-grade customer support and operations agent for Ola rides and services, built with **CrewAI**, **Autogen**, **FastAPI**, **LangChain**, and **ChromaDB**.

The entire repository runs under `MOCK_LLM=true` with **zero API keys and zero external network access**. All telemetry is strictly disabled via `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `ANONYMIZED_TELEMETRY=False`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` configured in `config.py`.

> [!NOTE]
> All policies, customer records, and ticket databases in this project are fictional and engineered specifically for architecture, governance, and evaluation benchmarking. For the exhaustive architectural specification and mathematical proofs, refer to [ola-support-agent.md](file:///d:/New%20folder%20(3)/ola-support-agent.md).

---

## 1. Quick Start

### Windows (PowerShell)
```powershell
# 1. Create and activate Python 3.11 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (CrewAI pins compatible chromadb)
pip install -r requirements.txt

# 3. One-time offline embedding model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 4. Run entire pipeline from clean slate
python run_all.py

# 5. Run test suite
pytest tests/ -v

# 6. Launch FastAPI server
uvicorn api.main:app --port 8000
```

### macOS / Linux (Bash)
```bash
# 1. Create and activate Python 3.11 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. One-time offline embedding model download
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 4. Run entire pipeline from clean slate
python3 run_all.py

# 5. Run test suite
pytest tests/ -v

# 6. Launch FastAPI server
uvicorn api.main:app --port 8000
```

---

## 2. Dataset Design (`dataset.py`)

The support ticketing dataset was synthetically generated with a fixed seed to ensure 100% deterministic reproducibility:
- **Seed:** `SEED = 42` (using Python's `random.Random(42)`)
- **Size:** `N_RECORDS = 50` records
- **Category Weights:** `Billing: 0.28`, `Technical Issue: 0.22`, `Account Access: 0.18`, `Product Defect: 0.14`, `General Inquiry: 0.18`
- **Status Weights:** `Open: 0.20`, `In Progress: 0.22`, `Escalated: 0.10`, `Resolved: 0.28`, `Closed: 0.20`
- **Severity Weights:** `P1: 0.10`, `P2: 0.25`, `P3: 0.40`, `P4: 0.25`
- **SLA Hours Target:** `P1: 4h`, `P2: 12h`, `P3: 24h`, `P4: 72h`
- **Base Escalation Probability:** `ESCALATED_BASE_PROB = 0.12`
- **Resolution Time Range:** Drawn uniformly from `[0.5 * SLA_HOURS, 1.5 * SLA_HOURS]`. *Reasoning: Captures realistic operational variance around SLA targets while maintaining bounded variability across severity tiers.*

### Final Empirical Distribution
- **Categories:** Billing: 17 (34.0%), Technical Issue: 11 (22.0%), Account Access: 11 (22.0%), General Inquiry: 8 (16.0%), Product Defect: 3 (6.0%) (all categories $\ge 3$)
- **Statuses:** Open: 14 (28.0%), In Progress: 12 (24.0%), Resolved: 12 (24.0%), Closed: 8 (16.0%), Escalated: 4 (8.0%) (all statuses $\ge 1$)
- **Severity:** P1: 6 (12.0%), P2: 17 (34.0%), P3: 21 (42.0%), P4: 6 (12.0%)
- **Total Escalated Tickets:** 10 / 50 (20.0%, within required $[10\%, 30\%]$ band)
- **Age Percentiles:** P50 (Median) = 14.5 days, P80 = 24.2 days

---

## 3. Knowledge Base (`kb/`)

The knowledge base consists of 12 distinct policy documents covering all core customer operations:
1. `01_ticket_priority.txt` — Ticket Priority Classification Policy (P1 critical, P2 high, P3 medium, P4 routine)
2. `02_sla_by_severity.txt` — Service Level Agreements by Severity Tier (initial response & resolution windows)
3. `03_escalation_matrix.txt` — Support Escalation Hierarchy and Routing Matrix (L1, L2, L3 operations manager)
4. `04_refund_compensation.txt` — Customer Refund and Goodwill Compensation Policy (Ola Money 24h, bank 5–7 days)
5. `05_communication_channels.txt` — Authorized Customer Communication Channels (app chat, registered email, no OTP requests)
6. `06_business_hours_holidays.txt` — Operational Business Hours and Holiday Coverage (24/7 for P1/P2, SLA pauses on holidays for P3/P4)
7. `07_repeat_complaints.txt` — Repeat Grievance and Chronic Complaint Policy (same category within 30 days escalated to Senior Specialist)
8. `08_service_credits.txt` — Service Credit Disbursement for SLA Defaults (Rs 150/100/50, 90-day expiry)
9. `09_feedback_collection.txt` — Customer Satisfaction Feedback and CSAT Survey Protocol (sent within 1 hour, <3 stars reopens task)
10. `10_vip_handling.txt` — Premium Rider and VIP Account Service Framework (Ola Select & corporate riders, direct L2 routing)
11. `11_outage_communication.txt` — Platform Incident and Outage Broadcast Protocol (>5% failure for 10 min declared major outage)
12. `12_ticket_data_retention.txt` — Customer Support Record Archival and Retention (3 years for billing disputes, 1 year standard)

Every document contains between 2 and 5 concise factual sentences with exact numbers and thresholds.

---

## 4. Chunking Strategies & Threshold Calibration

### Chunking Strategies
Both collections use explicit cosine similarity (`hnsw:space=cosine`, $similarity = 1 - distance$):
1. **`ola_kb_fixed`:** Fixed-size chunking with `chunk_size = 250` characters, `chunk_overlap = 50` characters, with boundaries snapped to whitespace to avoid word fragmentation. Generated **59 chunks**.
2. **`ola_kb_sentence`:** Semantic sentence-level chunking using regex `(?<=[.!?])\s+`, merged into target windows of 150–350 characters. Generated **50 chunks**.

### Calibration Measurements (`rag/calibrate.py`)
Tested across 3 in-scope queries and 3 out-of-scope negative probes:

| Metric | `ola_kb_fixed` | `ola_kb_sentence` |
|---|---|---|
| Min In-Scope Top-1 Similarity | 0.5475 | **0.5814** |
| Max Out-of-Scope Top-1 Similarity | 0.3149 | **0.2406** |
| Measured Separation Gap | 0.2326 | **0.3408** |
| Calibrated Midpoint Threshold | **0.43** | **0.41** |

**Reasoning:** `ola_kb_sentence` exhibited a significantly wider separation gap (0.3408 vs 0.2326) with lower out-of-scope similarity (0.2406 vs 0.3149). The deployed threshold `SIMILARITY_THRESHOLD = 0.41` represents the exact midpoint $(0.2406 + 0.5814) / 2 = 0.411 \to 0.41$.

### Chunking Evaluation (`rag/eval_chunking.py`)

| Query ID | Fixed Precision | Fixed Recall | Sentence Precision | Sentence Recall |
|---|---|---|---|---|
| Q1 (Priority) | 1.000 | 1.000 | 1.000 | 1.000 |
| Q2 (SLA) | 0.500 | 1.000 | 0.500 | 1.000 |
| Q3 (Escalation) | 0.333 | 1.000 | 0.333 | 1.000 |
| Q4 (Refund) | 0.667 | 0.667 | 0.500 | 0.333 |
| Q5 (Credits) | 0.000 | 0.000 | 0.333 | 0.500 |
| **AVERAGE** | **0.500** | **0.733** | **0.533** | **0.767** |

**Recommendation:** `ola_kb_sentence` achieved superior macro-average precision (0.533 vs 0.500) and recall (0.767 vs 0.733) with a composite mean of 0.650 vs 0.617. Sentence chunking avoids slicing across sentences, preserving complete policy conditions within single chunks. Thus, `DEPLOYED_COLLECTION = "ola_kb_sentence"`.

---

## 5. Support Ticket Tool & Escalation Formula

### Escalation Score Formula (`tools/ticket_tool.py`)
For open/active tickets, the score combines severity weight ($W_{sev}$), age relative to a 30-day ceiling, and existing escalation status:
$$\text{escalation\_score} = \text{round}\left(\min\left(1.0,\, 0.5 \times W_{sev} + 0.4 \times \min\left(1.0, \frac{\text{days\_since\_created}}{30.0}\right) + (0.1 \text{ if escalated else } 0.0)\right), 3\right)$$
Where $W_{sev}$: `P1 = 1.0`, `P2 = 0.6`, `P3 = 0.3`, `P4 = 0.1`.
- **Closed / Resolved tickets:** Always receive score `0.0`.
- **Active Escalated tickets:** Guaranteed `escalation_recommended = True`.

### P80 Threshold Justification
- **Formula:** `ESCALATION_THRESHOLD = 0.32`
- **Derivation:** Based on Phase 1 data, the 80th percentile ticket age is $24.2$ days.
  $$0.4 \times \frac{P80(24.2)}{30.0} = 0.3227 \approx 0.32$$
  This ensures that any active ticket whose aging exceeds the 80th percentile contributes enough age penalty to cross the recommendation boundary.
- **Empirical Validation:** 13 / 50 tickets (26.0%) in the dataset exceed the threshold.

---

## 6. Architecture & Multi-Agent Flow

```
[User Input]
     │
     ▼
[Step 1: Input Budget Check] (Reject if tokens > 256)
     │
     ▼
[Step 2: Prompt Injection Guardrail] (12 regex patterns -> Block 400)
     │
     ▼
[Step 3: PII Masking] (Mask phone numbers to [PHONE])
     │
     ▼
[Step 4: Memory Resolution] (Resolve multi-turn pronouns / ticket refs)
     │
     ▼
[Step 5: Cache Lookup] (Normalized key hit -> skip to Step 11)
     │
     ▼
[Step 6: Intent Routing] (policy | ticket | mixed)
     │
     ▼
[Step 7: CrewAI Orchestration] (Sequential process: Lookup -> Retrieval -> Compose)
     │
     ▼
[Step 8: Output Schema Validation] (Validate against SupportResponse)
     │
     ▼
[Step 9: Output Groundedness Guardrail] (Threshold + Claim-level word/number overlap)
     │
     ▼
[Step 10: Autogen Multi-Agent Review] (Critic & Final-Editor RoundRobinGroupChat)
     │
     ▼
[Step 11: Cache Store & Token Accounting] (Store in LRU cache, charge tokens)
     │
     ▼
[Step 12: Structured JSONL Logging] (Write per-request delta log to requests.jsonl)
```

### Agent & Tool Roles

| Agent | CrewAI Role | Allowed Tools | Governance Policy |
|---|---|---|---|
| **Retrieval Agent** | Searches Ola policy knowledge base | `rag_lookup` | Only RAG tool permitted |
| **Lookup Agent** | Fetches support ticket records | `check_support_ticket_status` | Only Ticket tool permitted |
| **Response Composer** | Synthesizes grounded customer reply | *None* | Zero tool access (least-autonomy) |

---

## 7. Mock LLM Design

The project uses deterministic mock LLMs (`llm/mock_crew_llm.py` and `llm/mock_autogen_client.py`) that simulate full ReAct agent loops:
1. **Avoiding Pitfall 1 (ReAct Template Hallucination):** CrewAI's prompt template contains placeholder text: `Observation: the result of the action`. A naive parser sees this and returns placeholder text. `MockCrewLLM` avoids this by searching for `Observation:` **only in the dynamically generated message region** (turns produced after the task prompt).
2. **Avoiding Pitfall 2 (Tool Argument Misclassification):** Rather than matching tool names (which breaks when tools are wrapped or renamed), the mock inspects the tool's Pydantic `args_schema` field names (`record_id` vs `query`).
3. **Query Rewrite for Mixed Intent:** For mixed intent queries, `MockCrewLLM` (retrieval role) strips ticket identifiers from the user query and injects the ticket context: `(for a {severity} {category} ticket open {days_since_created} days)`. This allows single-vector retrieval to locate the exact escalation matrix tier without altering knowledge base files.

---

## 8. Guardrails

1. **PII Masking (`guardrails/pii.py`):**
   - Matches Indian 10-digit mobile phone numbers with optional `+91`, leading `0`, spaces, or hyphens (`PHONE_RE`).
   - Masks matches to `[PHONE]`.
   - *Design Boundary:* Customer names, street addresses, and payment card details are explicitly out of scope for masking per design specification to prevent false-positive corruption of ticket descriptions.
2. **Prompt Injection Detection (`guardrails/injection.py`):**
   - Scans against 12 jailbreak and system-prompt extraction patterns (`ignore previous instructions`, `system prompt`, `you are now`, `developer mode`, etc.).
   - Rejection is immediate (`status: blocked`, 0 LLM calls made).
3. **Groundedness Verification (`guardrails/groundedness.py`):**
   - Validates that retrieved similarity meets or exceeds `SIMILARITY_THRESHOLD`.
   - Breaks answers into sentences and checks claim-level support: $\ge 60\%$ non-stopword overlap and 100% numeric consistency against retrieved context or ticket JSON.
   - Refuses ungrounded responses with fallback text.

---

## 9. Memory & Multi-Turn Session Management

- Uses LangChain's `InMemoryChatMessageHistory` with `RunnableWithMessageHistory` to store turn histories per `session_id`.
- Carries pronouns and ticket references across turns (e.g., Turn 1: "Check OLA-1027", Turn 2: "What is its resolution SLA?").
- *Note:* The LangChain `RunnableWithMessageHistory` deprecation warning is expected per the project brief and is preserved as designed. Memory is ephemeral and lives in-process.

---

## 10. Multi-Agent Review Stage (`review/autogen_review.py`)

Grounded answers pass through an Autogen `RoundRobinGroupChat`:
- **Configuration:** `max_turns=2`, `custom_message_types=[StructuredMessage[ReviewVerdict]]`.
- **Agents:** `Reviewer_Critic` evaluates factual alignment; `Final_Editor` outputs a structured `ReviewVerdict`:
  ```python
  class ReviewVerdict(BaseModel):
      approved: bool
      reason: str
      revised_answer: str | None = None
  ```
- **Behavior:** Approves grounded answers unchanged. If any sentence contains an unsupported claim or number, the editor excises the ungrounded sentence while preserving remaining source tags.

---

## 11. Governance & Security

### 1. Least-Autonomy Guard (`governance/tool_policy.py`)
`enforce_tool_policy` audits agent configurations before Crew kickoff. If an agent attempts to bind unauthorized tools (e.g. Composer binding tools or Lookup binding RAG), the guard raises `ToolPermissionError`.

### 2. Risk Classification
Classified as **MEDIUM RISK** under enterprise AI risk taxonomy. The agent handles customer support policies and accesses customer ticket status metadata, but operates strictly in a read-only capacity. It cannot execute financial disbursements, modify databases, or initiate rides.

### 3. Per-Request Budget (`governance/budget.py`)
A `TokenMeter` (ContextVar) enforces strict resource caps:
- **Measurements:** Normal policy request = 2,727 tokens; ticket lookup = 2,504 tokens; mixed intent = 4,586 tokens.
- **Caps:** `MAX_INPUT_TOKENS = 256` characters, `MAX_REQUEST_TOKENS = 7000` tokens ($\approx 1.5 \times 4,586$).
- Pre-flight rejection aborts oversized requests before invoking LLM calls.

---

## 12. Response Caching (`governance/cache.py`)

- **Cache Implementation:** LRU `OrderedDict` with `maxsize = 256`.
- **Key Normalization:** Lowercased, whitespace collapsed, punctuation stripped.
- **Scope:** Caches grounded generation stage (Steps 6–10); guardrails and logging still execute on hits.
- **Invalidation:** Automatically cleared whenever `/add-document` is called.
- **Empirical Evidence:**
  - 1st request (Miss): 10 LLM calls, 3 tool calls, latency $\approx 113$ ms.
  - 2nd request (Hit): 0 LLM calls, 0 tool calls, latency $\approx 0$ ms.

---

## 13. FastAPI Endpoints & Logging

### Endpoints (`api/main.py`)
- `GET /health` — Returns status and `mock_llm: true`.
- `POST /ask` — Validates `AskRequest` and returns `AskResponse`.
- `POST /add-document` — Ingests a new policy document, chunks it with sentence chunking, updates ChromaDB, and clears the cache.
- `WS /ws/chat` — Real-time multi-turn WebSocket chat. Survives sudden client disconnection without crashing concurrent sessions or server loop.

### Structured Logging (`logs/requests.jsonl`)
Every request produces one structured JSON line with per-request deltas, timings, and masked PII:
```json
{
  "trace_id": "649f1900-d121-4707-95c4-b7ba986fee51",
  "channel": "http",
  "endpoint": "/ask",
  "session_id": "log_sess_pii",
  "request_text": "My number is [PHONE], check OLA-1027",
  "status": "ok",
  "intent": "ticket",
  "cache_hit": false,
  "guardrail_flags": ["PII_MASKED:1"],
  "tokens_used": 2501,
  "cost_inr": 0.125,
  "llm_calls": 5,
  "tool_calls": 1,
  "latency_ms": 109.31,
  "timings_ms": {
    "guardrails_injection": 0.01,
    "guardrails_pii": 0.0,
    "memory_resolution": 8.17,
    "cache_lookup": 0.05,
    "crew_execution": 96.44,
    "schema_validation": 0.0,
    "groundedness_check": 0.18,
    "review": 4.39,
    "total": 109.31
  },
  "ts": "2026-11-08T15:25:13.787Z"
}
```

---

## 14. Evaluation at Scale (`eval/run_eval.py`)

Evaluated across 15 queries covering all 12 KB policy documents plus 3 edge cases:

| ID | Topic | Status | Intent | Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|---|---|---|---|
| 1 | Ticket priority | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | SLA by severity | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 3 | Escalation matrix | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | Refund/compensation | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 5 | Communication channels | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 6 | Business hours/holidays | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 7 | Repeat complaints | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 | Service credits | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 9 | Feedback collection | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 10 | VIP handling | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 11 | Outage communication | refused | refused | 0.00 | 1.00 | 0.00 | 1.00 |
| 12 | Data retention | ok | policy | 1.00 | 1.00 | 1.00 | 1.00 |
| 13 | Edge: mixed + ticket | ok | mixed | 0.50 | 0.00 | 1.00 | 1.00 |
| 14 | Out of scope | refused | refused | 1.00 | 1.00 | 1.00 | 1.00 |
| 15 | Edge: injection + PII | blocked | blocked | 1.00 | 1.00 | 1.00 | 1.00 |

### Macro-Averages (15 Cases)
- **Accuracy:** `0.900`
- **Grounding:** `0.933`
- **Completeness:** `0.933`
- **Safety:** `1.000`
- **Perfect Cases (1.0 across all 4 metrics):** `13 / 15`

---

## 15. Task → Transcript Map

| Task | Script | Transcript | Description & Proof |
|---|---|---|---|
| Task 1 | `dataset.py` | `transcripts/task01_dataset.txt` | 50 records, distributions, 20.0% escalated, validation PASS |
| Task 2 | `demos/task02_kb.py` | `transcripts/task02_kb.txt` | 12 documents, 2–5 sentences each |
| Task 3 | `demos/task03_index.py` | `transcripts/task03_index.txt` | ChromaDB collections built (fixed: 59, sentence: 50) |
| Task 4 | `rag/calibrate.py` & `demos/task04_grounded_generation.py` | `task04_calibration.txt`, `task04_grounded_generation.txt` | Calibrated thresholds & 5 in-scope + 1 out-of-scope answers |
| Task 5 | `rag/eval_chunking.py` | `task05_chunking_eval.txt` | P/R comparison favoring `ola_kb_sentence` (0.650 vs 0.617) |
| Task 6 | `demos/task06_ticket_tool.py` | `task06_ticket_tool.txt` | Ticket lookup tool, escalation score formula, P80 threshold |
| Task 7 | `demos/task07_crew.py` | `task07_crew.txt` | 3 CrewAI agents, ReAct loops, tool invocations |
| Task 8 | `demos/task08_memory.py` | `task08_memory_multiturn.txt`, `task08_memory_fresh.txt` | Multi-turn session memory vs fresh session reset |
| Task 9 | `demos/task09_schema.py` | `task09_schema.txt` | Pydantic schema validation & intentional error catch |
| Task 10 | `demos/task10_guardrails.py` | `task10_guardrails.txt` | PII masking, injection blocking, and groundedness refusal |
| Task 11 | `demos/task11_api.py` | `task11_api.txt` | FastAPI HTTP endpoints + WebSocket disconnect survival |
| Task 12 | `demos/task12_logging.py` | `task12_logging.txt` | JSONL request logging with trace_ids and zero raw PII |
| Task 13 | `eval/run_eval.py` | `task13_eval.txt` | 15-query evaluation benchmark with detailed rationales |
| Task 14 | `demos/task14_review.py` | `task14_review.txt` | Autogen RoundRobin review (approve & revision cases) |
| Task 15 | `demos/task15_governance.py` | `task15_governance.txt` | Tool permission enforcement & audit |
| Task 15b| `demos/task15_budget.py` | `task15_budget.txt` | Token budget meter & pre-flight oversized input rejection |
| Task 16 | `demos/task16_cache.py` | `task16_cache.txt` | LRU response cache before/after zero-call delta proof |

---

## 16. Limitations & Honest Notes

1. **Extractive Mock Generation:** Under `MOCK_LLM=true`, answer synthesis uses deterministic sentence extraction and verified templates rather than neural token generation.
2. **Heuristic Evaluation Judge:** `MockJudge` applies deterministic scoring based on exact factual keyword and numeric presence rather than subjective semantic interpretation.
3. **Multi-Topic Single-Vector Retrieval:** For multi-topic queries (e.g. Q4 and Q5 in chunking eval), single-vector top-3 retrieval retrieves chunks from one dominant topic, achieving partial recall on multi-topic gold sets.
4. **Near-Domain Threshold Margin:** Domain-adjacent out-of-scope queries (e.g. "pet-friendly cabs", similarity 0.3955) score close to the 0.41 threshold. The second line of defense (claim-level overlap and review) catches any borderline leakage.
5. **Eval Case 11 Retrieval Miss:** In evaluation query 11 ("When is an outage declared?"), the top similarity was 0.4069, just below the 0.41 threshold. The groundedness guardrail correctly executed a refusal (`status: refused`), which resulted in 0.0 accuracy for that specific case while maintaining 1.0 safety.
6. **In-Process Memory:** Session chat histories are stored in-memory and do not persist across process restarts.
7. **Expected Deprecation Warnings:** LangChain emits deprecation warnings regarding `RunnableWithMessageHistory` and `InMemoryChatMessageHistory`; these are expected and preserved per the architectural brief.

