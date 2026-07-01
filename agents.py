import os, logging, hashlib, pickle
from pathlib import Path
from dataclasses import dataclass
from groq import Groq
from ingest import IngestConfig, query_collection
from rag import build_context
from knowledge_graph import load_graph, enrich_query_with_graph

log = logging.getLogger("agents")

DB_DIR = "./voyage_faiss_db"
MODEL  = "llama-3.3-70b-versatile"
EMBED  = "voyage"
BASE_CFG = IngestConfig(db_dir=DB_DIR, embedding_backend=EMBED)

AGENTS = {
    "knowledge":   {"name": "Knowledge Copilot",      "icon": "📚", "color": "#3b82f6"},
    "rca":         {"name": "Maintenance RCA Agent",  "icon": "🔧", "color": "#f59e0b"},
    "compliance":  {"name": "Compliance Agent",       "icon": "📋", "color": "#10b981"},
    "lessons":     {"name": "Lessons Learned Agent",  "icon": "🧠", "color": "#8b5cf6"},
    "predictive":  {"name": "Predictive Maintenance", "icon": "📈", "color": "#ef4444"},
}

PROMPTS = {
"knowledge": """You are an Industrial Knowledge Copilot for a heavy manufacturing plant.
Answer questions from engineers, technicians, and safety officers using ONLY the provided document context.
RULES:
1. Answer ONLY from context. Never use general knowledge.
2. Every factual claim must include: [Source: filename, page/section]
3. If context is insufficient: "I could not find sufficient information in the available documents."
4. For safety-critical topics (LOTO, confined space, hot work, H2S): end with "Safety-critical — verify with current approved PTW before proceeding."
5. State confidence: HIGH / MEDIUM / LOW.
6. Be concise. Use bullet points for steps.""",

"rca": """You are a Maintenance Root Cause Analysis Agent for a heavy industrial plant.
Structure your response EXACTLY as:
FAILURE ANALYSIS — describe what failed and when
ROOT CAUSE — primary root cause with citation [Source: filename]
CORRECTIVE ACTIONS — immediate actions (numbered), long-term preventive measures
SIMILAR PAST INCIDENTS — reference similar failures from documents
SAFETY CONSIDERATIONS — relevant safety precautions
Confidence: HIGH / MEDIUM / LOW. If insufficient data, state what additional information is needed.""",

"compliance": """You are an Industrial Compliance Agent for Indian regulatory standards.
Regulations you enforce: Factories Act 1948, OISD Standards, PESO Rules, DGMS Guidelines.
Structure your response EXACTLY as:
COMPLIANT AREAS — what is correctly followed [Source: filename]
COMPLIANCE GAPS — Gap N: [Regulation reference] — what is missing
CRITICAL NON-CONFORMANCES — gaps posing immediate legal or safety risk
RECOMMENDED ACTIONS — prioritised corrective actions with regulatory references
Always cite specific regulation sections (e.g. Factories Act Section 38).
Confidence: HIGH / MEDIUM / LOW.""",

"lessons": """You are a Lessons Learned and Failure Intelligence Agent for a heavy industrial plant.
Structure your response EXACTLY as:
PATTERN ANALYSIS — recurring themes across incidents [Source: filename]
SYSTEMIC RISK FACTORS — root systemic causes appearing repeatedly
PROACTIVE WARNINGS — current conditions matching historical pre-incident patterns
KEY LESSONS — top 3-5 actionable lessons from historical data
RECOMMENDED PREVENTIVE ACTIONS — specific actions with priority level
Confidence: HIGH / MEDIUM / LOW. Think beyond individual incidents — look for systemic patterns.""",

"predictive": """You are a Predictive Maintenance Agent for a heavy industrial plant.
Structure your response EXACTLY as:
FAILURE PREDICTION — equipment at risk, predicted failure mode, time window (immediate/30 days/90 days), confidence HIGH/MEDIUM/LOW
DEGRADATION INDICATORS — warning signs in historical data [Source: filename]
RECOMMENDED MAINTENANCE ACTIONS — immediate inspections (numbered), scheduled maintenance, parts to pre-order
RISK ASSESSMENT — consequence of failure, priority: CRITICAL/HIGH/MEDIUM/LOW
Be specific about equipment tags (P-101, M-101). State what additional data would help if insufficient.""",
}

ORCHESTRATOR_PROMPT = """You are a query router for an Industrial Knowledge Platform.
Classify the question into exactly one agent type:
- "knowledge"   : general questions, procedures, specs, manuals, how-to
- "rca"         : root cause analysis, failure investigation, why did X fail
- "compliance"  : regulatory compliance, Factory Act, OISD, audits, permits
- "lessons"     : lessons learned, incident patterns, recurring failures, history
- "predictive"  : predict failures, maintenance forecasting, what needs attention
Respond with ONLY the agent key — one word lowercase."""

_KG = None

def get_graph():
    global _KG
    if _KG is None:
        _KG = load_graph()
    return _KG

# ──────────────────────────────────────────────────────────────
# RETRIEVAL CACHE
# This is the actual call path used by the live web app (server.py -> process_query
# -> run_agent -> query_collection). Caching here means repeated/demo questions
# skip the Voyage embedding API call entirely.
# Cache key now includes plant_id so different plants' cached results never mix.
# ──────────────────────────────────────────────────────────────

CACHE_DIR = Path("./query_cache")
CACHE_FILE = CACHE_DIR / "retrieval_cache.pkl"

_retrieval_cache = {}
_cache_loaded = False


def _cache_key(question, db_dir, top_k, plant_id):
    norm = question.strip().lower()
    raw = f"{norm}|{db_dir}|{top_k}|{plant_id or 'ALL'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_disk_cache():
    global _retrieval_cache, _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                _retrieval_cache = pickle.load(f)
            log.info(f"  Query cache loaded: {len(_retrieval_cache)} cached question(s)")
        except Exception as e:
            log.warning(f"  Could not load query cache: {e}")
            _retrieval_cache = {}


def _save_disk_cache():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(_retrieval_cache, f)
    except Exception as e:
        log.warning(f"  Could not save query cache: {e}")


def cached_query_collection(question, ingest_cfg, top_k, plant_id=None):
    _load_disk_cache()
    key = _cache_key(question, ingest_cfg.db_dir, top_k, plant_id)

    if key in _retrieval_cache:
        log.info("  Cache HIT — skipping embedding API call")
        return _retrieval_cache[key]

    log.info("  Cache MISS — calling embedding API")
    results = query_collection(question, ingest_cfg, top_k=top_k, plant_id=plant_id)
    _retrieval_cache[key] = results
    _save_disk_cache()
    return results


def clear_query_cache():
    """Call after re-ingesting documents so stale cached results aren't served."""
    global _retrieval_cache, _cache_loaded
    _retrieval_cache = {}
    _cache_loaded = True
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    log.info("  Query cache cleared")

# ──────────────────────────────────────────────────────────────

def route_query(question, client):
    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=10,
            messages=[{"role":"system","content":ORCHESTRATOR_PROMPT},{"role":"user","content":question}])
        agent = resp.choices[0].message.content.strip().lower()
        return agent if agent in AGENTS else "knowledge"
    except Exception as e:
        log.error(f"Orchestrator error: {e}")
        return "knowledge"

def run_agent(agent_key, question, client, conversation_history=None, top_k=5, plant_id=None):
    """
    plant_id: optional. If provided, only documents tagged with this plant are
    searched (multi-tenant filtering). If None (default), searches across all
    ingested documents — this is the current single-plant demo behavior and
    stays completely unchanged unless plant_id is explicitly passed in.
    """
    logging.getLogger("ingest").setLevel(logging.WARNING)
    results = cached_query_collection(question, BASE_CFG, top_k=top_k, plant_id=plant_id)
    logging.getLogger("ingest").setLevel(logging.INFO)
    context = build_context(results)
    try:
        graph_context = enrich_query_with_graph(get_graph(), question)
    except Exception as e:
        graph_context = ""
        log.warning(f"Graph enrichment failed: {e}")
    user_message = f"DOCUMENT CONTEXT:\n{context}\n{graph_context}\n\nQUESTION: {question}\n\nAnswer using the document and knowledge graph context above. Include citations [Source: filename, location] for document facts and [Knowledge Graph] for graph facts."
    messages = [{"role":"system","content":PROMPTS[agent_key]}]
    if conversation_history: messages.extend(conversation_history)
    messages.append({"role":"user","content":user_message})
    resp = client.chat.completions.create(model=MODEL, max_tokens=1500, messages=messages)
    answer = resp.choices[0].message.content
    if conversation_history is not None:
        conversation_history.append({"role":"user","content":user_message})
        conversation_history.append({"role":"assistant","content":answer})
    conf = "HIGH" if "HIGH" in answer else "MEDIUM" if "MEDIUM" in answer else "LOW"
    sources = [{"file":r["chunk"].source_file,"location":r["chunk"].page_or_sheet,
                "score":r["score"],"text":r["chunk"].text[:400]} for r in results]
    return {"answer":answer, "agent_key":agent_key, "agent_name":AGENTS[agent_key]["name"],
            "agent_icon":AGENTS[agent_key]["icon"], "agent_color":AGENTS[agent_key]["color"],
            "sources":sources, "confidence":conf}

def process_query(question, client, conversation_history=None, top_k=5, force_agent=None, plant_id=None):
    agent_key = force_agent if force_agent else route_query(question, client)
    log.info(f"Routing to: {AGENTS[agent_key]['name']}" + (f"  [plant_id={plant_id}]" if plant_id else ""))
    return run_agent(agent_key, question, client, conversation_history, top_k, plant_id=plant_id)