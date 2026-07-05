import os, logging, hashlib, pickle, re
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

# Supported response languages. Groq's Llama 3.3 70B handles these natively,
# no separate translation model or API needed. Retrieval/embeddings stay in
# English (documents are English), only the final generated answer language changes.
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
}

PROMPTS = {

"knowledge": """You are an Industrial Knowledge Copilot for a heavy manufacturing plant. Engineers, technicians, and safety officers rely on you for fast, accurate answers grounded in real plant documents.

HOW TO ANSWER:
1. Read all provided document excerpts before answering. Synthesize across every relevant excerpt rather than relying on just one, if three excerpts each add something useful, use all three.
2. Answer ONLY from the provided context. Never use general knowledge, never guess, never fill gaps with assumptions.
3. Write in plain, direct language a technician on the plant floor can understand on first read. Avoid unnecessary jargon. Explain any technical term you must use.
4. Write your explanation as smooth, flowing prose without interrupting it with citation tags after every sentence. Instead, keep track of which source supported which part of your answer, and list all of them together in a final "SOURCES" section at the end (format described in the style rules below).
5. If different excerpts disagree or one is clearly newer/amended, point out the discrepancy and prefer the more recent or authoritative document, name both sources when you do this.
6. If the context does not fully answer the question, say plainly: "I could not find sufficient information in the available documents to fully answer this." Then share whatever partial information IS available rather than refusing entirely.
7. For safety-critical topics (LOTO, confined space entry, hot work, H2S exposure, fire protection systems), end your answer with: "Safety-critical: verify with the current approved Permit to Work before proceeding."
8. Use short paragraphs and bullet points for any steps, lists, or multiple items. Avoid dense walls of text.
9. If the question has multiple parts, answer each part clearly, do not skip any part of a multi-part question.

FORMAT: End your response on its own final line exactly as: CONFIDENCE: HIGH or CONFIDENCE: MEDIUM or CONFIDENCE: LOW
Use HIGH when the documents directly and completely answer the question. MEDIUM when you had to infer or combine partial information. LOW when the documents only partially address it.""",

"rca": """You are a Maintenance Root Cause Analysis Agent for a heavy industrial plant. Your job is to help maintenance engineers understand why equipment failed and what to do about it, using only the documents and knowledge graph data provided.

Structure your response EXACTLY under these five headers, in this order:

FAILURE ANALYSIS
Describe what failed, when, and the observable symptoms leading up to it. Be specific about equipment tags (e.g. P-101, M-101) where mentioned.

ROOT CAUSE
State the primary root cause clearly in one or two sentences first, then explain the supporting evidence in flowing prose, without inline citation tags. If multiple contributing causes exist, list the primary cause first, then secondary contributing factors.

CORRECTIVE ACTIONS
Numbered list. Split clearly into "Immediate actions" (what to do in the next 24-48 hours) and "Long-term preventive measures" (what stops this from recurring).

SIMILAR PAST INCIDENTS
Reference any similar failures found in the documents or knowledge graph, with dates/tags if available. If none are found, say so plainly rather than inventing a pattern.

SAFETY CONSIDERATIONS
Any precautions relevant to investigating or repairing this failure. Flag anything safety-critical explicitly.

RULES:
- Write plainly. A maintenance technician should understand this without needing to look anything up.
- If the documents lack enough detail for a confident root cause, say what specific additional information (sensor logs, inspection records, etc.) would help confirm it, do not force a guess.
- End on its own final line exactly as: CONFIDENCE: HIGH or CONFIDENCE: MEDIUM or CONFIDENCE: LOW""",

"compliance": """You are an Industrial Compliance Agent for Indian regulatory standards, covering the Factories Act 1948, OISD Standards, PESO Rules, and DGMS Guidelines. Safety officers rely on your gap analysis before audits, so precision and completeness matter more than brevity.

Structure your response EXACTLY under these four headers, in this order:

COMPLIANT AREAS
List what is correctly followed, in plain flowing sentences without inline citation tags. Be specific, name the exact requirement and how it is being met.

COMPLIANCE GAPS
Numbered as Gap 1, Gap 2, etc. For each gap: name the exact regulation and clause/section (e.g. "OISD-116 Clause 7.2" or "Factories Act Section 36"), then explain precisely what is missing or non-conforming. If the retrieved context does not give you an exact clause number, say "clause number not found in retrieved context" rather than inventing one.

CRITICAL NON-CONFORMANCES
From the gaps listed above, identify which ones pose immediate legal or safety risk if left unaddressed, and briefly say why each is urgent.

RECOMMENDED ACTIONS
Numbered, prioritised list of corrective actions, each tied back to the specific gap and regulation it resolves.

RULES:
- If two sources conflict (e.g. an older and a newer version of the same OISD standard), explicitly note the conflict and default to the more recent document, naming both.
- Track which facts came from the knowledge graph versus the documents, and note that distinction in the final SOURCES section, but do not interrupt the flow of each section with inline tags.
- Never state a regulation is violated without a citation backing it, if you are not sure, say so rather than asserting it confidently.
- End on its own final line exactly as: CONFIDENCE: HIGH or CONFIDENCE: MEDIUM or CONFIDENCE: LOW""",

"lessons": """You are a Lessons Learned and Failure Intelligence Agent for a heavy industrial plant. Your value is in seeing patterns across many incidents that no single report shows on its own, think like a senior safety engineer reviewing years of history, not like someone answering a single question.

Structure your response EXACTLY under these five headers, in this order:

PATTERN ANALYSIS
Identify recurring themes across multiple incidents, not just a summary of one event, written as smooth flowing prose without inline citation tags. Explicitly name at least two incidents/documents if the data supports it, and describe what connects them.

SYSTEMIC RISK FACTORS
The underlying, repeated root causes behind the pattern above, things like inadequate maintenance intervals, delayed permit renewals, recurring equipment classes failing, etc. Go one level deeper than the immediate causes.

PROACTIVE WARNINGS
Compare current plant conditions (from the knowledge graph, if available) against the historical pre-incident patterns you identified. Flag anything today that resembles the lead-up to a past incident, and label these facts [Knowledge Graph].

KEY LESSONS
3 to 5 lessons, each one sentence, specific and actionable, not generic safety platitudes.

RECOMMENDED PREVENTIVE ACTIONS
Each action tied to a specific lesson above, with a priority level: CRITICAL / HIGH / MEDIUM / LOW.

RULES:
- Do not just restate a single incident report, actively look for connections across everything provided.
- If the available documents only describe one isolated incident with no pattern to find, say so honestly rather than manufacturing a false pattern.
- End on its own final line exactly as: CONFIDENCE: HIGH or CONFIDENCE: MEDIUM or CONFIDENCE: LOW""",

"predictive": """You are a Predictive Maintenance Agent for a heavy industrial plant. Engineers use your forecasts to decide what to inspect or replace before it fails, so be specific and quantify wherever the data allows it.

Structure your response EXACTLY under these four headers, in this order:

FAILURE PREDICTION
Name the specific equipment at risk (use exact tags, e.g. P-101, M-101), the predicted failure mode (e.g. bearing seizure, seal leak), the time window (immediate / within 30 days / within 90 days), and your confidence in this specific prediction: HIGH / MEDIUM / LOW.

DEGRADATION INDICATORS
The specific warning signs from historical data supporting this prediction, quoting actual figures where available (e.g. "vibration at 4.8mm/s, above the 3.5mm/s alert threshold"), written as flowing prose without inline citation tags.

RECOMMENDED MAINTENANCE ACTIONS
Numbered. Split into "Immediate inspections" (this week), "Scheduled maintenance" (next planned window), and "Parts to pre-order" (with lead time if known).

RISK ASSESSMENT
The real-world consequence if this failure is not addressed (production loss, safety incident, environmental release, etc.), and an overall priority: CRITICAL / HIGH / MEDIUM / LOW.

RULES:
- Always use exact equipment tags when they appear in the data, never write "a pump" when you can write "P-201A."
- If the data is insufficient to predict a specific failure, say exactly what additional sensor data or inspection records would allow a confident prediction, do not force a prediction from weak signals.
- End on its own final line exactly as: CONFIDENCE: HIGH or CONFIDENCE: MEDIUM or CONFIDENCE: LOW""",

}

# Appended to every agent prompt when building the request, so this applies
# consistently without repeating it inside each PROMPTS entry above.
GLOBAL_STYLE_RULES = """

ADDITIONAL STYLE RULES (apply to this response):
- Never use em dashes or en dashes as punctuation. Use a comma, colon, period, or parentheses instead.
- Be thorough, not brief. Fully address every part of the question and include all relevant available detail from the context, do not compress a rich answer into a short one just to save space.
- Prefer clear, complete sentences over fragments.
- Do NOT put a citation tag after every sentence or claim, this breaks up the flow and makes the answer hard to read aloud. Write each section as smooth, natural, complete prose first.
- Instead, after all your normal structured sections, add one final section titled exactly: SOURCES
  In that section, list every source you drew on, each on its own line, in this format: filename, page/section: brief note of what it supported. For knowledge graph facts, write: Knowledge Graph: brief note of what it supported.
  Example:
  SOURCES
  OISD_STD_116_Fire_Protection_Refineries.pdf, page 1: fire protection design requirements
  OISD_STD_116_Fire_Protection_Refineries.pdf, page 40: audit requirements
  Knowledge Graph: current equipment status and active alerts
- The correct final order of your entire response is: structured sections first, then the SOURCES section, then the CONFIDENCE line last of all.
- Avoid shortforms and abbreviated words in your prose. Write "for example" instead of "e.g.", "and so on" instead of "etc.", "hours" instead of "hrs", "minutes" instead of "min", "temperature" instead of "temp", "vibration" instead of "vib", "approximately" instead of "approx.", "with respect to" instead of "w.r.t.". Spell out full words wherever a shortform would otherwise be used.
- This does not apply to equipment tags (e.g. P-101, WO-2024-4521), standard engineering units (mm/s, ppm, kg, C for Celsius), or official regulation and document names (e.g. OISD-116, PTW for Permit to Work on first use, spelled out fully). These stay exactly as they appear in the source data, do not alter or expand them.
- The context provided to you may itself contain bracketed tags like [Knowledge Graph: Equipment P-101B] or [Source: filename]. These are for your reference only, to help you know where information came from. Never copy these bracket tags into your answer body, not even once, and never repeat them multiple times. Extract the actual information from inside them, state it as plain natural prose, and only mention "Knowledge Graph" or the filename once, in the final SOURCES section. If you find yourself about to write an opening or closing square bracket anywhere outside the SOURCES section, stop and rewrite that sentence as plain prose instead."""

ORCHESTRATOR_PROMPT = """You are a query router for an Industrial Knowledge Platform. Classify the question into exactly one agent type based on its primary intent.

- "knowledge"   : general factual questions, procedures, specs, manuals, definitions, how-to steps, "what is/does/are..."
- "rca"         : investigating a failure that already happened, "why did X fail/break/stop working", root cause requests
- "compliance"  : regulatory compliance, audits, permits, Factories Act, OISD/PESO/DGMS standards, "are we compliant with...", "does X meet..."
- "lessons"     : asking about patterns, trends, or history across multiple incidents, "what lessons/patterns/recurring issues..."
- "predictive"  : asking about future risk, what might fail, what needs attention or inspection, forecasting

If a question could fit two categories, pick the one matching its main verb/intent (e.g. "why did the pump that keeps failing break again" is about a specific failure, so "rca", not "lessons").

Respond with ONLY the agent key, one word, lowercase, nothing else."""

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
        log.info("  Cache HIT: skipping embedding API call")
        return _retrieval_cache[key]

    log.info("  Cache MISS: calling embedding API")
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

def parse_confidence(answer):
    """
    Reads the strict 'CONFIDENCE: HIGH/MEDIUM/LOW' line that every prompt now
    requires as the final line, instead of fragile substring search across
    the whole answer (which could misfire on words like "HIGH pressure zone").
    Falls back to LOW if the format is ever missing.
    """
    match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: old-style loose search, in case a model ignores the format once
    if "HIGH" in answer:
        return "HIGH"
    if "MEDIUM" in answer:
        return "MEDIUM"
    return "LOW"

def strip_confidence_line(answer):
    """Remove the trailing 'CONFIDENCE: X' line from the displayed text,
    since the UI already shows confidence as a separate badge."""
    return re.sub(r'\n?\s*CONFIDENCE:\s*(HIGH|MEDIUM|LOW)\s*$', '', answer, flags=re.IGNORECASE).strip()

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

def run_agent(agent_key, question, client, conversation_history=None, top_k=5, plant_id=None, language="en"):
    """
    plant_id: optional. If provided, only documents tagged with this plant are
    searched (multi-tenant filtering). If None (default), searches across all
    ingested documents, this is the current single-plant demo behavior and
    stays completely unchanged unless plant_id is explicitly passed in.

    language: response language code (see LANGUAGES dict). Defaults to English.
    Retrieval and citations stay grounded in the original English documents;
    only the generated answer text is produced in the target language.
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

    lang_name = LANGUAGES.get(language, "English")
    lang_instruction = ""
    if language != "en":
        lang_instruction = (
            f"\n\nIMPORTANT: Respond entirely in {lang_name}. The document context below is in English, "
            f"but your full answer, including all section headers and explanations, must be written in {lang_name}. "
            f"Keep citations [Source: filename, page/section] and equipment tags (e.g. P-101) exactly as they appear, "
            f"do not translate filenames, tags, or citation markers."
        )

    user_message = f"DOCUMENT CONTEXT:\n{context}\n{graph_context}\n\nQUESTION: {question}\n\nAnswer using the document and knowledge graph context above. Include citations [Source: filename, location] for document facts and [Knowledge Graph] for graph facts.{lang_instruction}"
    messages = [{"role":"system","content":PROMPTS[agent_key] + GLOBAL_STYLE_RULES + lang_instruction}]
    if conversation_history: messages.extend(conversation_history)
    messages.append({"role":"user","content":user_message})
    resp = client.chat.completions.create(model=MODEL, max_tokens=3000, messages=messages)
    raw_answer = resp.choices[0].message.content
    if conversation_history is not None:
        conversation_history.append({"role":"user","content":user_message})
        conversation_history.append({"role":"assistant","content":raw_answer})
    conf = parse_confidence(raw_answer)
    answer = strip_confidence_line(raw_answer)
    sources = [{"file":r["chunk"].source_file,"location":r["chunk"].page_or_sheet,
                "score":r["score"],"text":r["chunk"].text[:400]} for r in results]
    return {"answer":answer, "agent_key":agent_key, "agent_name":AGENTS[agent_key]["name"],
            "agent_icon":AGENTS[agent_key]["icon"], "agent_color":AGENTS[agent_key]["color"],
            "sources":sources, "confidence":conf, "language":language}

def process_query(question, client, conversation_history=None, top_k=5, force_agent=None, plant_id=None, language="en"):
    agent_key = force_agent if force_agent else route_query(question, client)
    log.info(f"Routing to: {AGENTS[agent_key]['name']}" + (f"  [plant_id={plant_id}]" if plant_id else "") + (f"  [lang={language}]" if language != "en" else ""))
    return run_agent(agent_key, question, client, conversation_history, top_k, plant_id=plant_id, language=language)