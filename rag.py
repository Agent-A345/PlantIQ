import os, sys, logging, argparse, hashlib, json, pickle
from pathlib import Path
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from ingest import IngestConfig, query_collection, load_db

log = logging.getLogger("rag")

@dataclass
class RAGConfig:
    db_dir: str = "./voyage_faiss_db"
    embedding_backend: str = "voyage"
    top_k: int = 5
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 1500
    min_similarity: float = 15.0

SYSTEM_PROMPT = """You are an Industrial Knowledge Copilot for a heavy manufacturing plant.
Answer questions from engineers, maintenance technicians, and safety officers using ONLY the provided document context.

RULES:
1. Answer ONLY from the provided context. Never use general knowledge.
2. Every factual claim must include a citation: [Source: filename, page/section]
3. If context is insufficient say: "I could not find sufficient information in the available documents."
4. For safety-critical topics (LOTO, confined space, hot work, H2S), end with:
   "Safety-critical — verify with current approved PTW before proceeding."
5. State confidence: HIGH (direct answer) / MEDIUM (inferred) / LOW (partial match).
6. Be concise. Use bullet points for steps."""

# ──────────────────────────────────────────────────────────────
# QUERY CACHE
# Caches retrieval results (query_collection output) by question text + plant_id,
# so repeated/demo questions skip the Voyage embedding API call entirely.
# Two layers: in-memory (fast, per-process) + on-disk (persists across restarts).
#
# Note: this module's cache is independent from agents.py's cache (the one
# actually used by the live web app via server.py). This one is used only
# when calling rag.py directly via CLI (python rag.py).
# ──────────────────────────────────────────────────────────────

CACHE_DIR = Path("./query_cache")
CACHE_FILE = CACHE_DIR / "retrieval_cache_cli.pkl"

_memory_cache = {}  # question_hash -> results list
_cache_loaded = False


def _cache_key(question, db_dir, top_k, plant_id=None):
    """Normalize + hash so trivial whitespace/case differences still hit cache."""
    norm = question.strip().lower()
    raw = f"{norm}|{db_dir}|{top_k}|{plant_id or 'ALL'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_disk_cache():
    global _memory_cache, _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                _memory_cache = pickle.load(f)
            log.info(f"  Query cache loaded: {len(_memory_cache)} cached question(s)")
        except Exception as e:
            log.warning(f"  Could not load query cache: {e}")
            _memory_cache = {}


def _save_disk_cache():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(_memory_cache, f)
    except Exception as e:
        log.warning(f"  Could not save query cache: {e}")


def cached_query_collection(question, ingest_cfg, top_k, plant_id=None):
    """
    Drop-in replacement for query_collection() with caching.
    Cache hit -> no Voyage API call at all.
    Cache miss -> calls query_collection() as normal, then stores result.
    """
    _load_disk_cache()
    key = _cache_key(question, ingest_cfg.db_dir, top_k, plant_id)

    if key in _memory_cache:
        log.info(f"  Cache HIT for query (no API call needed)")
        return _memory_cache[key]

    log.info(f"  Cache MISS — calling embedding API")
    results = query_collection(question, ingest_cfg, top_k=top_k, plant_id=plant_id)
    _memory_cache[key] = results
    _save_disk_cache()
    return results


def clear_query_cache():
    """Call this if documents are re-ingested and cached results would be stale."""
    global _memory_cache, _cache_loaded
    _memory_cache = {}
    _cache_loaded = True
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    log.info("  Query cache cleared")


# ──────────────────────────────────────────────────────────────

def build_context(results):
    if not results: return "No relevant documents found."
    lines = ["RETRIEVED DOCUMENT EXCERPTS:\n"]
    for i, r in enumerate(results, 1):
        c = r["chunk"]
        lines.append(f"[Excerpt {i}]\nSource: {c.source_file}\nLocation: {c.page_or_sheet}\nRelevance: {r['score']}%\nContent:\n{c.text}\n")
    return "\n---\n".join(lines)

def ask(question, cfg, client, conversation_history=None, plant_id=None):
    ingest_cfg = IngestConfig(db_dir=cfg.db_dir, embedding_backend=cfg.embedding_backend)
    logging.getLogger("ingest").setLevel(logging.WARNING)
    results = cached_query_collection(question, ingest_cfg, top_k=cfg.top_k, plant_id=plant_id)
    logging.getLogger("ingest").setLevel(logging.INFO)
    if cfg.embedding_backend != "hash":
        results = [r for r in results if r["score"] >= cfg.min_similarity]
    context = build_context(results)
    user_message = f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer using only the document context above. Include citations [Source: filename, location]."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history: messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(model=cfg.model, max_tokens=cfg.max_tokens, messages=messages)
    answer = response.choices[0].message.content
    if conversation_history is not None:
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": answer})
    return answer, results

def print_answer(question, answer, results):
    print(f"\n{'='*60}\nQ: {question}\n{'='*60}\n{answer}\n")
    if results:
        print(f"Sources: {', '.join({r['chunk'].source_file for r in results})}")
    print("-"*60)

WELCOME = """
PlantIQ — Industrial Knowledge Copilot
Commands: /clear  /sources  /quit  /clearcache
"""

def chat_loop(cfg, client):
    print(WELCOME)
    _, chunks = load_db(cfg.db_dir, cfg.embedding_backend)
    if not chunks:
        print("No documents indexed. Run: python demo_ingest.py")
        return
    sources = list({c.source_file for c in chunks})
    print(f"Loaded {len(chunks)} chunks from {len(sources)} document(s):")
    for s in sources: print(f"  {s}")
    print()
    history = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not q: continue
        if q.lower() in ("/quit", "quit", "exit"): print("Goodbye!"); break
        if q.lower() == "/clear": history.clear(); print("Cleared.\n"); continue
        if q.lower() == "/clearcache": clear_query_cache(); print("Query cache cleared.\n"); continue
        if q.lower() == "/sources":
            for s in sources: print(f"  {s}")
            print(); continue
        try:
            print("\nThinking...\n")
            answer, results = ask(q, cfg, client, history)
            print_answer(q, answer, results)
        except Exception as e:
            print(f"\nError: {e}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="./voyage_faiss_db")
    parser.add_argument("--embedding", default="voyage", choices=["hash","voyage","sentence"])
    parser.add_argument("--query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--plant-id", default=None, help="Optional: filter to a specific plant_id")
    args = parser.parse_args()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY not set in .env"); sys.exit(1)
    client = Groq(api_key=key)
    cfg = RAGConfig(db_dir=args.db, embedding_backend=args.embedding, top_k=args.top_k,
                    min_similarity=0.0 if args.embedding == "hash" else 15.0)
    if args.query:
        answer, results = ask(args.query, cfg, client, plant_id=args.plant_id)
        print_answer(args.query, answer, results)
    else:
        chat_loop(cfg, client)

if __name__ == "__main__":
    main()