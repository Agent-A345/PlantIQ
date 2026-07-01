import os, sys, json, tempfile, subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from ingest import IngestConfig, ingest_file, load_db, save_db
from agents import process_query, AGENTS, clear_query_cache
from knowledge_graph import load_graph, graph_to_json, get_equipment_context, get_compliance_context, get_critical_equipment
from pid_parser import parse_pid_with_gemini, enrich_graph_from_pid

app = Flask(__name__, static_folder="static")
CORS(app)

DB_DIR     = "./voyage_faiss_db"
EMBED      = "voyage"
INGEST_CFG = IngestConfig(db_dir=DB_DIR, embedding_backend=EMBED, chunk_size=500, chunk_overlap=50)
conversation_history = []

def get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key: raise ValueError("GROQ_API_KEY not set in .env")
    return Groq(api_key=key)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/stats")
def stats():
    try:
        _, chunks = load_db(DB_DIR, EMBED)
        sources = list({c.source_file for c in chunks})
        # plant_id breakdown — useful for multi-tenant visibility; falls back
        # to "default" for chunks ingested before plant_id existed
        plants = {}
        for c in chunks:
            pid = getattr(c, "plant_id", "default")
            plants[pid] = plants.get(pid, 0) + 1
        return jsonify({"chunks": len(chunks), "documents": len(sources), "sources": sources, "plants": plants})
    except Exception as e:
        return jsonify({"chunks": 0, "documents": 0, "sources": [], "plants": {}, "error": str(e)})

@app.route("/api/agents")
def get_agents():
    return jsonify(AGENTS)

def classify_error(e):
    """
    Turn raw exceptions into clear, demo-friendly messages instead of
    exposing stack traces / raw API errors to the frontend.
    Returns (user_message, error_type, http_status).
    """
    msg = str(e)
    low = msg.lower()

    if "429" in msg or "rate limit" in low or "too many requests" in low:
        return ("The embedding service is temporarily rate-limited. "
                "This usually resolves within a minute — please try again shortly.",
                "rate_limit", 429)

    if "groq_api_key" in low or "voyage_api_key" in low or "api key" in low and "not set" in low:
        return ("A required API key is missing on the server. "
                "Check the .env configuration.",
                "config_error", 500)

    if "getaddrinfo failed" in low or "urlerror" in low or "connection" in low and ("refused" in low or "reset" in low):
        return ("Could not reach an external service — check your internet connection "
                "and try again.",
                "network_error", 503)

    if "timeout" in low or "timed out" in low:
        return ("The request took too long to respond. Please try again.",
                "timeout", 504)

    if "no documents indexed" in low or ("chunks" in low and "0" in msg):
        return ("No documents are indexed yet. Load demo data or upload documents first.",
                "no_documents", 400)

    # Fallback — still informative but generic
    return ("Something went wrong while processing your question. Please try again.",
            "unknown_error", 500)


@app.route("/api/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "").strip()
    force_agent = data.get("agent")
    top_k = data.get("top_k", 5)
    plant_id = data.get("plant_id")  # optional — None means search all plants (current default demo behavior)
    if not question:
        return jsonify({"error": "No question provided", "error_type": "empty_question"}), 400
    try:
        client = get_client()
        result = process_query(question, client, conversation_history, top_k=top_k,
                               force_agent=force_agent if force_agent != "auto" else None,
                               plant_id=plant_id)
        return jsonify(result)
    except Exception as e:
        user_msg, error_type, status = classify_error(e)
        app.logger.error(f"[/api/ask] {error_type}: {e}")  # full detail still logged server-side
        return jsonify({"error": user_msg, "error_type": error_type, "detail": str(e)}), status

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    plant_id = request.form.get("plant_id", "default")
    suffix = Path(f.filename).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = Path(tmp.name)
        index, chunks = load_db(DB_DIR, EMBED)
        upload_cfg = IngestConfig(db_dir=DB_DIR, embedding_backend=EMBED, chunk_size=500,
                                   chunk_overlap=50, plant_id=plant_id)
        added = ingest_file(tmp_path, index, chunks, upload_cfg)
        if added > 0:
            for c in chunks[-added:]: c.source_file = f.filename
            save_db(DB_DIR, index, chunks)
            clear_query_cache()  # new docs added -> stale cached retrievals could miss them
        tmp_path.unlink(missing_ok=True)
        return jsonify({"success": True, "chunks_added": added, "filename": f.filename, "plant_id": plant_id})
    except Exception as e:
        user_msg, error_type, status = classify_error(e)
        app.logger.error(f"[/api/upload] {error_type}: {e}")
        return jsonify({"error": user_msg, "error_type": error_type, "detail": str(e)}), status

@app.route("/api/load-demo", methods=["POST"])
def load_demo():
    try:
        subprocess.run([sys.executable, "demo_ingest.py"], capture_output=True, text=True, timeout=120)
        clear_query_cache()  # demo docs (re)loaded -> stale cached retrievals could miss them
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/clear-docs", methods=["POST"])
def clear_docs():
    try:
        import shutil
        db_path = Path(DB_DIR)
        if db_path.exists(): shutil.rmtree(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        conversation_history.clear()
        clear_query_cache()  # docs wiped -> cached retrievals are now invalid
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/clear", methods=["POST"])
def clear_chat():
    conversation_history.clear()
    return jsonify({"success": True})

@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    try:
        clear_query_cache()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/plants")
def list_plants():
    """List all distinct plant_ids currently in the DB, with chunk counts.
    Lets the frontend build a plant selector dropdown for multi-tenant demos."""
    try:
        _, chunks = load_db(DB_DIR, EMBED)
        plants = {}
        for c in chunks:
            pid = getattr(c, "plant_id", "default")
            plants[pid] = plants.get(pid, 0) + 1
        return jsonify({"plants": plants})
    except Exception as e:
        return jsonify({"plants": {}, "error": str(e)})

@app.route("/api/pid/demo")
def pid_demo():
    return send_from_directory("static", "pid_cdu.svg", mimetype="image/svg+xml")

@app.route("/api/pid/parse", methods=["POST"])
def parse_pid():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set in .env file"}), 400
    if "file" in request.files:
        f = request.files["file"]
        suffix = Path(f.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name
        filename = f.filename
    else:
        tmp_path = os.path.join("static", "pid_cdu.svg")
        filename = "pid_cdu.svg"
    try:
        extracted = parse_pid_with_gemini(tmp_path, api_key)
        summary = enrich_graph_from_pid(extracted)
        if "file" in request.files: Path(tmp_path).unlink(missing_ok=True)
        return jsonify({"success": True, "filename": filename, "extracted": extracted, "graph_summary": summary})
    except Exception as e:
        if "file" in request.files: Path(tmp_path).unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/graph")
def graph_data():
    try:
        return jsonify(graph_to_json(load_graph()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/graph/equipment/<tag>")
def equipment_context(tag):
    try:
        return jsonify(get_equipment_context(load_graph(), tag))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/graph/critical")
def critical_equipment():
    try:
        return jsonify(get_critical_equipment(load_graph()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/alerts")
def get_alerts():
    """
    Proactive alerts for the dashboard banner.
    Filters get_critical_equipment() (which returns ALL equipment sorted by
    risk) down to only items that genuinely warrant surfacing unprompted —
    CRITICAL and MEDIUM-HIGH risk levels.
    """
    try:
        all_equipment = get_critical_equipment(load_graph())
        ALERT_LEVELS = {"CRITICAL", "MEDIUM-HIGH"}
        alerts = [
            {
                "tag": e.get("tag", "?"),
                "name": e.get("name", "Unknown equipment"),
                "risk_level": e.get("risk_level", "UNKNOWN"),
                "alert": e.get("alert", e.get("status", "")),
            }
            for e in all_equipment
            if e.get("risk_level") in ALERT_LEVELS
        ]
        return jsonify({"count": len(alerts), "alerts": alerts})
    except Exception as e:
        return jsonify({"count": 0, "alerts": [], "error": str(e)})

@app.route("/api/graph/compliance")
def compliance_status():
    try:
        return jsonify(get_compliance_context(load_graph()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    Path("static").mkdir(exist_ok=True)
    print("\nPlantIQ server starting...")
    print("Open http://localhost:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")