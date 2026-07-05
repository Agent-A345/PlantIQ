import os, sys, json, tempfile, subprocess, asyncio, io, hashlib, re
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from ingest import IngestConfig, ingest_file, load_db, save_db
from agents import process_query, AGENTS, clear_query_cache, LANGUAGES
from knowledge_graph import load_graph, graph_to_json, get_equipment_context, get_compliance_context, get_critical_equipment, get_equipment_health_scores, get_degradation_forecasts, compute_degradation_forecast, summarize_forecast, get_work_order_priorities, simulate_what_if, get_simulatable_equipment, find_similar_incidents, get_all_incident_signals, days_since_threshold_crossed, get_spare_parts_gaps, get_regulatory_deadlines, get_maintenance_window, get_maintenance_window_equipment, get_risk_cascade, get_cascade_equipment, get_carbon_energy_impact
from pid_parser import parse_pid_with_gemini, enrich_graph_from_pid

app = Flask(__name__, static_folder="static")
CORS(app)

DB_DIR     = "./voyage_faiss_db"
EMBED      = "voyage"
INGEST_CFG = IngestConfig(db_dir=DB_DIR, embedding_backend=EMBED, chunk_size=500, chunk_overlap=50)
conversation_history = []

# ──────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH (voice output)
# Primary: edge-tts — free, no API key, no billing. Uses Microsoft's actual
# neural voices (the ones in Edge browser's "Read Aloud"), genuinely
# native-sounding for Indian languages, unlike browser SpeechSynthesis.
# Fallback: gTTS — also free, no key, used only if the edge-tts voice for a
# given language fails for any reason (e.g. transient network issue).
# Audio is cached to disk by a hash of (text, language) so repeated reads
# of the same answer never regenerate audio.
# ──────────────────────────────────────────────────────────────

TTS_CACHE_DIR = Path("./query_cache/tts")
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Matches mixed alphanumeric codes: equipment tags (P-101B, M-101), work orders
# (WO-2024-4521), measurements (4.8mm/s, 81C, 48ppm, 26,000hrs), etc.
# Any token containing both a letter and a digit, optionally with hyphens.
_CODE_PATTERN = re.compile(r'\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]{2,}\b')


# Matches mixed alphanumeric codes: equipment tags (P-101B, M-101), work orders
# (WO-2024-4521), measurements (4.8mm/s, 81C, 48ppm, 26,000hrs), etc.
# Any token containing both a letter and a digit, optionally with hyphens.
_CODE_PATTERN = re.compile(r'\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]{2,}\b')

# Microsoft Edge neural voice per language. Female voices chosen for clarity;
# these are the same high-quality neural voices used in Edge's Read Aloud.
EDGE_VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "hi-IN-SwaraNeural",  # edge-tts has no stable pa-IN neural voice yet
}

# gTTS language codes
GTTS_LANG_CODES = {
    "en": "en", "hi": "hi", "mr": "mr", "gu": "gu", "ta": "ta",
    "te": "te", "bn": "bn", "kn": "kn", "ml": "ml", "pa": "pa",
}


def _tts_cache_path(text, language):
    key = hashlib.sha256(f"{text}|{language}".encode()).hexdigest()[:32]
    return TTS_CACHE_DIR / f"{key}.mp3"


def _synthesize_edge_tts(text, language):
    """Generate speech using edge-tts (Microsoft neural voices). Returns mp3 bytes."""
    voice = EDGE_VOICES.get(language, EDGE_VOICES["en"])

    async def _run():
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    return asyncio.run(_run())


def _synthesize_gtts(text, language):
    """Generate speech using gTTS. Returns mp3 bytes. Reliably reads
    alphanumeric codes (P-101B, WO-2024-4521) correctly, unlike edge-tts."""
    from gtts import gTTS
    lang_code = GTTS_LANG_CODES.get(language, "en")
    buf = io.BytesIO()
    tts = gTTS(text=text, lang=lang_code)
    tts.write_to_fp(buf)
    return buf.getvalue()


def _split_into_segments(text):
    """
    Splits text into ('prose', text) and ('code', text) segments based on
    _CODE_PATTERN. Adjacent prose pieces are kept separate by the regex
    split naturally; empty/whitespace-only segments are dropped.
    """
    segments = []
    last_end = 0
    for m in _CODE_PATTERN.finditer(text):
        prose = text[last_end:m.start()]
        if prose.strip():
            segments.append(("prose", prose))
        segments.append(("code", m.group(0)))
        last_end = m.end()
    tail = text[last_end:]
    if tail.strip():
        segments.append(("prose", tail))
    return segments


def synthesize_speech(text, language):
    """
    Hybrid synthesis: edge-tts (Microsoft neural voices) sounds more natural
    for regular prose, but reliably skips or mumbles Latin alphanumeric codes
    (equipment tags, work orders, measurements) when embedded in non-Latin
    script. gTTS handles those codes correctly but sounds less natural overall.

    So: split the answer into alternating prose/code segments, synthesize
    prose with edge-tts and codes with gTTS, then concatenate the resulting
    MP3 audio into one continuous clip. This means more API calls per answer
    (one per segment) so generation takes a little longer than a single call,
    but every equipment tag and measurement is actually spoken correctly.

    Falls back to a single whole-text gTTS pass if edge-tts is unavailable
    entirely (e.g. network issue), so voice output never just breaks.

    Result is cached to disk keyed on the full text, so repeat reads of the
    same answer never regenerate audio.
    """
    cache_path = _tts_cache_path(text, language)
    if cache_path.exists():
        return cache_path.read_bytes()

    segments = _split_into_segments(text)
    audio_parts = []
    edge_tts_available = True

    for seg_type, seg_text in segments:
        seg_text = seg_text.strip()
        if not seg_text:
            continue
        try:
            if seg_type == "prose" and edge_tts_available:
                audio_parts.append(_synthesize_edge_tts(seg_text, language))
            else:
                audio_parts.append(_synthesize_gtts(seg_text, language))
        except Exception as e:
            if seg_type == "prose":
                app.logger.warning(f"[TTS] edge-tts failed on segment ({e}), switching to gTTS for remainder")
                edge_tts_available = False
                try:
                    audio_parts.append(_synthesize_gtts(seg_text, language))
                except Exception as e2:
                    app.logger.error(f"[TTS] gTTS also failed on segment: {e2}")
            else:
                app.logger.error(f"[TTS] gTTS failed on code segment '{seg_text}': {e}")

    if not audio_parts:
        # Total failure across every segment: last-resort single whole-text gTTS pass
        app.logger.error("[TTS] All segment synthesis failed, attempting single whole-text gTTS pass")
        audio_bytes = _synthesize_gtts(text, language)
    else:
        # Simple byte concatenation of sequential MP3 segments. Not perfectly
        # gapless, but plays correctly in browsers for this use case.
        audio_bytes = b"".join(audio_parts)

    try:
        cache_path.write_bytes(audio_bytes)
    except Exception as e:
        app.logger.warning(f"[TTS] Could not cache audio: {e}")

    return audio_bytes

# ──────────────────────────────────────────────────────────────

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

@app.route("/api/languages")
def get_languages():
    return jsonify(LANGUAGES)

@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    """
    Voice output. Takes {text, language}, returns audio/mpeg bytes.
    Uses edge-tts (free, no key, Microsoft neural voices) with automatic
    gTTS fallback. Text is truncated to a safe length to avoid extremely
    long synthesis on very long answers.
    """
    data = request.json or {}
    text = (data.get("text") or "").strip()
    language = data.get("language", "en")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Cap length defensively — very long text makes for a very long audio clip
    MAX_CHARS = 3000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(".", 1)[0] + "."

    try:
        audio_bytes = synthesize_speech(text, language)
        return Response(audio_bytes, mimetype="audio/mpeg")
    except Exception as e:
        app.logger.error(f"[/api/tts] {e}")
        return jsonify({"error": "Could not generate audio. Please try again."}), 500

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
    language = data.get("language", "en")  # optional — defaults to English
    if not question:
        return jsonify({"error": "No question provided", "error_type": "empty_question"}), 400
    try:
        client = get_client()
        result = process_query(question, client, conversation_history, top_k=top_k,
                               force_agent=force_agent if force_agent != "auto" else None,
                               plant_id=plant_id, language=language)
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

@app.route("/api/health-scores")
def health_scores():
    """
    Equipment health dashboard data. Returns every equipment's computed
    0-100 health score, sorted worst-first, for a visual dashboard view
    rather than the text-based alert banner. Merges in a degradation
    forecast sentence for any equipment that has one, so the dashboard
    can show both current health and projected trend in one call.
    """
    try:
        G = load_graph()
        scores = get_equipment_health_scores(G)
        forecasts_by_tag = {f["tag"]: f["forecast"] for f in get_degradation_forecasts(G)}
        for e in scores:
            fc = forecasts_by_tag.get(e["tag"])
            if fc:
                e["forecast_text"] = fc["text"]
                e["forecast_days"] = fc["days"]
        # For equipment already past threshold, add crossing data
        for node_id, attrs in G.nodes(data=True):
            if attrs.get("type") != "EQUIPMENT":
                continue
            tag = attrs.get("tag", node_id)
            crossings = days_since_threshold_crossed(attrs)
            if crossings:
                # Find matching score entry
                for e in scores:
                    if e["tag"] == tag:
                        e["threshold_crossings"] = crossings
                        break
        return jsonify({"equipment": scores})
    except Exception as e:
        return jsonify({"equipment": [], "error": str(e)})

@app.route("/api/degradation-forecast")
def degradation_forecast():
    """
    Standalone endpoint listing only equipment with a meaningful degradation
    trend (genuinely projected to cross a threshold), sorted by urgency.
    Useful for a dedicated predictive maintenance view or report section.
    """
    try:
        forecasts = get_degradation_forecasts(load_graph())
        return jsonify({"count": len(forecasts), "forecasts": forecasts})
    except Exception as e:
        return jsonify({"count": 0, "forecasts": [], "error": str(e)})

# ──────────────────────────────────────────────────────────────
# SHIFT HANDOVER REPORT
# One-click PDF compiling current critical alerts, compliance gaps, and a
# summary of this session's Q&A, timestamped, for a shift supervisor to
# hand to the next shift. Uses reportlab, pure Python, no system dependency.
# ──────────────────────────────────────────────────────────────

def _build_shift_report_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.platypus.flowables import Flowable

    NAVY     = HexColor("#0A1628")
    TEAL     = HexColor("#0D9488")
    DARK     = HexColor("#1e293b")
    MID      = HexColor("#334155")
    RULE     = HexColor("#cbd5e0")
    GRAY_BG  = HexColor("#f8fafc")
    ALT_BG   = HexColor("#eef2f7")
    RED_T    = HexColor("#b91c1c")
    AMBER_T  = HexColor("#b45309")
    GREEN_T  = HexColor("#15803d")
    BLUE_T   = HexColor("#1d4ed8")
    RED_BG   = HexColor("#fff5f5")
    AMBER_BG = HexColor("#fffbeb")

    RISK_TEXT_COLORS = {
        "CRITICAL":    RED_T,
        "HIGH":        HexColor("#c2410c"),
        "MEDIUM-HIGH": AMBER_T,
        "MEDIUM":      BLUE_T,
        "LOW-MEDIUM":  GREEN_T,
        "LOW":         GREEN_T,
        "UNKNOWN":     MID,
    }

    class _SectionHead(Flowable):
        def __init__(self, number, title, width):
            Flowable.__init__(self)
            self.number = number
            self.title  = title
            self.width  = width
            self.height = 20

        def draw(self):
            c = self.canv
            c.setFillColor(NAVY)
            c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
            c.setFillColor(TEAL)
            c.rect(0, 0, 4, self.height, fill=1, stroke=0)
            c.setFillColor(white)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(12, 6, f"{self.number}. {self.title.upper()}")

    PAGE_W, _ = A4
    MARGIN    = 1.8 * cm
    CONTENT_W = PAGE_W - 2 * MARGIN

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=1.6*cm, bottomMargin=1.6*cm,
                            title="PlantIQ Shift Handover Report")

    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9, textColor=DARK, leading=14, spaceAfter=3)
        base.update(kw)
        return ParagraphStyle(name, **base)

    s_title  = S("title",  fontName="Helvetica-Bold", fontSize=18, textColor=NAVY, leading=24, spaceAfter=2)
    s_meta   = S("meta",   fontSize=9, textColor=MID, leading=13, spaceAfter=10)
    s_body   = S("body",   fontSize=9, textColor=DARK, leading=14, spaceAfter=3, alignment=TA_JUSTIFY)
    s_bold   = S("bold",   fontName="Helvetica-Bold", fontSize=9, textColor=DARK)
    s_tag    = S("tag",    fontName="Helvetica-Bold", fontSize=9, textColor=NAVY)
    s_cell   = S("cell",   fontSize=8.5, textColor=DARK, leading=12)
    s_italic = S("italic", fontName="Helvetica-Oblique", fontSize=9, textColor=MID, leading=14)
    s_foot   = S("foot",   fontSize=7.5, textColor=MID, leading=11)
    s_qa_q   = S("qaq",   fontName="Helvetica-Bold", fontSize=9, textColor=NAVY, spaceBefore=8)
    s_qa_a   = S("qaa",   fontSize=8.5, textColor=DARK, leading=13)

    def _clean_answer(text):
        """Strip SOURCES block and CONFIDENCE line from LLM answer."""
        for marker in ["\nSOURCES", "\n**SOURCES", "\nSources", "\nCONFIDENCE:", "\nCONFIDENCE :"]:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()
        for marker in ["SOURCES", "CONFIDENCE:"]:
            idx = text.rfind(marker)
            if idx != -1 and idx > len(text) * 0.6:
                text = text[:idx].strip()
        lines = text.splitlines()
        while lines and lines[-1].strip().upper().startswith("CONFIDENCE"):
            lines.pop()
        return "\n".join(lines).strip()

    story = []
    now   = datetime.now().strftime("%d %B %Y, %H:%M")

    story.append(Paragraph("Shift Handover Report", s_title))
    story.append(Paragraph(f"Generated by PlantIQ &nbsp;|&nbsp; {now}", s_meta))
    story.append(HRFlowable(width=CONTENT_W, thickness=1.5, color=TEAL, spaceAfter=12))

    # Section 1: Critical Equipment Alerts
    story.append(_SectionHead(1, "Critical Equipment Alerts", CONTENT_W))
    story.append(Spacer(1, 6))

    try:
        G_live = load_graph()
        health_by_tag = {e["tag"]: e["health_score"] for e in get_equipment_health_scores(G_live)}
        alerts = []
        for e in get_critical_equipment(G_live):
            if e.get("risk_level") in ("CRITICAL", "HIGH", "MEDIUM-HIGH", "MEDIUM"):
                e["health_score"] = health_by_tag.get(e.get("tag"), "")
                alerts.append(e)
    except Exception:
        G_live = None
        alerts = []

    if alerts:
        cw = [1.8*cm, 3.8*cm, 2.4*cm, 1.2*cm, CONTENT_W - 1.8*cm - 3.8*cm - 2.4*cm - 1.2*cm]
        hdr = [
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Tag</font>",         s_foot),
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Equipment</font>",    s_foot),
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Risk</font>",         s_foot),
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Score</font>",        s_foot),
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Alert Detail</font>", s_foot),
        ]
        rows = [hdr]
        for e in alerts:
            risk     = e.get("risk_level", "UNKNOWN")
            risk_col = RISK_TEXT_COLORS.get(risk, MID)
            score    = e.get("health_score", "")
            alert_txt = e.get("alert") or e.get("status") or "No detail available."
            rows.append([
                Paragraph(f"<b>{e.get('tag','')}</b>", s_tag),
                Paragraph(e.get("name", ""), s_cell),
                Paragraph(f'<font color="{risk_col.hexval()}" face="Helvetica-Bold">{risk}</font>', s_cell),
                Paragraph(str(score) if score != "" else "--", s_cell),
                Paragraph(alert_txt, s_cell),
            ])
        tbl = Table(rows, colWidths=cw, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1,-1), [GRAY_BG, ALT_BG]),
            ("TEXTCOLOR",      (0, 0), (-1, 0), white),
            ("VALIGN",         (0, 0), (-1,-1), "TOP"),
            ("TOPPADDING",     (0, 0), (-1,-1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1,-1), 5),
            ("LEFTPADDING",    (0, 0), (-1,-1), 5),
            ("RIGHTPADDING",   (0, 0), (-1,-1), 5),
            ("GRID",           (0, 0), (-1,-1), 0.35, RULE),
            ("LINEBELOW",      (0, 0), (-1, 0), 1.5, TEAL),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No critical or high-risk equipment alerts at this time.", s_italic))

    story.append(Spacer(1, 14))

    # Section 2: Compliance Gaps
    story.append(_SectionHead(2, "Compliance Gaps", CONTENT_W))
    story.append(Spacer(1, 6))

    try:
        findings   = get_compliance_context(G_live or load_graph())
        critical_f = findings.get("CRITICAL", [])
        high_f     = findings.get("HIGH", [])
    except Exception:
        critical_f, high_f = [], []

    if critical_f or high_f:
        for f in critical_f:
            desc  = f.get("description", "")
            reg   = f.get("regulation", "") or f.get("reference", "")
            label = f"{desc} &mdash; {reg}" if reg else desc
            row_p = Paragraph(f'<font color="{RED_T.hexval()}" face="Helvetica-Bold">CRITICAL:</font> {label}', s_body)
            row_t = Table([[row_p]], colWidths=[CONTENT_W])
            row_t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),RED_BG),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
                ("LINEBELOW",(0,0),(-1,-1),0.35,RULE),
            ]))
            story.append(row_t)
        for f in high_f:
            desc  = f.get("description", "")
            reg   = f.get("regulation", "") or f.get("reference", "")
            label = f"{desc} &mdash; {reg}" if reg else desc
            row_p = Paragraph(f'<font color="{AMBER_T.hexval()}" face="Helvetica-Bold">HIGH:</font> {label}', s_body)
            row_t = Table([[row_p]], colWidths=[CONTENT_W])
            row_t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),AMBER_BG),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
                ("LINEBELOW",(0,0),(-1,-1),0.35,RULE),
            ]))
            story.append(row_t)
    else:
        story.append(Paragraph("No critical or high severity compliance gaps at this time.", s_italic))

    story.append(Spacer(1, 14))

    # Section 3: Degradation Forecasts
    # Direct graph iteration — bypasses get_degradation_forecasts() which
    # excludes CRITICAL equipment. summarize_forecast returns dict not string.
    story.append(_SectionHead(3, "Equipment Degradation Forecasts", CONTENT_W))
    story.append(Spacer(1, 6))

    forecasts_all = []
    try:
        graph_for_fc = G_live or load_graph()
        for node_id, attrs in graph_for_fc.nodes(data=True):
            if attrs.get("type") != "EQUIPMENT":
                continue
            try:
                fc = compute_degradation_forecast(attrs)
                if not fc:
                    continue
                summary = summarize_forecast(fc)
                if not summary:
                    continue
                min_days = summary.get("days")
                forecasts_all.append({
                    "tag":      attrs.get("tag", node_id),
                    "name":     attrs.get("name", ""),
                    "text":     summary["text"],
                    "min_days": min_days,
                    "detail":   fc,
                })
            except Exception:
                continue
        forecasts_all.sort(key=lambda x: (x["min_days"] is None, x["min_days"] or 9999))
    except Exception:
        forecasts_all = []

    if forecasts_all:
        cw2 = [1.8*cm, 3.8*cm, 2.6*cm, CONTENT_W - 1.8*cm - 3.8*cm - 2.6*cm]
        hdr2 = [
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Tag</font>",              s_foot),
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Equipment</font>",         s_foot),
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Days to Threshold</font>", s_foot),
            Paragraph("<font color='#ffffff' face='Helvetica-Bold'>Forecast</font>",           s_foot),
        ]
        rows2 = [hdr2]
        for f in forecasts_all:
            d = f["min_days"]
            if d is None:
                days_str, days_col = "Monitoring", MID.hexval()
            elif d < 14:
                days_str, days_col = f"{d} days", RED_T.hexval()
            elif d < 30:
                days_str, days_col = f"{d} days", AMBER_T.hexval()
            else:
                days_str, days_col = f"~{d} days", DARK.hexval()
            rows2.append([
                Paragraph(f"<b>{f['tag']}</b>", s_tag),
                Paragraph(f["name"], s_cell),
                Paragraph(f'<font color="{days_col}" face="Helvetica-Bold">{days_str}</font>', s_cell),
                Paragraph(f["text"], s_cell),
            ])
        tbl2 = Table(rows2, colWidths=cw2, repeatRows=1)
        tbl2.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1,-1), [GRAY_BG, ALT_BG]),
            ("TEXTCOLOR",      (0, 0), (-1, 0), white),
            ("VALIGN",         (0, 0), (-1,-1), "TOP"),
            ("TOPPADDING",     (0, 0), (-1,-1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1,-1), 5),
            ("LEFTPADDING",    (0, 0), (-1,-1), 5),
            ("RIGHTPADDING",   (0, 0), (-1,-1), 5),
            ("GRID",           (0, 0), (-1,-1), 0.35, RULE),
            ("LINEBELOW",      (0, 0), (-1, 0), 1.5, TEAL),
        ]))
        story.append(tbl2)
    else:
        story.append(Paragraph(
            "No equipment currently shows a meaningful degradation trend toward a threshold.", s_italic))

    story.append(Spacer(1, 14))

    # Section 4: Q&A — full answers, SOURCES/CONFIDENCE stripped
    story.append(_SectionHead(4, "Session Question and Answer Log", CONTENT_W))
    story.append(Spacer(1, 6))

    if conversation_history:
        qa_pairs = []
        last_q = None
        for msg in conversation_history:
            role        = msg.get("role", "")
            content_msg = msg.get("content", "")
            if role == "user":
                q_match = re.search(r"QUESTION:\s*(.+?)(?:\n\n|$)", content_msg, re.DOTALL)
                last_q  = q_match.group(1).strip() if q_match else content_msg[:300]
            elif role == "assistant" and last_q:
                qa_pairs.append((last_q, content_msg))
                last_q = None

        if qa_pairs:
            for idx, (q, a) in enumerate(qa_pairs, 1):
                a_clean = _clean_answer(a)
                block = KeepTogether([
                    Paragraph(f"Q{idx}: {q}", s_qa_q),
                    Paragraph(a_clean, s_qa_a),
                    Spacer(1, 4),
                ])
                story.append(block)
                if idx < len(qa_pairs):
                    story.append(HRFlowable(width=CONTENT_W, thickness=0.35,
                                            color=RULE, spaceAfter=4, spaceBefore=2))
        else:
            story.append(Paragraph("No questions were asked during this session.", s_italic))
    else:
        story.append(Paragraph("No questions were asked during this session.", s_italic))

    story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=RULE, spaceBefore=12, spaceAfter=6))
    story.append(Paragraph(
        "Generated automatically by PlantIQ. Verify all critical items with current approved procedures before acting.",
        s_foot))

    doc.build(story)
    buf.seek(0)
    return buf


@app.route("/api/shift-report", methods=["GET"])
def shift_report():
    try:
        pdf_buf = _build_shift_report_pdf()
        filename = f"PlantIQ_Shift_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        return send_file(pdf_buf, mimetype="application/pdf", as_attachment=True, download_name=filename)
    except Exception as e:
        app.logger.error(f"[/api/shift-report] {e}")
        return jsonify({"error": "Could not generate shift report. Please try again."}), 500

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
                "risk_source": e.get("risk_source", "static_label"),
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


@app.route("/api/work-order-priorities")
def work_order_priorities():
    """
    Maintenance Work Order Prioritization Engine.
    Returns ranked list of actions for the current shift, scored by a
    composite of risk level, degradation urgency, compliance penalties,
    deferred WOs, H2S service, incident recency, and spare part lead times.
    """
    try:
        priorities = get_work_order_priorities(load_graph())
        return jsonify({"count": len(priorities), "priorities": priorities})
    except Exception as e:
        app.logger.error(f"[/api/work-order-priorities] {e}")
        return jsonify({"count": 0, "priorities": [], "error": str(e)})


@app.route("/api/whatif/equipment")
def whatif_equipment():
    """Returns all equipment suitable for What-If simulation with current readings."""
    try:
        equipment = get_simulatable_equipment(load_graph())
        return jsonify({"equipment": equipment})
    except Exception as e:
        return jsonify({"equipment": [], "error": str(e)})


@app.route("/api/whatif/simulate", methods=["POST"])
def whatif_simulate():
    """
    Runs a What-If simulation. Body: {tag, overrides: {param: value, ...}}
    Supported override keys: vibration_mm_s, bearing_temp_c, oil_iron_ppm,
    bearing_overdue_hours, oil_change_due, defer_days.
    Returns baseline vs simulated state, delta, cascade, and plain-language insights.
    """
    data = request.json or {}
    tag = (data.get("tag") or "").strip()
    overrides = data.get("overrides", {})
    if not tag:
        return jsonify({"error": "tag is required"}), 400
    try:
        result = simulate_what_if(load_graph(), tag, overrides)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"[/api/whatif/simulate] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/incidents/signals")
def incident_signals():
    try:
        signals = get_all_incident_signals(load_graph())
        return jsonify({"signals": signals})
    except Exception as e:
        return jsonify({"signals": [], "error": str(e)})


@app.route("/api/incidents/match", methods=["POST"])
def incident_match():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        matches = find_similar_incidents(load_graph(), query)
        return jsonify({"count": len(matches), "query": query, "matches": matches})
    except Exception as e:
        app.logger.error(f"[/api/incidents/match] {e}")
        return jsonify({"count": 0, "matches": [], "error": str(e)})


@app.route("/api/spare-parts")
def spare_parts():
    """
    Spare Parts Gap Analyzer. Returns stock status, gaps, financial exposure,
    and which equipment is at risk due to missing critical spares.
    """
    try:
        result = get_spare_parts_gaps(load_graph())
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"[/api/spare-parts] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regulatory-deadlines")
def regulatory_deadlines():
    """Regulatory Deadline Tracker — all compliance findings with deadline status."""
    try:
        result = get_regulatory_deadlines(load_graph())
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"[/api/regulatory-deadlines] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/maintenance-window/equipment")
def mw_equipment():
    try:
        return jsonify({"equipment": get_maintenance_window_equipment(load_graph())})
    except Exception as e:
        return jsonify({"equipment": [], "error": str(e)})


@app.route("/api/maintenance-window/plan", methods=["POST"])
def mw_plan():
    data = request.json or {}
    tag  = (data.get("tag") or "").strip()
    if not tag:
        return jsonify({"error": "tag is required"}), 400
    try:
        result = get_maintenance_window(load_graph(), tag)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"[/api/maintenance-window/plan] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/cascade/equipment")
def cascade_equipment():
    try:
        return jsonify({"equipment": get_cascade_equipment(load_graph())})
    except Exception as e:
        return jsonify({"equipment": [], "error": str(e)})


@app.route("/api/cascade/analyze", methods=["POST"])
def cascade_analyze():
    data = request.json or {}
    tag  = (data.get("tag") or "").strip()
    if not tag:
        return jsonify({"error": "tag is required"}), 400
    try:
        result = get_risk_cascade(load_graph(), tag)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"[/api/cascade/analyze] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/carbon-impact")
def carbon_impact():
    """Carbon and Energy Impact Calculator — estimates CO2 waste from degraded equipment."""
    try:
        result = get_carbon_energy_impact(load_graph())
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"[/api/carbon-impact] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    Path("static").mkdir(exist_ok=True)
    print("\nPlantIQ server starting...")
    print("Open http://localhost:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")