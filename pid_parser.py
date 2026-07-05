import os, json, base64, logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("pid_parser")

EXTRACTION_PROMPT = """You are an expert P&ID analyst for a petroleum refinery.
Analyse this P&ID and extract ALL equipment, instruments, connections and hazardous areas.
Respond ONLY with a valid JSON object in this exact structure — no other text:
{
  "equipment": [{"tag": "P-101A", "type": "Centrifugal Pump", "description": "Crude Feed Pump", "duty": "Continuous"}],
  "instruments": [{"tag": "PI-101", "type": "Pressure Indicator", "service": "Suction pressure P-101A"}],
  "connections": [{"from": "T-100", "to": "P-101A", "fluid": "Crude oil", "line_type": "process"}],
  "hazardous_areas": [{"zone": "Zone 1", "classification": "ATEX IIB T3", "description": "Pump bay area"}],
  "drawing_info": {"title": "CDU Pump Circuit", "drawing_number": "CDU-PID-001", "revision": "REV3"}
}"""

def read_image_as_base64(image_path):
    mime_types = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
                  ".gif":"image/gif",".webp":"image/webp",".svg":"image/svg+xml"}
    mime_type = mime_types.get(Path(image_path).suffix.lower(), "image/png")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type

def parse_pid_with_gemini(image_path, api_key=None):
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env file")

    import urllib.request, urllib.error

    log.info(f"Parsing P&ID: {image_path}")

    if str(image_path).endswith(".svg"):
        with open(image_path, "r") as f:
            svg_content = f.read()
        import re
        text_labels = [t.strip() for t in re.findall(r'<text[^>]*>([^<]+)</text>', svg_content) if t.strip()]
        pid_description = """P&ID DIAGRAM — Crude Distillation Unit (CDU), Vadodara Refinery

EQUIPMENT:
- T-100: Crude Feed Tank (storage vessel)
- P-101A: Crude Feed Pump 315kW 6.6kV motor M-101A (duty)
- P-101B: Crude Feed Pump 315kW (standby, dashed lines)
- E-101: Crude Pre-heater (shell and tube heat exchanger)
- T-101: Atmospheric Distillation Column (48 trays)
- P-102A: Atmospheric Residue Pump (bottom product)
- P-103A: Kerosene Product Pump (side draw)
- P-104A: Naphtha Reflux Pump (overhead)

INSTRUMENTS:
- LI-101: Level Indicator on T-100
- PI-101: Pressure Indicator P-101A suction
- PI-102: Pressure Indicator P-101A discharge
- TI-101: Temperature Indicator P-101A bearing
- TI-201: Temperature Indicator E-101 outlet
- PI-201: Pressure Indicator E-101
- FI-101: Flow Indicator P-101A seal flush
- AI-101: Analyser motor current P-101A
- AE-H2S-101: H2S Gas Detector pump bay
- FCV-101: Flow Control Valve P-101A discharge
- SDV-101: Shutdown Valve P-101A suction

PROCESS CONNECTIONS:
T-100 → P-101A (crude oil)
T-100 → P-101B (crude oil, standby)
P-101A → E-101 (crude oil via FCV-101)
P-101B → E-101 (crude oil, standby)
E-101 → T-101 (pre-heated crude)
T-101 → P-104A (naphtha overhead)
T-101 → P-103A (kerosene side draw)
T-101 → P-102A (atmospheric residue bottom)

HAZARDOUS AREAS:
Zone 1 ATEX IIB T3 — pump bay area around P-101 and P-102
H2S Hazard Area — around crude pump bays

DRAWING: CDU-PID-001-REV3, Vadodara Refinery Unit 3

All text labels in drawing: """ + ", ".join(text_labels)

        payload = {
            "contents": [{"parts": [{"text": f"{pid_description}\n\n{EXTRACTION_PROMPT}"}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
        }
    else:
        img_data, mime_type = read_image_as_base64(image_path)
        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": mime_type, "data": img_data}},
                {"text": EXTRACTION_PROMPT}
            ]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
        }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ValueError(f"Gemini API error {e.code}: {e.read().decode('utf-8')}")

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response: {result}")

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    text = text.strip()

    try:
        extracted = json.loads(text)
    except json.JSONDecodeError:
        import re as _re
        json_match = _re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                extracted = json.loads(json_match.group())
            except:
                extracted = {"equipment":[],"instruments":[],"connections":[],"hazardous_areas":[],"drawing_info":{"title":"Unknown","drawing_number":"Unknown"}}
        else:
            extracted = {"equipment":[],"instruments":[],"connections":[],"hazardous_areas":[],"drawing_info":{"title":"Unknown","drawing_number":"Unknown"}}

    log.info(f"Extracted: {len(extracted.get('equipment',[]))} equipment, {len(extracted.get('instruments',[]))} instruments, {len(extracted.get('connections',[]))} connections")
    return extracted

def enrich_graph_from_pid(extracted, graph_path="./knowledge_graph.pkl"):
    from knowledge_graph import load_graph, save_graph

    G = load_graph(graph_path)
    added_equipment, added_instruments, added_connections = [], [], []

    for eq in extracted.get("equipment", []):
        tag = eq.get("tag","").strip()
        if not tag: continue
        node_id = f"EQ-PID-{tag.replace('-','').replace(' ','')}"
        existing = next((nid for nid, d in G.nodes(data=True) if d.get("tag","").upper() == tag.upper()), None)
        if not existing:
            G.add_node(node_id, type="EQUIPMENT", tag=tag, name=eq.get("description",tag),
                       type_detail=eq.get("type","Unknown"), duty=eq.get("duty","Unknown"),
                       source="P&ID extraction", risk_level="UNKNOWN", criticality="UNKNOWN")
            added_equipment.append(tag)

    for inst in extracted.get("instruments", []):
        tag = inst.get("tag","").strip()
        if not tag: continue
        node_id = f"INST-{tag.replace('-','').replace(' ','')}"
        if not G.has_node(node_id):
            G.add_node(node_id, type="INSTRUMENT", tag=tag, name=inst.get("type",tag),
                       service=inst.get("service",""), source="P&ID extraction")
            added_instruments.append(tag)

    for conn in extracted.get("connections", []):
        src_tag = conn.get("from","").strip()
        dst_tag = conn.get("to","").strip()
        if not src_tag or not dst_tag: continue
        src_node = next((nid for nid, d in G.nodes(data=True) if d.get("tag","").upper() == src_tag.upper()), None)
        dst_node = next((nid for nid, d in G.nodes(data=True) if d.get("tag","").upper() == dst_tag.upper()), None)
        if src_node and dst_node and not G.has_edge(src_node, dst_node):
            G.add_edge(src_node, dst_node, relation="FEEDS_INTO",
                       fluid=conn.get("fluid",""), line_type=conn.get("line_type","process"),
                       source="P&ID extraction")
            added_connections.append(f"{src_tag} → {dst_tag}")

    for zone in extracted.get("hazardous_areas", []):
        zone_name = zone.get("zone","").strip()
        if not zone_name: continue
        node_id = f"LOC-PID-{zone_name.replace(' ','')}"
        if not G.has_node(node_id):
            G.add_node(node_id, type="LOCATION", name=f"{zone_name} — {zone.get('classification','')}",
                       classification=zone.get("classification",""), description=zone.get("description",""),
                       source="P&ID extraction")

    save_graph(G, graph_path)
    return {"equipment_added": added_equipment, "instruments_added": added_instruments,
            "connections_added": added_connections, "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(), "drawing_info": extracted.get("drawing_info",{})}
