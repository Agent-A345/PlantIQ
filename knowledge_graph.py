"""
PS8 Industrial Knowledge Intelligence — Knowledge Graph
========================================================
Builds and queries a rich industrial knowledge graph using NetworkX.

Node types:
  EQUIPMENT    — physical assets (pumps, vessels, motors)
  DOCUMENT     — source documents (manuals, procedures, reports)
  INCIDENT     — failure events, near-misses, H2S releases
  INSPECTION   — condition monitoring records
  REGULATION   — OISD standards, Factory Act sections
  SPARE_PART   — critical spare parts with lead times
  PERSON       — roles (Safety Manager, Maintenance Superintendent)
  LOCATION     — physical areas (CDU Pump Bay, Unit 3)
  CAPA         — corrective and preventive actions

Relationship types:
  MAINTAINS         — equipment → document (manual)
  FAILED_IN         — incident → equipment
  CAUSED_BY         — incident → root cause (another node)
  PRECEDED_BY       — incident → warning signs
  COMPLIES_WITH     — document → regulation
  VIOLATES          — finding → regulation
  LOCATED_IN        — equipment → location
  REQUIRES_PART     — equipment → spare_part
  INVESTIGATED_BY   — incident → person/role
  LED_TO_CAPA       — incident → capa
  PREVENTS          — capa → incident type
  MONITORS          — inspection → equipment
  CONNECTED_TO      — equipment → equipment (process connections)
  REFERENCES        — document → document
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Optional
import networkx as nx

log = logging.getLogger("knowledge_graph")

GRAPH_PATH = "./knowledge_graph.pkl"


def build_graph() -> nx.DiGraph:
    """
    Constructs the full industrial knowledge graph for Vadodara Refinery CDU.
    Returns a directed graph with rich node and edge attributes.
    """
    G = nx.DiGraph()

    locations = [
        ("LOC-CDU",      {"type":"LOCATION", "name":"Crude Distillation Unit (CDU)", "unit":"Unit 3", "hazard_zone":"Zone 1 ATEX"}),
        ("LOC-PUMPBAY-A",{"type":"LOCATION", "name":"CDU Pump Bay Area A",           "unit":"Unit 3", "hazard_zone":"Zone 1 ATEX"}),
        ("LOC-PUMPBAY-B",{"type":"LOCATION", "name":"CDU Pump Bay Area B",           "unit":"Unit 3", "hazard_zone":"Zone 1 ATEX"}),
        ("LOC-PUMPBAY-C",{"type":"LOCATION", "name":"CDU Pump Bay Area C",           "unit":"Unit 3", "hazard_zone":"Zone 1 ATEX"}),
        ("LOC-TANKFARM", {"type":"LOCATION", "name":"Tank Farm",                     "unit":"Unit 3", "hazard_zone":"Zone 1 ATEX"}),
        ("LOC-WAREHOUSE",{"type":"LOCATION", "name":"Plant Warehouse Rack B7 Bay 3", "unit":"Utilities","hazard_zone":"Non-classified"}),
    ]
    for node_id, attrs in locations:
        G.add_node(node_id, **attrs)

    equipment = [
        ("EQ-P101A", {"type":"EQUIPMENT","tag":"P-101A","name":"Crude Feed Pump A","make":"Flowserve","model":"DVMX 3x4x13",
                      "duty":"Continuous","status":"Running","power_kw":315,"fluid":"Arabian Light Crude",
                      "h2s_service":True,"atex_class":"Zone 1 IIB T3","vibration_mm_s":1.9,"bearing_temp_c":58,
                      "oil_change_hours":4000,"oil_change_due":"Dec 2024","seal_type":"API Plan 53B",
                      "risk_level":"LOW","criticality":"HIGH"}),
        ("EQ-P101B", {"type":"EQUIPMENT","tag":"P-101B","name":"Crude Feed Pump B (Standby)","make":"Flowserve","model":"DVMX 3x4x13",
                      "duty":"Standby","status":"Standby","power_kw":315,"fluid":"Arabian Light Crude",
                      "h2s_service":True,"atex_class":"Zone 1 IIB T3","vibration_mm_s":2.4,"bearing_temp_c":69,
                      "oil_change_hours":4000,"oil_change_due":"OVERDUE","seal_type":"API Plan 53B",
                      "risk_level":"MEDIUM-HIGH","criticality":"HIGH",
                      "alert":"NDE bearing temp elevated 69C, oil overdue, minor seal weep — WO-2024-4521 pending"}),
        ("EQ-P102A", {"type":"EQUIPMENT","tag":"P-102A","name":"Atmospheric Residue Pump A","make":"Flowserve",
                      "duty":"Continuous","status":"Running — post incident monitoring","power_kw":132,
                      "h2s_service":True,"vibration_mm_s":1.7,"bearing_temp_c":55,
                      "risk_level":"LOW","criticality":"HIGH","last_overhaul":"Oct 2024",
                      "note":"Returned to service after H2S incident Sep 2024 — increased monitoring until Jan 2025"}),
        ("EQ-P102B", {"type":"EQUIPMENT","tag":"P-102B","name":"Atmospheric Residue Pump B","make":"Flowserve",
                      "duty":"Continuous","status":"Running","power_kw":132,
                      "h2s_service":True,"vibration_mm_s":2.1,"bearing_temp_c":63,
                      "risk_level":"LOW","criticality":"HIGH","bearing_upgrade":"Labyrinth seal Jan 2024"}),
        ("EQ-P103A", {"type":"EQUIPMENT","tag":"P-103A","name":"Kerosene Product Pump A","make":"Flowserve",
                      "duty":"Continuous","status":"Running","power_kw":37,
                      "h2s_service":False,"vibration_mm_s":1.4,"bearing_temp_c":52,
                      "risk_level":"LOW","criticality":"MEDIUM"}),
        ("EQ-P103B", {"type":"EQUIPMENT","tag":"P-103B","name":"Kerosene Product Pump B","make":"Flowserve",
                      "duty":"Continuous","status":"Running — vibration trending up","power_kw":37,
                      "h2s_service":False,"vibration_mm_s":2.2,"bearing_temp_c":64,
                      "risk_level":"LOW-MEDIUM","criticality":"MEDIUM",
                      "alert":"Vibration up 37% in 9 months. Iron 12ppm in oil. Recommend quarterly monitoring."}),
        ("EQ-P201A", {"type":"EQUIPMENT","tag":"P-201A","name":"Vacuum Residue Transfer Pump A","make":"Flowserve",
                      "duty":"Continuous","status":"CRITICAL — immediate action required","power_kw":132,
                      "h2s_service":True,"vibration_mm_s":4.8,"bearing_temp_c":81,
                      "risk_level":"CRITICAL","criticality":"HIGH",
                      "alert":"Vibration 4.8mm/s (above 3.5 alert). Bearing temp 81C (above 75 limit). Oil iron 48ppm. Bearing overdue 26,000hrs. TAKE OUT OF SERVICE WITHIN 7 DAYS.",
                      "overdue_wos":"WO-2024-1847 (bearing), WO-2024-2156 (seal) — both deferred 3x"}),
        ("EQ-M101A",  {"type":"EQUIPMENT","tag":"M-101A","name":"Motor for P-101A","power_kw":315,"voltage_kv":6.6,"rpm":1480,
                       "type_detail":"Induction motor","atex_class":"Zone 1 IIB T3"}),
        ("EQ-E101",   {"type":"EQUIPMENT","tag":"E-101","name":"Crude Pre-heater","type_detail":"Shell and tube HX",
                       "last_inspection":"TA 2022","min_thickness_mm":4.2,"corrosion_rate":"0.3mm/yr",
                       "remaining_life_yrs":4,"risk_level":"LOW","criticality":"HIGH"}),
        ("EQ-T101",   {"type":"EQUIPMENT","tag":"T-101","name":"Atmospheric Distillation Column",
                       "last_inspection":"TA 2022","risk_level":"LOW","criticality":"HIGH"}),
        ("EQ-T104",   {"type":"EQUIPMENT","tag":"T-104","name":"Crude Storage Tank","type_detail":"Floating roof",
                       "recent_finding":"Roof drain valve found open Oct 2024 — closed immediately",
                       "risk_level":"LOW","criticality":"HIGH"}),
    ]
    for node_id, attrs in equipment:
        G.add_node(node_id, **attrs)

    documents = [
        ("DOC-PUMP-MANUAL",  {"type":"DOCUMENT","name":"Pump Maintenance Manual MM-PUMP-001-REV4",
                               "file":"pump_maintenance_manual.txt","covers":"P-101 to P-201 series",
                               "last_reviewed":"Jan 2025","dept":"Maintenance Engineering"}),
        ("DOC-IIR-2024-007", {"type":"DOCUMENT","name":"Incident Investigation Report IIR-2024-007",
                               "file":"incident_investigation_report.txt","date":"Sep 2024",
                               "classification":"HIGH POTENTIAL — H2S Release","equipment":"P-102A"}),
        ("DOC-PTW",          {"type":"DOCUMENT","name":"Permit to Work Procedure HSE-PTW-002-REV6",
                               "file":"permit_to_work_procedure.txt","standard":"OISD-105",
                               "last_reviewed":"Mar 2024","covers":"Hot Work, Cold Work, Confined Space, Electrical"}),
        ("DOC-COMPLIANCE",   {"type":"DOCUMENT","name":"OISD Compliance Checklist HSE-COMP-001-REV3",
                               "file":"oisd_compliance_checklist.txt","audit_period":"Oct 2024",
                               "critical_gaps":2,"high_gaps":2}),
        ("DOC-INSPECTION",   {"type":"DOCUMENT","name":"Equipment Inspection Records INSP-CDU-2024-Q3",
                               "file":"equipment_inspection_records.txt","period":"Jul-Sep 2024",
                               "critical_items":1,"attention_items":1}),
        ("DOC-LESSONS",      {"type":"DOCUMENT","name":"Lessons Learned Database HSE-LL-CDU-2024",
                               "file":"lessons_learned_database.txt","period":"2019-2024",
                               "patterns_identified":5,"proactive_warnings":3}),
    ]
    for node_id, attrs in documents:
        G.add_node(node_id, **attrs)

    incidents = [
        ("INC-2022-004", {"type":"INCIDENT","id":"INC-2022-004","date":"Mar 2022",
                          "equipment":"P-101A","type_detail":"H2S Release — Mechanical Seal Failure",
                          "h2s_ppm":3.5,"injury":False,"production_loss":False,
                          "root_cause":"Barrier fluid pressure loss — PRV-101 stuck open. Monthly checks skipped 2 months.",
                          "classification":"SIGNIFICANT","severity":"HIGH"}),
        ("INC-2023-007", {"type":"INCIDENT","id":"INC-2023-007","date":"Nov 2023",
                          "equipment":"P-102B","type_detail":"Bearing Failure — Shaft Seizure",
                          "h2s_ppm":0,"injury":False,"production_loss_hrs":4.2,
                          "root_cause":"Water contamination in oil. Oil analysis alert ignored for 6 weeks. WO deferred twice.",
                          "classification":"SIGNIFICANT","severity":"MEDIUM"}),
        ("INC-2024-006", {"type":"INCIDENT","id":"INC-2024-006","date":"Jul 2024",
                          "equipment":"P-101A","type_detail":"High Vibration — Planned Shutdown",
                          "h2s_ppm":0,"injury":False,"production_loss":False,
                          "root_cause":"Impeller imbalance from erosive wear. Arabian Heavy crude blend not assessed for equipment impact.",
                          "classification":"EQUIPMENT DEGRADATION","severity":"LOW"}),
        ("INC-2024-007", {"type":"INCIDENT","id":"INC-2024-007","date":"Sep 2024",
                          "equipment":"P-102A","type_detail":"H2S Release — Mechanical Seal Failure",
                          "h2s_ppm":12,"injury":True,"injury_detail":"Minor eye irritation — 1 operator",
                          "production_loss":False,"classification":"HIGH POTENTIAL",
                          "root_cause":"Chronic barrier pressure alarm normalised (23 alarms in 30 days). Seal flush fault deferred. Compound failure.",
                          "severity":"HIGH","precursors":"PAL-102 active 6hrs before incident. FI-102 low for 4 days."}),
    ]
    for node_id, attrs in incidents:
        G.add_node(node_id, **attrs)

    regulations = [
        ("REG-OISD-105",  {"type":"REGULATION","code":"OISD-105","name":"Work Permit System",
                            "authority":"OISD","applicability":"All petroleum installations","mandatory":True}),
        ("REG-OISD-116",  {"type":"REGULATION","code":"OISD-116","name":"Fire Protection Facilities",
                            "authority":"OISD","applicability":"Petroleum refineries","mandatory":True}),
        ("REG-OISD-117",  {"type":"REGULATION","code":"OISD-117","name":"Storage of Flammable Liquids",
                            "authority":"OISD","applicability":"Petroleum storage","mandatory":True}),
        ("REG-FA-S21",    {"type":"REGULATION","code":"Factories Act S.21","name":"Fencing of Machinery",
                            "authority":"Ministry of Labour","applicability":"All factories","mandatory":True}),
        ("REG-FA-S36",    {"type":"REGULATION","code":"Factories Act S.36","name":"Precautions against dangerous fumes",
                            "authority":"Ministry of Labour","applicability":"All factories","mandatory":True}),
        ("REG-FA-S38",    {"type":"REGULATION","code":"Factories Act S.38","name":"Explosive/inflammable substances",
                            "authority":"Ministry of Labour","applicability":"All factories","mandatory":True}),
        ("REG-FA-S41B",   {"type":"REGULATION","code":"Factories Act S.41B","name":"Information on hazardous substances",
                            "authority":"Ministry of Labour","applicability":"All factories","mandatory":True}),
    ]
    for node_id, attrs in regulations:
        G.add_node(node_id, **attrs)

    spare_parts = [
        ("SP-SEL-101",  {"type":"SPARE_PART","part_no":"FC-SEL-101-DUAL","name":"Mechanical Seal Complete Assembly",
                          "lead_time_weeks":8,"min_stock":1,"current_stock":1,"location":"Store"}),
        ("SP-IMP-101",  {"type":"SPARE_PART","part_no":"FC-IMP-101-330","name":"Impeller 330mm",
                          "lead_time_weeks":10,"min_stock":1,"current_stock":1,"location":"Store"}),
        ("SP-BRG-DE",   {"type":"SPARE_PART","part_no":"SKF-7316-BECBM","name":"Bearing Set DE",
                          "lead_time_weeks":2,"min_stock":2,"current_stock":2,"location":"Store"}),
        ("SP-BRG-NDE",  {"type":"SPARE_PART","part_no":"SKF-NU316-ECM","name":"Bearing Set NDE",
                          "lead_time_weeks":2,"min_stock":2,"current_stock":2,"location":"Store"}),
        ("SP-SHAFT",    {"type":"SPARE_PART","part_no":"FC-SHAFT-101","name":"Shaft 4140 Alloy — CRITICAL",
                          "lead_time_weeks":12,"min_stock":1,"current_stock":1,
                          "location":"Warehouse Rack B7 Bay 3","critical":True,
                          "note":"DO NOT ISSUE without Materials Manager approval Form SPR-ISSUE-001"}),
    ]
    for node_id, attrs in spare_parts:
        G.add_node(node_id, **attrs)

    capas = [
        ("CAPA-2022-001", {"type":"CAPA","id":"CAPA-2022-001","status":"COMPLETED",
                            "action":"PRV-101 replaced with position indicator","owner":"Mechanical Maint Supt","due":"Mar 2022"}),
        ("CAPA-2022-002", {"type":"CAPA","id":"CAPA-2022-002","status":"COMPLETED",
                            "action":"Low barrier pressure alarm PAL-101 added to DCS","owner":"Process Control Engr","due":"Apr 2022"}),
        ("CAPA-2024-001", {"type":"CAPA","id":"CAPA-2024-001","status":"COMPLETED",
                            "action":"Repair nitrogen leak in P-102A/B barrier fluid accumulators","owner":"Mech Maint Supt","due":"Sep 2024"}),
        ("CAPA-2024-002", {"type":"CAPA","id":"CAPA-2024-002","status":"IN PROGRESS",
                            "action":"Alarm audit CDU — identify all standing alarms >3 activations/day","owner":"Process Control Engr","due":"Oct 2024"}),
        ("CAPA-2024-003", {"type":"CAPA","id":"CAPA-2024-003","status":"COMPLETED",
                            "action":"Revise maintenance priority matrix for H2S service equipment","owner":"Maint Manager","due":"Oct 2024"}),
        ("CAPA-2024-005", {"type":"CAPA","id":"CAPA-2024-005","status":"IN PROGRESS",
                            "action":"Install auto barrier fluid replenishment P-101 and P-102 series","owner":"Projects Engr","due":"Dec 2024"}),
    ]
    for node_id, attrs in capas:
        G.add_node(node_id, **attrs)

    findings = [
        ("FIND-HYDRANT",    {"type":"FINDING","severity":"CRITICAL","description":"8 fire hydrants not tested in CDU — OISD-116 Clause 7.2 violation",
                              "status":"OVERDUE","due":"Nov 2024"}),
        ("FIND-H2SDETECT",  {"type":"FINDING","severity":"CRITICAL","description":"H2S detector AE-H2S-103 calibration overdue 23 days — Factories Act S.36 risk",
                              "status":"OVERDUE"}),
        ("FIND-PTWTRAIN",   {"type":"FINDING","severity":"HIGH","description":"3 Area Authorities PTW refresher overdue — OISD-105 non-compliance",
                              "status":"IN PROGRESS","due":"Nov 2024"}),
        ("FIND-MSDS",       {"type":"FINDING","severity":"HIGH","description":"MSDS for CA-2024-07 not posted at point of use — Factories Act S.41B",
                              "status":"COMPLETED","completed":"Oct 2024"}),
        ("FIND-GUARD",      {"type":"FINDING","severity":"HIGH","description":"Coupling guard missing on P-103B — Factories Act S.21 violation",
                              "status":"COMPLETED","completed":"Oct 2024"}),
        ("FIND-TANKDRAIN",  {"type":"FINDING","severity":"MEDIUM","description":"T-104 roof drain valve found open — OISD-117 non-compliance",
                              "status":"COMPLETED","completed":"Oct 2024"}),
    ]
    for node_id, attrs in findings:
        G.add_node(node_id, **attrs)


    loc_edges = [
        ("EQ-P101A","LOC-PUMPBAY-A","LOCATED_IN"),("EQ-P101B","LOC-PUMPBAY-A","LOCATED_IN"),
        ("EQ-P102A","LOC-PUMPBAY-B","LOCATED_IN"),("EQ-P102B","LOC-PUMPBAY-B","LOCATED_IN"),
        ("EQ-P103A","LOC-PUMPBAY-C","LOCATED_IN"),("EQ-P103B","LOC-PUMPBAY-C","LOCATED_IN"),
        ("EQ-P201A","LOC-PUMPBAY-B","LOCATED_IN"),("EQ-M101A","LOC-PUMPBAY-A","LOCATED_IN"),
        ("EQ-T104","LOC-TANKFARM","LOCATED_IN"),("SP-SHAFT","LOC-WAREHOUSE","STORED_AT"),
    ]
    for src, dst, rel in loc_edges:
        G.add_edge(src, dst, relation=rel)

    G.add_edge("EQ-P101A","DOC-PUMP-MANUAL",relation="MAINTAINED_BY",section="Section 2")
    G.add_edge("EQ-P101B","DOC-PUMP-MANUAL",relation="MAINTAINED_BY",section="Section 2")
    G.add_edge("EQ-P102A","DOC-PUMP-MANUAL",relation="MAINTAINED_BY",section="Section 2")
    G.add_edge("EQ-P102B","DOC-PUMP-MANUAL",relation="MAINTAINED_BY",section="Section 2")
    G.add_edge("EQ-P201A","DOC-PUMP-MANUAL",relation="MAINTAINED_BY",section="Section 2")
    G.add_edge("EQ-P101A","DOC-INSPECTION",relation="INSPECTED_IN",result="LOW risk")
    G.add_edge("EQ-P101B","DOC-INSPECTION",relation="INSPECTED_IN",result="MEDIUM-HIGH risk — URGENT")
    G.add_edge("EQ-P102A","DOC-INSPECTION",relation="INSPECTED_IN",result="LOW risk — post incident monitoring")
    G.add_edge("EQ-P201A","DOC-INSPECTION",relation="INSPECTED_IN",result="CRITICAL — immediate action required")

    G.add_edge("INC-2022-004","EQ-P101A",relation="OCCURRED_ON")
    G.add_edge("INC-2023-007","EQ-P102B",relation="OCCURRED_ON")
    G.add_edge("INC-2024-006","EQ-P101A",relation="OCCURRED_ON")
    G.add_edge("INC-2024-007","EQ-P102A",relation="OCCURRED_ON")

    G.add_edge("INC-2022-004","DOC-PUMP-MANUAL",relation="DOCUMENTED_IN",section="Section 4 Incident 1")
    G.add_edge("INC-2023-007","DOC-PUMP-MANUAL",relation="DOCUMENTED_IN",section="Section 4 Incident 2")
    G.add_edge("INC-2024-006","DOC-PUMP-MANUAL",relation="DOCUMENTED_IN",section="Section 4 Incident 3")
    G.add_edge("INC-2024-007","DOC-IIR-2024-007",relation="DOCUMENTED_IN",section="Full RCA Report")
    G.add_edge("INC-2022-004","DOC-LESSONS",relation="ANALYSED_IN",pattern="Pattern 1 — Seal failures")
    G.add_edge("INC-2023-007","DOC-LESSONS",relation="ANALYSED_IN",pattern="Pattern 2 — Maintenance deferrals")
    G.add_edge("INC-2024-007","DOC-LESSONS",relation="ANALYSED_IN",pattern="Pattern 1 + 2 + 3 — Compound failure")

    G.add_edge("INC-2022-004","CAPA-2022-001",relation="LED_TO_CAPA")
    G.add_edge("INC-2022-004","CAPA-2022-002",relation="LED_TO_CAPA")
    G.add_edge("INC-2024-007","CAPA-2024-001",relation="LED_TO_CAPA")
    G.add_edge("INC-2024-007","CAPA-2024-002",relation="LED_TO_CAPA")
    G.add_edge("INC-2024-007","CAPA-2024-003",relation="LED_TO_CAPA")
    G.add_edge("INC-2024-007","CAPA-2024-005",relation="LED_TO_CAPA")

    G.add_edge("INC-2022-004","INC-2024-007",relation="SIMILAR_TO",
               reason="Both H2S releases from barrier fluid system failure on H2S duty crude pumps")
    G.add_edge("INC-2023-007","INC-2024-007",relation="SIMILAR_TO",
               reason="Both involved deferred maintenance work orders on rotating equipment")

    G.add_edge("EQ-P101A","SP-SEL-101",relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=8)
    G.add_edge("EQ-P101A","SP-IMP-101",relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=10)
    G.add_edge("EQ-P101A","SP-BRG-DE", relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=2)
    G.add_edge("EQ-P101A","SP-BRG-NDE",relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=2)
    G.add_edge("EQ-P101A","SP-SHAFT",  relation="REQUIRES_PART",criticality="CRITICAL",lead_time_weeks=12)
    G.add_edge("EQ-P101B","SP-SEL-101",relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=8)
    G.add_edge("EQ-P101B","SP-SHAFT",  relation="REQUIRES_PART",criticality="CRITICAL",lead_time_weeks=12)

    G.add_edge("EQ-P101A","EQ-E101",relation="FEEDS_INTO",description="Crude feed to pre-heater")
    G.add_edge("EQ-P101B","EQ-E101",relation="FEEDS_INTO",description="Crude feed to pre-heater (standby)")
    G.add_edge("EQ-E101","EQ-T101", relation="FEEDS_INTO",description="Pre-heated crude to distillation column")
    G.add_edge("EQ-M101A","EQ-P101A",relation="DRIVES",description="6.6kV motor drives crude feed pump")

    G.add_edge("DOC-PTW","REG-OISD-105",relation="COMPLIES_WITH")
    G.add_edge("DOC-PTW","REG-FA-S36",  relation="COMPLIES_WITH")
    G.add_edge("DOC-PTW","REG-FA-S38",  relation="COMPLIES_WITH")

    G.add_edge("FIND-HYDRANT","REG-OISD-116",  relation="VIOLATES",clause="Clause 7.2")
    G.add_edge("FIND-H2SDETECT","REG-FA-S36",  relation="VIOLATES",clause="Section 36")
    G.add_edge("FIND-PTWTRAIN","REG-OISD-105",  relation="VIOLATES",clause="Clause 4.3")
    G.add_edge("FIND-MSDS","REG-FA-S41B",       relation="VIOLATES",clause="Section 41B")
    G.add_edge("FIND-GUARD","REG-FA-S21",       relation="VIOLATES",clause="Section 21")
    G.add_edge("FIND-TANKDRAIN","REG-OISD-117", relation="VIOLATES")

    G.add_edge("FIND-GUARD","EQ-P103A",relation="FOUND_ON")
    G.add_edge("FIND-TANKDRAIN","EQ-T104",relation="FOUND_ON")

    for fid in ["FIND-HYDRANT","FIND-H2SDETECT","FIND-PTWTRAIN","FIND-MSDS","FIND-GUARD","FIND-TANKDRAIN"]:
        G.add_edge("DOC-COMPLIANCE",fid,relation="CONTAINS_FINDING")

    log.info(f"Knowledge graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G



def save_graph(G: nx.DiGraph, path: str = GRAPH_PATH):
    with open(path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Graph saved to {path}")

def load_graph(path: str = GRAPH_PATH) -> nx.DiGraph:
    if Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    log.info("No saved graph found — building fresh")
    G = build_graph()
    save_graph(G, path)
    return G



def get_equipment_context(G: nx.DiGraph, tag: str) -> dict:
    """
    Returns full context for an equipment tag — specs, incidents, documents,
    spare parts, connected equipment, compliance findings.
    """
    node_id = None
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") == "EQUIPMENT" and attrs.get("tag","").upper() == tag.upper():
            node_id = nid
            break
    if not node_id:
        return {"found": False, "tag": tag}

    node_data = dict(G.nodes[node_id])
    result = {"found": True, "equipment": node_data, "tag": tag}

    incidents, documents, spare_parts, capas, locations, connected_eq = [], [], [], [], [], []

    for _, dst, edge_data in G.out_edges(node_id, data=True):
        rel = edge_data.get("relation","")
        dst_data = dict(G.nodes[dst])
        dtype = dst_data.get("type","")
        if dtype == "DOCUMENT":   documents.append({**dst_data, "relation": rel, **edge_data})
        if dtype == "SPARE_PART": spare_parts.append({**dst_data, "relation": rel, **edge_data})
        if dtype == "LOCATION":   locations.append({**dst_data, "relation": rel})
        if dtype == "EQUIPMENT":  connected_eq.append({**dst_data, "relation": rel, **edge_data})

    for src, _, edge_data in G.in_edges(node_id, data=True):
        rel = edge_data.get("relation","")
        src_data = dict(G.nodes[src])
        dtype = src_data.get("type","")
        if dtype == "INCIDENT": incidents.append({**src_data, "relation": rel})
        if dtype == "FINDING":  result.setdefault("findings",[]).append({**src_data, "relation": rel})

    for inc in incidents:
        inc_id = next((nid for nid, d in G.nodes(data=True)
                       if d.get("id") == inc.get("id")), None)
        if inc_id:
            for _, dst, ed in G.out_edges(inc_id, data=True):
                if G.nodes[dst].get("type") == "CAPA":
                    capas.append(dict(G.nodes[dst]))

    result["incidents"]     = incidents
    result["documents"]     = documents
    result["spare_parts"]   = spare_parts
    result["capas"]         = capas
    result["locations"]     = locations
    result["connected_equipment"] = connected_eq

    return result


def get_similar_incidents(G: nx.DiGraph, incident_id: str) -> list:
    """Returns incidents linked via SIMILAR_TO relationship."""
    similar = []
    for nid, data in G.nodes(data=True):
        if data.get("id") == incident_id:
            for _, dst, ed in G.out_edges(nid, data=True):
                if ed.get("relation") == "SIMILAR_TO":
                    similar.append({**dict(G.nodes[dst]), "similarity_reason": ed.get("reason","")})
            for src, _, ed in G.in_edges(nid, data=True):
                if ed.get("relation") == "SIMILAR_TO":
                    similar.append({**dict(G.nodes[src]), "similarity_reason": ed.get("reason","")})
    return similar


def get_compliance_context(G: nx.DiGraph) -> dict:
    """Returns all compliance findings grouped by severity."""
    findings = {"CRITICAL":[], "HIGH":[], "MEDIUM":[], "LOW":[]}
    for nid, data in G.nodes(data=True):
        if data.get("type") == "FINDING":
            sev = data.get("severity","MEDIUM")
            entry = dict(data)
            entry["regulations_violated"] = []
            for _, dst, ed in G.out_edges(nid, data=True):
                if ed.get("relation") == "VIOLATES":
                    reg = dict(G.nodes[dst])
                    entry["regulations_violated"].append(f"{reg.get('code','')} — {reg.get('name','')}")
            findings.get(sev, findings["MEDIUM"]).append(entry)
    return findings


def get_critical_equipment(G: nx.DiGraph) -> list:
    """Returns equipment sorted by risk level."""
    order = {"CRITICAL":0,"MEDIUM-HIGH":1,"HIGH":2,"MEDIUM":3,"LOW-MEDIUM":4,"LOW":5}
    equipment = []
    for nid, data in G.nodes(data=True):
        if data.get("type") == "EQUIPMENT":
            equipment.append(dict(data))
    equipment.sort(key=lambda x: order.get(x.get("risk_level","LOW"), 5))
    return equipment


def get_incident_patterns(G: nx.DiGraph) -> dict:
    """Analyses incident patterns from the graph."""
    incidents = [(nid, data) for nid, data in G.nodes(data=True) if data.get("type") == "INCIDENT"]
    h2s_incidents = [d for _, d in incidents if d.get("h2s_ppm", 0) > 0]
    seal_incidents = [d for _, d in incidents if "seal" in d.get("type_detail","").lower()]
    deferred_incidents = [d for _, d in incidents if "deferred" in d.get("root_cause","").lower()]
    return {
        "total_incidents": len(incidents),
        "h2s_releases": len(h2s_incidents),
        "seal_failures": len(seal_incidents),
        "deferred_maintenance_related": len(deferred_incidents),
        "h2s_incident_details": h2s_incidents,
        "pattern": "All H2S incidents involved barrier fluid system failure on H2S duty crude pumps."
    }


def enrich_query_with_graph(G: nx.DiGraph, question: str) -> str:
    """
    Extracts equipment tags and keywords from a question,
    queries the graph, and returns structured context to prepend to RAG.
    """
    question_upper = question.upper()

    tags_mentioned = []
    for nid, data in G.nodes(data=True):
        if data.get("type") == "EQUIPMENT":
            tag = data.get("tag","")
            if tag and tag.upper() in question_upper:
                tags_mentioned.append(tag)

    graph_context_parts = []

    for tag in tags_mentioned:
        ctx = get_equipment_context(G, tag)
        if ctx.get("found"):
            eq = ctx["equipment"]
            parts = [f"\n[KNOWLEDGE GRAPH — Equipment: {tag}]"]
            parts.append(f"Status: {eq.get('status','Unknown')} | Risk: {eq.get('risk_level','Unknown')} | Criticality: {eq.get('criticality','Unknown')}")
            if eq.get("alert"):
                parts.append(f"⚠️ ACTIVE ALERT: {eq['alert']}")
            if ctx.get("incidents"):
                parts.append(f"Incident history ({len(ctx['incidents'])} events):")
                for inc in ctx["incidents"][:3]:
                    parts.append(f"  - {inc.get('date','')} {inc.get('type_detail','')} | Root cause: {inc.get('root_cause','')[:100]}")
            if ctx.get("spare_parts"):
                critical_parts = [sp for sp in ctx["spare_parts"] if sp.get("critical")]
                if critical_parts:
                    parts.append(f"Critical spares: {', '.join(sp['name'] for sp in critical_parts)}")
            graph_context_parts.append("\n".join(parts))

    compliance_keywords = ["comply","compliance","oisd","factory act","regulation","audit","violation","permit"]
    if any(kw in question.lower() for kw in compliance_keywords):
        findings = get_compliance_context(G)
        critical = findings.get("CRITICAL",[])
        high = findings.get("HIGH",[])
        if critical or high:
            parts = ["\n[KNOWLEDGE GRAPH — Compliance Status]"]
            parts.append(f"Critical gaps: {len(critical)} | High gaps: {len(high)}")
            for f in critical:
                parts.append(f"  🔴 CRITICAL: {f.get('description','')} (Status: {f.get('status','')})")
            for f in high[:2]:
                parts.append(f"  🟠 HIGH: {f.get('description','')} (Status: {f.get('status','')})")
            graph_context_parts.append("\n".join(parts))

    maint_keywords = ["fail","attention","risk","critical","overdue","maintenance","predict"]
    if any(kw in question.lower() for kw in maint_keywords):
        critical_eq = [e for e in get_critical_equipment(G) if e.get("risk_level") in ["CRITICAL","MEDIUM-HIGH"]]
        if critical_eq:
            parts = ["\n[KNOWLEDGE GRAPH — High Risk Equipment]"]
            for eq in critical_eq[:3]:
                parts.append(f"  {eq.get('tag','')} ({eq.get('name','')}): {eq.get('risk_level','')} — {eq.get('alert',eq.get('note',''))[:120]}")
            graph_context_parts.append("\n".join(parts))

    pattern_keywords = ["pattern","trend","recurring","history","lessons","repeat"]
    if any(kw in question.lower() for kw in pattern_keywords):
        patterns = get_incident_patterns(G)
        parts = ["\n[KNOWLEDGE GRAPH — Incident Pattern Analysis]"]
        parts.append(f"Total incidents 2019-2024: {patterns['total_incidents']}")
        parts.append(f"H2S releases: {patterns['h2s_releases']} | Seal failures: {patterns['seal_failures']} | Deferred maintenance related: {patterns['deferred_maintenance_related']}")
        parts.append(f"Key pattern: {patterns['pattern']}")
        graph_context_parts.append("\n".join(parts))

    if graph_context_parts:
        return "\n\nKNOWLEDGE GRAPH CONTEXT (use this to enrich your answer):\n" + "\n".join(graph_context_parts)
    return ""



def graph_to_json(G: nx.DiGraph) -> dict:
    """Exports graph to JSON format for D3.js visualisation."""
    type_colors = {
        "EQUIPMENT":  "#3b82f6",
        "DOCUMENT":   "#10b981",
        "INCIDENT":   "#ef4444",
        "REGULATION": "#8b5cf6",
        "SPARE_PART": "#f59e0b",
        "CAPA":       "#06b6d4",
        "FINDING":    "#f97316",
        "LOCATION":   "#6b7280",
    }
    type_sizes = {
        "EQUIPMENT":20,"DOCUMENT":16,"INCIDENT":18,
        "REGULATION":14,"SPARE_PART":12,"CAPA":12,
        "FINDING":14,"LOCATION":10,
    }

    nodes = []
    for nid, data in G.nodes(data=True):
        ntype = data.get("type","UNKNOWN")
        label = data.get("tag") or data.get("code") or data.get("name","")
        if len(label) > 20: label = label[:18] + "…"
        risk = data.get("risk_level","")
        color = "#ef4444" if risk == "CRITICAL" else "#f59e0b" if "HIGH" in risk else type_colors.get(ntype,"#6b7280")
        nodes.append({
            "id":    nid,
            "label": label,
            "type":  ntype,
            "color": color,
            "size":  type_sizes.get(ntype, 10),
            "data":  {k:str(v) for k,v in data.items()},
        })

    rel_colors = {
        "LOCATED_IN":"#4b5563","MAINTAINED_BY":"#3b82f6","OCCURRED_ON":"#ef4444",
        "LED_TO_CAPA":"#10b981","SIMILAR_TO":"#f59e0b","REQUIRES_PART":"#f59e0b",
        "COMPLIES_WITH":"#8b5cf6","VIOLATES":"#ef4444","FEEDS_INTO":"#06b6d4",
        "DRIVES":"#6366f1","DOCUMENTED_IN":"#10b981","ANALYSED_IN":"#8b5cf6",
        "INSPECTED_IN":"#3b82f6","CONTAINS_FINDING":"#f97316","FOUND_ON":"#f97316",
    }
    links = []
    for src, dst, data in G.edges(data=True):
        rel = data.get("relation","RELATED_TO")
        links.append({
            "source": src,
            "target": dst,
            "relation": rel,
            "color": rel_colors.get(rel,"#374151"),
            "label": rel.replace("_"," "),
        })

    return {"nodes": nodes, "links": links,
            "stats": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    G = build_graph()
    save_graph(G)
    print(f"\nKnowledge Graph built:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"\nNode types:")
    from collections import Counter
    type_counts = Counter(data.get("type","?") for _, data in G.nodes(data=True))
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print(f"\nRelationship types:")
    rel_counts = Counter(data.get("relation","?") for _, _, data in G.edges(data=True))
    for r, c in sorted(rel_counts.items()):
        print(f"  {r}: {c}")

    print("\nTest graph enrichment for 'P-201A failure risk':")
    print(enrich_query_with_graph(G, "What is the failure risk for P-201A?"))
