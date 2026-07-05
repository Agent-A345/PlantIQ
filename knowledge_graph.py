"""
PS8 Industrial Knowledge Intelligence, Knowledge Graph
========================================================
Builds and queries a rich industrial knowledge graph using NetworkX.

Node types:
  EQUIPMENT   , physical assets (pumps, vessels, motors)
  DOCUMENT    , source documents (manuals, procedures, reports)
  INCIDENT    , failure events, near-misses, H2S releases
  INSPECTION  , condition monitoring records
  REGULATION  , OISD standards, Factory Act sections
  SPARE_PART  , critical spare parts with lead times
  PERSON      , roles (Safety Manager, Maintenance Superintendent)
  LOCATION    , physical areas (CDU Pump Bay, Unit 3)
  CAPA        , corrective and preventive actions

Relationship types:
  MAINTAINS        , equipment → document (manual)
  FAILED_IN        , incident → equipment
  CAUSED_BY        , incident → root cause (another node)
  PRECEDED_BY      , incident → warning signs
  COMPLIES_WITH    , document → regulation
  VIOLATES         , finding → regulation
  LOCATED_IN       , equipment → location
  REQUIRES_PART    , equipment → spare_part
  INVESTIGATED_BY  , incident → person/role
  LED_TO_CAPA      , incident → capa
  PREVENTS         , capa → incident type
  MONITORS         , inspection → equipment
  CONNECTED_TO     , equipment → equipment (process connections)
  REFERENCES       , document → document
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
                      "oil_iron_ppm":6,"bearing_overdue_hours":0,
                      "vibration_mm_s_history":[(180,1.7),(90,1.8),(30,1.85)],
                      "bearing_temp_c_history":[(180,56),(90,57),(30,57.5)],
                      "oil_iron_ppm_history":[(180,4),(90,5),(30,5.5)],
                      "risk_level":"LOW","criticality":"HIGH"}),
        ("EQ-P101B", {"type":"EQUIPMENT","tag":"P-101B","name":"Crude Feed Pump B (Standby)","make":"Flowserve","model":"DVMX 3x4x13",
                      "duty":"Standby","status":"Standby","power_kw":315,"fluid":"Arabian Light Crude",
                      "h2s_service":True,"atex_class":"Zone 1 IIB T3","vibration_mm_s":2.4,"bearing_temp_c":69,
                      "oil_change_hours":4000,"oil_change_due":"OVERDUE","seal_type":"API Plan 53B",
                      "oil_iron_ppm":15,"bearing_overdue_hours":0,
                      "vibration_mm_s_history":[(180,1.8),(90,2.1),(30,2.3)],
                      "bearing_temp_c_history":[(180,60),(90,65),(30,68)],
                      "oil_iron_ppm_history":[(180,8),(90,11),(30,13)],
                      "risk_level":"MEDIUM-HIGH","criticality":"HIGH",
                      "alert":"NDE bearing temp elevated 69C, oil overdue, minor seal weep, WO-2024-4521 pending"}),
        ("EQ-P102A", {"type":"EQUIPMENT","tag":"P-102A","name":"Atmospheric Residue Pump A","make":"Flowserve",
                      "duty":"Continuous","status":"Running, post incident monitoring","power_kw":132,
                      "h2s_service":True,"vibration_mm_s":1.7,"bearing_temp_c":55,
                      "oil_iron_ppm":8,"bearing_overdue_hours":0,
                      "vibration_mm_s_history":[(180,1.5),(90,1.6),(30,1.65)],
                      "bearing_temp_c_history":[(180,53),(90,54),(30,54.5)],
                      "oil_iron_ppm_history":[(180,6),(90,7),(30,7.5)],
                      "risk_level":"LOW","criticality":"HIGH","last_overhaul":"Oct 2024",
                      "note":"Returned to service after H2S incident Sep 2024, increased monitoring until Jan 2025"}),
        ("EQ-P102B", {"type":"EQUIPMENT","tag":"P-102B","name":"Atmospheric Residue Pump B","make":"Flowserve",
                      "duty":"Continuous","status":"Running","power_kw":132,
                      "h2s_service":True,"vibration_mm_s":2.1,"bearing_temp_c":63,
                      "oil_iron_ppm":7,"bearing_overdue_hours":0,
                      "vibration_mm_s_history":[(180,1.8),(90,1.95),(30,2.05)],
                      "bearing_temp_c_history":[(180,58),(90,60),(30,62)],
                      "oil_iron_ppm_history":[(180,5),(90,6),(30,6.5)],
                      "risk_level":"LOW","criticality":"HIGH","bearing_upgrade":"Labyrinth seal Jan 2024"}),
        ("EQ-P103A", {"type":"EQUIPMENT","tag":"P-103A","name":"Kerosene Product Pump A","make":"Flowserve",
                      "duty":"Continuous","status":"Running","power_kw":37,
                      "h2s_service":False,"vibration_mm_s":1.4,"bearing_temp_c":52,
                      "oil_iron_ppm":5,"bearing_overdue_hours":0,
                      "vibration_mm_s_history":[(180,1.2),(90,1.3),(30,1.35)],
                      "bearing_temp_c_history":[(180,49),(90,50),(30,51)],
                      "oil_iron_ppm_history":[(180,3),(90,4),(30,4.5)],
                      "risk_level":"LOW","criticality":"MEDIUM"}),
        ("EQ-P103B", {"type":"EQUIPMENT","tag":"P-103B","name":"Kerosene Product Pump B","make":"Flowserve",
                      "duty":"Continuous","status":"Running, vibration trending up","power_kw":37,
                      "h2s_service":False,"vibration_mm_s":2.2,"bearing_temp_c":64,
                      "oil_iron_ppm":12,"bearing_overdue_hours":0,
                      "vibration_mm_s_history":[(270,1.6),(180,1.85),(90,2.05)],
                      "bearing_temp_c_history":[(270,56),(180,59),(90,62)],
                      "oil_iron_ppm_history":[(270,7),(180,9),(90,10.5)],
                      "risk_level":"LOW-MEDIUM","criticality":"MEDIUM",
                      "alert":"Vibration up 37% in 9 months. Iron 12ppm in oil. Recommend quarterly monitoring."}),
        ("EQ-P201A", {"type":"EQUIPMENT","tag":"P-201A","name":"Vacuum Residue Transfer Pump A","make":"Flowserve",
                      "duty":"Continuous","status":"CRITICAL, immediate action required","power_kw":132,
                      "h2s_service":True,"vibration_mm_s":4.8,"bearing_temp_c":81,
                      "oil_iron_ppm":48,"bearing_overdue_hours":26000,
                      "vibration_mm_s_history":[(180,2.5),(90,3.4),(30,4.2)],
                      "bearing_temp_c_history":[(180,68),(90,74),(30,78)],
                      "oil_iron_ppm_history":[(180,15),(90,28),(30,38)],
                      "risk_level":"CRITICAL","criticality":"HIGH",
                      "alert":"Vibration 4.8mm/s (above 3.5 alert). Bearing temp 81C (above 75 limit). Oil iron 48ppm. Bearing overdue 26,000hrs. TAKE OUT OF SERVICE WITHIN 7 DAYS.",
                      "overdue_wos":"WO-2024-1847 (bearing), WO-2024-2156 (seal), both deferred 3x"}),
        ("EQ-M101A",  {"type":"EQUIPMENT","tag":"M-101A","name":"Motor for P-101A","power_kw":315,"voltage_kv":6.6,"rpm":1480,
                       "type_detail":"Induction motor","atex_class":"Zone 1 IIB T3"}),
        ("EQ-E101",   {"type":"EQUIPMENT","tag":"E-101","name":"Crude Pre-heater","type_detail":"Shell and tube HX",
                       "last_inspection":"TA 2022","min_thickness_mm":4.2,"corrosion_rate":"0.3mm/yr",
                       "remaining_life_yrs":4,"risk_level":"LOW","criticality":"HIGH"}),
        ("EQ-T101",   {"type":"EQUIPMENT","tag":"T-101","name":"Atmospheric Distillation Column",
                       "last_inspection":"TA 2022","risk_level":"LOW","criticality":"HIGH"}),
        ("EQ-T104",   {"type":"EQUIPMENT","tag":"T-104","name":"Crude Storage Tank","type_detail":"Floating roof",
                       "recent_finding":"Roof drain valve found open Oct 2024, closed immediately",
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
                               "classification":"HIGH POTENTIAL, H2S Release","equipment":"P-102A"}),
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
                          "equipment":"P-101A","type_detail":"H2S Release, Mechanical Seal Failure",
                          "h2s_ppm":3.5,"injury":False,"production_loss":False,
                          "root_cause":"Barrier fluid pressure loss, PRV-101 stuck open. Monthly checks skipped 2 months.",
                          "classification":"SIGNIFICANT","severity":"HIGH"}),
        ("INC-2023-007", {"type":"INCIDENT","id":"INC-2023-007","date":"Nov 2023",
                          "equipment":"P-102B","type_detail":"Bearing Failure, Shaft Seizure",
                          "h2s_ppm":0,"injury":False,"production_loss_hrs":4.2,
                          "root_cause":"Water contamination in oil. Oil analysis alert ignored for 6 weeks. WO deferred twice.",
                          "classification":"SIGNIFICANT","severity":"MEDIUM"}),
        ("INC-2024-006", {"type":"INCIDENT","id":"INC-2024-006","date":"Jul 2024",
                          "equipment":"P-101A","type_detail":"High Vibration, Planned Shutdown",
                          "h2s_ppm":0,"injury":False,"production_loss":False,
                          "root_cause":"Impeller imbalance from erosive wear. Arabian Heavy crude blend not assessed for equipment impact.",
                          "classification":"EQUIPMENT DEGRADATION","severity":"LOW"}),
        ("INC-2024-007", {"type":"INCIDENT","id":"INC-2024-007","date":"Sep 2024",
                          "equipment":"P-102A","type_detail":"H2S Release, Mechanical Seal Failure",
                          "h2s_ppm":12,"injury":True,"injury_detail":"Minor eye irritation, 1 operator",
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
                          "lead_time_weeks":8,"min_stock":2,"current_stock":1,"location":"Store A-3",
                          "critical":True,"unit_cost_inr":185000,"reorder_point":2,
                          "vendor":"Flowserve India","category":"Rotating"}),
        ("SP-IMP-101",  {"type":"SPARE_PART","part_no":"FC-IMP-101-330","name":"Impeller 330mm",
                          "lead_time_weeks":10,"min_stock":1,"current_stock":0,"location":"Store A-3",
                          "critical":True,"unit_cost_inr":95000,"reorder_point":1,
                          "vendor":"Flowserve India","category":"Rotating"}),
        ("SP-BRG-DE",   {"type":"SPARE_PART","part_no":"SKF-7316-BECBM","name":"Bearing Set DE",
                          "lead_time_weeks":2,"min_stock":4,"current_stock":4,"location":"Store B-1",
                          "critical":False,"unit_cost_inr":22000,"reorder_point":2,
                          "vendor":"SKF India","category":"Bearings"}),
        ("SP-BRG-NDE",  {"type":"SPARE_PART","part_no":"SKF-NU316-ECM","name":"Bearing Set NDE",
                          "lead_time_weeks":2,"min_stock":4,"current_stock":3,"location":"Store B-1",
                          "critical":False,"unit_cost_inr":18000,"reorder_point":2,
                          "vendor":"SKF India","category":"Bearings"}),
        ("SP-SHAFT",    {"type":"SPARE_PART","part_no":"FC-SHAFT-101","name":"Shaft 4140 Alloy",
                          "lead_time_weeks":12,"min_stock":1,"current_stock":1,
                          "location":"Warehouse Rack B7 Bay 3","critical":True,
                          "unit_cost_inr":340000,"reorder_point":1,
                          "vendor":"Bharat Forge","category":"Rotating",
                          "note":"DO NOT ISSUE without Materials Manager approval Form SPR-ISSUE-001"}),
        ("SP-OIL-101",  {"type":"SPARE_PART","part_no":"SHELL-TELLUS-46","name":"Hydraulic Oil 46 (200L drum)",
                          "lead_time_weeks":1,"min_stock":4,"current_stock":2,"location":"Lube Store",
                          "critical":False,"unit_cost_inr":12000,"reorder_point":4,
                          "vendor":"Shell India","category":"Consumables"}),
        ("SP-GLAND-101",{"type":"SPARE_PART","part_no":"FC-GLAND-101","name":"Gland Packing Set",
                          "lead_time_weeks":3,"min_stock":3,"current_stock":1,"location":"Store A-3",
                          "critical":True,"unit_cost_inr":8500,"reorder_point":3,
                          "vendor":"Garlock India","category":"Sealing"}),
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
                            "action":"Alarm audit CDU, identify all standing alarms >3 activations/day","owner":"Process Control Engr","due":"Oct 2024"}),
        ("CAPA-2024-003", {"type":"CAPA","id":"CAPA-2024-003","status":"COMPLETED",
                            "action":"Revise maintenance priority matrix for H2S service equipment","owner":"Maint Manager","due":"Oct 2024"}),
        ("CAPA-2024-005", {"type":"CAPA","id":"CAPA-2024-005","status":"IN PROGRESS",
                            "action":"Install auto barrier fluid replenishment P-101 and P-102 series","owner":"Projects Engr","due":"Dec 2024"}),
    ]
    for node_id, attrs in capas:
        G.add_node(node_id, **attrs)

    findings = [
        ("FIND-HYDRANT",    {"type":"FINDING","severity":"CRITICAL",
                              "description":"8 fire hydrants not tested in CDU",
                              "regulation":"OISD-116 Clause 7.2","category":"Fire Safety",
                              "status":"OVERDUE","due_date":"2024-11-30",
                              "overdue_days":217,"frequency_days":180,
                              "responsible":"Safety Officer","action_required":"Conduct pressure test on all 8 hydrants and submit Form FS-HYD-TEST"}),
        ("FIND-H2SDETECT",  {"type":"FINDING","severity":"CRITICAL",
                              "description":"H2S detector AE-H2S-103 calibration overdue 23 days",
                              "regulation":"Factories Act S.36","category":"Health & Safety",
                              "status":"OVERDUE","due_date":"2026-06-12",
                              "overdue_days":23,"frequency_days":90,
                              "responsible":"Instrument Engineer","action_required":"Calibrate AE-H2S-103 using certified gas standard, update calibration record"}),
        ("FIND-PTWTRAIN",   {"type":"FINDING","severity":"HIGH",
                              "description":"3 Area Authorities PTW refresher overdue",
                              "regulation":"OISD-105","category":"Permit to Work",
                              "status":"OVERDUE","due_date":"2026-06-01",
                              "overdue_days":34,"frequency_days":365,
                              "responsible":"Training Coordinator","action_required":"Schedule PTW refresher training for 3 Area Authorities within 7 days"}),
        ("FIND-MSDS",       {"type":"FINDING","severity":"HIGH",
                              "description":"MSDS for CA-2024-07 not posted at point of use",
                              "regulation":"Factories Act S.41B","category":"Chemical Safety",
                              "status":"OVERDUE","due_date":"2026-06-20",
                              "overdue_days":15,"frequency_days":365,
                              "responsible":"Chemical Store In-charge","action_required":"Print and post MSDS for CA-2024-07 at point of use immediately"}),
        ("FIND-GUARD",      {"type":"FINDING","severity":"HIGH",
                              "description":"Coupling guard missing on P-103B",
                              "regulation":"Factories Act S.21","category":"Mechanical Safety",
                              "status":"OVERDUE","due_date":"2026-07-01",
                              "overdue_days":4,"frequency_days":180,
                              "responsible":"Mechanical Maintenance","action_required":"Fabricate and install coupling guard on P-103B before equipment is returned to service"}),
        ("FIND-TANKDRAIN",  {"type":"FINDING","severity":"MEDIUM",
                              "description":"T-104 roof drain valve found open",
                              "regulation":"OISD-117","category":"Tank Safety",
                              "status":"COMPLETED","due_date":"2024-10-15",
                              "overdue_days":0,"frequency_days":90,
                              "responsible":"Operations","action_required":"Close roof drain valve and inspect for blockage"}),
        ("FIND-FIREAUDIT",  {"type":"FINDING","severity":"HIGH",
                              "description":"Annual fire safety audit due this month",
                              "regulation":"OISD-116 Clause 2.1","category":"Fire Safety",
                              "status":"UPCOMING","due_date":"2026-07-31",
                              "overdue_days":0,"days_remaining":26,"frequency_days":365,
                              "responsible":"Safety Manager","action_required":"Schedule and conduct annual fire safety audit per OISD-116"}),
        ("FIND-SIXMONTH",   {"type":"FINDING","severity":"MEDIUM",
                              "description":"6-month statutory inspection of pressure vessels due",
                              "regulation":"Factories Act S.31","category":"Pressure Vessels",
                              "status":"UPCOMING","due_date":"2026-08-15",
                              "overdue_days":0,"days_remaining":41,"frequency_days":180,
                              "responsible":"Inspection Engineer","action_required":"Schedule IBR inspection for T-100, T-101, T-104 with CCOE-approved inspector"}),
        ("FIND-MOCREVIEW",  {"type":"FINDING","severity":"LOW",
                              "description":"Management of Change review for CDU feed rate increase",
                              "regulation":"OISD Safety Framework Sec 6","category":"Management of Change",
                              "status":"IN_PROGRESS","due_date":"2026-07-20",
                              "overdue_days":0,"days_remaining":15,"frequency_days":None,
                              "responsible":"Process Engineer","action_required":"Complete HAZOP study and obtain approval before implementing feed rate change"}),
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
    G.add_edge("EQ-P101B","DOC-INSPECTION",relation="INSPECTED_IN",result="MEDIUM-HIGH risk, URGENT")
    G.add_edge("EQ-P102A","DOC-INSPECTION",relation="INSPECTED_IN",result="LOW risk, post incident monitoring")
    G.add_edge("EQ-P201A","DOC-INSPECTION",relation="INSPECTED_IN",result="CRITICAL, immediate action required")

    G.add_edge("INC-2022-004","EQ-P101A",relation="OCCURRED_ON")
    G.add_edge("INC-2023-007","EQ-P102B",relation="OCCURRED_ON")
    G.add_edge("INC-2024-006","EQ-P101A",relation="OCCURRED_ON")
    G.add_edge("INC-2024-007","EQ-P102A",relation="OCCURRED_ON")

    G.add_edge("INC-2022-004","DOC-PUMP-MANUAL",relation="DOCUMENTED_IN",section="Section 4 Incident 1")
    G.add_edge("INC-2023-007","DOC-PUMP-MANUAL",relation="DOCUMENTED_IN",section="Section 4 Incident 2")
    G.add_edge("INC-2024-006","DOC-PUMP-MANUAL",relation="DOCUMENTED_IN",section="Section 4 Incident 3")
    G.add_edge("INC-2024-007","DOC-IIR-2024-007",relation="DOCUMENTED_IN",section="Full RCA Report")
    G.add_edge("INC-2022-004","DOC-LESSONS",relation="ANALYSED_IN",pattern="Pattern 1, Seal failures")
    G.add_edge("INC-2023-007","DOC-LESSONS",relation="ANALYSED_IN",pattern="Pattern 2, Maintenance deferrals")
    G.add_edge("INC-2024-007","DOC-LESSONS",relation="ANALYSED_IN",pattern="Pattern 1 + 2 + 3, Compound failure")

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

    G.add_edge("EQ-P101A","SP-SEL-101",  relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=8)
    G.add_edge("EQ-P101A","SP-IMP-101",  relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=10)
    G.add_edge("EQ-P101A","SP-BRG-DE",  relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=2)
    G.add_edge("EQ-P101A","SP-BRG-NDE", relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=2)
    G.add_edge("EQ-P101A","SP-SHAFT",   relation="REQUIRES_PART",criticality="CRITICAL",lead_time_weeks=12)
    G.add_edge("EQ-P101B","SP-SEL-101", relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=8)
    G.add_edge("EQ-P101B","SP-SHAFT",   relation="REQUIRES_PART",criticality="CRITICAL",lead_time_weeks=12)
    G.add_edge("EQ-P201A","SP-SEL-101", relation="REQUIRES_PART",criticality="CRITICAL",lead_time_weeks=8)
    G.add_edge("EQ-P201A","SP-SHAFT",   relation="REQUIRES_PART",criticality="CRITICAL",lead_time_weeks=12)
    G.add_edge("EQ-P201A","SP-BRG-DE",  relation="REQUIRES_PART",criticality="CRITICAL",lead_time_weeks=2)
    G.add_edge("EQ-P201A","SP-GLAND-101",relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=3)
    G.add_edge("EQ-P102A","SP-BRG-DE",  relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=2)
    G.add_edge("EQ-P102A","SP-BRG-NDE", relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=2)
    G.add_edge("EQ-P102A","SP-OIL-101", relation="REQUIRES_PART",criticality="MEDIUM",lead_time_weeks=1)
    G.add_edge("EQ-P102B","SP-BRG-DE",  relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=2)
    G.add_edge("EQ-P102B","SP-OIL-101", relation="REQUIRES_PART",criticality="MEDIUM",lead_time_weeks=1)
    G.add_edge("EQ-P103A","SP-SEL-101", relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=8)
    G.add_edge("EQ-P103A","SP-GLAND-101",relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=3)
    G.add_edge("EQ-P103B","SP-SEL-101", relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=8)
    G.add_edge("EQ-P103B","SP-GLAND-101",relation="REQUIRES_PART",criticality="HIGH",lead_time_weeks=3)

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
    log.info("No saved graph found, building fresh")
    G = build_graph()
    save_graph(G, path)
    return G



def get_equipment_context(G: nx.DiGraph, tag: str) -> dict:
    """
    Returns full context for an equipment tag, specs, incidents, documents,
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

    # Apply the same live computed risk used by the alert banner and health
    # dashboard, so a per-tag Q&A answer never disagrees with what's shown
    # elsewhere in the app. Also attach a degradation forecast if one exists.
    computed_risk, computed_alert, has_telemetry = compute_anomaly_risk(node_data)
    if has_telemetry:
        node_data["risk_level"] = computed_risk
        if computed_alert:
            node_data["alert"] = computed_alert
        elif "alert" in node_data:
            del node_data["alert"]

    forecasts = compute_degradation_forecast(node_data)
    forecast_summary = summarize_forecast(forecasts) if forecasts else None
    if forecast_summary and computed_risk != "CRITICAL":
        node_data["degradation_forecast"] = forecast_summary["text"]

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
                    entry["regulations_violated"].append(f"{reg.get('code','')}, {reg.get('name','')}")
            findings.get(sev, findings["MEDIUM"]).append(entry)
    return findings


# ──────────────────────────────────────────────────────────────
# ANOMALY DETECTION ENGINE
# Threshold-based risk computation from real equipment sensor fields
# (vibration_mm_s, bearing_temp_c, oil_iron_ppm, bearing_overdue_hours,
# oil_change_due), rather than relying only on a pre-set risk_level label.
# This means risk_level and alert text are derived live from actual
# telemetry values, so if a P&ID upload or manual edit changes a sensor
# reading, the computed risk updates automatically without needing anyone
# to manually relabel the equipment as CRITICAL/HIGH/etc.
#
# Equipment without telemetry fields (motors, columns, heat exchangers,
# tanks) fall back to their static risk_level/alert exactly as before,
# this only activates for equipment that actually has sensor data.
# ──────────────────────────────────────────────────────────────

THRESHOLDS = {
    "vibration_mm_s":     {"watch": 2.0,  "alert": 3.5,  "critical": 4.5},
    "bearing_temp_c":     {"watch": 65,   "alert": 75,   "critical": 85},
    "oil_iron_ppm":       {"watch": 10,   "alert": 25,   "critical": 40},
    "bearing_overdue_hours": {"watch": 5000, "alert": 15000, "critical": 24000},
}

_RISK_ORDER = ["LOW", "LOW-MEDIUM", "MEDIUM", "MEDIUM-HIGH", "HIGH", "CRITICAL"]


def _classify_reading(value, param):
    """Returns 'critical', 'alert', 'watch', or None for a single reading against its thresholds."""
    t = THRESHOLDS.get(param)
    if t is None or value is None:
        return None
    if value >= t["critical"]:
        return "critical"
    if value >= t["alert"]:
        return "alert"
    if value >= t["watch"]:
        return "watch"
    return None


# ──────────────────────────────────────────────────────────────
# DEGRADATION TREND FORECASTING
# Extends the current-snapshot anomaly detection above with actual
# forecasting: fits a simple linear trend through each equipment's
# recent sensor history plus its current live reading, then projects
# forward to estimate when it will cross the alert or critical threshold.
#
# History is stored as (days_ago, value) tuples per parameter on each
# equipment node (e.g. vibration_mm_s_history). The live current reading
# (vibration_mm_s, etc.) is always treated as the final, most recent data
# point (days_ago effectively 0), so the trend always reflects the latest
# sensor value even if the history list itself is not updated.
# ──────────────────────────────────────────────────────────────

FORECAST_PARAMS = {
    "vibration_mm_s": ("Vibration", "mm/s"),
    "bearing_temp_c": ("Bearing temperature", "C"),
    "oil_iron_ppm": ("Oil iron content", "ppm"),
}


def _linear_trend_per_day(history, current_value):
    """
    Fits a simple least-squares line through history points (days_ago, value)
    plus the current reading (days_ago=0), and returns the slope in units
    per day. Positive slope means the reading is increasing over time
    (degrading, since higher is worse for all tracked parameters here).
    Returns 0.0 if there isn't enough data to fit a meaningful trend.
    """
    if not history:
        return 0.0
    points = [(-days_ago, value) for days_ago, value in history] + [(0, current_value)]
    n = len(points)
    if n < 2:
        return 0.0
    x_mean = sum(p[0] for p in points) / n
    y_mean = sum(p[1] for p in points) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    denominator = sum((x - x_mean) ** 2 for x in [p[0] for p in points])
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_degradation_forecast(attrs: dict):
    """
    Returns a dict of {param: {trend, rate_per_day, days_to_alert,
    days_to_critical}} for every trackable parameter that has both a
    history and a current reading. Only projects forward (positive days)
    for parameters that are genuinely trending upward and have not yet
    crossed the relevant threshold.
    """
    forecasts = {}
    for param in FORECAST_PARAMS:
        history = attrs.get(f"{param}_history")
        current = attrs.get(param)
        if not history or current is None:
            continue

        slope = _linear_trend_per_day(history, current)
        thresholds = THRESHOLDS.get(param, {})
        alert_th = thresholds.get("alert")
        critical_th = thresholds.get("critical")

        days_to_alert = None
        days_to_critical = None
        if slope > 0.0005:  # genuinely degrading, not flat/noise
            if alert_th is not None and current < alert_th:
                days_to_alert = max(0, round((alert_th - current) / slope))
            if critical_th is not None and current < critical_th:
                days_to_critical = max(0, round((critical_th - current) / slope))
            trend = "degrading"
        elif slope < -0.0005:
            trend = "improving"
        else:
            trend = "stable"

        forecasts[param] = {
            "trend": trend,
            "rate_per_day": round(slope, 4),
            "current_value": current,
            "days_to_alert": days_to_alert,
            "days_to_critical": days_to_critical,
        }
    return forecasts


def summarize_forecast(forecasts: dict):
    """
    Picks the single most urgent upcoming threshold crossing across all
    forecasted parameters for one equipment, and builds a plain-language
    forecast sentence. Returns None if nothing is meaningfully trending
    toward a threshold (equipment is stable or already past all thresholds).
    """
    candidates = []
    for param, f in forecasts.items():
        if f["days_to_critical"] is not None:
            candidates.append((f["days_to_critical"], param, "critical", f))
        elif f["days_to_alert"] is not None:
            candidates.append((f["days_to_alert"], param, "alert", f))
    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    days, param, level, f = candidates[0]
    label, unit = FORECAST_PARAMS[param]

    if days == 0:
        timing = "has already reached"
        text = (f"At the current degradation rate ({f['rate_per_day']:+.3f} {unit} per day), "
                f"{label.lower()} {timing} the {level} threshold.")
    elif days == 1:
        text = (f"At the current degradation rate ({f['rate_per_day']:+.3f} {unit} per day), "
                f"{label.lower()} is projected to reach the {level} threshold in approximately 1 day.")
    else:
        text = (f"At the current degradation rate ({f['rate_per_day']:+.3f} {unit} per day), "
                f"{label.lower()} is projected to reach the {level} threshold in approximately {days} days.")

    return {"days": days, "parameter": label, "level": level, "text": text}




def days_since_threshold_crossed(attrs: dict) -> list:
    """
    For equipment already past a threshold, works backwards using the
    linear trend slope to estimate when each parameter first crossed it.
    Returns a list of dicts sorted by most recent crossing first:
      [{param, label, unit, threshold_level, threshold_value,
        current_value, rate_per_day, days_ago}]
    Only returns params currently above a threshold AND trending upward.
    """
    results = []
    for param in FORECAST_PARAMS:
        history = attrs.get(f"{param}_history")
        current = attrs.get(param)
        if not history or current is None:
            continue

        slope = _linear_trend_per_day(history, current)
        if slope <= 0.0005:
            continue  # not trending up, skip

        thresholds = THRESHOLDS.get(param, {})
        label, unit = FORECAST_PARAMS[param]

        # Check which thresholds have already been crossed (current > threshold)
        for level in ["critical", "alert", "watch"]:
            thr_val = thresholds.get(level)
            if thr_val is None:
                continue
            if current > thr_val:
                # days_ago = (current - threshold) / slope
                days_ago = round((current - thr_val) / slope)
                results.append({
                    "param":            param,
                    "label":            label,
                    "unit":             unit,
                    "threshold_level":  level,
                    "threshold_value":  thr_val,
                    "current_value":    current,
                    "rate_per_day":     round(slope, 4),
                    "days_ago":         days_ago,
                })
                break  # only report the highest crossed threshold per param

    results.sort(key=lambda x: x["days_ago"])  # most recent first
    return results

def get_degradation_forecasts(G: nx.DiGraph) -> list:
    """
    Returns every equipment that has a meaningful degradation forecast
    (something genuinely trending toward a threshold), sorted by urgency,
    soonest threshold crossing first. Equipment that is stable, improving,
    or has no sensor history is excluded from this list entirely.
    """
    results = []
    for nid, data in G.nodes(data=True):
        if data.get("type") != "EQUIPMENT":
            continue

        # Skip equipment that's already CRITICAL overall (via compute_anomaly_risk
        # on its current readings), a forecast into an already-critical item adds
        # no predictive value, it's already flagged by the alert engine
        current_risk, _, has_telemetry = compute_anomaly_risk(data)
        if has_telemetry and current_risk == "CRITICAL":
            continue

        forecasts = compute_degradation_forecast(data)
        if not forecasts:
            continue
        summary = summarize_forecast(forecasts)
        if not summary:
            continue
        results.append({
            "tag": data.get("tag", nid),
            "name": data.get("name", "Unknown equipment"),
            "risk_level": data.get("risk_level", "LOW"),
            "forecast": summary,
            "parameter_forecasts": forecasts,
        })
    results.sort(key=lambda x: x["forecast"]["days"])
    return results

# ──────────────────────────────────────────────────────────────


def compute_anomaly_risk(attrs: dict):
    """
    Computes a dynamic risk_level and alert message from an equipment
    node's numeric sensor fields, comparing each against THRESHOLDS.

    Returns (risk_level, alert_text, has_telemetry). If the equipment has
    no recognised numeric telemetry fields at all, has_telemetry is False
    and the caller should fall back to the equipment's static risk_level.
    """
    readings = {
        "vibration_mm_s": attrs.get("vibration_mm_s"),
        "bearing_temp_c": attrs.get("bearing_temp_c"),
        "oil_iron_ppm": attrs.get("oil_iron_ppm"),
        "bearing_overdue_hours": attrs.get("bearing_overdue_hours"),
    }
    has_telemetry = any(v is not None for v in readings.values())
    if not has_telemetry:
        return None, None, False

    breaches = []      # (param, level, value)
    for param, value in readings.items():
        level = _classify_reading(value, param)
        if level:
            breaches.append((param, level, value))

    oil_overdue = str(attrs.get("oil_change_due", "")).upper() == "OVERDUE"

    critical_count = sum(1 for _, lvl, _ in breaches if lvl == "critical")
    alert_count = sum(1 for _, lvl, _ in breaches if lvl == "alert")
    watch_count = sum(1 for _, lvl, _ in breaches if lvl == "watch")

    # Risk classification from breach severity and count. Multiple simultaneous
    # watch-level readings compound into higher risk even if no single reading
    # crosses the hard alert line, which reflects real industrial practice:
    # several minor concurrent issues are worse than one in isolation.
    if critical_count >= 1 or alert_count >= 3:
        risk_level = "CRITICAL"
    elif alert_count >= 2 or (alert_count >= 1 and oil_overdue) or (watch_count >= 3 and oil_overdue):
        risk_level = "MEDIUM-HIGH"
    elif alert_count >= 1 or (watch_count >= 2 and oil_overdue):
        risk_level = "MEDIUM"
    elif watch_count >= 1 or oil_overdue:
        risk_level = "LOW-MEDIUM"
    else:
        risk_level = "LOW"

    # Build a plain-language alert message describing exactly which readings
    # triggered it, referencing the correct threshold for the band actually crossed
    PARAM_LABELS = {
        "vibration_mm_s": ("Vibration", "mm/s", "{label} is {value} {unit}, above the {threshold} {unit} {band} threshold"),
        "bearing_temp_c": ("Bearing temperature", "C", "{label} is {value}{unit}, above the {threshold}{unit} {band} threshold"),
        "oil_iron_ppm": ("Oil iron content", "ppm", "{label} is {value} {unit}, above the {threshold} {unit} {band} threshold"),
        "bearing_overdue_hours": ("Bearing service", "hours", "{label} is overdue by {value} {unit}, above the {threshold} {unit} {band} threshold"),
    }
    level_order = {"critical": 0, "alert": 1, "watch": 2}
    parts = []
    for param, level, value in sorted(breaches, key=lambda b: level_order[b[1]]):
        label, unit, template = PARAM_LABELS[param]
        threshold_val = THRESHOLDS[param][level]
        parts.append(template.format(label=label, value=value, unit=unit, threshold=threshold_val, band=level))
    if oil_overdue:
        parts.append("Oil change interval is overdue")

    if risk_level == "CRITICAL":
        parts.append("Recommend taking this equipment out of service for inspection")
    elif risk_level in ("MEDIUM-HIGH", "MEDIUM"):
        parts.append("Recommend scheduling inspection soon")

    alert_text = ". ".join(parts) + "." if parts else ""
    return risk_level, alert_text, True


def get_equipment_health_scores(G: nx.DiGraph) -> list:
    """
    Computes a smooth 0-100 health score per equipment, derived from the
    same threshold breaches used by compute_anomaly_risk, but expressed as
    a continuous score rather than discrete risk bands. Useful for a
    dashboard visual (health gauge cards) rather than the alert banner,
    which only needs CRITICAL/MEDIUM-HIGH labels.

    100 = fully healthy, no breaches. Points deducted per breach severity:
    watch -10, alert -25, critical -40, each capped so score never goes
    below 0. Oil overdue deducts a further 5 points.

    Equipment without telemetry fields get a score based on their static
    risk_level instead, so every equipment always has a score to display.
    """
    STATIC_RISK_SCORES = {
        "LOW": 95, "LOW-MEDIUM": 80, "MEDIUM": 65,
        "MEDIUM-HIGH": 45, "HIGH": 30, "CRITICAL": 10,
    }
    DEDUCTIONS = {"watch": 10, "alert": 25, "critical": 40}

    results = []
    for nid, data in G.nodes(data=True):
        if data.get("type") != "EQUIPMENT":
            continue
        entry = dict(data)

        readings = {
            "vibration_mm_s": entry.get("vibration_mm_s"),
            "bearing_temp_c": entry.get("bearing_temp_c"),
            "oil_iron_ppm": entry.get("oil_iron_ppm"),
            "bearing_overdue_hours": entry.get("bearing_overdue_hours"),
        }
        has_telemetry = any(v is not None for v in readings.values())

        if has_telemetry:
            # Score uses diminishing deductions so multiple breaches don't
            # collapse to 0. Primary breach (worst) takes full deduction,
            # each additional breach takes half, so a 4-breach CRITICAL
            # equipment still shows ~15-20 rather than 0.
            breach_scores = []
            for param, value in readings.items():
                level = _classify_reading(value, param)
                if level:
                    breach_scores.append(DEDUCTIONS[level])
            breach_scores.sort(reverse=True)
            total_deduction = 0
            for i, d in enumerate(breach_scores):
                total_deduction += d * (1.0 / (2 ** i))
            if str(entry.get("oil_change_due", "")).upper() == "OVERDUE":
                total_deduction += 5
            score = max(5, min(100, round(100 - total_deduction)))
            computed_risk, computed_alert, _ = compute_anomaly_risk(entry)
            display_risk_level = computed_risk or entry.get("risk_level", "LOW")
            display_alert = computed_alert or ""
            source = "computed_from_telemetry"
        else:
            score = STATIC_RISK_SCORES.get(entry.get("risk_level", "LOW"), 70)
            display_risk_level = entry.get("risk_level", "LOW")
            display_alert = entry.get("alert", "")
            source = "static_label"

        results.append({
            "tag": entry.get("tag", nid),
            "name": entry.get("name", "Unknown equipment"),
            "health_score": score,
            "risk_level": display_risk_level,
            "criticality": entry.get("criticality", "MEDIUM"),
            "risk_source": source,
            "alert": display_alert,
        })

    results.sort(key=lambda x: x["health_score"])
    return results


def get_critical_equipment(G: nx.DiGraph) -> list:
    """
    Returns equipment sorted by risk level. Risk level and alert text are
    computed live from real sensor thresholds (see compute_anomaly_risk)
    for any equipment with telemetry data. Equipment without telemetry
    (motors, columns, heat exchangers, tanks) keeps its static risk_level
    and alert fields exactly as authored.
    """
    order = {"CRITICAL":0,"MEDIUM-HIGH":1,"HIGH":2,"MEDIUM":3,"LOW-MEDIUM":4,"LOW":5}
    equipment = []
    for nid, data in G.nodes(data=True):
        if data.get("type") == "EQUIPMENT":
            entry = dict(data)
            computed_risk, computed_alert, has_telemetry = compute_anomaly_risk(entry)
            if has_telemetry:
                entry["risk_level"] = computed_risk
                if computed_alert:
                    entry["alert"] = computed_alert
                elif "alert" in entry:
                    del entry["alert"]  # clear a stale hand-written alert if telemetry now reads healthy
                entry["risk_source"] = "computed_from_telemetry"
            else:
                entry["risk_source"] = "static_label"
            equipment.append(entry)
    equipment.sort(key=lambda x: order.get(x.get("risk_level","LOW"), 5))
    return equipment

# ──────────────────────────────────────────────────────────────


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
            parts = [f"\nEquipment reference data for {tag} (use this information naturally in your answer, do not quote this heading):"]
            parts.append(f"Status: {eq.get('status','Unknown')} | Risk: {eq.get('risk_level','Unknown')} | Criticality: {eq.get('criticality','Unknown')}")
            if eq.get("alert"):
                parts.append(f"Active alert: {eq['alert']}")
            if eq.get("degradation_forecast"):
                parts.append(f"Degradation forecast: {eq['degradation_forecast']}")
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
            parts = ["\nCompliance status reference data (use this information naturally in your answer, do not quote this heading):"]
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
            parts = ["\nHigh risk equipment reference data (use this information naturally in your answer, do not quote this heading):"]
            for eq in critical_eq[:3]:
                parts.append(f"  {eq.get('tag','')} ({eq.get('name','')}): {eq.get('risk_level','')}, {eq.get('alert',eq.get('note',''))[:120]}")
            graph_context_parts.append("\n".join(parts))

    pattern_keywords = ["pattern","trend","recurring","history","lessons","repeat"]
    if any(kw in question.lower() for kw in pattern_keywords):
        patterns = get_incident_patterns(G)
        parts = ["\nIncident pattern reference data (use this information naturally in your answer, do not quote this heading):"]
        parts.append(f"Total incidents 2019-2024: {patterns['total_incidents']}")
        parts.append(f"H2S releases: {patterns['h2s_releases']} | Seal failures: {patterns['seal_failures']} | Deferred maintenance related: {patterns['deferred_maintenance_related']}")
        parts.append(f"Key pattern: {patterns['pattern']}")
        graph_context_parts.append("\n".join(parts))

    if graph_context_parts:
        return ("\n\nKNOWLEDGE GRAPH CONTEXT (this is reference data to inform your answer. "
                "Use the information naturally in flowing prose. Do not quote any of these headings "
                "or labels verbatim, and do not repeat 'Knowledge Graph' multiple times in your answer body. "
                "Mention 'Knowledge Graph' only once, in your final SOURCES section):\n"
                + "\n".join(graph_context_parts))
    return ""





# ──────────────────────────────────────────────────────────────
# MAINTENANCE WORK ORDER PRIORITIZATION ENGINE
# Synthesizes all signals (anomaly risk, degradation forecast,
# compliance findings, incident history, spare part availability,
# regulatory deadlines) into a ranked daily action list.
# This is what a maintenance planner does every morning — PlantIQ
# automates it into a scored, explainable priority queue.
#
# Scoring model (0-100 per action):
#   Base risk score      : CRITICAL=60, MEDIUM-HIGH=40, MEDIUM=25, else 10
#   Degradation urgency  : +25 if threshold in <14d, +15 if <30d, +8 if <90d
#   Compliance penalty   : +20 if CRITICAL finding, +10 if HIGH finding
#   Overdue WOs          : +15 if bearing/seal WOs deferred
#   H2S service          : +10 if equipment is in H2S duty service
#   Incident recency     : +8 if incident on this equipment in last 12 months
#   Spare part risk      : +5 if critical spare lead time > 8 weeks
#   Oil overdue          : +5 if oil change interval is overdue
# Score capped at 100. Ties broken by equipment tag alphabetically.
# ──────────────────────────────────────────────────────────────

def get_work_order_priorities(G: nx.DiGraph) -> list:
    """
    Returns a ranked list of maintenance actions for the current shift,
    each with a composite urgency score, rationale, and recommended action.
    Covers all equipment with any risk signal, not just CRITICAL.
    """
    # Build incident lookup by equipment tag (last incident date)
    recent_incident_tags = set()
    for nid, data in G.nodes(data=True):
        if data.get("type") != "INCIDENT":
            continue
        date_str = data.get("date", "")
        # Any 2024 or later incident counts as recent (within ~12 months of demo data)
        if "2024" in date_str or "2025" in date_str or "2026" in date_str:
            eq_tag = data.get("equipment", "")
            if eq_tag:
                recent_incident_tags.add(eq_tag)

    # Build compliance finding lookup by equipment tag
    compliance_by_tag = {}  # tag -> highest severity finding
    for nid, data in G.nodes(data=True):
        if data.get("type") != "FINDING":
            continue
        sev = data.get("severity", "MEDIUM")
        # Check if this finding is linked to specific equipment
        for _, dst, ed in G.out_edges(nid, data=True):
            dst_data = G.nodes[dst]
            if dst_data.get("type") == "EQUIPMENT":
                tag = dst_data.get("tag", "")
                if tag:
                    existing = compliance_by_tag.get(tag, "LOW")
                    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                    if order.index(sev) > order.index(existing):
                        compliance_by_tag[tag] = sev

    # Build spare part lead time lookup by equipment tag
    spare_lead_by_tag = {}  # tag -> max lead time weeks for critical spares
    for nid, data in G.nodes(data=True):
        if data.get("type") != "SPARE_PART":
            continue
        lead = data.get("lead_time_weeks", 0)
        is_critical = data.get("critical", False)
        if not is_critical or lead <= 8:
            continue
        # Find equipment that requires this spare
        for src, _, ed in G.in_edges(nid, data=True):
            if ed.get("relation") == "REQUIRES_PART":
                src_data = G.nodes[src]
                tag = src_data.get("tag", "")
                if tag:
                    spare_lead_by_tag[tag] = max(spare_lead_by_tag.get(tag, 0), lead)

    priorities = []

    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue

        tag  = attrs.get("tag", nid)
        name = attrs.get("name", "Unknown")

        # Compute live risk
        computed_risk, computed_alert, has_telemetry = compute_anomaly_risk(attrs)
        risk = computed_risk if has_telemetry else attrs.get("risk_level", "LOW")
        alert_text = computed_alert if has_telemetry else attrs.get("alert", "")

        # Skip genuinely low-risk equipment with no signals at all
        if risk in ("LOW", "UNKNOWN") and tag not in recent_incident_tags and tag not in compliance_by_tag:
            continue

        # ── Scoring ──────────────────────────────────────────────
        score = 0
        reasons = []
        action_parts = []

        # Base risk
        base = {"CRITICAL": 60, "HIGH": 50, "MEDIUM-HIGH": 40, "MEDIUM": 25, "LOW-MEDIUM": 12}.get(risk, 5)
        score += base
        if risk in ("CRITICAL", "HIGH", "MEDIUM-HIGH"):
            reasons.append(f"{risk} risk level from live sensor data" if has_telemetry else f"{risk} risk classification")

        # Degradation urgency
        fc = compute_degradation_forecast(attrs)
        forecast_summary = summarize_forecast(fc) if fc else None
        days_to_threshold = forecast_summary["days"] if forecast_summary else None
        if days_to_threshold is not None:
            if days_to_threshold < 14:
                score += 25
                reasons.append(f"threshold breach projected in {days_to_threshold} days")
                action_parts.append(f"Schedule immediate inspection — {forecast_summary['parameter']} crossing {forecast_summary['level']} threshold in {days_to_threshold} days")
            elif days_to_threshold < 30:
                score += 15
                reasons.append(f"threshold breach in ~{days_to_threshold} days")
                action_parts.append(f"Plan inspection within 2 weeks — {forecast_summary['parameter']} trend")
            elif days_to_threshold < 90:
                score += 8
                reasons.append(f"degradation trend, threshold in ~{days_to_threshold} days")

        # Compliance finding linked to this equipment
        comp_sev = compliance_by_tag.get(tag)
        if comp_sev == "CRITICAL":
            score += 20
            reasons.append("CRITICAL compliance finding linked to this equipment")
            action_parts.append("Resolve linked CRITICAL compliance finding immediately")
        elif comp_sev == "HIGH":
            score += 10
            reasons.append("HIGH compliance finding linked to this equipment")

        # Overdue work orders
        overdue_wos = attrs.get("overdue_wos", "")
        if overdue_wos:
            score += 15
            reasons.append(f"deferred WOs: {overdue_wos[:80]}")
            action_parts.append(f"Clear deferred work orders: {overdue_wos[:60]}")

        # H2S service
        if attrs.get("h2s_service"):
            score += 10
            reasons.append("H2S duty service — elevated consequence of failure")

        # Recent incident
        if tag in recent_incident_tags:
            score += 8
            reasons.append("incident on this equipment in last 12 months")

        # Long lead critical spare
        lead = spare_lead_by_tag.get(tag, 0)
        if lead > 8:
            score += 5
            reasons.append(f"critical spare lead time {lead} weeks — order early")
            action_parts.append(f"Verify critical spare stock (lead time {lead} weeks)")

        # Oil overdue
        if str(attrs.get("oil_change_due", "")).upper() == "OVERDUE":
            score += 5
            reasons.append("oil change interval overdue")
            action_parts.append("Complete overdue oil change")

        score = min(100, score)

        # ── Recommended action ────────────────────────────────────
        if not action_parts:
            if risk == "CRITICAL":
                action_parts = ["Take out of service for immediate inspection"]
            elif risk in ("HIGH", "MEDIUM-HIGH"):
                action_parts = ["Schedule inspection within 48 hours"]
            elif risk == "MEDIUM":
                action_parts = ["Monitor closely, schedule inspection this week"]
            else:
                action_parts = ["Continue enhanced monitoring"]

        # Primary action = highest-priority action_part
        recommended_action = action_parts[0]

        # Urgency tier label
        if score >= 75:
            urgency = "IMMEDIATE"
        elif score >= 50:
            urgency = "THIS SHIFT"
        elif score >= 30:
            urgency = "THIS WEEK"
        else:
            urgency = "MONITOR"

        priorities.append({
            "rank":               0,  # filled after sort
            "tag":                tag,
            "name":               name,
            "risk_level":         risk,
            "urgency":            urgency,
            "score":              score,
            "recommended_action": recommended_action,
            "reasons":            reasons,
            "alert":              alert_text,
            "has_telemetry":      has_telemetry,
            "days_to_threshold":  days_to_threshold,
            "forecast_text":      forecast_summary["text"] if forecast_summary else None,
        })

    # Sort by score descending, then tag ascending for stable ties
    priorities.sort(key=lambda x: (-x["score"], x["tag"]))
    for i, p in enumerate(priorities):
        p["rank"] = i + 1

    return priorities



# ──────────────────────────────────────────────────────────────
# WHAT-IF SIMULATOR
# Lets users hypothetically change sensor readings or defer
# maintenance on any equipment and instantly see the projected
# impact: new risk level, new health score, updated degradation
# forecast, compliance implications, and cascade effects on
# connected equipment — all computed from the same threshold
# and scoring logic used by the live anomaly engine.
# Zero LLM calls, zero API calls — pure graph math.
# ──────────────────────────────────────────────────────────────

def simulate_what_if(G: nx.DiGraph, tag: str, overrides: dict) -> dict:
    """
    Simulates a hypothetical scenario for one equipment tag.

    Parameters
    ----------
    G : nx.DiGraph
        The live knowledge graph.
    tag : str
        Equipment tag to simulate (e.g. "P-201A").
    overrides : dict
        Sensor or state overrides to apply, e.g.:
          {"vibration_mm_s": 5.5}
          {"bearing_temp_c": 90}
          {"defer_days": 30}          # project all readings forward N days
          {"oil_change_due": "OVERDUE"}

    Returns
    -------
    dict with keys:
      tag, name,
      baseline  : {risk_level, health_score, alert, forecast_text, forecast_days}
      simulated : {risk_level, health_score, alert, forecast_text, forecast_days}
      delta     : {risk_changed, score_delta, risk_escalated}
      cascade   : list of connected equipment that may be affected
      insights  : list of plain-language insight strings
      overrides_applied : the overrides dict actually used
    """
    # Find the equipment node
    target_node = None
    target_attrs = None
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") == "EQUIPMENT" and attrs.get("tag") == tag:
            target_node = nid
            target_attrs = dict(attrs)
            break

    if target_node is None:
        return {"error": f"Equipment tag {tag} not found in knowledge graph"}

    name = target_attrs.get("name", tag)

    # ── Baseline (current state) ──────────────────────────────
    baseline_risk, baseline_alert, has_telemetry = compute_anomaly_risk(target_attrs)
    if not has_telemetry:
        baseline_risk = target_attrs.get("risk_level", "LOW")
        baseline_alert = target_attrs.get("alert", "")

    baseline_fc   = compute_degradation_forecast(target_attrs)
    baseline_summ = summarize_forecast(baseline_fc) if baseline_fc else None

    STATIC_RISK_SCORES = {
        "LOW": 95, "LOW-MEDIUM": 80, "MEDIUM": 65,
        "MEDIUM-HIGH": 45, "HIGH": 30, "CRITICAL": 10, None: 70
    }
    DEDUCTIONS = {"watch": 10, "alert": 25, "critical": 40}

    def _compute_health(attrs_dict):
        readings = {
            "vibration_mm_s":       attrs_dict.get("vibration_mm_s"),
            "bearing_temp_c":       attrs_dict.get("bearing_temp_c"),
            "oil_iron_ppm":         attrs_dict.get("oil_iron_ppm"),
            "bearing_overdue_hours": attrs_dict.get("bearing_overdue_hours"),
        }
        has_tel = any(v is not None for v in readings.values())
        if not has_tel:
            return STATIC_RISK_SCORES.get(attrs_dict.get("risk_level"), 70)
        score = 100
        for param, value in readings.items():
            level = _classify_reading(value, param)
            if level:
                score -= DEDUCTIONS[level]
        if str(attrs_dict.get("oil_change_due", "")).upper() == "OVERDUE":
            score -= 5
        return max(0, min(100, score))

    baseline_score = _compute_health(target_attrs)

    baseline = {
        "risk_level":    baseline_risk,
        "health_score":  baseline_score,
        "alert":         baseline_alert or "",
        "forecast_text": baseline_summ["text"] if baseline_summ else None,
        "forecast_days": baseline_summ["days"] if baseline_summ else None,
    }

    # ── Apply overrides to a copy of attrs ───────────────────
    sim_attrs = dict(target_attrs)
    defer_days = overrides.pop("defer_days", None)

    # Direct sensor overrides
    for key, val in overrides.items():
        sim_attrs[key] = val

    # defer_days: project each history forward and advance current reading
    # using the existing linear trend slope
    if defer_days and defer_days > 0:
        for param in ["vibration_mm_s", "bearing_temp_c", "oil_iron_ppm"]:
            hist = sim_attrs.get(f"{param}_history")
            current = sim_attrs.get(param)
            if hist and current is not None:
                slope = _linear_trend_per_day(hist, current)
                if slope > 0:
                    projected = current + slope * defer_days
                    sim_attrs[param] = round(projected, 3)
        # Also push bearing_overdue_hours forward
        existing_overdue = sim_attrs.get("bearing_overdue_hours", 0) or 0
        sim_attrs["bearing_overdue_hours"] = existing_overdue + (defer_days * 24)
        overrides["defer_days"] = defer_days  # restore for response

    # ── Simulated state ───────────────────────────────────────
    sim_risk, sim_alert, sim_has_tel = compute_anomaly_risk(sim_attrs)
    if not sim_has_tel:
        sim_risk  = sim_attrs.get("risk_level", "LOW")
        sim_alert = sim_attrs.get("alert", "")

    sim_fc   = compute_degradation_forecast(sim_attrs)
    sim_summ = summarize_forecast(sim_fc) if sim_fc else None
    sim_score = _compute_health(sim_attrs)

    simulated = {
        "risk_level":    sim_risk,
        "health_score":  sim_score,
        "alert":         sim_alert or "",
        "forecast_text": sim_summ["text"] if sim_summ else None,
        "forecast_days": sim_summ["days"] if sim_summ else None,
    }

    # ── Delta ─────────────────────────────────────────────────
    RISK_ORDER = {"LOW": 0, "LOW-MEDIUM": 1, "MEDIUM": 2,
                  "MEDIUM-HIGH": 3, "HIGH": 4, "CRITICAL": 5}
    base_ord = RISK_ORDER.get(baseline_risk, 0)
    sim_ord  = RISK_ORDER.get(sim_risk, 0)
    delta = {
        "risk_changed":   baseline_risk != sim_risk,
        "score_delta":    sim_score - baseline_score,
        "risk_escalated": sim_ord > base_ord,
        "risk_improved":  sim_ord < base_ord,
        "levels_changed": sim_ord - base_ord,
    }

    # ── Cascade: connected equipment ──────────────────────────
    cascade = []
    for src, dst, ed in G.out_edges(target_node, data=True):
        rel = ed.get("relation", "")
        if rel in ("FEEDS_INTO", "DRIVES", "CONNECTED_TO"):
            dst_data = G.nodes[dst]
            if dst_data.get("type") == "EQUIPMENT":
                cascade.append({
                    "tag":      dst_data.get("tag", dst),
                    "name":     dst_data.get("name", ""),
                    "relation": rel,
                    "impact":   f"Downstream of {tag} via {rel.replace('_',' ').lower()} — at risk if {tag} fails",
                })
    for src, dst, ed in G.in_edges(target_node, data=True):
        rel = ed.get("relation", "")
        if rel in ("FEEDS_INTO", "DRIVES"):
            src_data = G.nodes[src]
            if src_data.get("type") == "EQUIPMENT":
                cascade.append({
                    "tag":      src_data.get("tag", src),
                    "name":     src_data.get("name", ""),
                    "relation": rel,
                    "impact":   f"Upstream of {tag} via {rel.replace('_',' ').lower()} — feeds into affected equipment",
                })

    # ── Plain language insights ───────────────────────────────
    insights = []

    if delta["risk_escalated"]:
        levels = delta["levels_changed"]
        insights.append(
            f"Risk escalates from {baseline_risk} to {sim_risk} "
            f"({levels} level{'s' if levels > 1 else ''} worse)."
        )
    elif delta["risk_improved"]:
        insights.append(f"Risk improves from {baseline_risk} to {sim_risk}.")
    else:
        insights.append(f"Risk level remains {sim_risk} — no change in risk band.")

    score_d = delta["score_delta"]
    if score_d != 0:
        direction = "drops" if score_d < 0 else "improves"
        insights.append(
            f"Health score {direction} from {baseline_score} to {sim_score} "
            f"({abs(score_d)} point{'s' if abs(score_d) != 1 else ''})."
        )

    if sim_summ and baseline_summ:
        bd, sd = baseline_summ["days"], sim_summ["days"]
        if sd < bd:
            insights.append(
                f"Threshold crossing accelerates: {baseline_summ['parameter']} "
                f"was projected at {bd} days, now projected at {sd} days."
            )
        elif sd > bd:
            insights.append(
                f"Threshold crossing delayed: {baseline_summ['parameter']} "
                f"now projected at {sd} days (was {bd} days)."
            )
    elif sim_summ and not baseline_summ:
        insights.append(
            f"Simulation creates a new degradation concern: "
            f"{sim_summ['parameter']} now projected to reach {sim_summ['level']} "
            f"threshold in {sim_summ['days']} days."
        )
    elif baseline_summ and not sim_summ:
        insights.append("Degradation concern resolved under this scenario.")

    if sim_risk == "CRITICAL" and baseline_risk != "CRITICAL":
        insights.append(
            "Equipment crosses into CRITICAL territory — immediate shutdown "
            "and inspection would be required under this scenario."
        )

    if defer_days:
        insights.append(
            f"Deferring maintenance by {defer_days} days projects sensor readings "
            f"forward along their current trend lines."
        )

    if cascade:
        connected_tags = ", ".join(c["tag"] for c in cascade)
        insights.append(
            f"Connected equipment at risk if {tag} fails: {connected_tags}."
        )

    # Check H2S risk
    if sim_attrs.get("h2s_service") and sim_risk in ("CRITICAL", "HIGH", "MEDIUM-HIGH"):
        insights.append(
            f"{tag} is in H2S duty service — failure under this scenario "
            "carries significant safety and regulatory consequence."
        )

    return {
        "tag":               tag,
        "name":              name,
        "baseline":          baseline,
        "simulated":         simulated,
        "delta":             delta,
        "cascade":           cascade,
        "insights":          insights,
        "overrides_applied": overrides,
    }


def get_simulatable_equipment(G: nx.DiGraph) -> list:
    """
    Returns all equipment with telemetry fields, suitable for the
    What-If simulator. Each entry includes current sensor readings
    and their threshold bands so the UI can show sensible slider ranges.
    """
    result = []
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue
        readings = {
            "vibration_mm_s":        attrs.get("vibration_mm_s"),
            "bearing_temp_c":        attrs.get("bearing_temp_c"),
            "oil_iron_ppm":          attrs.get("oil_iron_ppm"),
            "bearing_overdue_hours": attrs.get("bearing_overdue_hours"),
        }
        has_tel = any(v is not None for v in readings.values())
        risk, _, _ = compute_anomaly_risk(attrs)

        entry = {
            "tag":          attrs.get("tag", nid),
            "name":         attrs.get("name", ""),
            "risk_level":   risk or attrs.get("risk_level", "LOW"),
            "has_telemetry": has_tel,
            "readings":     {k: v for k, v in readings.items() if v is not None},
            "thresholds":   {k: THRESHOLDS[k] for k in readings if k in THRESHOLDS and readings[k] is not None},
            "h2s_service":  attrs.get("h2s_service", False),
        }
        result.append(entry)
    result.sort(key=lambda x: x["tag"])
    return result



# ──────────────────────────────────────────────────────────────
# INCIDENT PATTERN MATCHER
# ──────────────────────────────────────────────────────────────

_SYMPTOM_GROUPS = {
    "vibration":      ["vibration", "vibrating", "vib", "shaking", "imbalance", "resonance"],
    "bearing_temp":   ["bearing temp", "bearing temperature", "hot bearing", "temperature", "temp rising", "overheating", "thermal"],
    "seal_failure":   ["seal", "seal failure", "seal leak", "mechanical seal", "seal weep", "barrier fluid", "barrier pressure"],
    "h2s":            ["h2s", "hydrogen sulfide", "gas release", "toxic gas", "h2s release", "sour service"],
    "oil":            ["oil", "oil iron", "iron ppm", "oil contamination", "oil analysis", "lubrication", "lube"],
    "bearing_fail":   ["bearing failure", "bearing seized", "shaft seizure", "bearing", "nde bearing", "de bearing"],
    "alarm":          ["alarm", "alarms", "alarm flood", "normalised alarm", "alarm ignored", "alarm suppressed"],
    "deferred":       ["deferred", "overdue", "wo deferred", "work order", "maintenance deferred", "skipped"],
    "erosion":        ["erosion", "wear", "erosive", "impeller", "cavitation"],
    "water":          ["water", "contamination", "water contamination", "moisture"],
    "production":     ["production loss", "shutdown", "downtime", "trip", "unplanned"],
    "injury":         ["injury", "injured", "personnel", "operator", "near miss"],
}

_INCIDENT_SEARCH_FIELDS = ["type_detail", "root_cause", "precursors", "classification"]


def _extract_signals(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for signal, keywords in _SYMPTOM_GROUPS.items():
        if any(kw in text_lower for kw in keywords):
            found.add(signal)
    return found


def _score_incident(incident_attrs: dict, query_signals: set) -> float:
    if not query_signals:
        return 0.0
    incident_text = " ".join(str(incident_attrs.get(f, "")) for f in _INCIDENT_SEARCH_FIELDS)
    incident_signals = _extract_signals(incident_text)
    if incident_attrs.get("h2s_ppm", 0) > 0:
        incident_signals.add("h2s")
    if incident_attrs.get("injury"):
        incident_signals.add("injury")
    if incident_attrs.get("production_loss"):
        incident_signals.add("production")
    if not incident_signals:
        return 0.0
    overlap = query_signals & incident_signals
    score = len(overlap) / len(query_signals) * 100.0
    if incident_attrs.get("severity") in ("HIGH", "CRITICAL") and overlap:
        score = min(100.0, score * 1.15)
    return round(score, 1)


def find_similar_incidents(G: nx.DiGraph, symptom_query: str) -> list:
    """
    Given a free-text symptom description, finds and ranks historical incidents
    by similarity. Each match includes root cause, linked CAPAs, precursors,
    current equipment status, and matched signals.
    """
    query_signals = _extract_signals(symptom_query)
    results = []

    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "INCIDENT":
            continue
        score = _score_incident(attrs, query_signals)
        if score <= 0:
            continue

        # Linked CAPAs
        capas = []
        for _, dst, ed in G.out_edges(nid, data=True):
            if ed.get("relation") == "LED_TO_CAPA":
                capa = dict(G.nodes[dst])
                capas.append({
                    "id":     capa.get("id", dst),
                    "action": capa.get("action", ""),
                    "status": capa.get("status", ""),
                    "owner":  capa.get("owner", ""),
                })

        # Equipment current status
        eq_tag  = attrs.get("equipment", "")
        eq_info = {}
        if eq_tag:
            for enid, eattrs in G.nodes(data=True):
                if eattrs.get("type") == "EQUIPMENT" and eattrs.get("tag") == eq_tag:
                    computed_risk, _, has_tel = compute_anomaly_risk(eattrs)
                    eq_info = {
                        "tag":          eq_tag,
                        "name":         eattrs.get("name", ""),
                        "current_risk": computed_risk if has_tel else eattrs.get("risk_level", "UNKNOWN"),
                        "status":       eattrs.get("status", ""),
                    }
                    break

        # Matched signals
        incident_text = " ".join(str(attrs.get(f,"")) for f in _INCIDENT_SEARCH_FIELDS)
        incident_signals = _extract_signals(incident_text)
        if attrs.get("h2s_ppm", 0) > 0: incident_signals.add("h2s")
        if attrs.get("injury"):          incident_signals.add("injury")
        matched = list(query_signals & incident_signals)

        results.append({
            "score":           score,
            "matched_signals": matched,
            "id":              attrs.get("id", nid),
            "date":            attrs.get("date", ""),
            "type_detail":     attrs.get("type_detail", ""),
            "equipment":       eq_tag,
            "classification":  attrs.get("classification", ""),
            "severity":        attrs.get("severity", ""),
            "h2s_ppm":         attrs.get("h2s_ppm", 0),
            "injury":          attrs.get("injury", False),
            "root_cause":      attrs.get("root_cause", ""),
            "precursors":      attrs.get("precursors", ""),
            "capas":           capas,
            "equipment_now":   eq_info,
        })

    results.sort(key=lambda x: -x["score"])
    return results[:4]


def get_all_incident_signals(G: nx.DiGraph) -> list:
    """Returns symptom signal chips present across all incidents for the UI."""
    all_signals = set()
    for _, attrs in G.nodes(data=True):
        if attrs.get("type") != "INCIDENT":
            continue
        text = " ".join(str(attrs.get(f,"")) for f in _INCIDENT_SEARCH_FIELDS)
        all_signals |= _extract_signals(text)
        if attrs.get("h2s_ppm", 0) > 0: all_signals.add("h2s")
        if attrs.get("injury"):          all_signals.add("injury")

    SIGNAL_LABELS = {
        "vibration":    "High Vibration",
        "bearing_temp": "Bearing Temp Rising",
        "seal_failure": "Seal Failure / Leak",
        "h2s":          "H2S Release",
        "oil":          "Oil Contamination",
        "bearing_fail": "Bearing Failure",
        "alarm":        "Alarm Flood / Ignored",
        "deferred":     "Deferred Maintenance",
        "erosion":      "Erosion / Wear",
        "water":        "Water Contamination",
        "production":   "Production Loss",
        "injury":       "Personnel Injury",
    }
    return [{"signal": s, "label": SIGNAL_LABELS.get(s, s)} for s in sorted(all_signals)]



# ──────────────────────────────────────────────────────────────
# SPARE PARTS GAP ANALYZER
# Cross-references all CRITICAL/HIGH risk equipment against their
# required spare parts, flags stock shortfalls, quantifies the
# financial exposure of being caught without a part when equipment
# fails, and estimates production loss risk based on equipment
# criticality and spare lead time.
# ──────────────────────────────────────────────────────────────

# Estimated production loss per day (INR) if CDU equipment fails
# Based on typical 10,000 bpd CDU at ~3500 INR/barrel margin
_PRODUCTION_LOSS_PER_DAY = {
    "CRITICAL":    3500000,   # 35L/day — critical equipment, full CDU impact
    "HIGH":        1500000,   # 15L/day — significant but partial impact
    "MEDIUM-HIGH":  800000,   # 8L/day
    "MEDIUM":       400000,   # 4L/day
}


def get_spare_parts_gaps(G: nx.DiGraph) -> dict:
    """
    Analyses spare parts coverage for all equipment, identifies gaps,
    and quantifies financial exposure.

    Returns:
    {
      "summary": {total_parts, gaps_count, critical_gaps, total_exposure_inr},
      "gaps": [...],        # parts below reorder point, sorted by exposure
      "coverage": [...],    # all parts with full status
      "equipment_at_risk":  # equipment with CRITICAL/HIGH risk + missing parts
    }
    """
    # Build equipment risk map
    eq_risk = {}
    eq_name = {}
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue
        tag = attrs.get("tag", nid)
        risk, _, has_tel = compute_anomaly_risk(attrs)
        eq_risk[nid] = risk if has_tel else attrs.get("risk_level", "LOW")
        eq_name[nid] = attrs.get("name", "")

    # Build part -> list of requiring equipment
    part_to_equipment = {}
    for src, dst, ed in G.edges(data=True):
        if ed.get("relation") != "REQUIRES_PART":
            continue
        part_id = dst
        if part_id not in part_to_equipment:
            part_to_equipment[part_id] = []
        part_to_equipment[part_id].append({
            "node_id":    src,
            "tag":        G.nodes[src].get("tag", src),
            "name":       G.nodes[src].get("name", ""),
            "eq_risk":    eq_risk.get(src, "LOW"),
            "criticality": ed.get("criticality", "MEDIUM"),
        })

    gaps = []
    coverage = []
    total_exposure = 0

    for part_id, part_attrs in G.nodes(data=True):
        if part_attrs.get("type") != "SPARE_PART":
            continue

        name          = part_attrs.get("name", part_id)
        part_no       = part_attrs.get("part_no", "")
        current_stock = part_attrs.get("current_stock", 0) or 0
        min_stock     = part_attrs.get("min_stock", 1) or 1
        reorder_point = part_attrs.get("reorder_point", min_stock) or min_stock
        lead_weeks    = part_attrs.get("lead_time_weeks", 4) or 4
        unit_cost     = part_attrs.get("unit_cost_inr", 0) or 0
        is_critical   = part_attrs.get("critical", False)
        location      = part_attrs.get("location", "Unknown")
        vendor        = part_attrs.get("vendor", "Unknown")
        category      = part_attrs.get("category", "General")

        requiring_equipment = part_to_equipment.get(part_id, [])

        # Highest risk among requiring equipment
        risk_order = ["LOW","LOW-MEDIUM","MEDIUM","MEDIUM-HIGH","HIGH","CRITICAL"]
        worst_eq_risk = "LOW"
        for eq in requiring_equipment:
            r = eq.get("eq_risk", "LOW")
            if risk_order.index(r) > risk_order.index(worst_eq_risk):
                worst_eq_risk = r

        # Stock status
        if current_stock == 0:
            stock_status = "OUT_OF_STOCK"
        elif current_stock < reorder_point:
            stock_status = "BELOW_REORDER"
        elif current_stock < min_stock:
            stock_status = "BELOW_MINIMUM"
        else:
            stock_status = "ADEQUATE"

        # Financial exposure: if stock = 0 and equipment fails,
        # production stops for lead_time_weeks weeks
        exposure_inr = 0
        if stock_status in ("OUT_OF_STOCK", "BELOW_REORDER"):
            daily_loss = _PRODUCTION_LOSS_PER_DAY.get(worst_eq_risk, 0)
            lead_days  = lead_weeks * 7
            exposure_inr = daily_loss * lead_days
            total_exposure += exposure_inr

        # Gap severity
        if stock_status == "OUT_OF_STOCK" and is_critical:
            gap_severity = "CRITICAL"
        elif stock_status == "OUT_OF_STOCK":
            gap_severity = "HIGH"
        elif stock_status == "BELOW_REORDER" and is_critical:
            gap_severity = "HIGH"
        elif stock_status == "BELOW_REORDER":
            gap_severity = "MEDIUM"
        elif stock_status == "BELOW_MINIMUM":
            gap_severity = "LOW"
        else:
            gap_severity = None

        entry = {
            "part_id":       part_id,
            "part_no":       part_no,
            "name":          name,
            "category":      category,
            "location":      location,
            "vendor":        vendor,
            "current_stock": current_stock,
            "min_stock":     min_stock,
            "reorder_point": reorder_point,
            "lead_time_weeks": lead_weeks,
            "unit_cost_inr": unit_cost,
            "is_critical":   is_critical,
            "stock_status":  stock_status,
            "gap_severity":  gap_severity,
            "exposure_inr":  exposure_inr,
            "requiring_equipment": requiring_equipment,
            "worst_eq_risk": worst_eq_risk,
        }
        coverage.append(entry)
        if gap_severity:
            gaps.append(entry)

    # Sort gaps by exposure descending
    SEV_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    gaps.sort(key=lambda x: (-x["exposure_inr"], -SEV_ORDER.get(x["gap_severity"], 0)))
    coverage.sort(key=lambda x: (-SEV_ORDER.get(x["gap_severity"] or "LOW", 0), x["name"]))

    # Equipment at risk: CRITICAL/HIGH equipment with at least one gap part
    gap_part_ids = {g["part_id"] for g in gaps}
    eq_at_risk = {}
    for src, dst, ed in G.edges(data=True):
        if ed.get("relation") != "REQUIRES_PART":
            continue
        if dst not in gap_part_ids:
            continue
        eq_attrs = G.nodes[src]
        tag = eq_attrs.get("tag", src)
        risk = eq_risk.get(src, "LOW")
        if tag not in eq_at_risk:
            eq_at_risk[tag] = {
                "tag":       tag,
                "name":      eq_attrs.get("name", ""),
                "risk":      risk,
                "gap_parts": [],
            }
        part_attrs_map = dict(G.nodes[dst])
        eq_at_risk[tag]["gap_parts"].append({
            "name":          part_attrs_map.get("name", dst),
            "stock_status":  part_attrs_map.get("stock_status", "UNKNOWN"),
            "lead_weeks":    part_attrs_map.get("lead_time_weeks", 0),
            "gap_severity":  next((g["gap_severity"] for g in gaps if g["part_id"] == dst), None),
        })

    # Re-check stock_status on part nodes (they don't store it, compute inline)
    for eq in eq_at_risk.values():
        for gp in eq["gap_parts"]:
            matching = next((g for g in gaps if g["name"] == gp["name"]), None)
            if matching:
                gp["stock_status"] = matching["stock_status"]
                gp["gap_severity"] = matching["gap_severity"]

    eq_at_risk_list = sorted(eq_at_risk.values(),
                             key=lambda x: -risk_order.index(x["risk"]))

    critical_gaps = sum(1 for g in gaps if g["gap_severity"] == "CRITICAL")

    return {
        "summary": {
            "total_parts":      len(coverage),
            "gaps_count":       len(gaps),
            "critical_gaps":    critical_gaps,
            "total_exposure_inr": total_exposure,
        },
        "gaps":              gaps,
        "coverage":          coverage,
        "equipment_at_risk": eq_at_risk_list,
    }



# ──────────────────────────────────────────────────────────────
# REGULATORY DEADLINE TRACKER
# Pulls all compliance findings with deadline data, categorises
# them by status (OVERDUE / DUE_SOON / UPCOMING / COMPLETED),
# computes days overdue or days remaining, and returns a
# structured timeline sorted by urgency.
# ──────────────────────────────────────────────────────────────

from datetime import date as _date

_STATUS_ORDER = {"OVERDUE": 0, "DUE_SOON": 1, "UPCOMING": 2, "IN_PROGRESS": 3, "COMPLETED": 4}

def get_regulatory_deadlines(G: nx.DiGraph) -> dict:
    """
    Returns all compliance findings as a deadline tracker.
    Categorises each item:
      OVERDUE    - past due date
      DUE_SOON   - due within 30 days
      UPCOMING   - due in 31-90 days
      IN_PROGRESS - active but no imminent deadline
      COMPLETED  - resolved

    Returns:
    {
      "summary": {overdue, due_soon, upcoming, completed, total},
      "items": [...sorted by urgency...]
    }
    """
    today = _date.today()
    items = []

    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "FINDING":
            continue

        desc         = attrs.get("description", "")
        regulation   = attrs.get("regulation", "")
        category     = attrs.get("category", "General")
        severity     = attrs.get("severity", "MEDIUM")
        responsible  = attrs.get("responsible", "")
        action       = attrs.get("action_required", "")
        frequency    = attrs.get("frequency_days")
        raw_status   = attrs.get("status", "").upper().replace(" ", "_")
        due_date_str = attrs.get("due_date", "")

        overdue_days    = attrs.get("overdue_days", 0) or 0
        days_remaining  = attrs.get("days_remaining")

        # Compute days_remaining from due_date if available
        if due_date_str:
            try:
                due_dt = _date.fromisoformat(due_date_str)
                delta  = (due_dt - today).days
                if delta < 0:
                    overdue_days   = abs(delta)
                    days_remaining = 0
                    computed_status = "OVERDUE"
                elif delta <= 30:
                    days_remaining  = delta
                    overdue_days    = 0
                    computed_status = "DUE_SOON"
                elif delta <= 90:
                    days_remaining  = delta
                    overdue_days    = 0
                    computed_status = "UPCOMING"
                else:
                    days_remaining  = delta
                    overdue_days    = 0
                    computed_status = "UPCOMING"
            except ValueError:
                computed_status = raw_status or "UPCOMING"
        else:
            computed_status = raw_status or "UPCOMING"

        # Override: COMPLETED stays COMPLETED regardless of date
        if raw_status == "COMPLETED":
            computed_status = "COMPLETED"

        # Override: items explicitly marked OVERDUE in data
        if raw_status == "OVERDUE" and computed_status not in ("OVERDUE",):
            computed_status = "OVERDUE"

        items.append({
            "id":             nid,
            "description":    desc,
            "regulation":     regulation,
            "category":       category,
            "severity":       severity,
            "responsible":    responsible,
            "action_required": action,
            "status":         computed_status,
            "due_date":       due_date_str,
            "overdue_days":   overdue_days,
            "days_remaining": days_remaining,
            "frequency_days": frequency,
        })

    # Sort: OVERDUE first (most overdue first), then DUE_SOON, UPCOMING, rest
    def sort_key(x):
        s = x["status"]
        if s == "OVERDUE":
            return (0, -(x["overdue_days"] or 0))
        elif s == "DUE_SOON":
            return (1, x["days_remaining"] or 999)
        elif s == "UPCOMING":
            return (2, x["days_remaining"] or 999)
        elif s == "IN_PROGRESS":
            return (3, 0)
        else:
            return (4, 0)

    items.sort(key=sort_key)

    summary = {
        "overdue":     sum(1 for i in items if i["status"] == "OVERDUE"),
        "due_soon":    sum(1 for i in items if i["status"] == "DUE_SOON"),
        "upcoming":    sum(1 for i in items if i["status"] == "UPCOMING"),
        "in_progress": sum(1 for i in items if i["status"] == "IN_PROGRESS"),
        "completed":   sum(1 for i in items if i["status"] == "COMPLETED"),
        "total":       len(items),
    }

    return {"summary": summary, "items": items}



# ──────────────────────────────────────────────────────────────
# MAINTENANCE WINDOW OPTIMIZER
# Given a primary equipment shutdown, finds all other equipment
# that can be serviced in the same window to minimize total
# plant downtime. Uses process connections, shared isolation
# requirements, common trade/crew, and spare part availability
# to score co-maintenance candidates.
#
# Scoring per candidate (0-100):
#   Process proximity   : +30 if directly connected (same train)
#   Risk urgency        : +25 if CRITICAL/HIGH, +15 MEDIUM-HIGH
#   Degradation trend   : +20 if threshold <90 days away
#   Shared isolation    : +15 if same P&ID section/location
#   Overdue WOs         : +10 if has deferred work orders
#   Spare parts ready   : +5  if required spares are in stock
# ──────────────────────────────────────────────────────────────

# Estimated shutdown duration per equipment type (hours)
_SHUTDOWN_HOURS = {
    "pump":    8,
    "motor":   4,
    "vessel":  24,
    "tank":    48,
    "exchanger": 12,
    "column":  72,
    "default": 8,
}

# Trade/crew type per equipment
_TRADE_TYPE = {
    "P-101A": "mechanical", "P-101B": "mechanical",
    "P-102A": "mechanical", "P-102B": "mechanical",
    "P-103A": "mechanical", "P-103B": "mechanical",
    "P-201A": "mechanical", "M-101A": "electrical",
    "E-101":  "mechanical", "T-101":  "mechanical",
    "T-104":  "mechanical", "T-100":  "mechanical",
}

# P&ID section grouping (equipment that share isolation valves)
_PID_SECTIONS = {
    "CDU-FEED":  {"P-101A", "P-101B", "M-101A", "E-101"},
    "CDU-ATMOS": {"T-101", "P-102A", "P-102B"},
    "CDU-KERO":  {"P-103A", "P-103B"},
    "CDU-VAC":   {"P-201A"},
    "TANKFARM":  {"T-100", "T-104"},
}

def _get_equipment_type(tag: str, name: str) -> str:
    name_lower = name.lower()
    if "pump" in name_lower:      return "pump"
    if "motor" in name_lower:     return "motor"
    if "column" in name_lower or "distillation" in name_lower: return "column"
    if "tank" in name_lower:      return "tank"
    if "heater" in name_lower or "exchanger" in name_lower:    return "exchanger"
    return "default"

def _get_pid_section(tag: str):
    for section, tags in _PID_SECTIONS.items():
        if tag in tags:
            return section
    return None

def _get_connected_tags(G, node_id: str) -> set:
    """Returns tags of directly connected equipment (1 hop)."""
    connected = set()
    for src, dst, ed in G.edges(data=True):
        if ed.get("relation") in ("FEEDS_INTO", "DRIVES", "CONNECTED_TO", "BACKUP_FOR", "STANDBY_FOR"):
            if src == node_id:
                dst_d = G.nodes[dst]
                if dst_d.get("type") == "EQUIPMENT":
                    connected.add(dst_d.get("tag", dst))
            elif dst == node_id:
                src_d = G.nodes[src]
                if src_d.get("type") == "EQUIPMENT":
                    connected.add(src_d.get("tag", src))
    return connected


def get_maintenance_window(G: nx.DiGraph, primary_tag: str) -> dict:
    """
    Given a primary equipment tag requiring shutdown, returns an optimized
    maintenance window plan including:
    - Primary equipment details and estimated shutdown duration
    - Ranked co-maintenance candidates with scores and rationale
    - Total downtime saved vs doing each job separately
    - Required crew, tools, and spare parts summary
    - Process impact assessment (what stops running during the window)

    Parameters
    ----------
    primary_tag : str
        Equipment tag to shut down (e.g. "P-201A")

    Returns
    -------
    dict with keys: primary, candidates, window_summary, process_impact
    """
    # Find primary equipment node
    primary_node = None
    primary_attrs = None
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") == "EQUIPMENT" and attrs.get("tag") == primary_tag:
            primary_node = nid
            primary_attrs = dict(attrs)
            break

    if not primary_node:
        return {"error": f"Equipment {primary_tag} not found"}

    primary_name    = primary_attrs.get("name", primary_tag)
    primary_type    = _get_equipment_type(primary_tag, primary_name)
    primary_hours   = _SHUTDOWN_HOURS.get(primary_type, 8)
    primary_section = _get_pid_section(primary_tag)
    primary_trade   = _TRADE_TYPE.get(primary_tag, "mechanical")
    primary_connected = _get_connected_tags(G, primary_node)

    # Build spare parts map
    parts_in_stock = set()
    parts_needed   = {}  # tag -> list of part names needed
    for src, dst, ed in G.edges(data=True):
        if ed.get("relation") != "REQUIRES_PART":
            continue
        src_d = G.nodes[src]
        dst_d = G.nodes[dst]
        tag = src_d.get("tag", src)
        part_name = dst_d.get("name", dst)
        stock = dst_d.get("current_stock", 0) or 0
        if stock > 0:
            parts_in_stock.add(dst)
        if tag not in parts_needed:
            parts_needed[tag] = []
        parts_needed[tag].append({
            "name":    part_name,
            "in_stock": stock > 0,
            "stock":   stock,
            "lead_weeks": dst_d.get("lead_time_weeks", 0),
        })

    # Score all other equipment as co-maintenance candidates
    candidates = []
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue
        tag = attrs.get("tag", nid)
        if tag == primary_tag:
            continue

        name = attrs.get("name", "")
        score = 0
        reasons = []

        # Process proximity
        if tag in primary_connected:
            score += 30
            reasons.append("directly connected in process train — shared isolation")
        elif _get_pid_section(tag) == primary_section and primary_section:
            score += 20
            reasons.append(f"same P&ID section ({primary_section}) — shared isolation boundary")

        # Risk urgency
        computed_risk, _, has_tel = compute_anomaly_risk(attrs)
        risk = computed_risk if has_tel else attrs.get("risk_level", "LOW")
        if risk == "CRITICAL":
            score += 25
            reasons.append("CRITICAL risk — needs immediate attention")
        elif risk == "HIGH":
            score += 20
            reasons.append("HIGH risk — inspection overdue")
        elif risk == "MEDIUM-HIGH":
            score += 15
            reasons.append("MEDIUM-HIGH risk — deteriorating condition")
        elif risk == "MEDIUM":
            score += 8
            reasons.append("MEDIUM risk — monitor closely")

        # Degradation urgency
        fc = compute_degradation_forecast(attrs)
        summ = summarize_forecast(fc) if fc else None
        days = summ["days"] if summ else None
        if days is not None and days < 30:
            score += 20
            reasons.append(f"threshold breach in {days} days — urgent maintenance needed")
        elif days is not None and days < 90:
            score += 12
            reasons.append(f"threshold projected in {days} days — good window to service")
        elif days is not None and days < 180:
            score += 5
            reasons.append(f"degradation trend — {days} days to threshold")

        # Shared trade/crew
        candidate_trade = _TRADE_TYPE.get(tag, "mechanical")
        if candidate_trade == primary_trade:
            score += 8
            reasons.append(f"same crew type ({candidate_trade}) — no extra mobilization")

        # Overdue WOs
        overdue = attrs.get("overdue_wos", "")
        if overdue:
            score += 10
            reasons.append(f"deferred WOs pending: {str(overdue)[:50]}")

        # Spare parts ready
        tag_parts = parts_needed.get(tag, [])
        parts_ready = [p for p in tag_parts if p["in_stock"]]
        if parts_ready:
            score += 5
            reasons.append(f"{len(parts_ready)} required spare(s) in stock")

        # H2S service — worth co-servicing when area is already isolated
        if attrs.get("h2s_service") and primary_attrs.get("h2s_service"):
            score += 8
            reasons.append("both in H2S service — area gas test already required")

        score = min(100, score)
        if score < 5:
            continue  # Not worth suggesting

        eq_type   = _get_equipment_type(tag, name)
        est_hours = _SHUTDOWN_HOURS.get(eq_type, 8)

        candidates.append({
            "tag":          tag,
            "name":         name,
            "risk":         risk,
            "score":        score,
            "reasons":      reasons,
            "est_hours":    est_hours,
            "trade":        candidate_trade,
            "pid_section":  _get_pid_section(tag),
            "forecast_text": summ["text"] if summ else None,
            "days_to_threshold": days,
            "parts_needed": tag_parts,
            "overdue_wos":  overdue,
        })

    candidates.sort(key=lambda x: -x["score"])

    # ── Window summary ────────────────────────────────────────────────────────
    # If we do primary + top candidates together, what is total downtime saved?
    top_candidates = [c for c in candidates if c["score"] >= 20][:5]

    if top_candidates:
        # Doing separately: sum of individual shutdown hours (each needs own isolation)
        hours_separately = primary_hours + sum(c["est_hours"] for c in top_candidates)
        # Together: max of primary + largest candidate (parallel work within window)
        # plus a 20% overhead for coordination
        hours_together   = max(primary_hours, max(c["est_hours"] for c in top_candidates))
        hours_together   = round(hours_together * 1.2)
        hours_saved      = max(0, hours_separately - hours_together)
    else:
        hours_separately = primary_hours
        hours_together   = primary_hours
        hours_saved      = 0

    # Required crew types
    crew_types = {primary_trade}
    for c in top_candidates:
        crew_types.add(c["trade"])

    # All spares needed for primary + top candidates
    all_tags = [primary_tag] + [c["tag"] for c in top_candidates]
    all_parts = []
    for t in all_tags:
        for p in parts_needed.get(t, []):
            all_parts.append({**p, "for_tag": t})

    # ── Process impact ────────────────────────────────────────────────────────
    # What process units are affected during the window
    impacted = set()
    impacted.add(primary_tag)
    for c in top_candidates:
        impacted.add(c["tag"])
        # Add downstream equipment that loses feed
        for nid, attrs in G.nodes(data=True):
            if attrs.get("tag") == c["tag"]:
                for _, dst, ed in G.edges(nid, data=True):
                    if ed.get("relation") == "FEEDS_INTO":
                        dst_d = G.nodes[dst]
                        if dst_d.get("type") == "EQUIPMENT":
                            impacted.add(dst_d.get("tag", dst))

    # Check if any impacted equipment has a standby
    standby_available = {}
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue
        for _, dst, ed in G.edges(nid, data=True):
            if ed.get("relation") in ("BACKUP_FOR", "STANDBY_FOR"):
                dst_tag = G.nodes[dst].get("tag", dst)
                standby_available[dst_tag] = attrs.get("tag", nid)

    return {
        "primary": {
            "tag":          primary_tag,
            "name":         primary_name,
            "risk":         compute_anomaly_risk(primary_attrs)[0] or primary_attrs.get("risk_level",""),
            "est_hours":    primary_hours,
            "trade":        primary_trade,
            "pid_section":  primary_section,
            "h2s_service":  primary_attrs.get("h2s_service", False),
        },
        "candidates":   candidates,
        "top_candidates": top_candidates,
        "window_summary": {
            "hours_primary_only":   primary_hours,
            "hours_separately":     hours_separately,
            "hours_together":       hours_together,
            "hours_saved":          hours_saved,
            "crew_types":           sorted(crew_types),
            "total_parts":          len(all_parts),
            "parts_in_stock":       sum(1 for p in all_parts if p["in_stock"]),
        },
        "process_impact": {
            "impacted_tags":    sorted(impacted),
            "standby_available": standby_available,
        },
    }


def get_maintenance_window_equipment(G: nx.DiGraph) -> list:
    """Returns all equipment suitable as primary for maintenance window planning."""
    results = []
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue
        risk, _, has_tel = compute_anomaly_risk(attrs)
        results.append({
            "tag":       attrs.get("tag", nid),
            "name":      attrs.get("name", ""),
            "risk":      risk if has_tel else attrs.get("risk_level", "LOW"),
            "h2s":       attrs.get("h2s_service", False),
        })
    results.sort(key=lambda x: x["tag"])
    return results



# ──────────────────────────────────────────────────────────────
# RISK CASCADE ANALYZER
# "If X fails, what else fails or degrades?"
# Traverses the process graph in both directions:
#   Downstream: FEEDS_INTO, DRIVES chains (loss of feed/drive)
#   Upstream:   what was feeding into the failed equipment
#   Lateral:    shared spare parts, shared crew, shared utilities
# Each impacted node gets an impact level (DIRECT/INDIRECT/RESOURCE)
# and a plain-language explanation of why it is affected.
# ──────────────────────────────────────────────────────────────

# Production loss per hour by equipment type (INR) — approximate
_PROD_LOSS_PER_HOUR = {
    "T-101": 145000,   # Atm distillation column — full CDU stop
    "E-101": 120000,   # Pre-heater — feed preheat lost, throughput drop
    "P-101A": 85000,   # Primary crude feed pump
    "P-101B": 40000,   # Standby — affects resilience
    "P-102A": 50000,   # Atm residue
    "P-102B": 45000,
    "P-103A": 35000,   # Kerosene product
    "P-103B": 30000,
    "P-201A": 60000,   # Vacuum residue transfer
    "M-101A": 85000,   # Motor for P-101A
    "T-100":  20000,   # Feed tank — buffer
    "T-104":  15000,   # Storage
}

# Standby/backup pairs
_STANDBY_FOR = {
    "P-101A": "P-101B",   # P-101B is standby for P-101A
    "P-101B": "P-101A",
}

# Shared utility dependencies (equipment that share same utilities)
_SHARED_UTILITIES = {
    "CDU_LUBE":   {"P-101A","P-101B","P-102A","P-102B","P-103A","P-103B","P-201A"},
    "CDU_COOLING": {"E-101","T-101"},
    "CDU_POWER":  {"M-101A","P-101A","P-101B"},
}

def get_risk_cascade(G, failed_tag: str, max_depth: int = 4) -> dict:
    """
    Simulates failure of failed_tag and traces all cascade effects.

    Returns:
    {
      "failed":      {tag, name, risk, h2s, prod_loss_per_hour},
      "cascade":     [list of impacted nodes, sorted by severity],
      "summary":     {direct_count, indirect_count, resource_count,
                      total_prod_loss_per_hour, max_depth_reached},
      "chain":       ordered list of failure propagation steps,
      "mitigations": list of mitigation suggestions
    }
    """
    # Find failed node
    failed_node = None
    failed_attrs = None
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") == "EQUIPMENT" and attrs.get("tag") == failed_tag:
            failed_node = nid
            failed_attrs = dict(attrs)
            break

    if not failed_node:
        return {"error": f"Equipment {failed_tag} not found"}

    failed_name = failed_attrs.get("name", failed_tag)

    # Build equipment lookup: tag -> (node_id, attrs)
    eq_lookup = {}
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") == "EQUIPMENT":
            eq_lookup[attrs.get("tag", nid)] = (nid, attrs)

    impacted = {}   # tag -> impact dict
    chain    = []   # ordered list of cascade steps
    visited  = {failed_tag}

    def add_impact(tag, impact_type, reason, depth, via=None):
        if tag in impacted:
            # Keep highest severity
            existing = impacted[tag]
            if impact_type == "DIRECT" and existing["impact_type"] != "DIRECT":
                impacted[tag]["impact_type"] = "DIRECT"
                impacted[tag]["reason"] = reason
            return
        nid, attrs = eq_lookup.get(tag, (None, {}))
        risk, _, has_tel = compute_anomaly_risk(attrs)
        current_risk = risk if has_tel else attrs.get("risk_level", "UNKNOWN")
        prod_loss = _PROD_LOSS_PER_HOUR.get(tag, 20000)

        impacted[tag] = {
            "tag":         tag,
            "name":        attrs.get("name", tag),
            "impact_type": impact_type,
            "reason":      reason,
            "depth":       depth,
            "via":         via,
            "current_risk": current_risk,
            "prod_loss_per_hour": prod_loss,
            "h2s_service": attrs.get("h2s_service", False),
            "has_standby": _STANDBY_FOR.get(tag) is not None,
            "standby_tag": _STANDBY_FOR.get(tag),
        }
        chain.append({
            "step":        len(chain) + 1,
            "from":        via or failed_tag,
            "to":          tag,
            "relation":    impact_type,
            "reason":      reason,
            "depth":       depth,
        })

    # ── Downstream cascade (loss of output from failed equipment) ──────────
    def traverse_downstream(node_id, tag, depth):
        if depth > max_depth:
            return
        for _, dst, ed in G.out_edges(node_id, data=True):
            rel = ed.get("relation", "")
            if rel not in ("FEEDS_INTO", "DRIVES"):
                continue
            dst_d = G.nodes[dst]
            if dst_d.get("type") != "EQUIPMENT":
                continue
            dst_tag = dst_d.get("tag", dst)
            if dst_tag in visited:
                continue
            visited.add(dst_tag)

            if rel == "DRIVES":
                reason = f"{tag} drives {dst_tag} — motor failure stops the pump"
                itype  = "DIRECT"
            else:
                reason = f"{dst_tag} loses feed from {tag} — process throughput stops"
                itype  = "DIRECT"

            # Check if standby exists for the lost feed
            standby = _STANDBY_FOR.get(tag)
            if standby and standby not in visited:
                reason += f" (standby {standby} may compensate)"

            add_impact(dst_tag, itype, reason, depth, via=tag)
            traverse_downstream(dst, dst_tag, depth + 1)

    # ── Upstream cascade (equipment that was feeding the failed one) ────────
    def traverse_upstream(node_id, tag, depth):
        if depth > max_depth:
            return
        for src, _, ed in G.in_edges(node_id, data=True):
            rel = ed.get("relation", "")
            if rel not in ("FEEDS_INTO", "DRIVES"):
                continue
            src_d = G.nodes[src]
            if src_d.get("type") != "EQUIPMENT":
                continue
            src_tag = src_d.get("tag", src)
            if src_tag in visited:
                continue
            visited.add(src_tag)

            if rel == "DRIVES":
                reason = f"{src_tag} was driving {tag} — motor is now idle, potential overload on restart"
                itype  = "INDIRECT"
            else:
                reason = f"{src_tag} was feeding {tag} — product backs up, upstream pressure risk"
                itype  = "INDIRECT"

            add_impact(src_tag, itype, reason, depth, via=tag)

    # ── Lateral: shared spare parts ─────────────────────────────────────────
    def check_shared_spares():
        # Get all spare parts used by failed equipment
        failed_parts = set()
        for _, dst, ed in G.out_edges(failed_node, data=True):
            if ed.get("relation") == "REQUIRES_PART":
                failed_parts.add(dst)

        if not failed_parts:
            return

        # Find other equipment sharing those parts
        for src, dst, ed in G.edges(data=True):
            if ed.get("relation") != "REQUIRES_PART":
                continue
            src_d = G.nodes[src]
            if src_d.get("type") != "EQUIPMENT":
                continue
            src_tag = src_d.get("tag", src)
            if src_tag == failed_tag or src_tag in impacted:
                continue
            if dst in failed_parts:
                part_name = G.nodes[dst].get("name", dst)
                stock = G.nodes[dst].get("current_stock", 0) or 0
                if stock <= 1:  # Spare will be consumed by failed equipment repair
                    reason = (f"Shares spare part '{part_name}' with {failed_tag} — "
                              f"stock={stock}, may be depleted during {failed_tag} repair")
                    add_impact(src_tag, "RESOURCE", reason, 99, via="shared_spares")

    # ── Lateral: shared utilities ────────────────────────────────────────────
    def check_shared_utilities():
        for util_name, util_tags in _SHARED_UTILITIES.items():
            if failed_tag not in util_tags:
                continue
            for other_tag in util_tags:
                if other_tag == failed_tag or other_tag in impacted:
                    continue
                reason = (f"Shares {util_name.replace('_',' ').lower()} utility with {failed_tag} — "
                          f"utility disruption or maintenance access conflict likely")
                add_impact(other_tag, "RESOURCE", reason, 99, via="shared_utility")

    # Run all traversals
    traverse_downstream(failed_node, failed_tag, 1)
    traverse_upstream(failed_node, failed_tag, 1)
    check_shared_spares()
    check_shared_utilities()

    # Sort: DIRECT first, then INDIRECT, then RESOURCE; within each by prod_loss
    impact_order = {"DIRECT": 0, "INDIRECT": 1, "RESOURCE": 2}
    cascade_list = sorted(
        impacted.values(),
        key=lambda x: (impact_order.get(x["impact_type"], 3), -x["prod_loss_per_hour"])
    )

    # Summary stats
    direct   = [c for c in cascade_list if c["impact_type"] == "DIRECT"]
    indirect = [c for c in cascade_list if c["impact_type"] == "INDIRECT"]
    resource = [c for c in cascade_list if c["impact_type"] == "RESOURCE"]
    total_loss = sum(c["prod_loss_per_hour"] for c in cascade_list)

    # Mitigations
    mitigations = []
    for c in cascade_list:
        if c.get("has_standby") and c.get("standby_tag"):
            mitigations.append(f"Switch to standby {c['standby_tag']} for {c['tag']} to maintain feed to downstream equipment")
    if failed_attrs.get("h2s_service"):
        mitigations.append(f"Isolate {failed_tag} area and conduct H2S gas test before any work begins")
    if direct:
        direct_tags = ", ".join(c["tag"] for c in direct[:3])
        mitigations.append(f"Notify operations for {direct_tags} — direct process impact confirmed")
    mitigations.append(f"Issue emergency work order for {failed_tag} repair and update shift log")

    return {
        "failed": {
            "tag":               failed_tag,
            "name":              failed_name,
            "risk":              compute_anomaly_risk(failed_attrs)[0] or failed_attrs.get("risk_level",""),
            "h2s_service":       failed_attrs.get("h2s_service", False),
            "prod_loss_per_hour": _PROD_LOSS_PER_HOUR.get(failed_tag, 20000),
        },
        "cascade": cascade_list,
        "summary": {
            "direct_count":          len(direct),
            "indirect_count":        len(indirect),
            "resource_count":        len(resource),
            "total_impacted":        len(cascade_list),
            "total_prod_loss_per_hour": total_loss + _PROD_LOSS_PER_HOUR.get(failed_tag, 0),
        },
        "chain":       chain,
        "mitigations": mitigations,
    }


def get_cascade_equipment(G) -> list:
    """All equipment available for cascade simulation."""
    results = []
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue
        risk, _, has_tel = compute_anomaly_risk(attrs)
        results.append({
            "tag":  attrs.get("tag", nid),
            "name": attrs.get("name", ""),
            "risk": risk if has_tel else attrs.get("risk_level", "LOW"),
        })
    return sorted(results, key=lambda x: x["tag"])



# ──────────────────────────────────────────────────────────────
# CARBON AND ENERGY IMPACT CALCULATOR
# Estimates the carbon footprint and energy waste associated with:
#   1. Unplanned downtime from equipment failures (flaring + restart)
#   2. Degraded equipment efficiency (pumps running above optimal)
#   3. Deferred maintenance energy penalties
#   4. Compliance violations with environmental implications
#
# All figures are estimates based on typical petroleum refinery
# operational data. In production, these would come from DCS/SCADA.
# Disclosed honestly in the UI as estimates, not measured values.
# ──────────────────────────────────────────────────────────────

# CDU baseline parameters (Vadodara Refinery simulation)
_CDU_THROUGHPUT_BPD    = 10000          # barrels per day
_CDU_ENERGY_GJ_PER_DAY = 420            # GJ/day typical CDU energy use
_EMISSION_FACTOR_KG_CO2_PER_GJ = 56.1  # natural gas combustion (IPCC 2006)
_FLARE_CO2_KG_PER_HOUR = 180            # unplanned flaring during trip/restart
_FLARE_CH4_KG_PER_HOUR = 8             # methane slip (GWP 84 over 20yr)

# Equipment-specific energy penalties when degraded
# Extra energy consumed (GJ/day) vs healthy baseline
_DEGRADATION_ENERGY_PENALTY = {
    "pump_efficiency_loss": {
        # Each mm/s vibration above watch threshold: +X% energy consumption
        "vibration_pct_per_mms": 1.8,
        "baseline_kw": 75,              # typical CDU pump motor kW
    },
    "bearing_temp_penalty": {
        # Each degree C above alert threshold: extra friction losses
        "kwh_per_degree_per_day": 2.4,
    },
    "oil_degradation_penalty": {
        # Contaminated oil: increased friction, +Y% energy
        "pct_per_10ppm_above_watch": 0.9,
        "baseline_kw": 75,
    },
}

# Typical unplanned shutdown durations and impacts
_UNPLANNED_SHUTDOWN = {
    "CRITICAL": {"duration_hours": 18, "flare_hours": 3},
    "HIGH":     {"duration_hours": 10, "flare_hours": 1.5},
    "MEDIUM-HIGH": {"duration_hours": 6, "flare_hours": 0.5},
}

def _pump_vibration_penalty_kwh_day(vibration_mms, watch_threshold):
    """Extra kWh/day due to vibration-induced inefficiency."""
    if vibration_mms <= watch_threshold:
        return 0.0
    excess = vibration_mms - watch_threshold
    penalty_pct = excess * _DEGRADATION_ENERGY_PENALTY["pump_efficiency_loss"]["vibration_pct_per_mms"]
    baseline_kw = _DEGRADATION_ENERGY_PENALTY["pump_efficiency_loss"]["baseline_kw"]
    return baseline_kw * (penalty_pct / 100) * 24  # kWh/day

def _bearing_temp_penalty_kwh_day(temp_c, alert_threshold):
    """Extra kWh/day due to elevated bearing temperature friction."""
    if temp_c <= alert_threshold:
        return 0.0
    excess = temp_c - alert_threshold
    return excess * _DEGRADATION_ENERGY_PENALTY["bearing_temp_penalty"]["kwh_per_degree_per_day"]

def _oil_penalty_kwh_day(oil_ppm, watch_threshold, baseline_kw=75):
    """Extra kWh/day due to oil contamination friction losses."""
    if oil_ppm <= watch_threshold:
        return 0.0
    excess_10ppm = max(0, (oil_ppm - watch_threshold) / 10)
    penalty_pct = excess_10ppm * _DEGRADATION_ENERGY_PENALTY["oil_degradation_penalty"]["pct_per_10ppm_above_watch"]
    return baseline_kw * (penalty_pct / 100) * 24

def get_carbon_energy_impact(G) -> dict:
    """
    Calculates estimated carbon and energy impact across all plant equipment.

    Returns a structured report with:
    - Per-equipment energy waste and CO2 equivalent
    - Projected unplanned shutdown carbon impact
    - Total daily carbon footprint vs clean-running baseline
    - Top recommendations for carbon reduction
    - Disclaimer about estimate nature

    All CO2 figures in kg CO2-equivalent per day.
    Energy figures in kWh per day.
    """
    equipment_impacts = []
    total_energy_waste_kwh_day = 0.0
    total_co2_waste_kg_day     = 0.0
    total_shutdown_co2_kg      = 0.0  # one-time per projected failure

    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "EQUIPMENT":
            continue

        tag  = attrs.get("tag", nid)
        name = attrs.get("name", "")

        # Only pumps have telemetry for efficiency calculations
        vib   = attrs.get("vibration_mm_s")
        temp  = attrs.get("bearing_temp_c")
        oil   = attrs.get("oil_iron_ppm")

        if vib is None and temp is None and oil is None:
            continue  # no telemetry, skip

        energy_penalties = {}
        total_kwh_day    = 0.0

        # Vibration penalty
        vib_watch = THRESHOLDS.get("vibration_mm_s", {}).get("watch", 2.8)
        if vib is not None:
            vib_pen = _pump_vibration_penalty_kwh_day(vib, vib_watch)
            if vib_pen > 0:
                energy_penalties["vibration"] = round(vib_pen, 1)
                total_kwh_day += vib_pen

        # Bearing temp penalty
        temp_alert = THRESHOLDS.get("bearing_temp_c", {}).get("alert", 85)
        if temp is not None:
            temp_pen = _bearing_temp_penalty_kwh_day(temp, temp_alert)
            if temp_pen > 0:
                energy_penalties["bearing_temp"] = round(temp_pen, 1)
                total_kwh_day += temp_pen

        # Oil contamination penalty
        oil_watch = THRESHOLDS.get("oil_iron_ppm", {}).get("watch", 25)
        if oil is not None:
            oil_pen = _oil_penalty_kwh_day(oil, oil_watch)
            if oil_pen > 0:
                energy_penalties["oil_contamination"] = round(oil_pen, 1)
                total_kwh_day += oil_pen

        if total_kwh_day == 0:
            continue  # no penalties, skip

        # Convert kWh to GJ then to CO2
        gj_day  = total_kwh_day * 0.0036
        co2_day = gj_day * _EMISSION_FACTOR_KG_CO2_PER_GJ

        # Projected shutdown carbon impact (one-off, if equipment fails unplanned)
        risk, _, has_tel = compute_anomaly_risk(attrs)
        if not has_tel:
            risk = attrs.get("risk_level", "LOW")
        shutdown_data = _UNPLANNED_SHUTDOWN.get(risk, {})
        shutdown_co2  = 0.0
        flare_hours   = shutdown_data.get("flare_hours", 0)
        if flare_hours > 0:
            flare_co2 = flare_hours * _FLARE_CO2_KG_PER_HOUR
            flare_ch4_co2e = flare_hours * _FLARE_CH4_KG_PER_HOUR * 84  # GWP
            shutdown_co2 = flare_co2 + flare_ch4_co2e
            total_shutdown_co2_kg += shutdown_co2

        total_energy_waste_kwh_day += total_kwh_day
        total_co2_waste_kg_day     += co2_day

        equipment_impacts.append({
            "tag":              tag,
            "name":             name,
            "risk":             risk,
            "energy_waste_kwh_day": round(total_kwh_day, 1),
            "co2_waste_kg_day":     round(co2_day, 2),
            "co2_waste_annual_t":   round(co2_day * 365 / 1000, 2),
            "energy_penalties":     energy_penalties,
            "shutdown_co2_kg":      round(shutdown_co2, 1),
            "flare_hours":          flare_hours,
        })

    # Sort by energy waste descending
    equipment_impacts.sort(key=lambda x: -x["energy_waste_kwh_day"])

    # CDU baseline CO2 (clean running)
    baseline_gj_day = _CDU_ENERGY_GJ_PER_DAY
    baseline_co2_day = baseline_gj_day * _EMISSION_FACTOR_KG_CO2_PER_GJ

    # Waste percentage
    waste_pct = (total_co2_waste_kg_day / baseline_co2_day * 100) if baseline_co2_day else 0

    # Recommendations
    recommendations = []
    if equipment_impacts:
        worst = equipment_impacts[0]
        recommendations.append(
            f"Repair {worst['tag']} first — it accounts for the largest energy penalty "
            f"({worst['energy_waste_kwh_day']} kWh/day extra, {worst['co2_waste_kg_day']:.1f} kg CO2/day)."
        )
    vib_items = [e for e in equipment_impacts if "vibration" in e["energy_penalties"]]
    if vib_items:
        recommendations.append(
            f"Vibration-induced efficiency losses detected on {len(vib_items)} pump(s). "
            f"Balancing and alignment checks can recover up to 4% pump efficiency."
        )
    oil_items = [e for e in equipment_impacts if "oil_contamination" in e["energy_penalties"]]
    if oil_items:
        recommendations.append(
            f"Oil contamination on {len(oil_items)} unit(s) is increasing friction losses. "
            f"Oil change will reduce energy consumption and extend bearing life."
        )
    if total_shutdown_co2_kg > 0:
        recommendations.append(
            f"Preventing unplanned shutdowns on current CRITICAL/HIGH equipment would avoid "
            f"approximately {total_shutdown_co2_kg:.0f} kg CO2-equivalent from flaring."
        )
    recommendations.append(
        "All figures are estimates based on standard petroleum refinery operational benchmarks. "
        "Connect live DCS/SCADA data for measured values."
    )

    return {
        "summary": {
            "total_energy_waste_kwh_day":  round(total_energy_waste_kwh_day, 1),
            "total_co2_waste_kg_day":      round(total_co2_waste_kg_day, 2),
            "total_co2_waste_annual_t":    round(total_co2_waste_kg_day * 365 / 1000, 2),
            "baseline_co2_kg_day":         round(baseline_co2_day, 1),
            "waste_percentage":            round(waste_pct, 2),
            "shutdown_co2_risk_kg":        round(total_shutdown_co2_kg, 1),
            "equipment_with_penalties":    len(equipment_impacts),
        },
        "equipment":       equipment_impacts,
        "recommendations": recommendations,
        "disclaimer": (
            "Figures are engineering estimates derived from equipment telemetry and standard "
            "petroleum refinery benchmarks (IPCC emission factors, typical CDU energy profiles). "
            "They represent the additional carbon burden from degraded vs optimal operation, "
            "not absolute plant emissions. Connect to live DCS/SCADA for measured values."
        ),
    }
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