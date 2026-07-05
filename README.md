<div align="center">

# 🏭 PlantIQ
### Industrial Knowledge Intelligence Platform

**ET AI Hackathon 2026 · Problem Statement 8**

*Unified AI Brain for India's Heavy Industry*

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square&logo=flask)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange?style=flat-square)
![Voyage](https://img.shields.io/badge/Voyage_AI-voyage--4-teal?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![Languages](https://img.shields.io/badge/Languages-10-green?style=flat-square)
![Features](https://img.shields.io/badge/Features-22+-purple?style=flat-square)

</div>

---

## 🔥 The Problem

India's industrial plants are drowning in fragmented knowledge.

| Stat | Impact |
|------|--------|
| **35%** of working hours | Wasted searching for information *(McKinsey 2024)* |
| **7-12** disconnected systems | Average Indian heavy industrial plant *(NASSCOM-EY)* |
| **18-22%** of unplanned downtime | Caused by knowledge fragmentation *(BIS Research)* |
| **25%** of engineers retiring | Taking irreplaceable operational knowledge with them |

> 💡 **The Knowledge Cliff:** Once experienced engineers retire, their undocumented knowledge is gone forever. PlantIQ exists to prevent this.

---

## ✨ What PlantIQ Does

PlantIQ ingests real regulatory documents and plant data, builds a structured knowledge graph, and makes everything queryable through 5 specialist AI agents, on any device, in any language.

```
Ask: "Does our CDU fire protection meet OISD-116 and Factories Act 1948?"

PlantIQ: Searches 729 chunks across 6 real government PDFs
         Cross-references knowledge graph for compliance gaps
         Returns structured answer with page-level citations
         Flags 7 critical non-conformances with recommended actions
         Time to answer: < 30 seconds
         Traditional method: 4-6 hours
```

---

## 🤖 5 Specialist AI Agents

| Agent | Purpose |
|-------|---------|
| 📚 **Knowledge Copilot** | RAG Q&A with real document citations and confidence scores |
| 🔧 **Maintenance RCA** | Root cause analysis and corrective action recommendations |
| 📋 **Compliance Agent** | Factories Act 1948 and OISD standards gap detection |
| 🧠 **Lessons Learned** | Systemic incident pattern analysis and proactive warnings |
| 📈 **Predictive Maintenance** | Linear regression forecasting with threshold breach timelines |

All routed automatically by an **LLM Orchestrator**. No need to pick the right agent manually.

---

## 💥 22+ Intelligence Features

PlantIQ goes far beyond a RAG chatbot. Every feature below is implemented and running:

### 🔴 Core Intelligence
| Feature | What It Does |
|---------|-------------|
| ⚡ **Proactive Alert Engine** | Critical equipment risks surface automatically on page load before any question is asked |
| 💚 **Health Dashboard** | Live computed health scores (0-100) from real sensor telemetry, worst equipment first |
| 📄 **Shift Handover Report** | Auto-generated PDF with alerts, compliance gaps, degradation forecasts, and Q&A log |
| 🕸️ **Graph Explorer** | Force-directed 66-node knowledge graph, fully interactive |
| 📐 **P&ID Vision Parser** | Gemini Vision extracts equipment tags and connections from engineering drawings |

### 🧠 Analytical Intelligence
| Feature | What It Does |
|---------|-------------|
| 💥 **Risk Cascade Analyzer** | Simulate equipment failure: trace DIRECT, INDIRECT, and RESOURCE cascade effects with production loss in INR/hr |
| 🔮 **What-If Simulator** | Adjust any sensor value and instantly see projected risk, health score, and forecast. Zero API calls |
| 🔍 **Incident Pattern Matcher** | Describe symptoms in plain language, find matching historical incidents, root causes, and CAPAs |
| ⚡ **Work Order Priority Engine** | 8-signal composite scoring: risk + degradation + compliance + H2S duty + overdue WOs + spares + incidents |
| 🌿 **Carbon Impact Calculator** | IPCC 2006 emission factors applied to sensor degradation data. Identifies avoidable CO2 waste in tonnes/year |

### 🔧 Maintenance Intelligence
| Feature | What It Does |
|---------|-------------|
| 🔩 **Maintenance Window Optimizer** | Given a primary shutdown, find all co-maintenance candidates to minimize total plant downtime |
| 🔧 **Spare Parts Gap Analyzer** | Cross-reference equipment vs stock levels with financial exposure if equipment fails without spares |
| 📅 **Regulatory Deadline Tracker** | All compliance deadlines tracked with days-overdue computed live from today's date |
| 📉 **Degradation Forecasting** | Real linear regression on sensor history projects threshold breach dates 30 to 2,600 days ahead |
| 📊 **Equipment Anomaly Detection** | Composite risk scoring from 4 telemetry streams: vibration, bearing temp, oil iron, bearing hours |

### 🌐 Platform Features
| Feature | What It Does |
|---------|-------------|
| 🌐 **10 Languages** | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, English |
| 🎙️ **Voice In + Out** | Web Speech API for input, edge-tts + gTTS hybrid for output |
| 🤚 **Hands-Free Mode** | Continuous voice loop for field technicians |
| 🏭 **Multi-Tenant Architecture** | plant_id namespacing built in, ready for dozens of plants |
| ⚡ **Query Caching** | Repeated queries skip the embedding API entirely |
| 📱 **Mobile-First Design** | Fully tested on phone. Thumb-friendly, responsive |

---

## 📄 Real Regulatory Documents

PlantIQ is backed by **6 real government-issued PDFs** (729 chunks, Voyage-4 embeddings):

| # | Document | Source |
|---|----------|--------|
| 1 | OISD-STD-116: Fire Protection for Petroleum Refineries (amended 2022) | oisd.gov.in |
| 2 | OISD-STD-116 Revised Edition (June 2025) | oisd.gov.in |
| 3 | OISD-STD-117: Fire Protection for Petroleum Depots and Terminals | oisd.gov.in |
| 4 | OISD-STD-118: Layouts for Oil and Gas Installations (2025) | oisd.gov.in |
| 5 | OISD Safety Framework Report (2023) | oisd.gov.in |
| 6 | The Factories Act, 1948 | indiacode.nic.in |

> Not synthetic. Not fabricated. Real government standards used by Indian refineries.

> ⚠️ **Honest disclosure:** Plant operational data (equipment telemetry, incidents, CAPAs) is simulated for the demo. The regulatory documents are 100% real. In production, the plant connects its own DCS/SCADA data.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Document Sources                              │
│  PDF · DOCX · XLSX · TXT · P&ID Drawings (Gemini)      │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: Knowledge Store                               │
│  Voyage-4 Embeddings → FAISS Vector DB                  │
│  NetworkX Knowledge Graph (66 nodes, 68 edges,          │
│  16 relationship types, 8 node types)                   │
│  Multi-tenant plant_id namespacing · Query caching      │
├─────────────────────────────────────────────────────────┤
│  LAYER 3: AI Agent Layer                                │
│  Groq Llama 3.3 70B · LLM Orchestrator                 │
│  5 Specialist Agents · Graph-enriched RAG               │
│  Proactive alert engine · 22+ intelligence features     │
├─────────────────────────────────────────────────────────┤
│  LAYER 4: User Interfaces                               │
│  Web Chat · Mobile · 10 Panel Dashboards                │
│  Voice I/O · 10 Languages · Shift Report PDF            │
│  Graph Explorer · P&ID Parser                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| LLM | Groq Llama 3.3 70B | 10x faster than GPT-4o, handles 10 languages natively |
| Embeddings | Voyage AI voyage-4 | Best-in-class for technical domain text |
| Vector DB | FAISS | Windows-compatible, production-replaceable with Pinecone |
| Knowledge Graph | NetworkX | Full traversal API, 16 relationship types |
| P&ID Vision | Google Gemini 2.5 Flash | Best multimodal for engineering drawings |
| Voice TTS | edge-tts + gTTS hybrid | Microsoft neural voices + gTTS fallback |
| Backend | Flask + Python 3.9 | Lightweight, fully deployable |
| PDF Reports | ReportLab | Programmatic Shift Handover PDF generation |
| Deployment | Docker + docker-compose | Container-ready |

---

## 🚀 Setup

### Prerequisites
- Python 3.9+
- pipenv

### Installation

```bash
git clone https://github.com/Agent-A345/PlantIQ
cd PlantIQ
pip install -r requirements.txt
```

### 🔑 API Keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
VOYAGE_API_KEY=your_voyage_key_here
VOYAGE_API_KEY_2=your_voyage_key_2_here   # recommended for ingestion
GEMINI_API_KEY=your_gemini_key_here
```

- **Groq**: Free at [console.groq.com](https://console.groq.com)
- **Voyage AI**: Free tier at [voyageai.com](https://www.voyageai.com). Multiple keys recommended. Free tier has a 3 RPM limit, keys rotate automatically during ingestion.
- **Gemini**: Free tier at [aistudio.google.com](https://aistudio.google.com)

### ▶️ Run

```bash
# Step 1: Ingest documents into vector DB
pipenv run python demo_ingest.py

# Step 2: Start the server
pipenv run python server.py

# Step 3: Open in browser
# http://localhost:5000
```

### 📱 Access on Mobile

```bash
# Find your IP
ipconfig    # Windows
ifconfig    # Mac/Linux

# Open on phone (same WiFi or hotspot)
# http://YOUR_IP:5000
```

### 🐳 Docker

```bash
docker compose build
docker compose up
# Open http://localhost:5000
```

---

## 📁 Project Structure

```
PlantIQ/
├── agents.py              5 specialist agents + LLM orchestrator + query cache
├── knowledge_graph.py     Industrial knowledge graph (NetworkX, 22+ analytical functions)
├── knowledge_graph.pkl    Pre-built graph (66 nodes, 68 edges, 16 relationship types)
├── server.py              Flask backend, 30+ API routes
├── ingest.py              Document ingestion pipeline (FAISS + Voyage-4, multi-key rotation)
├── demo_ingest.py         Ingests demo_docs_refinery/ into voyage_faiss_db/
├── rag.py                 RAG chain with Groq LLM
├── pid_parser.py          P&ID parser using Gemini Vision
├── requirements.txt       Python dependencies
├── Dockerfile             Container definition
├── docker-compose.yml     Multi-service orchestration
├── static/
│   ├── index.html         Full frontend (10 panels + chat + graph + P&ID parser)
│   └── pid_cdu.svg        Demo CDU P&ID drawing
└── demo_docs_refinery/    6 real regulatory PDFs
```

---

## 💬 Try These Queries

```
# Compliance Agent
Does our CDU fire protection meet OISD-116 and Factories Act 1948?

# Knowledge Copilot
What safety monitoring systems protect fuel tank areas?

# Lessons Learned
What systemic safety gaps does the OISD Safety Framework Report identify?

# Maintenance RCA
Why did pump P-201A reach critical status and what should we do?

# Predictive Maintenance
Which equipment is most likely to fail in the next 30 days?
```

### 🎛️ Try These Panels (no query needed)

| Panel | How to Access | What to Look For |
|-------|--------------|-----------------|
| 💥 Cascade | Click **Cascade** in topbar, select P-101A | 9 impacted units, INR 6.95L/hr loss |
| 🔮 What-If | Click **What-If**, select P-201A, raise vibration slider | Instant risk recalculation |
| ⚡ Priorities | Click **Priorities** in topbar | P-201A at score 93, IMMEDIATE |
| 🌿 Carbon | Click the green **🌿** button, bottom-right | 13.4t avoidable CO2/year |
| 📅 Deadlines | Click **Deadlines** | 5 overdue items, H2S calibration 23 days late |
| 🔧 Spares | Click **Spares** | Impeller out of stock, INR 14.28 Cr exposure |

---

## 🌍 Business Impact

| Before PlantIQ | After PlantIQ |
|----------------|---------------|
| 4 to 6 hours per compliance query | Under 30 seconds |
| 35% of work hours lost to search | Under 5% |
| Post-audit compliance discovery | Proactive, continuous |
| Reactive equipment failure response | 30 to 90 days early warning |
| Manual maintenance planning | 8-signal automated prioritization |
| Carbon impact unmeasured | 13.4t avoidable CO2/year identified |

**TAM:** 6,000+ large Indian industrial plants · **SaaS:** INR 5 to 15L/plant/year · **Scale path:** Docker to AWS/Azure · Pinecone · Neo4j · OPC-UA DCS/SCADA

---

## ⚠️ Known Limitations

- **Voyage rate limits:** Free tier is 3 RPM per account. Use multiple API keys (VOYAGE_API_KEY through VOYAGE_API_KEY_9) for large document ingestion. The pipeline rotates keys automatically.
- **Groq 413 error:** Conversation history too long. Use the Clear Chat button between sessions.
- **Gemini 429/503:** Free tier quota on P&ID parsing. Wait and retry.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built by Team Razorbacks (solo) for ET AI Hackathon 2026*

**🏭 PlantIQ · Your plant's institutional knowledge, one question away.**

</div>
