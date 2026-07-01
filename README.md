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

> 💡 **The Knowledge Cliff:** Once experienced engineers retire, their knowledge is gone forever. PlantIQ exists to prevent this.

---

## ✨ What PlantIQ Does

PlantIQ ingests real regulatory documents and plant data, builds a structured knowledge graph, and makes everything queryable through 5 specialist AI agents — on any device.

```
Ask: "Does our CDU fire protection meet OISD-116 and Factories Act 1948?"

PlantIQ: Searches 729 chunks across 6 real government PDFs
         Cross-references knowledge graph for compliance gaps
         Returns structured answer with page-level citations
         Flags critical non-conformances with recommended actions
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
| 📈 **Predictive Maintenance** | Equipment failure forecasting from historical data |

All routed automatically by an **LLM Orchestrator** — no need to pick the right agent manually.

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

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Document Sources                              │
│  PDF · DOCX · XLSX · TXT · P&ID Drawings (Gemini)      │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: Knowledge Store                               │
│  Voyage-4 → FAISS Vector DB                            │
│  NetworkX Knowledge Graph (66 nodes, 68 edges)          │
│  Multi-tenant plant_id namespacing · Query caching      │
├─────────────────────────────────────────────────────────┤
│  LAYER 3: AI Agent Layer                                │
│  Groq Llama 3.3 70B · LLM Orchestrator                 │
│  5 Specialist Agents · Graph-enriched RAG               │
│  Proactive alert engine                                 │
├─────────────────────────────────────────────────────────┤
│  LAYER 4: User Interfaces                               │
│  Web Chat · Mobile · Graph Explorer · P&ID Parser       │
│  Plant Selector (multi-tenant) · Alert Banner           │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Capabilities

**🔴 Proactive Alert Engine**
Critical equipment risks surface automatically on page load, before the user asks anything. Click any alert to instantly fire a deep-dive agent query.

**🕸️ Interactive Knowledge Graph**
66+ nodes across 8 types. Force-directed graph explorer in the UI. Every agent response is enriched with graph context automatically.

**📐 P&ID Vision Parser**
Upload any engineering drawing. Gemini Vision extracts equipment tags, instruments, and process connections. Entities auto-populate the knowledge graph.

**🏭 Multi-Tenant Architecture**
Every document tagged with `plant_id`. Queries filter by site. `/api/plants` endpoint lists all active sites. Ready to scale across dozens of plants from one deployment.

**⚡ Query Caching**
Repeated queries skip the embedding API entirely. Cache persists across server restarts.

**📱 Mobile-First Design**
Fully tested on phone via hotspot. Hamburger nav, agent strip, plant selector, thumb-friendly touch targets.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq Llama 3.3 70B |
| Embeddings | Voyage AI (voyage-4) |
| Vector DB | FAISS |
| Knowledge Graph | NetworkX |
| P&ID Vision | Google Gemini 2.5 Flash |
| Backend | Flask + Python 3.9 |
| Frontend | Vanilla HTML/CSS/JS |
| Document Parsing | PyPDF2, python-docx, openpyxl, pytesseract |
| Deployment | Docker + docker-compose |

---

## 🚀 Setup

### Prerequisites
- Python 3.9+
- pipenv or pip

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
VOYAGE_API_KEY_2=your_voyage_key_2_here   # recommended, see note below
GEMINI_API_KEY=your_gemini_key_here
```

- **Groq**: Free at [console.groq.com](https://console.groq.com)
- **Voyage AI**: Free tier (200M tokens/account) at [voyageai.com](https://www.voyageai.com). Multiple keys recommended — Voyage free tier has a 3 RPM limit, multiple accounts rotate automatically during ingestion.
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
├── ingest.py              Document ingestion pipeline (FAISS + Voyage-4, multi-key rotation)
├── rag.py                 RAG chain with Groq LLM
├── server.py              Flask backend, all API routes
├── demo_ingest.py         Ingests demo_docs_refinery/ into voyage_faiss_db/
├── pid_parser.py          P&ID parser using Gemini Vision
├── knowledge_graph.py     Industrial knowledge graph (NetworkX)
├── knowledge_graph.pkl    Pre-built graph (66 nodes, 68 edges)
├── requirements.txt       Python dependencies
├── Dockerfile             Container definition
├── docker-compose.yml     Multi-service orchestration
├── static/
│   ├── index.html         Full frontend (chat + graph explorer + P&ID parser)
│   └── pid_cdu.svg        Demo CDU P&ID drawing (illustrative)
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
Why did pump P-201A reach critical status?

# Predictive Maintenance
Which equipment needs urgent attention?
```

---

## ⚠️ Known Limitations

- **Voyage rate limits**: Free tier is 3 RPM per account. Use multiple API keys (VOYAGE_API_KEY through VOYAGE_API_KEY_9) for ingestion of large document sets. The ingestion pipeline rotates keys automatically.
- **Groq 413 error**: Conversation history too long. Use the Clear Chat button between sessions.
- **Gemini 429/503**: Free tier quota on P&ID parsing. Wait and retry.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built by Team Razorbacks (solo) for ET AI Hackathon 2026*

</div>
