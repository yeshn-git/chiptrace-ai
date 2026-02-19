# ChipTrace AI

### Metric Tree-Driven Supply Chain Performance Analysis System

> Built for the DSU Ideathon — AI-driven solutions for India's semiconductor self-reliance.
> Automotive Legacy Semiconductor Supply Chain | OEM to N-Tier Visibility | Real-Time Risk Intelligence

---

## Table of Contents

1. [What is ChipTrace AI](#1-what-is-chiptrace-ai)
2. [Quick Start](#2-quick-start)
3. [Project Structure](#3-project-structure)
4. [Backend — FastAPI + SQLAlchemy](#4-backend--fastapi--sqlalchemy)
5. [Metric Tree Engine](#5-metric-tree-engine)
6. [Machine Learning Models](#6-machine-learning-models)
7. [API Reference](#7-api-reference)
8. [Frontend — React + Glassmorphism UI](#8-frontend--react--glassmorphism-ui)
9. [Frontend UI Overhaul (Branch: Frontend-Overhaul)](#9-frontend-ui-overhaul-branch-frontend-overhaul)
10. [Data Pipeline](#10-data-pipeline)
11. [Infrastructure & Docker](#11-infrastructure--docker)
12. [Tech Stack](#12-tech-stack)

---

## 1. What is ChipTrace AI

ChipTrace AI is a supply chain intelligence platform built specifically for **automotive legacy semiconductor supply chains**. It solves a fundamental problem with traditional flat supply chain reporting: a delayed shipment notification tells you *what* went wrong, but gives no visibility into *why* or *how far upstream* the root cause originated.

**The core innovation** is a **Metric Tree** — a weighted, hierarchical scoring graph that spans from raw Tier-3 material suppliers all the way up through Tier-2 fabs, Tier-1 chip suppliers, and finally the OEM assembly line. Every leaf node in the tree is scored 0–100 from live database records. Those scores propagate upward using weighted aggregation, so a single critical node failure (e.g., Taiwan fab utilization overload) cascades visibly to the OEM-level health score in real time.

**Key capabilities:**
- Real-time metric tree with 76+ nodes across 4 hierarchical tiers
- ML-powered disruption prediction (delay, resolution time, OEM impact)
- One-click scenario simulation to test disruption cascades
- Flat vs. hierarchical reporting comparison to demonstrate intelligence gap
- Supplier network visualization with geopolitical and financial risk scoring
- Auto-refreshing disruption log with severity classification

---

## 2. Quick Start

### With Docker (Recommended)

```bash
git clone https://github.com/rohitnor/chiptrace-ai.git
cd chiptrace-ai
docker-compose up --build
```

| Service  | URL                              |
|----------|----------------------------------|
| Frontend | http://localhost:3000            |
| Backend  | http://localhost:8000            |
| API Docs | http://localhost:8000/docs       |

### Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# Optional: set PostgreSQL URL (defaults to SQLite if not set)
set DATABASE_URL=postgresql://chiptrace:chiptrace123@localhost:5432/chiptrace_db

# Seed the database
python data/generate_dataset.py
python data/seed_db.py

# Train ML models
python ml/train_delay_model.py
python ml/train_resolution_model.py
python ml/train_impact_model.py

# Start the API server
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

---

## 3. Project Structure

```
chiptrace-ai/
│
├── backend/
│   ├── main.py                          # FastAPI application entry point, CORS, router registration
│   ├── database.py                      # SQLAlchemy engine — SQLite (dev) / PostgreSQL (prod)
│   ├── requirements.txt                 # Python dependencies
│   │
│   ├── models/
│   │   └── db_models.py                 # ORM table definitions (Supplier, DisruptionEvent, etc.)
│   │
│   ├── services/
│   │   ├── metric_tree.py               # Core metric tree scoring and propagation engine
│   │   ├── alert_engine.py              # RED/AMBER alert detection and root cause tracing
│   │   └── ml_service.py               # ML model loading, inference, and prediction dispatch
│   │
│   ├── routers/
│   │   ├── tree.py                      # GET /api/metric-tree/* — snapshot, alerts, nodes
│   │   ├── disruptions.py              # GET /api/disruptions — historical disruption log
│   │   ├── predict.py                  # GET /api/predict/* — ML predictions
│   │   ├── compare.py                  # GET /api/compare/flat-vs-tree — reporting comparison
│   │   ├── simulate.py                 # POST /api/simulate/disruption — scenario injection
│   │   └── suppliers.py               # GET /api/suppliers — N-tier supplier network
│   │
│   ├── data/
│   │   ├── generate_dataset.py          # Synthetic dataset generation (Faker + domain rules)
│   │   └── seed_db.py                  # Populates the database from generated CSVs
│   │
│   └── ml/
│       ├── train_delay_model.py         # XGBoost model: disruption → delay days
│       ├── train_resolution_model.py    # LightGBM model: disruption → resolution days
│       ├── train_impact_model.py        # scikit-learn model: disruption → OEM impact days
│       └── models/                      # Saved .pkl model files (auto-generated on training)
│
├── frontend/
│   ├── tailwind.config.js               # Tailwind config with custom colors, shadows, fonts
│   └── src/
│       ├── index.js                     # App shell, BrowserRouter, Nav component
│       ├── index.css                    # Global CSS — glass utilities, cinematic gradient, animations
│       │
│       ├── api/
│       │   └── index.js                 # Axios API client (all endpoint calls)
│       │
│       ├── hooks/
│       │   └── index.js                 # useMetricTree, useAlerts, usePredictions (with polling)
│       │
│       ├── pages/
│       │   ├── Dashboard.jsx            # Main view — KPI grid, metric tree, sidebar panels
│       │   ├── CompareView.jsx          # Flat vs. hierarchical reporting comparison
│       │   ├── DisruptionDetail.jsx     # Full disruption event log table
│       │   └── SupplierNetwork.jsx      # N-tier supplier risk map and table
│       │
│       └── components/
│           ├── MetricTree/
│           │   ├── TreeCanvas.jsx       # D3.js interactive metric tree canvas
│           │   ├── AlertBanner.jsx      # Dismissible critical alert banners
│           │   ├── DisruptionsPanel.jsx # Live-polling disruptions sidebar panel
│           │   └── SimulatePanel.jsx    # Scenario simulation controls
│           └── Predictions/
│               └── PredictionCards.jsx  # ML prediction metric cards + node inspector
│
├── docker-compose.yml                   # Three-service stack: db, backend, frontend
└── README.md
```

---

## 4. Backend — FastAPI + SQLAlchemy

### Entry Point (`main.py`)

The FastAPI application registers six routers under `/api/*`, configures CORS to allow the React dev server on port 3000, and auto-creates all database tables on startup via `Base.metadata.create_all()`.

### Database Layer (`database.py`)

The database layer uses **dual-mode configuration**: if the `DATABASE_URL` environment variable is set to a PostgreSQL URI, it uses PostgreSQL (production/Docker mode). If the variable is absent, it automatically falls back to a local **SQLite** file (`chiptrace.db`) for zero-config development. This makes local testing require no infrastructure setup whatsoever.

```python
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")
```

### Routers

| Router | Prefix | Purpose |
|--------|--------|---------|
| `tree.py` | `/api/metric-tree` | Returns scored metric tree snapshot, node details, alerts |
| `disruptions.py` | `/api/disruptions` | Paginated disruption event log |
| `predict.py` | `/api/predict` | ML inference endpoints for delay, resolution, OEM impact |
| `compare.py` | `/api/compare` | Side-by-side flat vs. hierarchical reporting data |
| `simulate.py` | `/api/simulate` | Injects synthetic disruptions and recalculates tree scores |
| `suppliers.py` | `/api/suppliers` | Full supplier network with tier, country, risk scores |

---

## 5. Metric Tree Engine

**File:** `backend/services/metric_tree.py`

This is the analytical core of ChipTrace AI. It defines a static directed acyclic graph (DAG) with **76 nodes** across 4 levels of the automotive semiconductor supply chain.

### Tree Structure

```
Root: Supply Chain Health Score
├── Delivery Timeliness (weight: 0.35)
│   ├── Supplier Lead Time
│   │   ├── Wafer Fabrication Cycle Time          [leaf]
│   │   ├── OSAT Assembly & Test Duration         [leaf]
│   │   ├── Tier-2 Raw Material Arrival Lag       [leaf]
│   │   └── Tier-3 Chemical/Gas Supply Delay      [leaf]
│   ├── Transit & Logistics
│   │   ├── Port Congestion Index                 [leaf]
│   │   ├── Air Freight Availability              [leaf]
│   │   ├── Incoterms Compliance Rate             [leaf]
│   │   └── Last-Mile Delivery to OEM Plant       [leaf]
│   └── OEM Line Readiness
│       ├── Kanban Signal Accuracy                [leaf]
│       ├── Dock-to-Stock Processing Time         [leaf]
│       └── Line-Side Buffer Stock (days)         [leaf]
│
├── Order Accuracy & Quality (weight: 0.30)
│   ├── Chip-Level Quality
│   │   ├── Incoming Inspection Reject Rate (ppm) [leaf]
│   │   ├── AEC-Q100 Electrical Parameter Drift   [leaf]
│   │   ├── ESD Damage Rate During Transit        [leaf]
│   │   └── Counterfeit Component Detection Rate  [leaf]
│   ├── Batch & Traceability
│   │   ├── Lot Traceability Score (fab to OEM)   [leaf]
│   │   ├── Date Code Mismatch %                  [leaf]
│   │   └── Certificate of Conformance %          [leaf]
│   └── Demand Signal Accuracy
│       ├── Forecast vs Actual Deviation          [leaf]
│       ├── Bullwhip Effect Index                 [leaf]
│       └── ECN Response Lag                      [leaf]
│
├── Supply Resilience (weight: 0.25)
│   ├── Fab Concentration & Capacity Risk
│   │   ├── Geographic Fab Concentration Index
│   │   │   ├── % Volume from Taiwan Fabs         [leaf]
│   │   │   ├── % Volume from China Fabs          [leaf]
│   │   │   └── % Volume from Japan/Korea Fabs    [leaf]
│   │   ├── Node Obsolescence Risk
│   │   │   ├── Legacy Node Retirement Timeline   [leaf]
│   │   │   ├── Active Fabs Running Target Node   [leaf]
│   │   │   └── Fab Capex Investment Trend        [leaf]
│   │   └── Fab Utilization Rate
│   │       ├── Current Fab Load vs Rated %       [leaf]
│   │       ├── Priority Queue (Auto vs Consumer) [leaf]
│   │       └── Wafer Start Flexibility           [leaf]
│   ├── OSAT Vulnerability, Material Traceability,
│   │   Demand Shock Absorption, Logistics Infra,
│   │   and Early Warning Signals ...             [leaves]
│
└── Regulatory & Compliance Risk (weight: 0.10)
    ├── Export Control Exposure (ITAR/EAR)        [leaf]
    ├── REACH/RoHS Material Compliance %          [leaf]
    ├── Conflict Mineral Audit Score (3TG)        [leaf]
    └── Carbon Footprint per Chip (Scope 3)       [leaf]
```

### Scoring Algorithm

1. **Leaf scoring** — Each leaf node is scored 0–100 from real database records (disruption history, supplier risk scores, logistics events)
2. **Upward propagation** — Internal nodes are computed bottom-up using weighted averages via NetworkX topological sort
3. **Status thresholds** — `score >= 70` → Green, `score >= 40` → Amber, `score < 40` → Red
4. **Root cause tracing** — Starting from any node, the engine follows the lowest-scoring child at each level to identify the single most critical leaf causing the parent degradation

---

## 6. Machine Learning Models

**Directory:** `backend/ml/`  
**Inference service:** `backend/services/ml_service.py`

Three independent supervised models handle different prediction targets. All models are trained on the synthetic disruption dataset and serialized to `.pkl` files.

| Model | File | Algorithm | Target | Features |
|-------|------|-----------|--------|----------|
| Delay Predictor | `train_delay_model.py` | XGBoost | `predicted_delay_days` | disruption type, severity, affected node count, supplier tier |
| Resolution Predictor | `train_resolution_model.py` | LightGBM | `predicted_resolution_days` | same feature set |
| OEM Impact Predictor | `train_impact_model.py` | scikit-learn (GBM) | `oem_impact_days` | same feature set |

### Prediction Endpoint

```
GET /api/predict/full
```

Returns a consolidated `summary` object used by the frontend `PredictionCards` component:

```json
{
  "summary": {
    "predicted_delay_days": 14.0,
    "predicted_resolution_days": 25.0,
    "oem_impact_days": 8.5,
    "disruption_type": "fab_capacity",
    "severity": "medium"
  }
}
```

### Scenario Simulation

```
POST /api/simulate/disruption?disruption_type=fab_capacity&severity=critical
```

The simulation endpoint:
1. Creates a synthetic `DisruptionEvent` record in the database
2. Identifies which metric tree leaf nodes are affected by the disruption type
3. Degrades their scores proportionally to the severity level
4. Re-runs the full tree propagation
5. Returns the count of cascaded RED nodes and the deepest root-cause leaf

---

## 7. API Reference

All endpoints are also available via Swagger UI at **http://localhost:8000/docs**.

### Metric Tree

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/metric-tree/snapshot` | Full scored tree (all nodes, root score, status) |
| `GET` | `/api/metric-tree/alerts` | All RED nodes with root cause traces |
| `GET` | `/api/metric-tree/node/{node_id}` | Detail for a specific node |
| `GET` | `/api/metric-tree/tree-definition` | Static tree structure (no scores) |

### Disruptions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/disruptions?limit=N` | Paginated disruption event log |
| `GET` | `/api/disruptions/{id}` | Single disruption detail |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/predict/full` | All three ML predictions consolidated |
| `GET` | `/api/predict/delay` | Delay prediction only |
| `GET` | `/api/predict/resolution/{type}/{severity}` | Resolution time for given scenario |

### Simulation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/simulate/disruption?disruption_type=X&severity=Y` | Inject disruption and cascade scores |

### Suppliers

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/suppliers` | Full supplier list |
| `GET` | `/api/suppliers?tier=N` | Filter by tier (0–3) |
| `GET` | `/api/suppliers/{id}/events` | Events for a specific supplier |

### Compare

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/compare/flat-vs-tree` | Side-by-side data for the Compare page |

---

## 8. Frontend — React + Glassmorphism UI

**Stack:** React 18, React Router v6, D3.js (v7), Recharts, Axios, Tailwind CSS, Inter (Google Fonts)

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | KPI grid, interactive D3 metric tree, ML predictions, disruptions panel, scenario simulator |
| Compare | `/compare` | Side-by-side flat versus hierarchical reporting with root cause trace |
| Disruptions | `/disruptions` | Full historical disruption event log with severity badges |
| Suppliers | `/suppliers` | N-tier supplier table with financial health and geopolitical risk bars |

### Custom React Hooks (`src/hooks/index.js`)

| Hook | Polling | Description |
|------|---------|-------------|
| `useMetricTree` | Every 30s | Fetches `/api/metric-tree/snapshot`, exposes `data`, `loading`, `error`, `refetch` |
| `useAlerts` | WebSocket + HTTP fallback | Subscribes to `/ws/alerts`, falls back to polling gracefully |
| `usePredictions` | Every 60s | Fetches `/api/predict/full`, provides structured fallback on error |

### API Client (`src/api/index.js`)

Centralized Axios instance pointed at `REACT_APP_API_URL` (defaults to `http://localhost:8000`). All API calls go through named exports for easy mocking and testing.

---

## 9. Frontend UI Overhaul (Branch: `Frontend-Overhaul`)

This is a complete visual redesign of the entire frontend, committed to the `Frontend-Overhaul` branch. It does not touch any backend code.

### Global Styles (`src/index.css`)

**Background:**
A full-screen, fixed cinematic gradient flowing from pure `#000000` at the top, through dark zinc (`#131313`), and fading out to a warm silver-gray (`#3a3a3a`) at the bottom. A radial white-light bloom is layered at the top to create depth. The gradient is `background-attachment: fixed` so it stays still while content scrolls.

**Inter Font:**
Loaded from Google Fonts and applied globally via CSS `font-family`. Replaces the previous system font stack.

**Custom Glass Utility Classes (`@layer utilities`):**

| Class | Use Case | Blur | Background |
|-------|----------|------|------------|
| `.glass` | Active nav links, chips, callout boxes | `blur(20px)` | `rgba(255,255,255, 0.05)` |
| `.glass-dark` | Primary panels, tree container, sidebar | `blur(24px)` | `rgba(0,0,0, 0.45)` |
| `.glass-card` | KPI metric cards (with hover lift) | `blur(16px)` | `rgba(255,255,255, 0.04)` |
| `.glass-nav` | Sticky navigation bar | `blur(28px)` | `rgba(0,0,0, 0.55)` |

All glass classes include:
- A subtle `inset` top-edge highlight (`rgba(255,255,255, 0.07)`) to simulate a light refraction edge
- A deep `box-shadow` underneath to create elevation
- Semi-transparent `border` using white alpha instead of hard grays

**Muted Badge System — No Bright Neons:**

| Class | Color | Use Case |
|-------|-------|----------|
| `.badge-critical` | Dark burgundy `#8B2020` → text `#D97070` | Critical severity, RED nodes, active alerts |
| `.badge-warn` | Dark ochre `#7A5C1E` → text `#C9A84C` | Amber/medium severity, warning states |
| `.badge-ok` | Deep forest `#1E5C3A` → text `#5BAD82` | Green/resolved/healthy states |
| `.badge-info` | Steel blue `#1E3A5C` → text `#6AABCF` | Neutral informational callouts |

**Animations:**

| Class / Keyframe | Type | Description |
|-----------------|------|-------------|
| `.fade-up` | Entrance | Slides content up 16px + fades in, with a spring curve |
| `.fade-in` | Entrance | Simple opacity fade |
| `.scale-in` | Entrance | Scales from 96% + fades in |
| `.shimmer` | Loading skeleton | Horizontal sweep shimmer over a dark base |
| `.alert-pulse` | Looping | Critical node glow pulse using `drop-shadow` filter |
| `.animate-spin` | Looping | Spinner for loading states |

### Navigation (`src/index.js`)

- **Glass nav bar** with `glass-nav` class and `sticky top-0 z-50` positioning
- **Logo mark** — a 2x2 grid of four squares SVG rendered in translucent white
- **Brand text** — "ChipTrace" in bold followed by "AI" in muted gray weight
- **Nav links** — uppercase, `tracking-widest`, `text-xs`. Active link uses `.glass` pill background
- **Live status indicator** — a pulsing green dot with the text "LIVE" in a small glass pill on the right

### Dashboard (`src/pages/Dashboard.jsx`)

- **KPI Cards** (`glass-card` class) — 4-column grid with staggered `fade-up` animations (0ms, 60ms, 120ms, 180ms delays)
- System Health score color-coded by status (burgundy/ochre/forest per the badge palette)
- **Metric Tree panel** — `glass-dark` with an "Interactive" live-pulse chip in the header
- **Right sidebar** — three stacked `glass-dark` panels: ML Predictions, Recent Disruptions, Scenario Simulation
- Loading and error screens match the cinematic aesthetic (spinner, glass error card)

### Compare View (`src/pages/CompareView.jsx`)

- Two full-height `glass-dark` panels side by side
- **Flat Report** header tinted burgundy with a "Legacy" badge
- **Metric Tree** header tinted forest-green with an "Intelligent" badge
- Root cause trace rendered as cascading arrow chains with per-status badges
- Insight boxes use matching tinted glass (red-tinted for flat, green-tinted for hierarchical)

### Disruption Detail (`src/pages/DisruptionDetail.jsx`)

- Three stat cards at the top: Total Events, Active (brick red), Resolved (forest green)
- Full-width `glass-dark` disruption table with muted badge severity indicators
- Date formatted as `DD Mon YY` using `en-IN` locale
- Empty state uses a centered glass tile with an "OK" label

### Supplier Network (`src/pages/SupplierNetwork.jsx`)

- Filter pill buttons at the top with per-tier accent colors (blue/green/amber/purple)
- Tier summary cards are clickable — clicking a tier both filters the table and highlights the card border with that tier's color
- **Mini progress bars** instead of raw numbers for Financial Health and Geo Risk — bars change color based on value thresholds
- Tier badges (`T0`–`T3`) each have a unique color scheme

### Prediction Cards (`src/components/Predictions/PredictionCards.jsx`)

- Three cards with a color hierarchy: amber (delay) → orange (resolution) → burgundy (OEM impact)
- **Shimmer skeletons** shown while data loads instead of blank cards
- Selected Node inspector panel appears with score, status, and raw node ID

### Disruptions Panel (`src/components/MetricTree/DisruptionsPanel.jsx`)

- Each disruption card uses a per-severity tinted glass background and border
- Severity accent: critical/high (burgundy), medium (ochre), low (forest green)
- Live indicator dot in the panel header pulses in the critical-red color
- Empty state shows a green "OK" tile

### Simulate Panel (`src/components/MetricTree/SimulatePanel.jsx`)

- Disruption type select styled as a `glass-card` dropdown
- Severity buttons change background and text color based on their own severity level
- Trigger button styled as a muted burgundy glass button with hover darkening via `onMouseEnter`
- Result card animates in with `.scale-in` and uses a tinted glass style

### Alert Banner (`src/components/MetricTree/AlertBanner.jsx`)

- Replaces hard colored boxes with muted burgundy glass panels
- "CRITICAL" label is a small dark pill with tight uppercase tracking
- The entire banner has `.alert-pulse` (periodic glow via CSS `drop-shadow`)

### Tailwind Config (`tailwind.config.js`)

- `fontFamily.sans` set to `['Inter', ...]`
- Custom color tokens: `critical`, `warn`, `ok`, `info` (all desaturated, dark base + text variant)
- Extended `backdropBlur` values: `xs`, `sm`, `md`, `lg`, `xl`
- Custom `boxShadow.glass` and `boxShadow.glass-lg`
- `maxWidth['8xl']` for wide-layout support
- `fadeUp` keyframe registered in Tailwind for use as `animate-fade-up`

---

## 10. Data Pipeline

**Directory:** `backend/data/`

### `generate_dataset.py`

Generates synthetic but domain-accurate data using the `Faker` library and custom semiconductor supply chain rules:

- **Suppliers** across all 4 tiers with realistic country distributions (Taiwan, Japan, South Korea, Germany, USA, Netherlands, Switzerland)
- **DisruptionEvents** with type (`fab_capacity`, `logistics`, `quality`, `material`, `financial`), severity, OEM impact days, predicted vs actual resolution days
- **LeafScores** mapped to metric tree node IDs with realistic score distributions

Output goes to `data/generated/` as CSV files.

### `seed_db.py`

Reads the generated CSVs and inserts records into the database using SQLAlchemy bulk operations. Safe to re-run (clears and re-seeds).

---

## 11. Infrastructure & Docker

**File:** `docker-compose.yml`

Three services in a defined dependency chain:

| Service | Image / Build | Port | Notes |
|---------|--------------|------|-------|
| `db` | `postgres:16` | 5432 | Health-checked before backend starts |
| `backend` | `./backend/Dockerfile` | 8000 | Waits for `db` healthy signal |
| `frontend` | `./frontend/Dockerfile` | 3000 | Depends on backend |

The backend volume mounts `./backend:/app` so code changes hot-reload without rebuilding the image. The frontend mounts `./frontend/src:/app/src` for the same reason.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | SQLite fallback | PostgreSQL connection string |
| `REACT_APP_API_URL` | `http://localhost:8000` | Backend base URL for the frontend |
| `REACT_APP_WS_URL` | `ws://localhost:8000` | WebSocket URL for real-time alerts |

---

## 12. Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **API Framework** | FastAPI | 0.111.0 |
| **ASGI Server** | Uvicorn | 0.30.1 |
| **ORM** | SQLAlchemy | 2.0.31 |
| **Database (dev)** | SQLite | built-in |
| **Database (prod)** | PostgreSQL | 16 |
| **Validation** | Pydantic | 2.8.2 |
| **ML — Gradient Boost** | XGBoost | 2.0.3 |
| **ML — Gradient Boost** | LightGBM | 4.4.0 |
| **ML — Utilities** | scikit-learn | 1.5.1 |
| **ML — Explainability** | SHAP | 0.45.1 |
| **Data Processing** | pandas, numpy | 2.2.2 / 1.26.4 |
| **Graph Engine** | NetworkX | 3.3 |
| **Frontend Framework** | React | 18.3.1 |
| **Routing** | React Router | 6.24.1 |
| **Tree Visualization** | D3.js | 7.9.0 |
| **Charts** | Recharts | 2.12.7 |
| **HTTP Client** | Axios | 1.7.2 |
| **Styling** | Tailwind CSS | (react-scripts) |
| **Font** | Inter (Google Fonts) | — |
| **Containerization** | Docker + Compose | 3.9 |

---

> ChipTrace AI — Built at DSU Ideathon, February 2026.
> Demonstrating how hierarchical intelligence outperforms flat reporting for India's semiconductor supply chain resilience.
