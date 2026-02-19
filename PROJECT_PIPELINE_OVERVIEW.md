# ChipTrace AI – End-to-End Project & Pipeline Overview

This document explains the full flow of the ChipTrace AI project so you can confidently present it end‑to‑end.

---

## 1. Problem & Goal

- **Problem**: Automotive OEMs rely on a fragile, globally distributed semiconductor supply chain. Disruptions in fabs, logistics, materials, or finance can translate into severe production delays.
- **Goal**: Provide a **live, explainable cockpit** for supply‑chain performance and risk:
  - A **metric tree** that visualizes health across multiple layers of the supply chain.
  - **ML models** that predict supply delay, disruption resolution time, and OEM production impact.
  - **Interactive UI** for alerts, drill‑downs, and what‑if simulations.

Tech stack (at a glance):

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL, Python (scikit‑learn, XGBoost, LightGBM).
- **Frontend**: React, TailwindCSS, D3.js.
- **Data/ML**: Synthetic dataset generation + three supervised ML models.

---

## 2. Data & Database Layer

### 2.1 Synthetic Data Generation

Location: `backend/data/`

- `generate_dataset.py`
  - Creates realistic **synthetic JSON datasets**:
    - `events.json` – events with demand, supply, and disruption context.
    - `disruptions.json` – disruption type, severity, durations.
    - `inventory.json` – inventory and stocking metrics.
    - `suppliers.json` – supplier risk/health metrics.
    - `macro_signals.json` – macroeconomic / geopolitical risk signals.
- The features resemble real‑world operational metrics (fab utilization, wafer supplier count, LTA coverage, port congestion, etc.).

### 2.2 Database & ORM

Locations:

- `backend/models/db_models.py` – SQLAlchemy models, e.g.:
  - `MetricSnapshot`: node_id, score, status, evaluated_at.
  - Disruption/supplier‑related tables (depending on schema).
- `backend/database.py` – engine + session management.
- In `backend/main.py`:
  - `Base.metadata.create_all(bind=engine)` creates all tables on startup.

### 2.3 Seeding the Database

Location: `backend/data/seed_db.py`

- Reads the generated JSON files.
- Inserts **metric snapshots** and sample disruptions into the DB.
- Result: the DB holds a time series of leaf metric scores and disruption history, which powers the metric tree and ML features.

**How to present this**: “We bootstrap a realistic supply‑chain universe using synthetic but structured data, then store it in a relational database as the source of truth for metrics and disruptions.”

---

## 3. Metric Tree Engine

### 3.1 Definition

Location: `backend/services/metric_tree.py`

- `METRIC_TREE_DEFINITION` describes the full metric tree:
  - Each node has `node_id`, `label`, optional `parent`, `category`, `weight`, etc.
  - **Leaf nodes** = concrete metrics (e.g., fab utilization, port congestion, LTA utilization, die bank, tier‑1 stock).
  - **Internal nodes** = roll‑ups (e.g., "Resilience", "Demand Shock", "Logistics").

### 3.2 Score Propagation

- `get_leaf_nodes()` returns all leaf node IDs.
- `propagate_scores(leaf_scores: Dict[node_id, score])`:
  - Starts from the leaf scores.
  - Rolls them up the tree using configured weights.
  - Produces a structure `node_id -> { score, status, label, parent }`.
  - `status` is derived from score thresholds (e.g., green / amber / red).

### 3.3 Snapshots & Alerts

Location: `backend/routers/tree.py`

- `get_current_leaf_scores(db)`:
  - For each leaf node, pulls the **most recent** `MetricSnapshot` from the DB.
  - If there is no historical data yet, it **simulates scores** with realistic ranges and injects some stressed nodes for demo.
- `/api/metric-tree/snapshot`:
  - Uses `get_current_leaf_scores` + `propagate_scores`.
  - Returns the entire tree (root + all nodes) with scores and statuses.
- `/api/metric-tree/alerts`:
  - Calls `get_all_alerts(all_scores)` to extract red/amber nodes.

**How to present this**: “We maintain a hierarchical metric tree, roll up leaf metrics into higher‑level resilience and logistics scores, and expose both the full tree and a filtered list of alerts.”

---

## 4. ML Pipeline (Training)

There are three main ML models, one for each prediction task. All training scripts live under `backend/ml/`.

### 4.1 Delay Prediction Model – `train_delay_model.py`

- **Purpose**: Predict `delay_days` (supply delay) from current supply‑chain state.
- **Features** (examples):
  - `fab_utilization_score`
  - `wafer_supplier_score`
  - `port_congestion_score`
  - `lta_coverage_pct`
  - `spot_exposure_pct`
  - `macro_signal_severity`
  - `die_bank_score`
  - `tier1_stock_days`
  - `financial_health`
  - `bullwhip_index`
  - plus encoded disruption risk and fill rate.
- **Model**: `xgboost.XGBRegressor`.
- **Outputs**:
  - `backend/ml/models/delay_model.pkl` – trained model.
  - `backend/ml/models/delay_model_features.json` – list of features + metrics.

### 4.2 Resolution Time Model – `train_resolution_model.py`

- **Purpose**: Predict how long a disruption will take to resolve.
- **Features**:
  - Encoded `disruption_type` (e.g., fab_capacity, logistics, material, financial).
  - Encoded `severity` (low / medium / high / critical).
  - Tree‑derived resilience metrics (die bank, tier‑1 finished goods, LTA utilization, etc.).
  - Binary structural features like single‑sourcing.
- **Model**: Random Forest regressor.
- **Outputs**:
  - `backend/ml/models/resolution_model.pkl` + `resolution_model_features.json`.

### 4.3 OEM Impact Model – `train_impact_model.py`

- **Purpose**: Translate delay and resolution into **OEM production impact days**.
- **Features**: Functions of delay, resolution, and chip criticality, plus context from generated data.
- **Model**: LightGBM regressor (`LGBMRegressor`).
- **Outputs**:
  - `backend/ml/models/impact_model.pkl` + `impact_model_features.json`.

**How to present this**: “We train three specialized ML models—one for delay, one for resolution time, and one for OEM impact—on top of synthetic supply‑chain event data. Each model has its own engineered feature set aligned with how that risk behaves in reality.”

---

## 5. ML Serving Layer

Location: `backend/services/ml_service.py`

### 5.1 Model Loading

- Model directory: `backend/ml/models/`.
- Functions:
  - `get_delay_model()` → loads `delay_model.pkl` lazily.
  - `get_resolution_model()` → loads `resolution_model.pkl`.
  - `get_impact_model()` → loads `impact_model.pkl`.
- Uses a shared `load_model(filename)` helper and caches instances in module‑level variables.

### 5.2 Feature Construction at Runtime

- `build_delay_features(tree_state, supplier_data=None)`:
  - Reads scores from the **current propagated tree** (e.g., `resilience.fab_concentration.utilization_rate.load_pct`).
  - Adds supplier health where available.
  - Produces a NumPy feature vector aligned with `delay_model_features.json`.
- `build_resolution_features(disruption_type, severity, tree_state=None)`:
  - Encodes disruption type and severity as integers.
  - Pulls tree‑based resilience features.
  - Adds structural defaults (single source, macro active, etc.).

This is where we fixed **feature shape mismatches** so runtime vectors match the training schema.

### 5.3 Prediction APIs (Python Level)

- `predict_delay(tree_state)` → returns:
  - `predicted_delay_days`
  - `confidence` level
  - `features_used`
  - `fallback` (true/false) if model not available
- `predict_resolution(disruption_type, severity, tree_state)` → similar contract.
- `predict_oem_impact(delay_days, resolution_days, chip_criticality)` → returns impact days + metadata.

**How to present this**: “The ML service converts the live metric tree into the exact feature vectors our models expect, runs the models, and returns structured JSON responses with predictions and metadata for the UI.”

---

## 6. API Layer (FastAPI)

### 6.1 App Setup – `backend/main.py`

- Creates the FastAPI app with title, description, version.
- Enables CORS for the React dev server (`http://localhost:3000`).
- Includes routers with prefixes and tags:
  - `/api/metric-tree` → tree visualization + alerts.
  - `/api/disruptions` → disruption listing.
  - `/api/predict` → ML predictions.
  - `/api/compare`, `/api/simulate`, `/api/suppliers` → comparison & simulation endpoints.
- Exposes health endpoints `/` and `/health`.

### 6.2 Prediction Routes – `backend/routers/predict.py`

- `GET /api/predict/delay`
  - Fetches current leaf scores via `get_current_leaf_scores(db)`.
  - Propagates them into full tree.
  - Calls `ml_service.predict_delay` and returns result.

- `GET /api/predict/resolution/{disruption_type}/{severity}`
  - Same tree state; adds the chosen disruption context.

- `GET /api/predict/impact/{delay_days}/{resolution_days}`
  - Uses provided durations + chip criticality (default).

- `GET /api/predict/full`
  - Runs all three models and returns a combined payload:
    - `delay`, `resolution`, `impact`, and a `summary` object.

### 6.3 Metric Tree Routes – `backend/routers/tree.py`

- `/api/metric-tree/snapshot`
  - Full tree flattened into a node list (used by D3 in the frontend).
- `/api/metric-tree/node/{node_id}`
  - Single node with history from `MetricSnapshot`.
- `/api/metric-tree/alerts`
  - All red/amber nodes, driving the alerts panel.

**How to present this**: “FastAPI exposes a clean set of endpoints that the frontend consumes: one family for the metric tree, one for disruptions, and one for ML predictions.”

---

## 7. Frontend Architecture (React + D3)

Frontend root: `frontend/`

### 7.1 Data Access

- `src/api/index.js`
  - Wraps HTTP requests to the backend APIs.
- `src/hooks/index.js`
  - Custom hooks like `useMetricTree`, `usePredictions`, etc.
  - Responsible for loading state, error handling, and refetching.

### 7.2 Main Pages – `src/pages/`

- `Dashboard.jsx`
  - Main “cockpit” view.
  - **Left**: “Supply Chain Metric Tree” panel with D3 visualization.
  - **Right**: ML predictions (cards) + disruptions and supporting panels.
- `CompareView.jsx`
  - Compare different scenarios / time periods.
- `DisruptionDetail.jsx`
  - Deep dive into a specific disruption path.
- `SupplierNetwork.jsx`
  - Overview of suppliers and their risk/health.

### 7.3 Key Components

- `components/MetricTree/TreeCanvas.jsx`
  - D3‑powered rendering of the metric tree.
  - Features:
    - Collapsible nodes: click to expand/collapse children.
    - Colored nodes by status (green / amber / red).
    - Zoom & pan (scroll to zoom, drag to pan, double‑click to reset).
    - Hover effects and subtle animations.
  - Consumes `treeData.nodes` from `/api/metric-tree/snapshot`.

- `components/MetricTree/AlertBanner.jsx`
  - Displays top alerts (red/amber nodes) from `/api/metric-tree/alerts`.

- `components/MetricTree/SimulatePanel.jsx`
  - UI for running what‑if simulations (adjusting metrics, re‑querying APIs).

- `components/Predictions/PredictionCards.jsx`
  - Shows three ML cards:
    - Predicted delay.
    - Resolution time.
    - OEM impact.
  - Data source: `/api/predict/full`.

### 7.4 Styling – `src/index.css` + Tailwind

- Dark, glassy aesthetic with gradients and glow effects.
- Custom D3 styles:
  - `.node`, `.link`, `.alert-pulse` animations.
  - Hover transitions, drop shadows, and pulse for critical nodes.

**How to present this**: “The React frontend uses custom hooks to fetch data and D3 to render a collapsible metric tree on the left, while Tailwind and CSS deliver a polished, dark‑mode analytics dashboard.”

---

## 8. End‑to‑End Runtime Flow

Use this as your spoken walkthrough during a demo.

1. **Load Dashboard (Frontend)**
   - React calls `/api/metric-tree/snapshot` and `/api/predict/full` via hooks.

2. **Backend Computes Current State**
   - `get_current_leaf_scores` reads the most recent `MetricSnapshot`s (or simulates if empty).
   - `propagate_scores` rolls leaf metrics up to compute all node scores and statuses.

3. **Metric Tree Rendered (Frontend)**
   - `TreeCanvas` receives `nodes` and builds a hierarchy.
   - Renders an interactive D3 tree with collapse/expand behavior.

4. **ML Predictions Generated (Backend)**
   - `/api/predict/full`:
     - Builds feature vectors from the current tree.
     - Runs delay, resolution, and impact models.
     - Returns a combined JSON result.

5. **Predictions Displayed (Frontend)**
   - `PredictionCards` show predicted delay days, resolution time, and OEM impact.
   - Confidence levels and statuses are shown for quick interpretation.

6. **User Interaction & Drill‑down**
   - User clicks nodes to expand parts of the tree or focus on a particular risk.
   - Alerts banner highlights critical nodes.
   - Simulation panel can be used (if wired) to tweak inputs and request updated predictions.

**One‑sentence summary**: “ChipTrace AI ingests synthetic but realistic supply‑chain data into a metric tree, trains three ML models on historical events, and exposes an interactive React+D3 dashboard where users can visualize health, see predicted delay and impact, and explore disruptions in real time.”

---

## 9. Suggested 60–90 Second Pitch

You can adapt this as a spoken explanation:

> “ChipTrace AI is a supply‑chain risk cockpit for automotive semiconductors. We model the supply chain as a metric tree, where each leaf is a real‑world metric like fab utilization, port congestion, or die bank, and internal nodes roll these up into resilience and logistics health scores.
>
> Under the hood, we generate realistic synthetic event and disruption data, store it in a relational database, and maintain time‑series snapshots for each metric. On top of that, we train three supervised ML models: one predicts supply delay, one predicts how long disruptions will take to resolve, and one estimates OEM production impact.
>
> At runtime, a FastAPI backend pulls the latest metric snapshots, propagates them through the tree, and then feeds the current state into the models. A React frontend with D3 renders an interactive, collapsible tree on the left and ML prediction cards on the right, with alerts and drill‑downs for risky nodes. The result is a live, explainable view of supply‑chain health and predicted impact that supply‑chain teams can use for monitoring and scenario planning.”
