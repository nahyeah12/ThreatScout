# 🛡️ ThreatScout
> **Explainable Hybrid Insider Threat Detection using Fuzzy Behavioral Fingerprinting**

ThreatScout models every employee as a **behavioral fingerprint** and detects insider threats by tracking deviations from their personal baseline using a hybrid Isolation Forest + Fuzzy Logic pipeline — with full plain-English explainability.

---

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [Implementation Details](#implementation-details)
   - [System Pipeline](#system-pipeline)
   - [Simulation Layer](#1-simulation-layer)
   - [Behavioral Fingerprint](#2-behavioral-fingerprint-corefeatuespy)
   - [Anomaly Detection](#3-anomaly-detection-coreanomalypy)
   - [Fuzzy Logic Engine](#4-fuzzy-logic-engine-corefuzzypy)
   - [Risk Aggregation](#5-risk-aggregation-corerisspy)
   - [Streamlit Dashboard](#6-streamlit-dashboard-apppy)
5. [Key Design Decisions](#key-design-decisions)
6. [CERT Dataset Integration](#cert-dataset-integration-planned)

---

## Setup & Installation

### Prerequisites

- Python **3.10+**
- `pip` (comes with Python)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/threatscout.git
cd threatscout
```

### 2. Create a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
streamlit>=1.32.0
plotly>=5.20.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

---

## Project Structure

```
threatscout/
│
├── app.py                   # Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
│
├── simulation/
│   ├── __init__.py
│   ├── agents.py            # Employee class + role baselines
│   └── behaviour.py         # Normal/insider behaviour simulation
│
└── core/
    ├── __init__.py
    ├── features.py          # Behavioral fingerprint (baseline/current/deviation)
    ├── anomaly.py           # Isolation Forest anomaly detector
    ├── fuzzy.py             # Fuzzy membership functions + rule base
    └── risk.py              # Risk aggregation + explainability
```

> Make sure each folder has an `__init__.py` (can be empty) so Python treats them as packages.

```bash
touch simulation/__init__.py core/__init__.py
```

---

## How to Run

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**.

### Using the Dashboard

| Action | How |
|---|---|
| Advance the simulation one time step | Click **▶ Advance One Step** in the sidebar |
| Inject an insider threat | Click **💉 Inject Insider** — picks one employee randomly and starts their behavioural drift |
| Inspect an employee | Select their ID from the dropdown in the main panel |
| Reset everything | Click **🔄 Reset Simulation** |

> **Tip:** Advance 8–10 steps after injecting an insider to see their risk score climb and the fuzzy rules fire.

---

## Implementation Details

### System Pipeline

```
Employee Simulation (agents.py + behaviour.py)
            ↓
    Behavioral Features
            ↓
  Behavioral Fingerprint          ← core/features.py
  (baseline | current | deviation)
            ↓
   Anomaly Detection              ← core/anomaly.py
   (Isolation Forest score 0–1)
            ↓
   Fuzzy Risk Inference           ← core/fuzzy.py
   (membership functions + rules)
            ↓
   Final Risk Score + Label       ← core/risk.py
   + Plain-English Explanation
            ↓
   Streamlit Dashboard            ← app.py
```

---

### 1. Simulation Layer

**Files:** `simulation/agents.py`, `simulation/behaviour.py`

#### Employees & Roles

Four roles are defined, each with a unique **baseline probability** for four behavioural signals:

| Signal | Meaning |
|---|---|
| `after_hours_login` | Probability of logging in outside business hours |
| `file_access` | Normalised file access frequency |
| `usb_usage` | USB device connection frequency |
| `job_sites` | Visits to recruitment / job-search websites |

Each `Employee` object stores:
- `baseline` — fixed role-based normal behaviour
- `current` — live behaviour updated each timestep
- `risk_score`, `risk_label`, `risk_history` — written by the risk layer
- `is_insider` — flag toggled by `inject_insider()`

#### Normal vs Insider Behaviour

**Normal employees** (`simulate_normal_behaviour`): each signal is sampled from a small Gaussian noise window (`±0.05`) around their personal baseline — modelling realistic day-to-day variation.

**Insider employees** (`simulate_insider_behaviour`): the same baseline noise is applied first, then a **gradual escalation factor** is added on top:

```
factor = min(1.0, step × 0.1)

after_hours_login += 0.3 × factor
file_access       += 0.4 × factor
usb_usage         += 0.5 × factor
job_sites         += 0.3 × factor
```

This models a realistic insider who starts normally and becomes progressively riskier over time — not a sudden step change.

---

### 2. Behavioral Fingerprint (`core/features.py`)

The fingerprint is the central data structure of the system. For each employee it holds three parallel vectors:

| Vector | Description |
|---|---|
| `baseline_vec` | Their historical normal (from role baseline) |
| `current_vec` | Observed behaviour at this timestep |
| `deviation_vec` | `\|current − baseline\|` unsigned, per-feature |

A scalar `deviation_score` (mean of `deviation_vec`) summarises total drift in one number.

The `feature_vector` property concatenates all three vectors into shape `(12,)`, which is the input fed to the anomaly model. This means the model sees not just *what* the employee is doing, but also *how far they've drifted* from their own norm.

---

### 3. Anomaly Detection (`core/anomaly.py`)

**Model:** `sklearn.ensemble.IsolationForest`

Isolation Forest detects anomalies by randomly partitioning the feature space — anomalous points are isolated in fewer splits and receive lower decision function scores.

#### Training

The detector is trained **once** on the clean initial population (before any insider is injected), using only the 4-dimensional `current_vec`. This establishes what "normal activity" looks like across the organisation.

#### Scoring

Isolation Forest's raw `decision_function` output is a real number where:
- Positive values → more normal
- Negative values → more anomalous

This is remapped to `[0, 1]`:

```
anomaly_score = clip(1.0 − (raw_score + 0.5), 0, 1)
```

So `anomaly_score = 0` means perfectly normal, and `anomaly_score = 1` means maximally anomalous.

---

### 4. Fuzzy Logic Engine (`core/fuzzy.py`)

The fuzzy layer converts raw numerical scores into **interpretable linguistic categories** and produces a human-readable explanation alongside every risk decision. It is implemented in pure NumPy — no external fuzzy library required.

#### Membership Functions

Both input variables (`anomaly_score` and `deviation_score`) are fuzzified using **trapezoidal membership functions** into three linguistic terms: `Low`, `Medium`, `High`.

Example — Deviation membership functions:

```
Low:    ▓▓▓░░░░░░░  0.0 ──── 0.10 ──── 0.20
Medium: ░░▓▓▓▓░░░░  0.10 ─── 0.20 ─── 0.30 ─── 0.40
High:   ░░░░░▓▓▓▓▓  0.30 ─── 0.40 ─────────── 1.0
```

#### Rule Base (9 rules, Mamdani-style AND)

| Anomaly | Deviation | → Risk |
|---|---|---|
| Low | Low | **Low** (0.10) |
| Low | Medium | **Low** (0.20) |
| Low | High | **Medium** (0.45) |
| Medium | Low | **Low** (0.25) |
| Medium | Medium | **Medium** (0.50) |
| Medium | High | **High** (0.75) |
| High | Low | **Medium** (0.50) |
| High | Medium | **High** (0.80) |
| High | High | **High** (0.95) |

Rules are combined using the `AND = min(μ_A, μ_D)` T-norm. The final `fuzzy_score` is the **weighted centroid** (centre-of-gravity defuzzification) across all fired rules.

The `explain()` function reads the membership degrees and dominant rule to generate a plain-English paragraph — no template strings, derived entirely from the live inference state.

---

### 5. Risk Aggregation (`core/risk.py`)

Combines both signals into a single `final_score`:

```
final_score = 0.4 × anomaly_score + 0.6 × fuzzy_score
```

The fuzzy score is weighted more heavily because it incorporates *both* the anomaly signal and the personal deviation — making it richer than the raw IF score alone.

The `risk_label` (`Low` / `Medium` / `High`) is derived from the fuzzy output (not the final score), since the fuzzy layer is the explainable component.

`update_risk_scores(employees, detector)` is the single call that runs every simulation step — it builds fingerprints, scores them, runs fuzzy inference, and writes everything back onto each `Employee` object in-place.

Employees with `final_score ≥ 0.65` are flagged by `get_alerts()`.

---

### 6. Streamlit Dashboard (`app.py`)

The UI is built entirely in Streamlit with Plotly charts. State is managed via `st.session_state` so the simulation persists across rerenders.

| Panel | What it shows |
|---|---|
| **KPI row** | Total employees, High/Medium risk counts, average score |
| **Alert banners** | Any employee above the 0.65 threshold |
| **Employee Risk Board** | Sortable table, colour-coded by risk label |
| **Radar chart** | Three overlapping polygons — baseline (blue), current (green), deviation (red) — visualising the behavioral fingerprint |
| **Fuzzy Explanation** | Plain-English paragraph generated by `fuzzy.explain()` |
| **Fired Rules panel** | Each rule that fired, with its strength as a progress bar |
| **Risk History** | Line chart across all timesteps; the insider is highlighted in red; alert threshold shown as dotted line |

---

## Key Design Decisions

**Why Isolation Forest?** It is unsupervised — no labelled insider data is needed for training. It works well on small tabular datasets and is fast enough to retrain in-browser if the CERT dataset is integrated.

**Why Fuzzy Logic over a second ML model?** Fuzzy logic makes the *decision boundary* transparent. A user can read the fired rules and understand exactly why someone is rated High risk. A neural network risk scorer would not offer this.

**Why personal baseline deviation instead of population deviation?** Comparing an employee only to the population mean would flag power users (e.g., an Admin with high file access) as risky even when behaving normally. Anchoring each score to the individual's own baseline eliminates this role-based false positive.

**Why weight fuzzy higher (60%) than anomaly (40%)?** The fuzzy score synthesises two signals (anomaly + deviation), so it carries more information. The weighting is exposed as a constant in `risk.py` and can be tuned.

---

## CERT Dataset Integration (Planned)

The CERT Insider Threat Dataset provides real-world logs that map directly to the four simulation features:

| CERT file | Feature |
|---|---|
| `logon.csv` | `after_hours_login` — filter by off-hours timestamps |
| `file.csv` | `file_access` — normalise daily file operation count |
| `device.csv` | `usb_usage` — USB connect events per day |
| `http.csv` | `job_sites` — filter URLs matching job/career domains |

To integrate:
1. Extract and normalise the four signals per user per day into a DataFrame
2. Call `detector.train_matrix(X)` with the clean-user rows
3. Feed each day's row through `BehavioralFingerprint` manually (bypassing the simulation layer)
4. The fuzzy and risk layers require no changes

---