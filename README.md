# Notebook Seven · Well Anomaly Monitor

Browser-native **Isolation Forest** anomaly detection dashboard for offshore well sensor data.  
No server. No cloud. No dependencies beyond a browser.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [File structure](#file-structure)
4. [How to build and run](#how-to-build-and-run)
5. [How inference works](#how-inference-works)
6. [CSV format](#csv-format)
7. [Latency benchmarking](#latency-benchmarking)
8. [CSV export (thesis data)](#csv-export)
9. [Extending the project](#extending-the-project)
10. [Design decisions & trade-offs](#design-decisions--trade-offs)

---

## What it does

Notebook Seven takes a window of 180 rows from an offshore well telemetry CSV (sensors: **P-TPT**, **T-TPT**, **P-MON-CKP**), extracts 15 statistical features (mean, std, min, max, median × 3 sensors), runs them through a pre-trained **Isolation Forest** (200 trees, exported from scikit-learn), and displays:

- Anomaly score (0–1, where > threshold = anomaly)
- Which feature is the primary driver (highest absolute z-score)
- A live telemetry line chart (sliding 180-sample window)
- A feature deviation bar chart (z-scores, colour-coded by severity)
- A streaming simulation (normal + anomaly injection modes)
- **Row-by-row CSV streaming** — plays back a real CSV file as if it were a live feed
- **Latency benchmark** — runs 1 000 inferences and reports mean, median, p95, p99
- **Full inference log** exportable as CSV for thesis analysis

Everything runs in the browser. The model is embedded at build time. No data leaves the device.

---

## Architecture

```
notebook7/
├── src/
│   ├── index.html          # Markup only — no inline JS or CSS
│   ├── style.css           # All styles — warm light theme
│   └── app.js              # UI controller, chart management, event wiring
├── workers/
│   └── inference.worker.js # Isolation Forest engine — runs off main thread
├── build/                  # Output directory (git-ignored)
│   ├── index.html          # Copied verbatim from src/
│   ├── style.css           # Copied verbatim from src/
│   └── bundle.js           # Generated: model + worker shim + app
├── build.js                # Build script (Node.js, no external deps)
├── model_config.json       # Exported IF model (from Python/sklearn)
└── package.json
```

### Data flow

```
CSV upload / simulation tick
        │
        ▼
  telemetryBuffer (sliding 180-sample window)
        │
        ▼  postMessage({ type:'infer', windowData, threshold })
  ┌─────────────────────────┐
  │  inference.worker.js    │  ← runs on a separate thread
  │  extractFeatures()      │    (UI stays responsive)
  │  200 × pathLength()     │
  │  anomalyScore()         │
  └─────────────────────────┘
        │
        ▼  postMessage({ type:'result', score, stats, latencyMs, … })
  handleInferenceResult()
        │
        ├── update score / banner / decision text
        ├── update sensor cards
        ├── refreshFeatChart()
        └── recordInference() → infLog[]  →  Export CSV
```

### Why a Web Worker?

The 200-tree traversal is synchronous and CPU-bound (~5–15 ms). Without a worker, this blocks the main thread every 600 ms, causing chart redraws to stutter. The worker runs the entire inference pipeline off-thread. The main thread only updates the DOM with the result.

The worker is embedded as a **Blob URL** at build time — no extra HTTP request, no server required, works on `file://`.

---

## File structure

| File | Purpose |
|------|---------|
| `src/index.html` | Pure markup. References `style.css` and `build/bundle.js`. No inline scripts. |
| `src/style.css` | All visual styles. CSS custom properties for theming. Responsive breakpoints at 1050px and 680px. |
| `src/app.js` | UI controller. Owns: chart init, worker communication, CSV validation/streaming, simulation, event log, CSV export. Does **not** contain any inference math. |
| `workers/inference.worker.js` | Isolation Forest engine. Contains: `extractFeatures`, `pathLength`, `computeScore`, `topContributor`, benchmark runner. Receives/posts structured messages. Does **not** touch the DOM. |
| `build.js` | Build step. Reads model JSON, injects it into the worker, creates Blob URL shim, assembles `bundle.js`. Run with `node build.js`. |
| `model_config.json` | Exported sklearn Isolation Forest. Keys: `scaler_mean`, `scaler_scale`, `max_samples`, `trees`. |

---

## How to build and run

### Prerequisites

- Node.js ≥ 16 (only for the build step — runtime has zero Node dependency)
- A `model_config.json` exported from your Python training script (see below)

### Build

```bash
# Clone / unzip the project
cd notebook7

# Place your model file here:
cp /path/to/your/model_config.json .

# Run the build
node build.js

# Optional: specify a different model path
node build.js --model /path/to/other_model.json
```

Build output goes to `build/`. The build script will:
1. Validate the model JSON (checks required keys + feature count)
2. Inject the model into the worker source
3. Create a Blob URL shim so the worker runs without a server
4. Assemble `bundle.js` (model + worker + app)
5. Copy `index.html` and `style.css`

### Run

```bash
# Option A — local server (recommended)
cd build
python3 -m http.server 8080
# open http://localhost:8080

# Option B — double-click
# Open build/index.html directly in Chrome/Edge/Firefox.
# The Blob URL shim means no server is needed.

# Option C — npm shortcut
npm run dev   # builds then serves
npm run build # build only
npm run serve # serve only (assumes already built)
```

### Exporting the model from Python

```python
import json, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ... train your model ...

def tree_to_dict(node, tree, feature_names):
    if tree.children_left[node] == -1:  # leaf
        return {"is_leaf": True, "size": int(tree.n_node_samples[node])}
    return {
        "is_leaf": False,
        "feature_idx": int(tree.feature[node]),
        "split_value": float(tree.threshold[node]),
        "left":  tree_to_dict(tree.children_left[node],  tree, feature_names),
        "right": tree_to_dict(tree.children_right[node], tree, feature_names),
    }

model_config = {
    "scaler_mean":  scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "max_samples":  int(clf.max_samples_),
    "offset":       float(clf.offset_),
    "trees": [
        {
            "features": est.tree_.feature[est.tree_.feature >= 0].tolist(),
            "tree": tree_to_dict(0, est.tree_, feature_names)
        }
        for est in clf.estimators_
    ]
}

with open("model_config.json", "w") as f:
    json.dump(model_config, f)
```

---

## How inference works

### Feature extraction

For each of the 3 sensors over the last 180 rows:

| Feature | Description |
|---------|-------------|
| mean | Arithmetic mean of the window |
| std | Population standard deviation |
| min | Minimum value |
| max | Maximum value |
| median | Middle value of sorted window |

This gives **15 features** total (5 × 3). They are z-score scaled using the `scaler_mean` and `scaler_scale` saved from training.

### Isolation Forest scoring

Each of the 200 trees is traversed recursively. At each internal node, the feature value is compared to a split threshold; at a leaf, the path length is corrected using the expected average path length `c(n)`:

```
c(n) = 2 × (ln(n-1) + 0.5772) - (2(n-1)/n)   [Euler–Mascheroni constant]
```

The anomaly score is:

```
score = 2^( -avg_path_length / c(max_samples) )
```

Scores close to 1.0 are anomalies; scores around 0.5 are normal.

### Anomaly explanation

After scoring, the dashboard finds the feature with the largest absolute z-score deviation from its training distribution. This is shown as the **primary driver** in the decision support text and highlighted in the bar chart (red bars = z > 2, amber = z > 1, green = normal).

---

## CSV format

The dashboard accepts CSV files with **at minimum** these columns (exact names):

| Column | Unit | Typical range |
|--------|------|--------------|
| `P-TPT` | MPa | 4.5 – 6.0 |
| `T-TPT` | °C | 65 – 85 |
| `P-MON-CKP` | MPa | 15 – 21 |

- Minimum **180 rows** required (one inference window)
- Additional columns are ignored
- The dashboard streams the file **row by row** at 600 ms/row, simulating a live feed
- A progress bar shows streaming position

Example header row:
```
timestamp,P-TPT,T-TPT,P-MON-CKP
2024-01-01T00:00:00,5.12,74.3,18.2
```

---

## Latency benchmarking

Click **Run 1000×** in the Latency Benchmark panel after any inference window is available (CSV or simulation).

The worker runs 1 000 consecutive inferences on the same window, collects each `performance.now()` delta (sub-millisecond resolution), sorts them, and returns:

| Metric | Description |
|--------|-------------|
| **Mean** | Average inference time |
| **Median** | 50th percentile — most representative single-call time |
| **p95** | 95th percentile — worst-case under normal conditions |
| **p99** | 99th percentile — tail latency |

These values correspond directly to the latency table in the thesis (§ Results). Expected values on a typical 2020+ laptop: median ~7–10 ms, p95 ~12–18 ms.

All benchmark results are also logged to the event log and can be exported via **Export CSV**.

---

## CSV export

Click **Export CSV** to download every inference recorded in the current session.

Each row contains:

| Column | Notes |
|--------|-------|
| `timestamp` | ISO 8601 |
| `source` | `csv-stream` or `sim` |
| `sim_mode` | `normal` / `anomaly` / `n/a` |
| `p_tpt_last`, `t_tpt_last`, `p_mon_last` | Last sensor value in window |
| `anomaly_score` | 6 decimal places |
| `threshold` | Threshold at time of inference |
| `classification` | `NORMAL` or `ANOMALY` |
| `top_feature` | e.g. `P-TPT_mean` |
| `top_z_score` | z-score of primary driver |
| `latency_ms` | **4 decimal places** — use for thesis average |
| `window_size` | Always 180 once full |
| `ptpt_mean/std/min/max/median` | All 5 window stats |
| `ttpt_mean/std/min/max/median` | All 5 window stats |
| `pmon_mean/std/min/max/median` | All 5 window stats |

To compute average latency in Python:
```python
import pandas as pd
df = pd.read_csv('notebook7_2024-01-01.csv')
print(df['latency_ms'].describe())
print('mean:  ', df['latency_ms'].mean())
print('median:', df['latency_ms'].median())
print('p95:   ', df['latency_ms'].quantile(0.95))
```

---

## Extending the project

### Add a new sensor

1. Retrain the model in Python with the new feature columns.
2. Re-export `model_config.json`.
3. Update `SENSORS` array in `workers/inference.worker.js`.
4. Update `extractFeatures()` to read the new column.
5. Add a sensor card in `src/index.html`.
6. Re-run `node build.js`.

### Change inference window size

Update `WINDOW_SIZE` in `src/app.js` (currently `180`).

### Change streaming speed

Update `SIM_INTERVAL_MS` in `src/app.js` (currently `600` ms per row).

### Retrain the model

Re-run your Python training script and replace `model_config.json`, then re-run `node build.js`. The dashboard re-embeds the new model automatically.

---

## Design decisions & trade-offs

| Decision | Rationale |
|----------|-----------|
| **Web Worker for inference** | Keeps UI at 60 fps during 200-tree traversal. Worker is skipped (not queued) if busy — this prevents a backlog during fast streaming. |
| **Model embedded at build time** | No `fetch()` needed → works on `file://`. No second file to distribute. Build step makes this reproducible rather than manual. |
| **Blob URL for worker** | Workers can't load from `file://` with a URL. A Blob URL is the standard cross-origin workaround. |
| **Sliding window, not batch** | Reflects real-time monitoring intent. Only the last 180 rows matter for the current well state. |
| **No framework (React/Vue)** | Dashboard is a single-purpose tool. Vanilla JS is faster to load, easier to audit, has zero dependency drift risk. |
| **PapaParse for CSV** | Handles edge cases (BOM, CRLF, quoted fields, type coercion) that manual split() would not. |
| **Separate HTML/CSS/JS** | Allows IDE syntax highlighting, linting, and diffing on each file independently. The build step reassembles them. |

---

*Notebook Seven · built for thesis use · offline · no telemetry*
