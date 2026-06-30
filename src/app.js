/**
 * app.js — Notebook Seven
 * Main UI controller. Delegates inference to inference.worker.js.
 *
 * Responsibilities:
 *  - CSV upload + streaming playback (row-by-row with sliding window)
 *  - Simulation (synthetic data, normal + anomaly modes)
 *  - Chart management (telemetry line chart + feature deviation bar chart)
 *  - Worker communication (send window → receive result)
 *  - Event logging + CSV export of all inference records
 *  - Latency benchmarking display
 *  - All error handling + user feedback (toast system)
 */

'use strict';

// ── CONSTANTS ────────────────────────────────────────────────────────────────

const WINDOW_SIZE = 180;       // rows per inference window
const SIM_INTERVAL_MS = 600;   // streaming tick speed
const BENCH_RUNS = 1000;       // inferences for latency benchmark

// ── STATE ────────────────────────────────────────────────────────────────────

let threshold    = 0.60;
let lastScore, lastStats;
let totalAlarms  = 0;
let totalInf     = 0;
let telBuffer    = [];          // circular buffer of {ptpt,ttpt,pmon}
let sensorChart, featChart;
let simInterval  = null;
let simTick      = 0;
let simMode      = 'normal';
let workerBusy   = false;
let worker       = null;

// CSV streaming state
let csvRows       = [];         // all parsed rows from uploaded CSV
let streamPointer = 0;         // current row index
let streamInterval = null;

// Inference log — every row is one inference run (for thesis CSV export)
let infLog = [];

// ── DOM REFS ─────────────────────────────────────────────────────────────────

const dom = {
  gDot:        () => document.getElementById('gDot'),
  gSt:         () => document.getElementById('gSt'),
  bufC:        () => document.getElementById('bufC'),
  alarmC:      () => document.getElementById('alarmC'),
  infC:        () => document.getElementById('infC'),
  latEl:       () => document.getElementById('latEl'),
  workerBadge: () => document.getElementById('workerBadge'),
  scoreEl:     () => document.getElementById('scoreEl'),
  bannerEl:    () => document.getElementById('bannerEl'),
  tSlider:     () => document.getElementById('tSlider'),
  tVal:        () => document.getElementById('tVal'),
  decEl:       () => document.getElementById('decEl'),
  ackBtn:      () => document.getElementById('ackBtn'),
  logEl:       () => document.getElementById('logEl'),
  simTkEl:     () => document.getElementById('simTk'),
  toast:       () => document.getElementById('toast'),
  streamProgress: () => document.getElementById('streamFill'),
  // bench cells
  bMean:       () => document.getElementById('bMean'),
  bMedian:     () => document.getElementById('bMedian'),
  bP95:        () => document.getElementById('bP95'),
  bP99:        () => document.getElementById('bP99'),
};

// ── TOAST NOTIFICATIONS ──────────────────────────────────────────────────────

let toastTimer = null;
/**
 * Show a transient notification.
 * @param {string} msg
 * @param {'error'|'warn'|'info'} [level='error']
 */
function toast(msg, level = 'error') {
  const el = dom.toast();
  el.className = `show ${level === 'error' ? '' : level}`;
  el.innerHTML = `<i class="fas fa-${level === 'error' ? 'circle-xmark' : level === 'warn' ? 'triangle-exclamation' : 'circle-info'}"></i> ${msg}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ''; }, 4000);
}

// ── EVENT LOG ────────────────────────────────────────────────────────────────

function addLog(msg, type = '') {
  const el  = dom.logEl();
  const d   = document.createElement('div');
  d.className = 'lentry';
  const ts  = new Date().toLocaleTimeString('en-GB', { hour12: false });
  d.innerHTML = `<span class="lts">${ts}</span><span class="lmsg ${type}">${msg}</span>`;
  el.prepend(d);
  while (el.children.length > 100) el.removeChild(el.lastChild);
}

// ── WORKER SETUP ─────────────────────────────────────────────────────────────

/**
 * Initialise the Web Worker.
 * The worker script URL is injected by the build step.
 * Falls back to same-origin URL if WORKER_URL is not defined.
 */
function initWorker() {
  // WORKER_URL is set by build.js as a global (Blob URL with model embedded)
  const url = (typeof WORKER_URL !== 'undefined') ? WORKER_URL : 'workers/inference.worker.js';

  worker = new Worker(url);

  worker.onmessage = (e) => {
    workerBusy = false;
    dom.workerBadge().className = 'worker-badge';
    dom.workerBadge().innerText = 'worker · idle';

    const msg = e.data;

    if (msg.type === 'result') {
      handleInferenceResult(msg);
    } else if (msg.type === 'bench') {
      handleBenchResult(msg);
    } else if (msg.type === 'error') {
      toast(`Worker error: ${msg.message}`);
      addLog(`Worker error: ${msg.message}`, 'bad');
    }
  };

  worker.onerror = (e) => {
    toast('Web Worker failed. Inference unavailable.');
    addLog('Worker fatal error: ' + e.message, 'bad');
  };
}

/**
 * Send a window to the worker for inference.
 * Skips if worker is already processing (prevents queue build-up).
 */
function requestInference(windowData, source) {
  if (!worker) { toast('Worker not ready.', 'warn'); return; }
  if (workerBusy) return;  // skip frame — UI stays responsive

  workerBusy = true;
  dom.workerBadge().className = 'worker-badge busy';
  dom.workerBadge().innerText = 'worker · busy';

  worker.postMessage({ type: 'infer', windowData, threshold, _source: source });
  // We stash source in the window so handleInferenceResult can use it
  worker._pendingSource = source;
}

function requestBench(windowData) {
  if (!worker || workerBusy) { toast('Worker busy — try after current inference.', 'warn'); return; }
  workerBusy = true;
  dom.workerBadge().className = 'worker-badge busy';
  dom.workerBadge().innerText = 'worker · benchmarking…';
  addLog(`Running ${BENCH_RUNS} inference benchmark…`, 'warn');
  worker.postMessage({ type: 'bench', windowData, runs: BENCH_RUNS });
}

// ── RESULT HANDLERS ──────────────────────────────────────────────────────────

function handleInferenceResult(msg) {
  const { score, stats, latencyMs, topFeature, topZ, isAnomaly } = msg;
  lastScore = score;
  lastStats = stats;

  // Latency display
  dom.latEl().innerText = latencyMs.toFixed(3) + ' ms';

  // Score display
  const near = score > threshold - 0.06;
  dom.scoreEl().innerText  = score.toFixed(4);
  dom.scoreEl().className  = 'snum' + (isAnomaly ? ' bad' : near ? ' warn' : ' ok');

  // Banner
  if (isAnomaly) {
    dom.bannerEl().className = 'banner bad';
    dom.bannerEl().innerHTML = '<i class="fas fa-triangle-exclamation"></i>Anomaly detected';
    dom.gDot().className     = 'dot alert';
    dom.gSt().innerText      = 'anomaly';
    dom.ackBtn().disabled    = false;
    totalAlarms++;
    dom.alarmC().innerText   = totalAlarms;
    addLog(`ALARM · score ${score.toFixed(4)} · driver: ${topFeature} (${parseFloat(topZ).toFixed(2)}σ)`, 'bad');
  } else {
    dom.bannerEl().className = 'banner ok';
    dom.bannerEl().innerHTML = '<i class="fas fa-circle-check"></i>Normal operation';
    dom.gDot().className     = near ? 'dot warn' : 'dot';
    dom.gSt().innerText      = near ? 'elevated' : 'nominal';
    dom.ackBtn().disabled    = true;
    addLog(`Score ${score.toFixed(4)} — nominal`, 'ok');
  }

  // Decision text
  const sLabel = { P_TPT: 'P-TPT', T_TPT: 'T-TPT', P_MON: 'P-MON-CKP' };
  dom.decEl().innerText   = isAnomaly
    ? `Score ${score.toFixed(4)} exceeds threshold ${threshold.toFixed(2)}. Primary driver: ${topFeature.replace('_',' ')} (${parseFloat(topZ).toFixed(2)}σ). Recommend reviewing sensor trends.`
    : `Score ${score.toFixed(4)} within normal bounds. No significant deviations detected across all 15 features.`;
  dom.decEl().className   = 'dtxt' + (isAnomaly ? ' bad' : '');

  // Counters
  totalInf++;
  dom.infC().innerText = totalInf;

  // Feature chart
  if (stats) refreshFeatChart(stats);

  // Record for CSV export
  const source = worker._pendingSource || 'unknown';
  recordInference({ score, stats, latencyMs, topFeature, topZ, isAnomaly, source });
}

function handleBenchResult(msg) {
  const { mean, median, p95, p99, min, max, samples } = msg;
  dom.bMean().innerText   = mean.toFixed(3)   + ' ms';
  dom.bMedian().innerText = median.toFixed(3) + ' ms';
  dom.bP95().innerText    = p95.toFixed(3)    + ' ms';
  dom.bP99().innerText    = p99.toFixed(3)    + ' ms';
  addLog(`Benchmark (${samples} runs) — mean: ${mean.toFixed(2)}ms · median: ${median.toFixed(2)}ms · p95: ${p95.toFixed(2)}ms · p99: ${p99.toFixed(2)}ms`, 'ok');
  toast(`Benchmark complete. Median ${median.toFixed(2)} ms, p95 ${p95.toFixed(2)} ms`, 'info');
}

// ── INFERENCE LOG (CSV EXPORT) ───────────────────────────────────────────────

function recordInference({ score, stats, latencyMs, topFeature, topZ, isAnomaly, source }) {
  const last  = telBuffer[telBuffer.length - 1] || {};
  const pmon  = stats[0]; // order: P_MON, T_TPT, P_TPT
  const ttpt  = stats[1];
  const ptpt  = stats[2];

  infLog.push({
    timestamp:       new Date().toISOString(),
    source:          source,
    sim_mode:        source === 'sim' ? simMode : 'n/a',
    p_tpt_last:      (last.ptpt || 0).toFixed(4),
    t_tpt_last:      (last.ttpt || 0).toFixed(4),
    p_mon_last:      (last.pmon || 0).toFixed(4),
    anomaly_score:   score.toFixed(6),
    threshold:       threshold.toFixed(2),
    classification:  isAnomaly ? 'ANOMALY' : 'NORMAL',
    top_feature:     topFeature,
    top_z_score:     parseFloat(topZ).toFixed(4),
    latency_ms:      latencyMs.toFixed(4),
    window_size:     telBuffer.length,
    // P-TPT stats
    ptpt_mean:   ptpt.mean.toFixed(4),   ptpt_std: ptpt.std.toFixed(4),
    ptpt_min:    ptpt.min.toFixed(4),    ptpt_max: ptpt.max.toFixed(4),   ptpt_median: ptpt.median.toFixed(4),
    // T-TPT stats
    ttpt_mean:   ttpt.mean.toFixed(4),   ttpt_std: ttpt.std.toFixed(4),
    ttpt_min:    ttpt.min.toFixed(4),    ttpt_max: ttpt.max.toFixed(4),   ttpt_median: ttpt.median.toFixed(4),
    // P-MON stats
    pmon_mean:   pmon.mean.toFixed(4),   pmon_std: pmon.std.toFixed(4),
    pmon_min:    pmon.min.toFixed(4),    pmon_max: pmon.max.toFixed(4),   pmon_median: pmon.median.toFixed(4),
  });
}

function exportCSV() {
  if (!infLog.length) {
    toast('No inference data to export yet.', 'warn');
    return;
  }
  const cols = Object.keys(infLog[0]);
  const rows = [
    cols.join(','),
    ...infLog.map(r =>
      cols.map(k => {
        const v = String(r[k]);
        return v.includes(',') || v.includes('"')
          ? `"${v.replace(/"/g, '""')}"` : v;
      }).join(',')
    )
  ];
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `notebook7_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
  a.click();
  addLog(`Exported ${infLog.length} inference records`, 'ok');
  toast(`Exported ${infLog.length} records`, 'info');
}

// ── SENSOR CARD UPDATE ───────────────────────────────────────────────────────

function updateCards(ptpt, ttpt, pmon) {
  function set(cardId, stId, valId, val, ok) {
    document.getElementById(cardId).classList.toggle('alert', !ok);
    const s = document.getElementById(stId);
    s.innerText   = ok ? 'stable' : 'abnormal';
    s.className   = 'sstate' + (ok ? '' : ' bad');
    document.getElementById(valId).innerText = isNaN(val) ? '—' : val.toFixed(2);
  }
  set('cp', 'sp', 'vp', ptpt, ptpt >= 4.5  && ptpt <= 6.0);
  set('ct', 'st', 'vt', ttpt, ttpt >= 65   && ttpt <= 85);
  set('cm', 'sm', 'vm', pmon, pmon >= 15   && pmon <= 21);
}

// ── CHARTS ───────────────────────────────────────────────────────────────────

function initCharts() {
  const gc = 'rgba(143,188,143,.14)';
  const font = { family: 'Nunito', size: 9 };

  sensorChart = new Chart(
    document.getElementById('sensorCanvas').getContext('2d'), {
      type: 'line',
      data: {
        labels: Array.from({ length: WINDOW_SIZE }, (_, i) => i),
        datasets: [
          { label: 'P-TPT',  borderColor: '#2a4d3a', data: [], fill: false, borderWidth: 1.7, pointRadius: 0, tension: 0.3 },
          { label: 'T-TPT',  borderColor: '#b45309', data: [], fill: false, borderWidth: 1.7, pointRadius: 0, tension: 0.3 },
          { label: 'P-MON',  borderColor: '#2e86c1', data: [], fill: false, borderWidth: 1.7, pointRadius: 0, tension: 0.3 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { labels: { color: '#7a9b8a', font, boxWidth: 12 } } },
        scales: {
          x: { ticks: { display: false }, grid: { color: gc } },
          y: { ticks: { color: '#7a9b8a', font }, grid: { color: gc } }
        }
      }
    }
  );

  featChart = new Chart(
    document.getElementById('featCanvas').getContext('2d'), {
      type: 'bar',
      data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 0, borderRadius: 4 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 280 },
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#7a9b8a', font: { ...font, size: 7 }, maxRotation: 55 }, grid: { display: false } },
          y: { ticks: { color: '#7a9b8a', font }, grid: { color: gc } }
        }
      }
    }
  );
}

function refreshSensorChart() {
  sensorChart.data.datasets[0].data = telBuffer.map(d => d.ptpt);
  sensorChart.data.datasets[1].data = telBuffer.map(d => d.ttpt);
  sensorChart.data.datasets[2].data = telBuffer.map(d => d.pmon);
  sensorChart.update('none');
}

function refreshFeatChart(stats) {
  const sN   = { P_TPT: 'P-TPT', T_TPT: 'T-TPT', P_MON: 'P-MON' };
  const stN  = ['μ', 'σ', 'min', 'max', 'med'];
  const labels = [], zs = [], colors = [];

  // MODEL is available globally (injected by build step)
  for (let i = 0; i < stats.length; i++) {
    const s = stats[i], base = i * 5;
    ['mean', 'std', 'min', 'max', 'median'].forEach((key, j) => {
      const z = Math.abs((s[key] - MODEL.scaler_mean[base + j]) / MODEL.scaler_scale[base + j]);
      labels.push(`${sN[s.sensor]}.${stN[j]}`);
      zs.push(z);
      colors.push(z > 2 ? 'rgba(192,57,43,.7)' : z > 1 ? 'rgba(180,83,9,.65)' : 'rgba(143,188,143,.7)');
    });
  }

  featChart.data.labels                        = labels;
  featChart.data.datasets[0].data             = zs;
  featChart.data.datasets[0].backgroundColor  = colors;
  featChart.update();
}

// ── PROCESS A FULL WINDOW ────────────────────────────────────────────────────

function processWindow(source) {
  if (telBuffer.length < WINDOW_SIZE) return;

  dom.bufC().innerText = `${WINDOW_SIZE}/${WINDOW_SIZE}`;
  const last = telBuffer[telBuffer.length - 1];
  updateCards(last.ptpt, last.ttpt, last.pmon);
  refreshSensorChart();

  const win = {
    P_TPT: telBuffer.map(d => d.ptpt),
    T_TPT: telBuffer.map(d => d.ttpt),
    P_MON: telBuffer.map(d => d.pmon),
  };

  requestInference(win, source);
}

// ── CSV UPLOAD & STREAMING ───────────────────────────────────────────────────

/**
 * Validate a parsed CSV result object.
 * Throws a descriptive Error on any problem.
 */
function validateCSV(result) {
  if (!result.data || result.data.length === 0) {
    throw new Error('The CSV file is empty.');
  }

  const required = ['P-TPT', 'T-TPT', 'P-MON-CKP'];
  const cols     = Object.keys(result.data[0]);
  const missing  = required.filter(c => !cols.includes(c));

  if (missing.length) {
    throw new Error(
      `Missing required columns: ${missing.join(', ')}. ` +
      `File has: ${cols.join(', ')}.`
    );
  }

  if (result.data.length < WINDOW_SIZE) {
    throw new Error(
      `Need at least ${WINDOW_SIZE} rows. File only has ${result.data.length}.`
    );
  }

  // Check for non-numeric values in key columns
  const sample = result.data[0];
  for (const col of required) {
    if (typeof sample[col] !== 'number' || isNaN(sample[col])) {
      throw new Error(
        `Column "${col}" contains non-numeric values. ` +
        `Ensure your CSV has no header rows mixed into data.`
      );
    }
  }
}

/**
 * Load a CSV and begin streaming playback row-by-row.
 * Each tick advances streamPointer and runs inference on the current window.
 */
function loadAndStreamCSV(csvString) {
  stopStream();
  stopSim();

  const result = Papa.parse(csvString, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  // Surface any parse warnings
  if (result.errors && result.errors.length) {
    const serious = result.errors.filter(e => e.type !== 'FieldMismatch');
    if (serious.length) {
      toast(`CSV parse warning: ${serious[0].message}`, 'warn');
    }
  }

  try {
    validateCSV(result);
  } catch (err) {
    toast(err.message);
    addLog('CSV error: ' + err.message, 'bad');
    return;
  }

  csvRows       = result.data;
  streamPointer = 0;
  telBuffer     = [];

  addLog(`Loaded ${csvRows.length} rows — starting streaming playback`, 'ok');
  toast(`Streaming ${csvRows.length} rows at 1 row / ${SIM_INTERVAL_MS}ms`, 'info');

  streamInterval = setInterval(() => {
    if (streamPointer >= csvRows.length) {
      stopStream();
      addLog('CSV stream complete', 'ok');
      toast('Stream finished.', 'info');
      return;
    }

    const row = csvRows[streamPointer++];

    // Guard against bad rows mid-stream
    const ptpt = parseFloat(row['P-TPT']);
    const ttpt = parseFloat(row['T-TPT']);
    const pmon = parseFloat(row['P-MON-CKP']);

    if (isNaN(ptpt) || isNaN(ttpt) || isNaN(pmon)) {
      addLog(`Row ${streamPointer}: bad values — skipped`, 'warn');
      return;
    }

    telBuffer.push({ ptpt, ttpt, pmon });
    if (telBuffer.length > WINDOW_SIZE) telBuffer.shift();

    // Progress bar
    const pct = (streamPointer / csvRows.length) * 100;
    dom.streamProgress().style.width = pct + '%';

    dom.bufC().innerText = `${telBuffer.length}/${WINDOW_SIZE}`;
    updateCards(ptpt, ttpt, pmon);
    refreshSensorChart();

    if (telBuffer.length >= WINDOW_SIZE) {
      processWindow('csv-stream');
    }
  }, SIM_INTERVAL_MS);
}

function stopStream() {
  if (streamInterval) { clearInterval(streamInterval); streamInterval = null; }
  dom.streamProgress().style.width = '0%';
}

// ── SIMULATION ───────────────────────────────────────────────────────────────

/** Generate a realistic synthetic sensor sample. */
function simSample(t, mode) {
  if (mode === 'normal') {
    return {
      ptpt: 5.2 + 0.15 * Math.sin(t / 20) + (Math.random() - 0.5) * 0.08,
      ttpt: 74  + 1.5  * Math.sin(t / 15) + (Math.random() - 0.5) * 0.6,
      pmon: 18  + 0.4  * Math.sin(t / 25) + (Math.random() - 0.5) * 0.2,
    };
  }
  // Anomaly: periodic pressure spike
  const spike = (t % 30 < 5) ? 2.8 : 0;
  return {
    ptpt: 5.2 + spike + (Math.random() - 0.5) * 0.3,
    ttpt: 74  + spike * 7 + (Math.random() - 0.5) * 2,
    pmon: 18  + spike * 0.9 + (Math.random() - 0.5) * 0.5,
  };
}

function simStep() {
  simTick++;
  const s = simSample(simTick, simMode);
  telBuffer.push(s);
  if (telBuffer.length > WINDOW_SIZE) telBuffer.shift();

  dom.simTkEl().innerText  = `t = ${simTick}`;
  dom.bufC().innerText     = `${Math.min(telBuffer.length, WINDOW_SIZE)}/${WINDOW_SIZE}`;
  updateCards(s.ptpt, s.ttpt, s.pmon);
  refreshSensorChart();

  if (telBuffer.length >= WINDOW_SIZE) processWindow('sim');
}

function startSim(mode) {
  stopStream();
  simMode = mode;
  if (simInterval) clearInterval(simInterval);

  // Pre-fill buffer so inference starts immediately
  if (telBuffer.length === 0) {
    for (let i = 0; i < WINDOW_SIZE - 1; i++) {
      telBuffer.push(simSample(i, 'normal'));
    }
  }

  document.getElementById('simN').classList.toggle('on', mode === 'normal');
  document.getElementById('simA').classList.toggle('on', mode === 'anomaly');
  addLog(`Simulation started: ${mode}`);
  simInterval = setInterval(simStep, SIM_INTERVAL_MS);
}

function stopSim() {
  if (simInterval) { clearInterval(simInterval); simInterval = null; }
  dom.simTkEl().innerText = '';
  document.getElementById('simN').classList.remove('on');
  document.getElementById('simA').classList.remove('on');
}

// ── EVENT WIRING ─────────────────────────────────────────────────────────────

function wireEvents() {
  // CSV file upload
  document.getElementById('csvFile').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.endsWith('.csv')) {
      toast('Please select a .csv file.'); return;
    }
    addLog(`Reading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…`);
    const reader = new FileReader();
    reader.onload  = ev => loadAndStreamCSV(ev.target.result);
    reader.onerror = ()  => toast('File read failed.');
    reader.readAsText(file);
    // Reset input so same file can be re-selected
    e.target.value = '';
  });

  // Threshold slider
  document.getElementById('tSlider').addEventListener('input', e => {
    threshold = parseFloat(e.target.value);
    dom.tVal().innerText = threshold.toFixed(2);
    // Re-evaluate last score with new threshold
    if (lastScore !== undefined && lastStats) {
      handleInferenceResult({
        score: lastScore, stats: lastStats,
        latencyMs: parseFloat(dom.latEl().innerText) || 0,
        topFeature: '', topZ: '0',
        isAnomaly: lastScore > threshold,
      });
    }
  });

  // Acknowledge
  dom.ackBtn().addEventListener('click', () => {
    dom.ackBtn().disabled = true;
    addLog('Alarm acknowledged by operator');
    dom.bannerEl().innerHTML = '<i class="fas fa-eye"></i>Acknowledged — monitoring';
  });

  // Clear log
  document.getElementById('clearBtn').addEventListener('click', () => {
    dom.logEl().innerHTML = '';
    addLog('Log cleared');
  });

  // Export CSV
  document.getElementById('dlBtn').addEventListener('click', exportCSV);

  // Simulation buttons
  document.getElementById('simN').addEventListener('click',    () => startSim('normal'));
  document.getElementById('simA').addEventListener('click',    () => startSim('anomaly'));
  document.getElementById('simStop').addEventListener('click', () => { stopSim(); stopStream(); addLog('Stopped'); });

  // Benchmark
  document.getElementById('benchBtn').addEventListener('click', () => {
    if (telBuffer.length < WINDOW_SIZE) {
      toast('Need a full window first. Run simulation or upload a CSV.', 'warn');
      return;
    }
    const win = {
      P_TPT: telBuffer.map(d => d.ptpt),
      T_TPT: telBuffer.map(d => d.ttpt),
      P_MON: telBuffer.map(d => d.pmon),
    };
    requestBench(win);
  });

  // Drag & drop
  const dropEl = document.getElementById('drop');
  document.addEventListener('dragover',  e => { e.preventDefault(); dropEl.classList.add('on'); });
  document.addEventListener('dragleave', e => { if (!e.relatedTarget) dropEl.classList.remove('on'); });
  document.addEventListener('drop', e => {
    e.preventDefault(); dropEl.classList.remove('on');
    const file = e.dataTransfer.files[0];
    if (!file || !file.name.endsWith('.csv')) {
      toast('Drop a .csv file.'); return;
    }
    addLog(`Dropped ${file.name}…`);
    const reader = new FileReader();
    reader.onload  = ev => loadAndStreamCSV(ev.target.result);
    reader.onerror = ()  => toast('File read failed.');
    reader.readAsText(file);
  });
}

// ── BOOT ─────────────────────────────────────────────────────────────────────

function boot() {
  initCharts();
  initWorker();
  wireEvents();
  addLog('Notebook Seven ready · 200 trees · Web Worker active', 'ok');
}

document.addEventListener('DOMContentLoaded', boot);
