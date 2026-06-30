/**
 * inference.worker.js
 * Notebook Seven — Isolation Forest Web Worker
 *
 * Runs entirely off the main UI thread.
 * Receives messages of shape:
 *   { type: 'infer', windowData, threshold }
 *   { type: 'bench', windowData, runs }
 *
 * Posts messages of shape:
 *   { type: 'result',  score, stats, latencyMs, topFeature, topZ, isAnomaly }
 *   { type: 'bench',   mean, median, p95, p99, min, max, samples }
 *   { type: 'error',   message }
 */

'use strict';

// MODEL is injected by the build step as a global variable
// (build.js wraps this file: const MODEL = {...}; <worker code>)
/* global MODEL */

const SENSORS  = ['P_MON', 'T_TPT', 'P_TPT'];
const STAT_KEYS = ['mean', 'std', 'min', 'max', 'median'];
const SENSOR_LABELS = { P_TPT: 'P-TPT', T_TPT: 'T-TPT', P_MON: 'P-MON-CKP' };

// ── ISOLATION FOREST MATH ───────────────────────────────────────────────────

/**
 * Expected average path length for a dataset of size n.
 * Matches sklearn's implementation exactly.
 */
function expectedPathLength(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n);
}

/**
 * Recursively traverse a single isolation tree.
 * Returns the path length (depth + leaf correction).
 */
function pathLength(sample, node, depth) {
  if (node.is_leaf) return depth + expectedPathLength(node.size);
  return sample[node.feature_idx] <= node.split_value
    ? pathLength(sample, node.left,  depth + 1)
    : pathLength(sample, node.right, depth + 1);
}

// ── FEATURE EXTRACTION ──────────────────────────────────────────────────────

/**
 * From a window of 180 rows extract 15 statistical features
 * (mean, std, min, max, median) × 3 sensors, then z-score scale.
 */
function extractFeatures(win) {
  const features = [];
  const stats    = [];

  for (const sensor of SENSORS) {
    const arr  = win[sensor];
    const n    = arr.length;

    if (!arr || n === 0) throw new Error(`Sensor ${sensor} has no data.`);

    const mean   = arr.reduce((a, b) => a + b, 0) / n;
    const std    = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    const min    = Math.min(...arr);
    const max    = Math.max(...arr);
    const sorted = [...arr].sort((a, b) => a - b);
    const median = sorted[Math.floor(n / 2)];

    features.push(mean, std, min, max, median);
    stats.push({ sensor, mean, std, min, max, median });
  }

  const scaled = features.map(
    (v, i) => (v - MODEL.scaler_mean[i]) / MODEL.scaler_scale[i]
  );

  return { scaled, stats };
}

// ── SCORING ─────────────────────────────────────────────────────────────────

/**
 * Core inference: traverse all 200 trees, compute anomaly score.
 * Returns { score, stats, latencyMs }.
 */
function computeScore(windowData) {
  const t0 = performance.now();

  const { scaled, stats } = extractFeatures(windowData);
  let totalPathLength = 0;

  for (const item of MODEL.trees) {
    const subsample = item.features.map(i => scaled[i]);
    totalPathLength += pathLength(subsample, item.tree, 0);
  }

  const avgPath = totalPathLength / MODEL.trees.length;
  const score   = Math.pow(2, -(avgPath / expectedPathLength(MODEL.max_samples)));
  const latencyMs = performance.now() - t0;

  return { score, stats, latencyMs };
}

// ── EXPLANATION ─────────────────────────────────────────────────────────────

/**
 * Find the feature with the highest absolute z-score deviation.
 */
function topContributor(stats) {
  let maxZ = 0, topFeature = '', topZ = 0;

  for (let i = 0; i < stats.length; i++) {
    const s = stats[i];
    const base = i * 5;

    STAT_KEYS.forEach((key, j) => {
      const z = Math.abs(
        (s[key] - MODEL.scaler_mean[base + j]) / MODEL.scaler_scale[base + j]
      );
      if (z > maxZ) {
        maxZ       = z;
        topFeature = `${SENSOR_LABELS[s.sensor]}_${key}`;
        topZ       = z;
      }
    });
  }

  return { topFeature, topZ: topZ.toFixed(4) };
}

// ── MESSAGE HANDLER ─────────────────────────────────────────────────────────

self.onmessage = function (e) {
  const { type, windowData, threshold, runs } = e.data;

  try {
    if (type === 'infer') {
      // Single inference
      const { score, stats, latencyMs } = computeScore(windowData);
      const { topFeature, topZ }        = topContributor(stats);
      const isAnomaly                   = score > threshold;

      self.postMessage({
        type: 'result',
        score,
        stats,
        latencyMs,
        topFeature,
        topZ,
        isAnomaly,
      });

    } else if (type === 'bench') {
      // Benchmark: run N inferences and collect latency distribution
      const n       = runs || 1000;
      const samples = [];

      for (let i = 0; i < n; i++) {
        const { latencyMs } = computeScore(windowData);
        samples.push(latencyMs);
      }

      samples.sort((a, b) => a - b);

      const mean   = samples.reduce((a, b) => a + b, 0) / n;
      const median = samples[Math.floor(n / 2)];
      const p95    = samples[Math.floor(n * 0.95)];
      const p99    = samples[Math.floor(n * 0.99)];
      const min    = samples[0];
      const max    = samples[n - 1];

      self.postMessage({ type: 'bench', mean, median, p95, p99, min, max, samples: n });

    } else {
      throw new Error(`Unknown message type: ${type}`);
    }

  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};
