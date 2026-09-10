// wikilite scorer — dependency-free JS inference for the distilled model
// (results/<mode>/wikilite/wikilite_s<seed>.json, LightGBM dump_model format).
// Proof that the model runs client-side (browser extension / userscript / Toolforge).
// Usage: node scorer.mjs <model.json> [verify_test.json]
import { readFileSync } from 'node:fs';

export function loadModel(path) {
  const d = JSON.parse(readFileSync(path, 'utf8'));
  return { features: d.features, nClasses: d.n_classes, trees: d.model.tree_info };
}

function walk(node, x) {
  while (node.split_feature !== undefined) {
    const v = x[node.split_feature];
    let left;
    if (node.decision_type === '<=') left = v <= node.threshold;
    else if (node.decision_type === '==') left = String(v) === String(node.threshold);
    else left = v <= node.threshold;
    if (node.missing_type !== 'None' && (v === null || Number.isNaN(v))) {
      left = node.default_left;
    }
    node = left ? node.left_child : node.right_child;
  }
  return node.leaf_value;
}

export function predictProba(model, x) {
  const raw = new Array(model.nClasses).fill(0);
  model.trees.forEach((t, i) => { raw[i % model.nClasses] += walk(t.tree_structure, x); });
  const mx = Math.max(...raw);
  const ex = raw.map((r) => Math.exp(r - mx));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / s);
}

// CLI verify: compare against python probs, report agreement + speed
const [modelPath, verifyPath] = process.argv.slice(2);
if (modelPath) {
  const model = loadModel(modelPath);
  if (verifyPath) {
    const v = JSON.parse(readFileSync(verifyPath, 'utf8'));
    let agree = 0; let maxDiff = 0;
    const t0 = performance.now();
    v.rows.forEach((row, i) => {
      const p = predictProba(model, row);
      const py = v.py_probs[i];
      const am = p.indexOf(Math.max(...p));
      const amPy = py.indexOf(Math.max(...py));
      if (am === amPy) agree += 1;
      py.forEach((q, k) => { maxDiff = Math.max(maxDiff, Math.abs(q - p[k])); });
    });
    const us = ((performance.now() - t0) / v.rows.length) * 1000;
    console.log(JSON.stringify({
      n: v.rows.length, argmax_agreement: agree / v.rows.length,
      max_prob_diff: +maxDiff.toExponential(2), us_per_article: +us.toFixed(1),
    }));
  } else {
    console.log(`loaded: ${model.trees.length} trees, ${model.features.length} features`);
  }
}
