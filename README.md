# Copy-on-Write Memory Analysis in Machine Learning Systems

Linux/WSL2 experiment comparing independent model allocation and parent-preloaded CoW using trained float32 MLP parameters. Not an inference benchmark.

## Method

Eight configurations, five repeats, two strategies, three phases: 240 aggregate and 1,320 per-process records. Model mapping metrics from smaps; whole-process RSS/PSS/Private Dirty from smaps_rollup. RSS double-counts shared pages. Predictions: independent W*S; CoW S + Wm*round(p*N)*B, with retained parent model. Four-worker continuous break-even 75%.

## Rerun results

Mean post-modification MiB.

|Exp|Ind model PSS|CoW model PSS|Ind total PSS|CoW total PSS|Ind RSS|CoW RSS|Error KiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|E1|17.203|8.594|37.285|27.529|100.157|101.486|-4.000|
|E2|17.203|4.297|37.053|22.954|100.064|101.727|-4.000|
|E3|17.203|12.891|37.432|31.986|100.236|102.096|-4.000|
|E4|17.203|21.504|37.501|40.629|100.193|102.102|0.000|
|E5|34.406|12.894|60.986|37.298|181.299|180.853|-1.000|
|E6|260.766|130.376|282.134|150.555|344.462|407.205|-3.000|
|E7|17.203|8.596|37.423|27.701|100.345|102.110|-2.000|
|E8|17.203|17.202|37.624|36.496|100.579|102.505|-5.000|

## Reproduce

Use Ubuntu/WSL2 with Python, NumPy and scikit-learn. Keep existing results; run in a fresh checkout without rerun output files.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn
bash run_all.sh
```

`predictions.json` contains configuration-only predictions; `cow_ml_memory_v2.py` is the measurement implementation. `rerun/` contains 32 files (raw, summary, metadata, per-process for each E1–E8). Original results remain historical.

## Limitations

Minor faults are indirect evidence, timing excludes parent preload/fork/measurement, and small PSS errors are consistent with reporting precision but not causally isolated. One WSL2 environment; no inference or GPU claims.
