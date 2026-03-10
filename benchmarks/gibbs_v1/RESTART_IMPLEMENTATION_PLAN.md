# MozJPEG Gibbs Restart Protocol v6 (Implemented)

This document is the pinned execution protocol now implemented in:

- `benchmarks/gibbs_v1/scripts/run_protocol_v6.py`

It replaces vague restart language with fixed thresholds, fixed model class, fixed split policy, fixed calibration rules, and hard-stop feasibility gates.

## Pinned Decisions

1. `tau_B` is an image-level eligibility threshold calibrated on validation only.
2. `tau_C` is a block-level routing threshold calibrated on the same validation partition, filtered to `tau_B`-eligible images.
3. Score direction is fixed:
   - higher `score_B` => more likely `safe=1`
   - higher `score_C` => more eligible block for skip
4. Routing policy is fixed:
   - `score_B < tau_B` => no routing in image
   - `score_B >= tau_B` and `score_C >= tau_C` => guarded skip allowed for block
5. C acceptance is anchored to `scripts/gibbs_large_common.py` gates.

## Stage B (Predictor)

`run_protocol_v6.py stage-b` implements:

- Label definition: `safe=1` iff full-image all-trellis-off size delta is `<= +0.20%`.
- Split policy: deterministic hash split `60/20/20` with seed `1729`.
- Model class (all models, theorem and baselines):
  - `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, class_weight=None, random_state=1729)`
- Baselines:
  - gradient-only
  - HF-only
  - combined non-theorem (HF + gradient)
- Threshold policy:
  - calibrate `tau_B` on validation by maximizing recall subject to precision `>= 95%`.
  - freeze `tau_B` before test/C.
- Required reporting:
  - test prevalence
  - AUROC / PR-AUC / precision@`tau_B`
  - bootstrap 95% CI for each (`10000` resamples, seed `1729`)
- Acceptance checks:
  - AUROC `>= 0.70`
  - AUROC lift `>= 0.05` vs best baseline
  - PR-AUC lift `>= 0.03` vs best baseline
  - precision@`tau_B` `>= 0.95`
  - recall non-inferiority at precision floor: theorem recall `>= baseline_recall - 0.005`

Outputs:

- `records.csv`
- `model.json`
- `summary.json`

## Stage C (Routing)

`run_protocol_v6.py stage-c` implements:

- Load frozen `tau_B` and theorem model from stage B artifacts.
- Calibrate `tau_C` once on validation images predicted safe by `tau_B`.
- `tau_C` rule: deterministic sweep, select most aggressive threshold satisfying eligible-validation size constraints:
  - mean size delta `<= +0.20%`
  - p95 size delta `<= +0.50%`
- Compute feasibility upper bound (not a forecast):
  - `projected_max_runtime_gain_pct = 100 * r_trellis * f_eligible * s_block`
- Hard stop before C benchmark if upper bound `< 3.0`.
- If feasible, run single-seam Huffman policy benchmark and evaluate harness gates.

Outputs:

- `rows.csv`
- `summary.json`

## Reproducibility Pins

- Split assignment is deterministic and stored in `model.json`.
- `tau_B` and `tau_C` are serialized and reused; no post-hoc retuning.
- Positive class is explicit (`safe=1`).
- Delta sign convention is fixed globally:
  - `delta_pct = 100 * (candidate - baseline) / baseline`

## Usage

### B only

```bash
python3 benchmarks/gibbs_v1/scripts/run_protocol_v6.py stage-b \
  --bin /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/cjpeg \
  --manifest benchmarks/gibbs_v1/manifests/local_stress.txt \
  --quality 85 \
  --results-dir benchmarks/gibbs_v1/results
```

### C only (requires B summary)

```bash
python3 benchmarks/gibbs_v1/scripts/run_protocol_v6.py stage-c \
  --bin /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/cjpeg \
  --djpeg /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/djpeg \
  --b-summary benchmarks/gibbs_v1/results/<b_tag>/summary.json \
  --benchmark-manifest benchmarks/gibbs_v1/manifests/large_mixed.txt \
  --quality 85 \
  --repeats 7 \
  --results-dir benchmarks/gibbs_v1/results
```

### End-to-end

```bash
python3 benchmarks/gibbs_v1/scripts/run_protocol_v6.py run-all \
  --bin /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/cjpeg \
  --djpeg /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/djpeg \
  --b-manifest benchmarks/gibbs_v1/manifests/local_stress.txt \
  --c-manifest benchmarks/gibbs_v1/manifests/large_mixed.txt \
  --quality 85 \
  --repeats 7 \
  --results-dir benchmarks/gibbs_v1/results
```
