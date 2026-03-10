# Gibbs v6 Effort and Current Repo State

Date: 2026-03-10 (America/New_York)

## Objective
Implement and execute `MozJPEG Gibbs Restart Protocol v6` with pinned decisions, frozen thresholds, fixed model class, fixed split policy, hard-stage gating, and explicit negative-result handling.

## What Was Implemented

1. Protocol runner:
   - `benchmarks/gibbs_v1/scripts/run_protocol_v6.py`
   - Commands:
     - `stage-b`
     - `stage-c`
     - `run-all`

2. Protocol documentation:
   - `benchmarks/gibbs_v1/RESTART_IMPLEMENTATION_PLAN.md`
   - `benchmarks/gibbs_v1/README.md` (new v6 usage section)
   - `benchmarks/gibbs_v1/REVIEWER_TECH_MEMO.md` (reviewer context memo)

3. v6 protocol mechanics now enforced in code:
   - `tau_B` frozen from validation only.
   - `tau_C` calibrated once on `tau_B`-eligible validation images only.
   - Single-seam policy (`Huffman` path via existing guarded seam behavior).
   - Fixed metric sign convention: `delta_pct = 100 * (candidate - baseline) / baseline`.
   - Feasibility bound: `100 * r_trellis * f_eligible * s_block`.
   - Hard stop before C benchmark if feasibility bound `< 3.0`.
   - Bootstrap CI reporting (`n_resamples`, `seed`) for B metrics.

## Execution Summary

### B Stage
A passing B run was obtained:
- `benchmarks/gibbs_v1/results/v6_final_try_b_local_stress_q85/summary.json`
- `acceptance.all_pass = true`

### C Stage
C remained infeasible under pinned rules:
- `benchmarks/gibbs_v1/results/v6_final_try_c_large_mixed_q85/summary.json`
- `feasibility.feasible = false`
- `s_block = 0.0` across tau_C sweep candidates for eligible validation images.

Interpretation under protocol:
- This is a valid transfer/fesibility negative result in v6 terms.

## Current Branch / Remote State

- Branch: `codex/gibbs-invariant-pilot-v1`
- Remotes:
  - `origin`: `https://github.com/mozilla/mozjpeg.git`
  - `public-origin`: `https://github.com/zfifteen/mozjpeg-gibbs-invariant-pilot.git`

## Files Touched in This Effort

- `benchmarks/gibbs_v1/scripts/run_protocol_v6.py`
- `benchmarks/gibbs_v1/README.md`
- `benchmarks/gibbs_v1/RESTART_IMPLEMENTATION_PLAN.md`
- `benchmarks/gibbs_v1/REVIEWER_TECH_MEMO.md`
- `benchmarks/gibbs_v1/EFFORT_AND_REPO_STATE.md`
- `benchmarks/gibbs_v1/STATUS_REPORT_FOR_REVIEWERS.md`

## Reviewer Notes

- The implementation intentionally prioritizes reproducibility and falsifiability over optimization opportunism.
- Result quality should be judged against pinned protocol contracts, not informal speed impressions.
- The critical reviewer artifacts are:
  - B pass summary (`v6_final_try_b_local_stress_q85`)
  - C infeasibility summary (`v6_final_try_c_large_mixed_q85`)
