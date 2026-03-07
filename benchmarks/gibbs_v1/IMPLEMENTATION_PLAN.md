# MozJPEG Gibbs-Invariant Pilot Plan (Implemented v1)

## Integration-Point Audit

- `cjpeg.c`
  - `parse_switches()` owns CLI feature gating (`-gibbs-mode`, `-gibbs-log`).
  - `main()` owns end-to-end per-image telemetry emission (JSONL), encode timing, and output byte capture.
- `jcmaster.c`
  - `prepare_for_pass()` is the pass-scheduling seam for trellis-pass counting.
  - `finish_pass_master()` is the correct place to increment completed-loop counters.
- `jccoefct.c`
  - `compress_data()` and `compress_first_pass()` are the correct seams for single-visit block AC feature extraction.
  - `compress_trellis_pass()` is the correct seam for counting trellis blocks touched.
- `jpeglib.h`
  - Left unchanged for v1 (no new public API/ABI).
- `jpegint.h`
  - Internal-only telemetry fields added to `jpeg_comp_master` for cross-module aggregation.

## v1 (Instrumentation-Only) Delivered

- Added CLI flags in `cjpeg`:
  - `-gibbs-mode off|log-only` (default `off`)
  - `-gibbs-log <path>`
- Added JSONL per-image telemetry in log-only mode with:
  - AC magnitude aggregates (`R16`, `R32`, `R32-R16`) over block zigzag AC coefficients.
  - Trellis activity counters (passes, loops completed/estimated, blocks touched).
  - Encode time and output bytes.
- Verified `A` vs `B1` bitstream invariance on smoke test (`cmp` equality).
- Added reproducibility harness under `benchmarks/gibbs_v1/`:
  - `build_freeze_binaries.sh`
  - `record_manifest.sh`
  - `make_dataset_manifest.py`
  - `generate_stress_set.py`
  - `run_ab.py`
  - `summarize_ab.py`
  - `evaluate_metrics.py`

## A/B Benchmark Protocol (v1)

- A/B-0: baseline repeatability (`A_run1` vs `A_run2`).
- A/B-1: baseline vs instrumentation (`A_run1` vs `B1`) with invariance + overhead checks.
- Matrix:
  - Quality: `75`, `85`, `92`
  - Modes: `trellis`, `notrellis`
  - Datasets: `dev`, `test`, `stress` manifests

## v2 Design (Planned, Not Implemented)

- Compute image-level score from v1 aggregates (dev-set calibrated).
- Guarded routing policy:
  - low-score: taper/disable expensive trellis path
  - high-score: retain current trellis behavior
- Keep feature-flag fallback to baseline behavior.

