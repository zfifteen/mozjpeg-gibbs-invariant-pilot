# Gibbs-Invariant v1 Benchmark Harness (mozjpeg)

This folder contains reproducible tooling for:

- Freezing binary `A` (upstream baseline) and `B1` (gibbs log-only instrumentation).
- Creating deterministic corpus manifests (`dev`, `test`, `stress`).
- Running A/B-0 and A/B-1 experiments.
- Summarizing runtime/size/invariance metrics.

## Scope

v1 is **instrumentation-only**:

- `B1` adds `-gibbs-mode off|log-only` and `-gibbs-log <jsonl>`.
- No encoding decision changes are introduced in log-only mode.

## Quick Start

1. Freeze binaries and build manifests:

```bash
cd /Users/velocityworks/IdeaProjects/mozjpeg
./benchmarks/gibbs_v1/scripts/build_freeze_binaries.sh
./benchmarks/gibbs_v1/scripts/record_manifest.sh
```

2. Prepare dataset manifests:

```bash
# Example paths; adjust to your local corpus staging
python3 benchmarks/gibbs_v1/scripts/make_dataset_manifest.py \
  --clic-train-dir /data/clic/train \
  --kodak-dir /data/kodak24 \
  --tecnick-dir /data/tecnick100 \
  --stress-dir /data/gibbs_stress \
  --out-dir benchmarks/gibbs_v1/manifests \
  --seed 1729
```

3. Generate deterministic stress set (if needed):

```bash
python3 benchmarks/gibbs_v1/scripts/generate_stress_set.py \
  --out-dir /data/gibbs_stress \
  --count 200 \
  --seed 1729
```

4. Run A/B-0 and A/B-1:

```bash
python3 benchmarks/gibbs_v1/scripts/run_ab.py \
  --bin-a benchmarks/gibbs_v1/bin/cjpeg_A_baseline \
  --bin-b benchmarks/gibbs_v1/bin/cjpeg_B1_logonly \
  --dev-manifest benchmarks/gibbs_v1/manifests/dev.txt \
  --test-manifest benchmarks/gibbs_v1/manifests/test.txt \
  --stress-manifest benchmarks/gibbs_v1/manifests/stress.txt \
  --qualities 75,85,92 \
  --modes trellis,notrellis \
  --results-dir benchmarks/gibbs_v1/results
```

5. Summarize:

```bash
python3 benchmarks/gibbs_v1/scripts/summarize_ab.py \
  --results-csv benchmarks/gibbs_v1/results/ab_results.csv
```

## Experiments

- `A/B-0`: baseline repeatability (run `A` twice, compare runtime noise + bitstream equality).
- `A/B-1`: `A` vs `B1` invariance and overhead in log-only mode.
- `A/B-2` (future): `A` vs `B2` guarded-routing policy.

## Acceptance Gates (v1)

- Bitstream identity for `A` vs `B1` under identical options.
- Instrumentation overhead <= 3% median on `test` corpus.
- JSONL log records present for >= 99% of `B1` encodes.

## Large-Image Speed/Size-Safe Protocol

This repo now includes a dedicated large-image protocol for Gibbs guarded routing:

- Manifest: `benchmarks/gibbs_v1/manifests/large_mixed.txt` (6-10 large inputs).
- Quality: `85`, repeats: `7`.
- Comparison: patched `-gibbs-mode off` vs patched `-gibbs-mode guarded`.
- Outputs per run:
  - `rows.csv`
  - `summary.json`
  - `decode_validation.json`

Commands:

```bash
python3 benchmarks/gibbs_v1/scripts/check_off_invariance.py \
  --bin-a benchmarks/gibbs_v1/bin/cjpeg_A_baseline \
  --bin-b /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/cjpeg \
  --djpeg /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/djpeg \
  --manifest benchmarks/gibbs_v1/manifests/large_mixed.txt \
  --quality 85 \
  --results-dir benchmarks/gibbs_v1/results

python3 benchmarks/gibbs_v1/scripts/run_large_ab.py \
  --bin /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/cjpeg \
  --djpeg /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/djpeg \
  --manifest benchmarks/gibbs_v1/manifests/large_mixed.txt \
  --quality 85 \
  --repeats 7 \
  --results-dir benchmarks/gibbs_v1/results

python3 benchmarks/gibbs_v1/scripts/sweep_large_ab.py \
  --bin /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/cjpeg \
  --djpeg /Users/velocityworks/IdeaProjects/mozjpeg/build-patched/djpeg \
  --manifest benchmarks/gibbs_v1/manifests/large_mixed.txt \
  --quality 85 \
  --repeats 7 \
  --thresholds 0.14,0.18,0.22 \
  --cliff-mins 0.9,1.0,1.1 \
  --tail-activity-maxes 2,3,4 \
  --results-dir benchmarks/gibbs_v1/results
```
