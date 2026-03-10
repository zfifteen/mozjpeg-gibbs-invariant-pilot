# Technical Memo for External Reviewers

## Project Context

This repository is a public working fork of `mozjpeg` used for a targeted R&D exercise:

- Upstream base: `master`
- Experimental branch: `codex/gibbs-invariant-pilot-v1`
- Public repo: `https://github.com/zfifteen/mozjpeg-gibbs-invariant-pilot`

The goal is **not** to redesign JPEG encoding broadly. The goal is to test whether two Gibbs-invariant-inspired signals can safely drive a narrow optimization in `mozjpeg`.

## Purpose of the Exercise

We are evaluating whether theorem-derived diagnostics can identify blocks where expensive trellis behavior can be reduced without violating size/quality safety gates.

In plain terms:

1. Detect likely discontinuity/tail behavior from block spectral structure.
2. Use that signal to route encoder effort more selectively.
3. Preserve safety (decode correctness, bounded size delta, bounded tail latency).

## Prior Work Summary

A substantial prior attempt was already implemented on `codex/gibbs-invariant-pilot-v1`:

- Instrumentation harness and reproducibility tooling: `benchmarks/gibbs_v1/`
- Experimental CLI and routing controls in `cjpeg.c`
- Guarded routing logic in trellis quantization paths
- Many benchmark sweeps and tuning runs in `benchmarks/gibbs_v1/results/`

### What succeeded

Instrumentation/invariance stage was successful:

- Bitstream invariance and logging completeness passed in
  `benchmarks/gibbs_v1/results/local_ab_20260306/ab_summary.json`.

### What failed

Optimization stage did not meet acceptance criteria:

- Safe-ish configurations produced insufficient median runtime gain.
- Aggressive configurations improved runtime but caused unacceptable output-size regression.
- No configuration met all gates consistently.

## Why We Are Requesting Review

The prior attempt drifted into broad tuning and multi-parameter search. We now need a strict, high-discipline retry plan before further code edits.

Primary document for review:

- `benchmarks/gibbs_v1/RESTART_IMPLEMENTATION_PLAN.md`

This plan is intentionally conservative and stage-gated.

## What We Need Reviewers to Evaluate

Please evaluate the restart plan for:

1. **Correct seam selection**
   - Are the proposed file/function touch points the right ones for a narrow, causal experiment?

2. **Theorem-to-code mapping quality**
   - Are proposed feature proxies faithful enough to justify policy decisions?

3. **Scope control**
   - Does the plan avoid broad invasive changes before predictive validity is proven?

4. **Hypothesis testability**
   - Are hypotheses falsifiable with the existing harness and artifacts?

5. **Acceptance-gate rigor**
   - Are stop conditions and pass/fail criteria adequate to prevent another uncontrolled tuning cycle?

6. **Risk of repeating prior failure mode**
   - Identify where the plan may still allow drift (parameter fishing, hidden scope expansion, conflated objectives).

## Non-Goals for This Review

Please do **not** optimize for:

- New codec features unrelated to this hypothesis
- Broad architecture refactors
- Cross-codec portability
- Publication framing

This review is strictly about making the next experiment technically disciplined and decision-useful.

## Current Constraints

- Must preserve decode correctness.
- Must preserve off-mode invariance checks.
- Must keep optimization acceptance gates explicit and unchanged unless justified.
- Must avoid touching public ABI/APIs for this phase.

## Suggested Review Output Format

To keep reviews actionable, we prefer:

1. Findings ranked by severity (`P1`, `P2`, `P3`)
2. For each finding:
   - File/function reference
   - Why it matters
   - Concrete correction
3. Final verdict:
   - `approve`, `approve with required changes`, or `reject`

## Quick Navigation

- Restart plan: `benchmarks/gibbs_v1/RESTART_IMPLEMENTATION_PLAN.md`
- Prior harness overview: `benchmarks/gibbs_v1/README.md`
- Prior implementation summary: `benchmarks/gibbs_v1/IMPLEMENTATION_PLAN.md`
- Core acceptance logic: `benchmarks/gibbs_v1/scripts/gibbs_large_common.py`
- Representative failed optimization run:
  `benchmarks/gibbs_v1/results/large_ab_q85_r7_persistenttail_lazy_tuned/summary.json`
- Representative aggressive regression run:
  `benchmarks/gibbs_v1/results/guarded_detailed_fix3_20260306/summary.json`
- Representative successful instrumentation/invariance run:
  `benchmarks/gibbs_v1/results/local_ab_20260306/ab_summary.json`

## Bottom Line

This exercise is a controlled retry, not a fresh greenfield effort.

The core review question is:

> Does the restart plan enforce enough discipline to produce a clear yes/no outcome on theorem-guided routing effectiveness, without repeating the previous broad and costly iteration cycle?
