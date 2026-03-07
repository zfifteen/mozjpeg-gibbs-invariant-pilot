#!/usr/bin/env python3
import argparse
import pathlib
import time

from gibbs_large_common import (
    compare_off_vs_guarded,
    dump_json,
    evaluate_acceptance,
    read_manifest,
    run_encode,
    validate_decode,
    write_rows_csv,
)


def main():
    ap = argparse.ArgumentParser(description="Run large-image Off vs Guarded benchmark")
    ap.add_argument("--bin", type=pathlib.Path, required=True,
                    help="Patched cjpeg binary that supports -gibbs-mode")
    ap.add_argument("--djpeg", type=pathlib.Path, required=True,
                    help="djpeg binary for decode validation")
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--results-dir", type=pathlib.Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gibbs-threshold", type=float, default=0.18)
    ap.add_argument("--gibbs-cliff-min", type=float, default=1.0)
    ap.add_argument("--gibbs-tail-min-flips", type=int, default=2)
    ap.add_argument("--gibbs-tail-activity-max", type=int, default=3)
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    images = read_manifest(args.manifest)
    if not images:
        raise SystemExit(f"manifest has no images: {args.manifest}")

    run_tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = args.results_dir / run_tag
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    decode_failures = []

    for repeat in range(1, args.repeats + 1):
        for image in images:
            image_key = image.stem
            for variant in ("off", "guarded"):
                out_file = artifacts_dir / f"{image_key}__{variant}__r{repeat}.jpg"
                enc = run_encode(
                    bin_path=args.bin,
                    image_path=image,
                    out_path=out_file,
                    quality=args.quality,
                    mode=variant,
                    gibbs_threshold=args.gibbs_threshold,
                    gibbs_cliff_min=args.gibbs_cliff_min,
                    gibbs_tail_min_flips=args.gibbs_tail_min_flips,
                    gibbs_tail_activity_max=args.gibbs_tail_activity_max,
                )

                decode = {"ok": False, "exit_code": -1, "stderr": ""}
                if enc["output_exists"] and enc["exit_code"] == 0:
                    decode = validate_decode(args.djpeg, out_file)

                row = {
                    "image": str(image),
                    "image_key": image_key,
                    "repeat": repeat,
                    "variant": variant,
                    "quality": args.quality,
                    "runtime_ms": enc["runtime_ms"],
                    "output_bytes": enc["output_bytes"],
                    "sha256": enc["sha256"],
                    "encode_exit_code": enc["exit_code"],
                    "encode_stderr": enc["stderr"],
                    "decode_ok": int(decode["ok"]),
                    "decode_exit_code": decode["exit_code"],
                    "decode_stderr": decode["stderr"],
                    "output_path": str(out_file),
                }
                rows.append(row)

                if enc["exit_code"] != 0 or not decode["ok"]:
                    decode_failures.append({
                        "image": str(image),
                        "repeat": repeat,
                        "variant": variant,
                        "output_path": str(out_file),
                        "encode_exit_code": enc["exit_code"],
                        "encode_stderr": enc["stderr"],
                        "decode_exit_code": decode["exit_code"],
                        "decode_stderr": decode["stderr"],
                    })

                if not args.keep_artifacts and out_file.exists():
                    out_file.unlink()

    fieldnames = [
        "image", "image_key", "repeat", "variant", "quality",
        "runtime_ms", "output_bytes", "sha256",
        "encode_exit_code", "encode_stderr",
        "decode_ok", "decode_exit_code", "decode_stderr",
        "output_path",
    ]
    write_rows_csv(out_dir / "rows.csv", rows, fieldnames)

    metrics, per_image = compare_off_vs_guarded(rows)
    acceptance = evaluate_acceptance(metrics, decode_failures=len(decode_failures))

    summary = {
        "schema": "mozjpeg-gibbs-large-ab-v1",
        "tag": run_tag,
        "manifest": str(args.manifest),
        "num_inputs": len(images),
        "quality": args.quality,
        "repeats": args.repeats,
        "params": {
            "gibbs_threshold": args.gibbs_threshold,
            "gibbs_cliff_min": args.gibbs_cliff_min,
            "gibbs_tail_min_flips": args.gibbs_tail_min_flips,
            "gibbs_tail_activity_max": args.gibbs_tail_activity_max,
            "gibbs_low_loops": 0,
        },
        "metrics": metrics,
        "acceptance": acceptance,
        "per_image": per_image,
    }

    decode_report = {
        "schema": "mozjpeg-gibbs-decode-validation-v1",
        "tag": run_tag,
        "total_outputs": len(rows),
        "decode_failures": len(decode_failures),
        "failed_outputs": decode_failures,
    }

    dump_json(out_dir / "summary.json", summary)
    dump_json(out_dir / "decode_validation.json", decode_report)

    print(f"rows: {out_dir / 'rows.csv'}")
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"decode: {out_dir / 'decode_validation.json'}")
    print(f"runtime_delta_median_pct: {metrics['runtime_delta_median_pct']}")
    print(f"size_delta_mean_pct: {metrics['size_delta_mean_pct']}")
    print(f"acceptance_all_pass: {acceptance['all_pass']}")


if __name__ == "__main__":
    main()
