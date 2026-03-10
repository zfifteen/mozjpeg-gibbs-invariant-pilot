#!/usr/bin/env python3
import argparse
import pathlib
import time

from gibbs_large_common import (
    compare_off_vs_guarded,
    dump_json,
    evaluate_acceptance,
    parse_csv_floats,
    parse_csv_ints,
    read_manifest,
    run_encode,
    validate_decode,
    write_rows_csv,
)


def run_off_baseline(bin_path, djpeg_path, images, quality, repeats, artifacts_root, keep_artifacts):
    rows = []
    decode_failures = []
    for repeat in range(1, repeats + 1):
        for image in images:
            image_key = image.stem
            out_file = artifacts_root / f"off__{image_key}__r{repeat}.jpg"
            enc = run_encode(
                bin_path=bin_path,
                image_path=image,
                out_path=out_file,
                quality=quality,
                mode="off",
                gibbs_threshold=0.18,
                gibbs_cliff_min=1.0,
                gibbs_tail_min_flips=2,
                gibbs_tail_activity_max=3,
            )
            decode = {"ok": False, "exit_code": -1, "stderr": ""}
            if enc["output_exists"] and enc["exit_code"] == 0:
                decode = validate_decode(djpeg_path, out_file)

            row = {
                "image": str(image),
                "image_key": image_key,
                "repeat": repeat,
                "variant": "off",
                "quality": quality,
                "runtime_ms": enc["runtime_ms"],
                "output_bytes": enc["output_bytes"],
                "sha256": enc["sha256"],
                "encode_exit_code": enc["exit_code"],
                "encode_stderr": enc["stderr"],
                "decode_ok": int(decode["ok"]),
                "decode_exit_code": decode["exit_code"],
                "decode_stderr": decode["stderr"],
                "output_path": str(out_file),
                "gibbs_threshold": "",
                "gibbs_cliff_min": "",
                "gibbs_tail_activity_max": "",
                "gibbs_tail_min_flips": "",
            }
            rows.append(row)

            if enc["exit_code"] != 0 or not decode["ok"]:
                decode_failures.append({
                    "image": str(image),
                    "repeat": repeat,
                    "variant": "off",
                    "output_path": str(out_file),
                    "encode_exit_code": enc["exit_code"],
                    "encode_stderr": enc["stderr"],
                    "decode_exit_code": decode["exit_code"],
                    "decode_stderr": decode["stderr"],
                })

            if not keep_artifacts and out_file.exists():
                out_file.unlink()

    return rows, decode_failures


def run_guarded_tuple(bin_path, djpeg_path, images, quality, repeats, artifacts_root,
                      threshold, cliff_min, tail_min_flips, tail_activity_max,
                      keep_artifacts):
    rows = []
    decode_failures = []

    tuple_tag = f"th{threshold:.4f}_cl{cliff_min:.4f}_ta{tail_activity_max}_tf{tail_min_flips}"

    for repeat in range(1, repeats + 1):
        for image in images:
            image_key = image.stem
            out_file = artifacts_root / f"{tuple_tag}__{image_key}__r{repeat}.jpg"
            enc = run_encode(
                bin_path=bin_path,
                image_path=image,
                out_path=out_file,
                quality=quality,
                mode="guarded",
                gibbs_threshold=threshold,
                gibbs_cliff_min=cliff_min,
                gibbs_tail_min_flips=tail_min_flips,
                gibbs_tail_activity_max=tail_activity_max,
            )
            decode = {"ok": False, "exit_code": -1, "stderr": ""}
            if enc["output_exists"] and enc["exit_code"] == 0:
                decode = validate_decode(djpeg_path, out_file)

            row = {
                "image": str(image),
                "image_key": image_key,
                "repeat": repeat,
                "variant": "guarded",
                "quality": quality,
                "runtime_ms": enc["runtime_ms"],
                "output_bytes": enc["output_bytes"],
                "sha256": enc["sha256"],
                "encode_exit_code": enc["exit_code"],
                "encode_stderr": enc["stderr"],
                "decode_ok": int(decode["ok"]),
                "decode_exit_code": decode["exit_code"],
                "decode_stderr": decode["stderr"],
                "output_path": str(out_file),
                "gibbs_threshold": threshold,
                "gibbs_cliff_min": cliff_min,
                "gibbs_tail_activity_max": tail_activity_max,
                "gibbs_tail_min_flips": tail_min_flips,
            }
            rows.append(row)

            if enc["exit_code"] != 0 or not decode["ok"]:
                decode_failures.append({
                    "image": str(image),
                    "repeat": repeat,
                    "variant": "guarded",
                    "output_path": str(out_file),
                    "encode_exit_code": enc["exit_code"],
                    "encode_stderr": enc["stderr"],
                    "decode_exit_code": decode["exit_code"],
                    "decode_stderr": decode["stderr"],
                })

            if not keep_artifacts and out_file.exists():
                out_file.unlink()

    return rows, decode_failures


def main():
    ap = argparse.ArgumentParser(description="Sweep guarded Gibbs parameters on large-image set")
    ap.add_argument("--bin", type=pathlib.Path, required=True)
    ap.add_argument("--djpeg", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--thresholds", default="0.14,0.18,0.22")
    ap.add_argument("--cliff-mins", default="0.9,1.0,1.1")
    ap.add_argument("--tail-activity-maxes", default="2,3,4")
    ap.add_argument("--tail-min-flips", type=int, default=2)
    ap.add_argument("--results-dir", type=pathlib.Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    images = read_manifest(args.manifest)
    if not images:
        raise SystemExit(f"manifest has no images: {args.manifest}")

    thresholds = parse_csv_floats(args.thresholds)
    cliff_mins = parse_csv_floats(args.cliff_mins)
    tail_activity_maxes = parse_csv_ints(args.tail_activity_maxes)

    run_tag = args.tag or time.strftime("sweep_%Y%m%d_%H%M%S")
    out_dir = args.results_dir / run_tag
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("running off baseline once for all tuples...")
    off_rows, off_decode_failures = run_off_baseline(
        bin_path=args.bin,
        djpeg_path=args.djpeg,
        images=images,
        quality=args.quality,
        repeats=args.repeats,
        artifacts_root=artifacts_dir,
        keep_artifacts=args.keep_artifacts,
    )

    tuple_summaries = []
    all_rows = list(off_rows)
    all_decode_failures = list(off_decode_failures)

    selected_tuple = None

    for threshold in thresholds:
        for cliff_min in cliff_mins:
            for tail_activity_max in tail_activity_maxes:
                print(
                    "evaluating tuple:",
                    f"threshold={threshold}",
                    f"cliff_min={cliff_min}",
                    f"tail_activity_max={tail_activity_max}",
                )

                guarded_rows, guarded_decode_failures = run_guarded_tuple(
                    bin_path=args.bin,
                    djpeg_path=args.djpeg,
                    images=images,
                    quality=args.quality,
                    repeats=args.repeats,
                    artifacts_root=artifacts_dir,
                    threshold=threshold,
                    cliff_min=cliff_min,
                    tail_min_flips=args.tail_min_flips,
                    tail_activity_max=tail_activity_max,
                    keep_artifacts=args.keep_artifacts,
                )

                combined_rows = off_rows + guarded_rows
                metrics, per_image = compare_off_vs_guarded(combined_rows)
                acceptance = evaluate_acceptance(
                    metrics,
                    decode_failures=len(off_decode_failures) + len(guarded_decode_failures),
                )

                record = {
                    "gibbs_threshold": threshold,
                    "gibbs_cliff_min": cliff_min,
                    "gibbs_tail_min_flips": args.tail_min_flips,
                    "gibbs_tail_activity_max": tail_activity_max,
                    "metrics": metrics,
                    "acceptance": acceptance,
                    "per_image": per_image,
                    "decode_failures": len(off_decode_failures) + len(guarded_decode_failures),
                }
                tuple_summaries.append(record)
                all_rows.extend(guarded_rows)
                all_decode_failures.extend(guarded_decode_failures)

                if selected_tuple is None and acceptance["all_pass"]:
                    selected_tuple = record

    write_rows_csv(
        out_dir / "rows.csv",
        all_rows,
        [
            "image", "image_key", "repeat", "variant", "quality",
            "runtime_ms", "output_bytes", "sha256",
            "encode_exit_code", "encode_stderr",
            "decode_ok", "decode_exit_code", "decode_stderr",
            "output_path",
            "gibbs_threshold", "gibbs_cliff_min",
            "gibbs_tail_activity_max", "gibbs_tail_min_flips",
        ],
    )

    tuple_rows = []
    for t in tuple_summaries:
        tuple_rows.append({
            "gibbs_threshold": t["gibbs_threshold"],
            "gibbs_cliff_min": t["gibbs_cliff_min"],
            "gibbs_tail_min_flips": t["gibbs_tail_min_flips"],
            "gibbs_tail_activity_max": t["gibbs_tail_activity_max"],
            "runtime_delta_median_pct": t["metrics"]["runtime_delta_median_pct"],
            "runtime_delta_p95_pct": t["metrics"]["runtime_delta_p95_pct"],
            "size_delta_mean_pct": t["metrics"]["size_delta_mean_pct"],
            "size_delta_p95_pct": t["metrics"]["size_delta_p95_pct"],
            "decode_failures": t["decode_failures"],
            "all_pass": int(t["acceptance"]["all_pass"]),
        })

    write_rows_csv(
        out_dir / "tuple_summary_rows.csv",
        tuple_rows,
        [
            "gibbs_threshold", "gibbs_cliff_min", "gibbs_tail_min_flips", "gibbs_tail_activity_max",
            "runtime_delta_median_pct", "runtime_delta_p95_pct",
            "size_delta_mean_pct", "size_delta_p95_pct",
            "decode_failures", "all_pass",
        ],
    )

    sweep_summary = {
        "schema": "mozjpeg-gibbs-large-sweep-v1",
        "tag": run_tag,
        "manifest": str(args.manifest),
        "quality": args.quality,
        "repeats": args.repeats,
        "num_inputs": len(images),
        "search_space": {
            "thresholds": thresholds,
            "cliff_mins": cliff_mins,
            "tail_activity_maxes": tail_activity_maxes,
            "tail_min_flips": args.tail_min_flips,
        },
        "selected_tuple": selected_tuple,
        "tuple_summaries": tuple_summaries,
        "decode_failures_total": len(all_decode_failures),
    }

    decode_report = {
        "schema": "mozjpeg-gibbs-decode-validation-v1",
        "tag": run_tag,
        "total_outputs": len(all_rows),
        "decode_failures": len(all_decode_failures),
        "failed_outputs": all_decode_failures,
    }

    dump_json(out_dir / "sweep_summary.json", sweep_summary)
    dump_json(out_dir / "decode_validation.json", decode_report)

    print(f"rows: {out_dir / 'rows.csv'}")
    print(f"tuple summary: {out_dir / 'tuple_summary_rows.csv'}")
    print(f"sweep summary: {out_dir / 'sweep_summary.json'}")
    print(f"decode: {out_dir / 'decode_validation.json'}")
    if selected_tuple:
        print("selected tuple found")
    else:
        print("no tuple met all acceptance gates")


if __name__ == "__main__":
    main()
