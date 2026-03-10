#!/usr/bin/env python3
import argparse
import pathlib
import time

from gibbs_large_common import dump_json, read_manifest, run_encode, validate_decode, write_rows_csv


def main():
    ap = argparse.ArgumentParser(description="Check A vs patched-off bitstream invariance")
    ap.add_argument("--bin-a", type=pathlib.Path, required=True, help="Baseline cjpeg")
    ap.add_argument("--bin-b", type=pathlib.Path, required=True, help="Patched cjpeg")
    ap.add_argument("--djpeg", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--results-dir", type=pathlib.Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    images = read_manifest(args.manifest)
    if not images:
        raise SystemExit(f"manifest has no images: {args.manifest}")

    run_tag = args.tag or time.strftime("off_invariance_%Y%m%d_%H%M%S")
    out_dir = args.results_dir / run_tag
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mismatches = []
    decode_failures = []

    for image in images:
        key = image.stem
        out_a = artifacts_dir / f"{key}__A.jpg"
        out_b = artifacts_dir / f"{key}__B_off.jpg"

        # Baseline run (no Gibbs flags).
        enc_a = run_encode(
            bin_path=args.bin_a,
            image_path=image,
            out_path=out_a,
            quality=args.quality,
            mode="plain",
            gibbs_threshold=0.18,
            gibbs_cliff_min=1.0,
            gibbs_tail_min_flips=2,
            gibbs_tail_activity_max=3,
        )

        # Patched run with explicit off mode.
        enc_b = run_encode(
            bin_path=args.bin_b,
            image_path=image,
            out_path=out_b,
            quality=args.quality,
            mode="off",
            gibbs_threshold=0.18,
            gibbs_cliff_min=1.0,
            gibbs_tail_min_flips=2,
            gibbs_tail_activity_max=3,
        )

        decode_a = {"ok": False, "exit_code": -1, "stderr": ""}
        decode_b = {"ok": False, "exit_code": -1, "stderr": ""}
        if enc_a["output_exists"] and enc_a["exit_code"] == 0:
            decode_a = validate_decode(args.djpeg, out_a)
        if enc_b["output_exists"] and enc_b["exit_code"] == 0:
            decode_b = validate_decode(args.djpeg, out_b)

        same_hash = enc_a["sha256"] == enc_b["sha256"] and enc_a["sha256"] != ""

        row = {
            "image": str(image),
            "quality": args.quality,
            "a_runtime_ms": enc_a["runtime_ms"],
            "b_runtime_ms": enc_b["runtime_ms"],
            "a_output_bytes": enc_a["output_bytes"],
            "b_output_bytes": enc_b["output_bytes"],
            "a_sha256": enc_a["sha256"],
            "b_sha256": enc_b["sha256"],
            "bit_identical": int(same_hash),
            "a_encode_exit": enc_a["exit_code"],
            "b_encode_exit": enc_b["exit_code"],
            "a_decode_ok": int(decode_a["ok"]),
            "b_decode_ok": int(decode_b["ok"]),
            "a_output_path": str(out_a),
            "b_output_path": str(out_b),
        }
        rows.append(row)

        if not same_hash:
            mismatches.append({
                "image": str(image),
                "a_sha256": enc_a["sha256"],
                "b_sha256": enc_b["sha256"],
                "a_output_bytes": enc_a["output_bytes"],
                "b_output_bytes": enc_b["output_bytes"],
            })

        if not decode_a["ok"] or not decode_b["ok"]:
            decode_failures.append({
                "image": str(image),
                "a_encode_exit": enc_a["exit_code"],
                "b_encode_exit": enc_b["exit_code"],
                "a_decode_exit": decode_a["exit_code"],
                "b_decode_exit": decode_b["exit_code"],
                "a_decode_stderr": decode_a["stderr"],
                "b_decode_stderr": decode_b["stderr"],
            })

        if not args.keep_artifacts:
            if out_a.exists():
                out_a.unlink()
            if out_b.exists():
                out_b.unlink()

    write_rows_csv(
        out_dir / "rows.csv",
        rows,
        [
            "image", "quality",
            "a_runtime_ms", "b_runtime_ms",
            "a_output_bytes", "b_output_bytes",
            "a_sha256", "b_sha256", "bit_identical",
            "a_encode_exit", "b_encode_exit",
            "a_decode_ok", "b_decode_ok",
            "a_output_path", "b_output_path",
        ],
    )

    report = {
        "schema": "mozjpeg-gibbs-off-invariance-v1",
        "tag": run_tag,
        "manifest": str(args.manifest),
        "num_inputs": len(images),
        "quality": args.quality,
        "bit_identical_pairs": sum(r["bit_identical"] for r in rows),
        "bit_identical_rate": (sum(r["bit_identical"] for r in rows) / len(rows)) if rows else None,
        "mismatch_count": len(mismatches),
        "decode_failures": len(decode_failures),
        "mismatches": mismatches,
        "decode_failure_details": decode_failures,
        "acceptance": {
            "all_pairs_bit_identical": len(mismatches) == 0,
            "decode_failures_zero": len(decode_failures) == 0,
            "all_pass": len(mismatches) == 0 and len(decode_failures) == 0,
        },
    }
    dump_json(out_dir / "invariance_report.json", report)

    print(f"rows: {out_dir / 'rows.csv'}")
    print(f"report: {out_dir / 'invariance_report.json'}")
    print(f"bit_identical_rate: {report['bit_identical_rate']}")
    print(f"acceptance_all_pass: {report['acceptance']['all_pass']}")


if __name__ == "__main__":
    main()
