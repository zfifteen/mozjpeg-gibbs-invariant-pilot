#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import pathlib
import statistics
import subprocess
import time


def read_manifest(path: pathlib.Path):
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(pathlib.Path(s))
    return lines


def sha256_file(path: pathlib.Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_cmd(bin_path, image_path, out_path, quality, mode, gibbs_log_path=None):
    cmd = [str(bin_path), "-quality", str(quality), "-outfile", str(out_path)]
    if mode == "notrellis":
        cmd.append("-notrellis")
    if gibbs_log_path is not None:
        cmd.extend(["-gibbs-mode", "log-only", "-gibbs-log", str(gibbs_log_path)])
    cmd.append(str(image_path))
    return cmd


def key_for(row):
    return (row["dataset"], row["image"], row["quality"], row["mode"])


def median(xs):
    return statistics.median(xs) if xs else None


def p95(xs):
    if not xs:
        return None
    ys = sorted(xs)
    i = max(0, min(len(ys) - 1, int(round(0.95 * (len(ys) - 1)))))
    return ys[i]


def count_jsonl_records(path: pathlib.Path):
    if not path.exists():
        return 0, 0

    total = 0
    valid = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                json.loads(s)
                valid += 1
            except json.JSONDecodeError:
                pass
    return total, valid


def main():
    ap = argparse.ArgumentParser(description="Run mozjpeg Gibbs A/B-0 and A/B-1 experiments")
    ap.add_argument("--bin-a", type=pathlib.Path, required=True)
    ap.add_argument("--bin-b", type=pathlib.Path, required=True)
    ap.add_argument("--dev-manifest", type=pathlib.Path, required=True)
    ap.add_argument("--test-manifest", type=pathlib.Path, required=True)
    ap.add_argument("--stress-manifest", type=pathlib.Path, required=True)
    ap.add_argument("--qualities", default="75,85,92")
    ap.add_argument("--modes", default="trellis,notrellis")
    ap.add_argument("--results-dir", type=pathlib.Path, required=True)
    ap.add_argument("--max-images-per-dataset", type=int, default=0,
                    help="0 means use all images")
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    qualities = [int(x) for x in parse_csv_list(args.qualities)]
    modes = parse_csv_list(args.modes)

    datasets = {
        "dev": read_manifest(args.dev_manifest),
        "test": read_manifest(args.test_manifest),
        "stress": read_manifest(args.stress_manifest),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.results_dir / "gibbs_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prepared_logs = set()
    expected_log_records = {}

    # A/B-0: baseline repeatability (A run 1 vs A run 2)
    run_defs = [
        ("A_run1", args.bin_a, False),
        ("A_run2", args.bin_a, False),
        ("B1", args.bin_b, True),
    ]

    for dataset_name, images in datasets.items():
        if args.max_images_per_dataset > 0:
            images = images[:args.max_images_per_dataset]

        for quality in qualities:
            for mode in modes:
                for image_path in images:
                    image_key = image_path.stem
                    for run_label, bin_path, use_gibbs in run_defs:
                        out_file = artifacts_dir / f"{dataset_name}__q{quality}__{mode}__{run_label}__{image_key}.jpg"
                        if out_file.exists():
                            out_file.unlink()

                        gibbs_log_file = None
                        if use_gibbs:
                            gibbs_log_file = logs_dir / f"{dataset_name}__q{quality}__{mode}.jsonl"
                            log_key = (dataset_name, quality, mode)
                            if log_key not in prepared_logs:
                                if gibbs_log_file.exists():
                                    gibbs_log_file.unlink()
                                prepared_logs.add(log_key)
                            expected_log_records[log_key] = expected_log_records.get(log_key, 0) + 1

                        cmd = build_cmd(
                            bin_path=bin_path,
                            image_path=image_path,
                            out_path=out_file,
                            quality=quality,
                            mode=mode,
                            gibbs_log_path=gibbs_log_file,
                        )

                        t0 = time.perf_counter()
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        dt_ms = (time.perf_counter() - t0) * 1000.0

                        size = out_file.stat().st_size if out_file.exists() else -1
                        digest = sha256_file(out_file) if out_file.exists() else ""

                        rows.append({
                            "dataset": dataset_name,
                            "image": str(image_path),
                            "quality": quality,
                            "mode": mode,
                            "run_label": run_label,
                            "binary": str(bin_path),
                            "runtime_ms": dt_ms,
                            "output_bytes": size,
                            "sha256": digest,
                            "exit_code": proc.returncode,
                            "stderr": proc.stderr.decode("utf-8", errors="replace").strip(),
                        })

                        if not args.keep_artifacts and out_file.exists():
                            out_file.unlink()

    csv_path = args.results_dir / "ab_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "image", "quality", "mode", "run_label", "binary",
                "runtime_ms", "output_bytes", "sha256", "exit_code", "stderr",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    a1 = {key_for(r): r for r in rows if r["run_label"] == "A_run1"}
    a2 = {key_for(r): r for r in rows if r["run_label"] == "A_run2"}
    b1 = {key_for(r): r for r in rows if r["run_label"] == "B1"}

    noise_pct = []
    invariance = []
    overhead_pct = []
    invariance_test = []
    overhead_pct_test = []

    for k, r1 in a1.items():
        r2 = a2.get(k)
        rb = b1.get(k)
        if r2 is not None and r1["runtime_ms"] > 0:
            noise_pct.append(abs(r2["runtime_ms"] - r1["runtime_ms"]) / r1["runtime_ms"] * 100.0)
        if rb is not None:
            inv = 1.0 if rb["sha256"] == r1["sha256"] and rb["exit_code"] == 0 and r1["exit_code"] == 0 else 0.0
            invariance.append(inv)
            if k[0] == "test":
                invariance_test.append(inv)
            if r1["runtime_ms"] > 0:
                ov = (rb["runtime_ms"] - r1["runtime_ms"]) / r1["runtime_ms"] * 100.0
                overhead_pct.append(ov)
                if k[0] == "test":
                    overhead_pct_test.append(ov)

    total_expected_logs = 0
    total_found_logs = 0
    total_valid_logs = 0
    log_completeness_entries = []
    for dataset_name, quality, mode in sorted(expected_log_records):
        expected = expected_log_records[(dataset_name, quality, mode)]
        log_path = logs_dir / f"{dataset_name}__q{quality}__{mode}.jsonl"
        found, valid = count_jsonl_records(log_path)
        total_expected_logs += expected
        total_found_logs += found
        total_valid_logs += valid
        completeness = (found / expected) if expected > 0 else 1.0
        validity = (valid / found) if found > 0 else 0.0
        log_completeness_entries.append({
            "dataset": dataset_name,
            "quality": quality,
            "mode": mode,
            "expected_records": expected,
            "found_records": found,
            "valid_json_records": valid,
            "completeness_rate": completeness,
            "valid_json_rate": validity,
            "path": str(log_path),
        })

    invariance_rate = (sum(invariance) / len(invariance)) if invariance else None
    invariance_rate_test = (sum(invariance_test) / len(invariance_test)) if invariance_test else None
    overhead_median = median(overhead_pct)
    overhead_p95 = p95(overhead_pct)
    overhead_median_test = median(overhead_pct_test)
    overhead_p95_test = p95(overhead_pct_test)
    log_completeness_rate = (total_found_logs / total_expected_logs) if total_expected_logs > 0 else None
    log_valid_json_rate = (total_valid_logs / total_found_logs) if total_found_logs > 0 else None

    accept_invariance = invariance_rate == 1.0 if invariance_rate is not None else False
    accept_overhead = overhead_median_test is not None and overhead_median_test <= 3.0
    accept_logs = log_completeness_rate is not None and log_completeness_rate >= 0.99

    summary = {
        "schema": "mozjpeg-gibbs-ab-summary-v1",
        "rows": len(rows),
        "ab0_baseline_runtime_noise_median_pct": median(noise_pct),
        "ab0_baseline_runtime_noise_p95_pct": p95(noise_pct),
        "ab1_bitstream_invariance_rate": invariance_rate,
        "ab1_bitstream_invariance_rate_test": invariance_rate_test,
        "ab1_overhead_median_pct": overhead_median,
        "ab1_overhead_p95_pct": overhead_p95,
        "ab1_overhead_median_pct_test": overhead_median_test,
        "ab1_overhead_p95_pct_test": overhead_p95_test,
        "ab1_log_records_expected": total_expected_logs,
        "ab1_log_records_found": total_found_logs,
        "ab1_log_records_valid_json": total_valid_logs,
        "ab1_log_completeness_rate": log_completeness_rate,
        "ab1_log_valid_json_rate": log_valid_json_rate,
        "ab1_log_completeness_detail": log_completeness_entries,
        "v1_acceptance": {
            "bitstream_identical": accept_invariance,
            "overhead_median_test_lte_3pct": accept_overhead,
            "logs_completeness_gte_99pct": accept_logs,
            "pass": (accept_invariance and accept_overhead and accept_logs),
        },
    }

    summary_path = args.results_dir / "ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
