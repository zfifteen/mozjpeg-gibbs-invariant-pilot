#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict


def median(xs):
    return statistics.median(xs) if xs else None


def p95(xs):
    if not xs:
        return None
    ys = sorted(xs)
    i = max(0, min(len(ys) - 1, int(round(0.95 * (len(ys) - 1)))))
    return ys[i]


def main():
    ap = argparse.ArgumentParser(description="Summarize mozjpeg Gibbs A/B results")
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rows = []
    with open(args.results_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["quality"] = int(row["quality"])
            row["runtime_ms"] = float(row["runtime_ms"])
            row["output_bytes"] = int(row["output_bytes"])
            row["exit_code"] = int(row["exit_code"])
            rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["quality"], row["mode"], row["run_label"])].append(row)

    # Build A/B pairing map.
    by_key = defaultdict(dict)
    for row in rows:
        k = (row["dataset"], row["quality"], row["mode"], row["image"])
        by_key[k][row["run_label"]] = row

    section = []
    overhead_all = []
    invariance_all = []

    for dataset in sorted({r["dataset"] for r in rows}):
        for quality in sorted({r["quality"] for r in rows}):
            for mode in sorted({r["mode"] for r in rows}):
                overhead = []
                invariance = []
                for k, m in by_key.items():
                    if k[0] != dataset or k[1] != quality or k[2] != mode:
                        continue
                    if "A_run1" not in m or "B1" not in m:
                        continue
                    a = m["A_run1"]
                    b = m["B1"]
                    if a["runtime_ms"] > 0:
                        overhead.append((b["runtime_ms"] - a["runtime_ms"]) / a["runtime_ms"] * 100.0)
                    invariance.append(1.0 if a["sha256"] == b["sha256"] and a["exit_code"] == 0 and b["exit_code"] == 0 else 0.0)

                overhead_all.extend(overhead)
                invariance_all.extend(invariance)

                if overhead or invariance:
                    section.append({
                        "dataset": dataset,
                        "quality": quality,
                        "mode": mode,
                        "ab1_overhead_median_pct": median(overhead),
                        "ab1_overhead_p95_pct": p95(overhead),
                        "ab1_invariance_rate": (sum(invariance) / len(invariance)) if invariance else None,
                        "pairs": len(invariance),
                    })

    summary = {
        "schema": "mozjpeg-gibbs-summary-v1",
        "overall": {
            "ab1_overhead_median_pct": median(overhead_all),
            "ab1_overhead_p95_pct": p95(overhead_all),
            "ab1_invariance_rate": (sum(invariance_all) / len(invariance_all)) if invariance_all else None,
            "pairs": len(invariance_all),
        },
        "by_dataset_quality_mode": section,
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
