#!/usr/bin/env python3
import csv
import hashlib
import json
import pathlib
import statistics
import subprocess
import time


def read_manifest(path: pathlib.Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(pathlib.Path(s))
    return items


def sha256_file(path: pathlib.Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def median(xs):
    return statistics.median(xs) if xs else None


def p95(xs):
    if not xs:
        return None
    ys = sorted(xs)
    idx = max(0, min(len(ys) - 1, int(round(0.95 * (len(ys) - 1)))))
    return ys[idx]


def mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def parse_csv_floats(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_rows_csv(path: pathlib.Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_encode_cmd(bin_path: pathlib.Path, image_path: pathlib.Path, out_path: pathlib.Path,
                     quality: int, mode: str, gibbs_threshold: float,
                     gibbs_cliff_min: float, gibbs_tail_min_flips: int,
                     gibbs_tail_activity_max: int):
    cmd = [
        str(bin_path),
        "-quality", str(quality),
        "-outfile", str(out_path),
    ]

    if mode == "plain":
        pass
    elif mode == "off":
        cmd.extend(["-gibbs-mode", "off"])
    elif mode == "guarded":
        cmd.extend([
            "-gibbs-mode", "guarded",
            "-gibbs-threshold", f"{gibbs_threshold:.6g}",
            "-gibbs-cliff-min", f"{gibbs_cliff_min:.6g}",
            "-gibbs-tail-min-flips", str(gibbs_tail_min_flips),
            "-gibbs-tail-activity-max", str(gibbs_tail_activity_max),
            "-gibbs-low-loops", "0",
        ])
    else:
        raise ValueError(f"unknown mode: {mode}")

    cmd.append(str(image_path))
    return cmd


def run_encode(bin_path: pathlib.Path, image_path: pathlib.Path, out_path: pathlib.Path,
               quality: int, mode: str, gibbs_threshold: float,
               gibbs_cliff_min: float, gibbs_tail_min_flips: int,
               gibbs_tail_activity_max: int):
    cmd = build_encode_cmd(
        bin_path=bin_path,
        image_path=image_path,
        out_path=out_path,
        quality=quality,
        mode=mode,
        gibbs_threshold=gibbs_threshold,
        gibbs_cliff_min=gibbs_cliff_min,
        gibbs_tail_min_flips=gibbs_tail_min_flips,
        gibbs_tail_activity_max=gibbs_tail_activity_max,
    )

    if out_path.exists():
        out_path.unlink()

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    exists = out_path.exists()
    size = out_path.stat().st_size if exists else -1
    digest = sha256_file(out_path) if exists else ""

    return {
        "cmd": cmd,
        "exit_code": proc.returncode,
        "stderr": proc.stderr.decode("utf-8", errors="replace").strip(),
        "runtime_ms": dt_ms,
        "output_exists": exists,
        "output_bytes": size,
        "sha256": digest,
    }


def validate_decode(djpeg_path: pathlib.Path, jpeg_path: pathlib.Path):
    proc = subprocess.run(
        [str(djpeg_path), "-ppm", str(jpeg_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return {
        "ok": (proc.returncode == 0),
        "exit_code": proc.returncode,
        "stderr": proc.stderr.decode("utf-8", errors="replace").strip(),
    }


def per_image_medians(rows):
    grouped = {}
    for row in rows:
        key = (row["image"], row["variant"])
        bucket = grouped.setdefault(key, {"runtime_ms": [], "output_bytes": []})
        bucket["runtime_ms"].append(float(row["runtime_ms"]))
        bucket["output_bytes"].append(int(row["output_bytes"]))

    med = {}
    for key, bucket in grouped.items():
        med[key] = {
            "runtime_ms": median(bucket["runtime_ms"]),
            "output_bytes": median(bucket["output_bytes"]),
        }
    return med


def compare_off_vs_guarded(rows):
    med = per_image_medians(rows)
    images = sorted({row["image"] for row in rows})
    per_image = []

    for image in images:
        off = med.get((image, "off"))
        guarded = med.get((image, "guarded"))
        if off is None or guarded is None:
            continue

        off_rt = off["runtime_ms"]
        guarded_rt = guarded["runtime_ms"]
        off_sz = off["output_bytes"]
        guarded_sz = guarded["output_bytes"]

        rt_delta_pct = None
        sz_delta_pct = None
        if off_rt and off_rt > 0:
            rt_delta_pct = (guarded_rt - off_rt) / off_rt * 100.0
        if off_sz and off_sz > 0:
            sz_delta_pct = (guarded_sz - off_sz) / off_sz * 100.0

        per_image.append({
            "image": image,
            "off_runtime_ms_median": off_rt,
            "guarded_runtime_ms_median": guarded_rt,
            "runtime_delta_pct": rt_delta_pct,
            "off_bytes_median": off_sz,
            "guarded_bytes_median": guarded_sz,
            "size_delta_pct": sz_delta_pct,
        })

    runtime_deltas = [x["runtime_delta_pct"] for x in per_image if x["runtime_delta_pct"] is not None]
    size_deltas = [x["size_delta_pct"] for x in per_image if x["size_delta_pct"] is not None]

    metrics = {
        "runtime_delta_median_pct": median(runtime_deltas),
        "runtime_delta_p95_pct": p95(runtime_deltas),
        "size_delta_mean_pct": mean(size_deltas),
        "size_delta_p95_pct": p95(size_deltas),
        "num_images": len(per_image),
    }
    return metrics, per_image


def evaluate_acceptance(metrics, decode_failures):
    rt_med = metrics["runtime_delta_median_pct"]
    rt_p95 = metrics["runtime_delta_p95_pct"]
    sz_mean = metrics["size_delta_mean_pct"]
    sz_p95 = metrics["size_delta_p95_pct"]

    # Guarded is faster when delta is negative.
    speed_gate = (rt_med is not None) and (rt_med <= -3.0)
    tail_gate = (rt_p95 is not None) and (rt_p95 <= 1.0)
    size_mean_gate = (sz_mean is not None) and (sz_mean <= 0.20)
    size_p95_gate = (sz_p95 is not None) and (sz_p95 <= 0.50)
    decode_gate = (decode_failures == 0)

    return {
        "runtime_median_improvement_ge_3pct": speed_gate,
        "runtime_p95_delta_le_1pct": tail_gate,
        "size_mean_delta_le_0_20pct": size_mean_gate,
        "size_p95_delta_le_0_50pct": size_p95_gate,
        "decode_failures_zero": decode_gate,
        "all_pass": speed_gate and tail_gate and size_mean_gate and size_p95_gate and decode_gate,
    }


def dump_json(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
