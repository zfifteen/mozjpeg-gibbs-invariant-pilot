#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gibbs_large_common import (
    compare_off_vs_guarded,
    dump_json,
    evaluate_acceptance,
    p95,
    read_manifest,
    run_encode,
    validate_decode,
    write_rows_csv,
)


DEFAULT_SEED = 1729
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_PRECISION_FLOOR = 0.95
DEFAULT_SAFE_SIZE_DELTA_PCT = 0.20
DEFAULT_RECALL_NONINFERIORITY_MARGIN = 0.005

# zigzag index -> natural order index
JPEG_NATURAL_ORDER = np.array([
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int32)


GRAD_FEATURES = [
    "cliff_norm_p95",
]

HF_FEATURES = [
    "hf_ratio_mean",
]

COMBINED_FEATURES = [
    "hf_ratio_mean",
    "cliff_norm_p95",
]

THEOREM_FEATURES = [
    "hf_ratio_mean",
    "hf_ratio_p95",
    "tail_ratio_mean",
    "tail_ratio_p95",
    "cliff_norm_p95",
    "tail_flips_mean",
    "low_activity_frac",
    "persistent_frac_tau018",
    "grad_mean",
    "grad_p95",
]


@dataclass
class ThresholdSelection:
    tau: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass
class FittedModel:
    feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    tau: float


def stable_split_bucket(path: pathlib.Path, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{str(path)}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10


def assign_split(path: pathlib.Path, seed: int) -> str:
    bucket = stable_split_bucket(path, seed)
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def split_images(images: Sequence[pathlib.Path], seed: int) -> Dict[str, str]:
    return {str(p): assign_split(p, seed) for p in images}


def delta_pct(candidate: float, baseline: float) -> Optional[float]:
    if baseline is None or baseline <= 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def quant_table_for_quality(bin_path: pathlib.Path, sample_image: pathlib.Path, quality: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="gibbs_v6_qtbl_") as td:
        out_path = pathlib.Path(td) / "probe.jpg"
        cmd = [
            str(bin_path),
            "-quality",
            str(quality),
            "-outfile",
            str(out_path),
            str(sample_image),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not out_path.exists():
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"failed to probe quant table: {stderr}")

        with Image.open(out_path) as img:
            qtables = img.quantization
            if not qtables or 0 not in qtables:
                raise RuntimeError("no luma quantization table found in probe JPEG")
            q0 = np.asarray(qtables[0], dtype=np.float64)
            if q0.shape[0] != 64:
                raise RuntimeError("unexpected luma quantization table size")
            return q0


def image_to_luma(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as img:
        ycc = img.convert("YCbCr")
        y = np.asarray(ycc, dtype=np.float64)[:, :, 0]
    return y


def pad_to_multiple_of_8(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    ph = (8 - (h % 8)) % 8
    pw = (8 - (w % 8)) % 8
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, ph), (0, pw)), mode="edge")


def split_blocks_8x8(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    bh = h // 8
    bw = w // 8
    blocks = arr.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    return blocks


def count_tail_flips(signs: np.ndarray) -> np.ndarray:
    flips = np.zeros(signs.shape[0], dtype=np.int32)
    for i in range(signs.shape[0]):
        nz = signs[i][signs[i] != 0]
        if nz.shape[0] <= 1:
            flips[i] = 0
        else:
            flips[i] = int(np.sum(nz[1:] != nz[:-1]))
    return flips


def block_cliff_norm(dc_grid: np.ndarray, q0: float) -> np.ndarray:
    cliff = np.zeros_like(dc_grid, dtype=np.float64)

    lr = np.abs(dc_grid[:, 1:] - dc_grid[:, :-1])
    cliff[:, 1:] = np.maximum(cliff[:, 1:], lr)
    cliff[:, :-1] = np.maximum(cliff[:, :-1], lr)

    up = np.abs(dc_grid[1:, :] - dc_grid[:-1, :])
    cliff[1:, :] = np.maximum(cliff[1:, :], up)

    return cliff / (8.0 * q0 + 1.0)


def dc_gradient_stats(dc_grid: np.ndarray, q0: float) -> Tuple[float, float]:
    grads = []
    lr = np.abs(dc_grid[:, 1:] - dc_grid[:, :-1])
    if lr.size:
        grads.append(lr.reshape(-1))
    up = np.abs(dc_grid[1:, :] - dc_grid[:-1, :])
    if up.size:
        grads.append(up.reshape(-1))
    if not grads:
        return 0.0, 0.0
    g = np.concatenate(grads)
    norm = 8.0 * q0 + 1.0
    return float(np.mean(g) / norm), float(np.percentile(g, 95.0) / norm)


def image_gradient_features(y: np.ndarray) -> Tuple[float, float]:
    gx = np.abs(np.diff(y, axis=1))
    gy = np.abs(np.diff(y, axis=0))
    grad_mean = float(gx.mean() + gy.mean())

    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    if h == 0 or w == 0:
        return grad_mean, 0.0

    gm = np.sqrt(gx[:h, :w] * gx[:h, :w] + gy[:h, :w] * gy[:h, :w])
    grad_p95 = float(np.percentile(gm, 95.0))
    return grad_mean, grad_p95


def compute_features_and_blocks(
    image_path: pathlib.Path,
    qzig: np.ndarray,
    tau_ref: float = 0.18,
    cliff_min: float = 1.0,
    tail_min_flips: int = 2,
    tail_activity_max: int = 3,
) -> Dict[str, object]:
    y = image_to_luma(image_path)
    grad_mean, grad_p95 = image_gradient_features(y)

    y_pad = pad_to_multiple_of_8(y)
    blocks = split_blocks_8x8(y_pad)
    bh, bw = blocks.shape[0], blocks.shape[1]

    dct_blocks = dctn(blocks - 128.0, axes=(-2, -1), norm="ortho")
    coeff_nat = dct_blocks.reshape(-1, 64)
    coeff_zig = coeff_nat[:, JPEG_NATURAL_ORDER]

    abs_zig = np.abs(coeff_zig)
    r16 = np.sum(abs_zig[:, 1:17], axis=1)
    tail = abs_zig[:, 17:33]
    tail_sum = np.sum(tail, axis=1)

    tail_ratio = tail_sum / (r16 + 1.0e-9)

    hf_ratio = tail_sum / (tail_sum + r16 + 1.0e-9)

    signs = np.sign(coeff_zig[:, 17:33]).astype(np.int8)
    tail_flips = count_tail_flips(signs)

    qtail = qzig[17:33]
    tail_activity = np.sum(tail >= (8.0 * qtail[None, :]), axis=1)

    dc_grid = coeff_nat[:, 0].reshape(bh, bw)
    cliff_norm_grid = block_cliff_norm(dc_grid, qzig[0])
    cliff_norm = cliff_norm_grid.reshape(-1)
    dc_grad_mean, dc_grad_p95 = dc_gradient_stats(dc_grid, qzig[0])

    base_cond = (
        (cliff_norm >= cliff_min)
        & (tail_flips >= tail_min_flips)
        & (tail_activity <= tail_activity_max)
    )
    persistent_ref = base_cond & (tail_ratio >= tau_ref)

    features = {
        "grad_mean": grad_mean,
        "grad_p95": grad_p95,
        "dc_grad_mean": dc_grad_mean,
        "dc_grad_p95": dc_grad_p95,
        "hf_ratio_mean": float(np.mean(hf_ratio)),
        "hf_ratio_p95": float(np.percentile(hf_ratio, 95.0)),
        "tail_ratio_mean": float(np.mean(tail_ratio)),
        "tail_ratio_p95": float(np.percentile(tail_ratio, 95.0)),
        "cliff_norm_p95": float(np.percentile(cliff_norm, 95.0)),
        "tail_flips_mean": float(np.mean(tail_flips)),
        "low_activity_frac": float(np.mean(tail_activity <= tail_activity_max)),
        "persistent_frac_tau018": float(np.mean(persistent_ref)),
        "num_blocks": int(coeff_nat.shape[0]),
    }

    return {
        "features": features,
        "block_score_c": tail_ratio,
        "block_base_cond": base_cond,
        "num_blocks": int(coeff_nat.shape[0]),
    }


def model_scores(X: np.ndarray, model: FittedModel) -> np.ndarray:
    Xn = (X - model.scaler_mean[None, :]) / model.scaler_scale[None, :]
    logits = Xn @ model.coef.reshape(-1, 1)
    logits = logits.reshape(-1) + model.intercept
    return 1.0 / (1.0 + np.exp(-logits))


def confusion_at_threshold(y_true: np.ndarray, score: np.ndarray, tau: float) -> Dict[str, float]:
    pred = score >= tau
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def calibrate_threshold(
    y_val: np.ndarray,
    score_val: np.ndarray,
    precision_floor: float,
) -> Optional[ThresholdSelection]:
    thresholds = sorted(set(float(x) for x in score_val), reverse=True)
    best = None
    for tau in thresholds:
        c = confusion_at_threshold(y_val, score_val, tau)
        if c["precision"] < precision_floor:
            continue
        candidate = (
            c["recall"],
            c["precision"],
            -tau,
        )
        if best is None or candidate > best[0]:
            best = (
                candidate,
                ThresholdSelection(
                    tau=tau,
                    precision=c["precision"],
                    recall=c["recall"],
                    tp=c["tp"],
                    fp=c["fp"],
                    fn=c["fn"],
                    tn=c["tn"],
                ),
            )
    return None if best is None else best[1]


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def safe_pr_auc(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, score))


def bootstrap_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    tau: float,
    n_resamples: int,
    seed: int,
) -> Dict[str, Optional[List[float]]]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]

    auroc_vals = []
    pr_vals = []
    prec_vals = []
    attempts = 0
    max_attempts = n_resamples * 30

    while len(prec_vals) < n_resamples and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        sb = score[idx]

        c = confusion_at_threshold(yb, sb, tau)
        prec_vals.append(c["precision"])

        if len(np.unique(yb)) >= 2:
            auroc_vals.append(float(roc_auc_score(yb, sb)))
            pr_vals.append(float(average_precision_score(yb, sb)))

    def ci(vals: List[float]) -> Optional[List[float]]:
        if not vals:
            return None
        return [
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
        ]

    return {
        "auroc_ci95": ci(auroc_vals),
        "pr_auc_ci95": ci(pr_vals),
        "precision_ci95": ci(prec_vals),
        "samples_used": {
            "auroc": len(auroc_vals),
            "pr_auc": len(pr_vals),
            "precision": len(prec_vals),
            "attempts": attempts,
        },
    }


def fit_model(
    records: List[Dict[str, object]],
    split_map: Dict[str, str],
    feature_names: List[str],
    precision_floor: float,
    require_threshold: bool,
) -> Dict[str, object]:
    train_rows = [r for r in records if split_map[r["image"]] == "train"]
    val_rows = [r for r in records if split_map[r["image"]] == "val"]
    test_rows = [r for r in records if split_map[r["image"]] == "test"]

    def build_xy(rows):
        X = np.asarray([[r["features"][f] for f in feature_names] for r in rows], dtype=np.float64)
        y = np.asarray([int(r["safe_label"]) for r in rows], dtype=np.int32)
        images = [r["image"] for r in rows]
        return X, y, images

    X_train, y_train, train_images = build_xy(train_rows)
    X_val, y_val, val_images = build_xy(val_rows)
    X_test, y_test, test_images = build_xy(test_rows)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("training split has a single class; cannot fit logistic regression")
    if len(np.unique(y_val)) < 2:
        raise RuntimeError("validation split has a single class; cannot calibrate tau")

    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n = scaler.transform(X_val)
    X_test_n = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=DEFAULT_SEED,
    )
    clf.fit(X_train_n, y_train)

    score_train = clf.predict_proba(X_train_n)[:, 1]
    score_val = clf.predict_proba(X_val_n)[:, 1]
    score_test = clf.predict_proba(X_test_n)[:, 1]

    sel = calibrate_threshold(y_val, score_val, precision_floor)
    if sel is None:
        if require_threshold:
            raise RuntimeError("no validation threshold satisfies precision floor")
        # For baseline models, allow threshold absence and mark recall/precision as 0.
        sel = ThresholdSelection(
            tau=float("inf"),
            precision=0.0,
            recall=0.0,
            tp=0,
            fp=0,
            fn=int(np.sum(y_val == 1)),
            tn=int(np.sum(y_val == 0)),
        )

    test_conf = confusion_at_threshold(y_test, score_test, sel.tau)

    model = FittedModel(
        feature_names=feature_names,
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        coef=clf.coef_.reshape(-1).copy(),
        intercept=float(clf.intercept_[0]),
        tau=float(sel.tau),
    )

    return {
        "model": model,
        "splits": {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        },
        "scores": {
            "train": {img: float(s) for img, s in zip(train_images, score_train)},
            "val": {img: float(s) for img, s in zip(val_images, score_val)},
            "test": {img: float(s) for img, s in zip(test_images, score_test)},
        },
        "metrics": {
            "auroc_test": safe_auc(y_test, score_test),
            "pr_auc_test": safe_pr_auc(y_test, score_test),
            "precision_test_at_tau": test_conf["precision"],
            "recall_test_at_tau": test_conf["recall"],
            "confusion_test_at_tau": {
                "tp": test_conf["tp"],
                "fp": test_conf["fp"],
                "fn": test_conf["fn"],
                "tn": test_conf["tn"],
            },
            "prevalence_test_safe1": float(np.mean(y_test)) if y_test.size else None,
            "tau_val": float(sel.tau),
            "tau_val_precision": float(sel.precision),
            "tau_val_recall": float(sel.recall),
            "tau_exists_at_precision_floor": bool(math.isfinite(sel.tau)),
        },
        "y_test": y_test,
        "score_test": score_test,
    }


def median_of(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def measure_plain_notrellis(
    bin_path: pathlib.Path,
    image: pathlib.Path,
    quality: int,
    repeats: int,
    artifacts_dir: pathlib.Path,
) -> Dict[str, float]:
    rt_plain = []
    rt_not = []
    sz_plain = []
    sz_not = []

    for r in range(1, repeats + 1):
        out_plain = artifacts_dir / f"{image.stem}__plain__r{r}.jpg"
        out_not = artifacts_dir / f"{image.stem}__notrellis__r{r}.jpg"

        enc_plain = run_encode(
            bin_path=bin_path,
            image_path=image,
            out_path=out_plain,
            quality=quality,
            mode="plain",
            gibbs_threshold=0.18,
            gibbs_cliff_min=1.0,
            gibbs_tail_min_flips=2,
            gibbs_tail_activity_max=3,
        )
        cmd = [
            str(bin_path),
            "-quality",
            str(quality),
            "-notrellis",
            "-outfile",
            str(out_not),
            str(image),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        exists = out_not.exists()
        size = out_not.stat().st_size if exists else -1

        if enc_plain["exit_code"] != 0 or proc.returncode != 0:
            stderr = enc_plain["stderr"] or proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"encode failure for {image}: {stderr}")

        rt_plain.append(float(enc_plain["runtime_ms"]))
        rt_not.append(float(dt_ms))
        sz_plain.append(float(enc_plain["output_bytes"]))
        sz_not.append(float(size))

        if out_plain.exists():
            out_plain.unlink()
        if out_not.exists():
            out_not.unlink()

    plain_rt = median_of(rt_plain)
    not_rt = median_of(rt_not)
    plain_sz = median_of(sz_plain)
    not_sz = median_of(sz_not)
    if plain_rt is None or not_rt is None or plain_sz is None or not_sz is None:
        raise RuntimeError("failed to measure plain/notrellis medians")

    label_delta = delta_pct(not_sz, plain_sz)
    if label_delta is None:
        raise RuntimeError("invalid baseline size for label delta")

    r_trellis = 0.0
    if plain_rt > 0.0:
        r_trellis = max(0.0, (plain_rt - not_rt) / plain_rt)

    return {
        "plain_runtime_ms": plain_rt,
        "notrellis_runtime_ms": not_rt,
        "plain_bytes": plain_sz,
        "notrellis_bytes": not_sz,
        "label_size_delta_pct": label_delta,
        "runtime_trellis_share": r_trellis,
    }


def model_to_json_obj(model: FittedModel) -> Dict[str, object]:
    return {
        "feature_names": model.feature_names,
        "scaler_mean": [float(x) for x in model.scaler_mean],
        "scaler_scale": [float(x) for x in model.scaler_scale],
        "coef": [float(x) for x in model.coef],
        "intercept": float(model.intercept),
        "tau": float(model.tau),
    }


def model_from_json_obj(obj: Dict[str, object]) -> FittedModel:
    return FittedModel(
        feature_names=[str(x) for x in obj["feature_names"]],
        scaler_mean=np.asarray(obj["scaler_mean"], dtype=np.float64),
        scaler_scale=np.asarray(obj["scaler_scale"], dtype=np.float64),
        coef=np.asarray(obj["coef"], dtype=np.float64),
        intercept=float(obj["intercept"]),
        tau=float(obj["tau"]),
    )


def score_image(feature_dict: Dict[str, float], model: FittedModel) -> float:
    x = np.asarray([feature_dict[f] for f in model.feature_names], dtype=np.float64)
    x = (x - model.scaler_mean) / model.scaler_scale
    logit = float(x @ model.coef + model.intercept)
    return 1.0 / (1.0 + math.exp(-logit))


def git_meta(repo_root: pathlib.Path) -> Dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "commit": commit.stdout.decode("utf-8", errors="replace").strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.decode("utf-8", errors="replace").strip()) if dirty.returncode == 0 else None,
    }


def run_stage_b(args) -> pathlib.Path:
    images = read_manifest(args.manifest)
    if not images:
        raise SystemExit(f"manifest has no images: {args.manifest}")

    run_tag = args.tag or time.strftime("protocol_v6_b_%Y%m%d_%H%M%S")
    out_dir = args.results_dir / run_tag
    artifacts_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    qzig = quant_table_for_quality(args.bin, images[0], args.quality)

    split_map = split_images(images, args.seed)

    records = []
    for image in images:
        fb = compute_features_and_blocks(image, qzig)
        meas = measure_plain_notrellis(
            bin_path=args.bin,
            image=image,
            quality=args.quality,
            repeats=args.label_repeats,
            artifacts_dir=artifacts_dir,
        )
        safe_label = int(meas["label_size_delta_pct"] <= args.safe_size_delta_pct)
        records.append({
            "image": str(image),
            "split": split_map[str(image)],
            "features": fb["features"],
            "safe_label": safe_label,
            "label_size_delta_pct": meas["label_size_delta_pct"],
            "plain_runtime_ms": meas["plain_runtime_ms"],
            "notrellis_runtime_ms": meas["notrellis_runtime_ms"],
            "plain_bytes": meas["plain_bytes"],
            "notrellis_bytes": meas["notrellis_bytes"],
            "runtime_trellis_share": meas["runtime_trellis_share"],
        })

    models = {
        "theorem": fit_model(records, split_map, THEOREM_FEATURES, args.precision_floor, require_threshold=True),
        "baseline_grad": fit_model(records, split_map, GRAD_FEATURES, args.precision_floor, require_threshold=False),
        "baseline_hf": fit_model(records, split_map, HF_FEATURES, args.precision_floor, require_threshold=False),
        "baseline_combined": fit_model(records, split_map, COMBINED_FEATURES, args.precision_floor, require_threshold=False),
    }

    theorem = models["theorem"]
    baseline_names = ["baseline_grad", "baseline_hf", "baseline_combined"]

    baseline_aurocs = [
        models[n]["metrics"]["auroc_test"] for n in baseline_names
        if models[n]["metrics"]["auroc_test"] is not None
    ]
    baseline_prs = [
        models[n]["metrics"]["pr_auc_test"] for n in baseline_names
        if models[n]["metrics"]["pr_auc_test"] is not None
    ]
    best_baseline_auroc = max(baseline_aurocs) if baseline_aurocs else None
    best_baseline_pr = max(baseline_prs) if baseline_prs else None
    best_baseline_recall = max(models[n]["metrics"]["recall_test_at_tau"] for n in baseline_names)

    theorem_metrics = theorem["metrics"]
    theorem_auroc = theorem_metrics["auroc_test"]
    theorem_pr = theorem_metrics["pr_auc_test"]
    theorem_precision = theorem_metrics["precision_test_at_tau"]
    theorem_recall = theorem_metrics["recall_test_at_tau"]

    auroc_lift = None
    if theorem_auroc is not None and best_baseline_auroc is not None:
        auroc_lift = float(theorem_auroc - best_baseline_auroc)
    pr_lift = None
    if theorem_pr is not None and best_baseline_pr is not None:
        pr_lift = float(theorem_pr - best_baseline_pr)
    recall_margin = float(theorem_recall - best_baseline_recall)

    ci = bootstrap_ci(
        y_true=theorem["y_test"],
        score=theorem["score_test"],
        tau=theorem["model"].tau,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    acceptance = {
        "auroc_ge_0_70": bool(theorem_auroc is not None and theorem_auroc >= 0.70),
        "auroc_lift_ge_0_05": bool(auroc_lift is not None and auroc_lift >= 0.05),
        "pr_auc_lift_ge_0_03": bool(pr_lift is not None and pr_lift >= 0.03),
        "precision_ge_0_95": bool(theorem_precision >= args.precision_floor),
        "recall_noninferior_margin_0_005": bool(recall_margin >= -args.recall_noninferiority_margin),
    }
    acceptance["all_pass"] = all(acceptance.values())

    split_counts = {
        "train": sum(1 for r in records if r["split"] == "train"),
        "val": sum(1 for r in records if r["split"] == "val"),
        "test": sum(1 for r in records if r["split"] == "test"),
    }

    model_payload = {
        "schema": "mozjpeg-gibbs-protocol-v6-b-model-v1",
        "seed": args.seed,
        "precision_floor": args.precision_floor,
        "safe_size_delta_pct": args.safe_size_delta_pct,
        "recall_noninferiority_margin": args.recall_noninferiority_margin,
        "model_class": {
            "name": "LogisticRegression",
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": None,
            "random_state": args.seed,
            "normalization": "train_split_standard_scaler",
            "score": "P(safe=1)",
        },
        "feature_sets": {
            "theorem": THEOREM_FEATURES,
            "baseline_grad": GRAD_FEATURES,
            "baseline_hf": HF_FEATURES,
            "baseline_combined": COMBINED_FEATURES,
        },
        "models": {name: model_to_json_obj(models[name]["model"]) for name in models},
        "split_assignments": split_map,
        "records": records,
    }

    summary = {
        "schema": "mozjpeg-gibbs-protocol-v6-b-summary-v1",
        "tag": run_tag,
        "manifest": str(args.manifest),
        "quality": args.quality,
        "seed": args.seed,
        "split_counts": split_counts,
        "bootstrap": {
            "n_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "positive_class": "safe=1",
        "model_class": model_payload["model_class"],
        "metrics": {
            "theorem": theorem_metrics,
            "baseline_grad": models["baseline_grad"]["metrics"],
            "baseline_hf": models["baseline_hf"]["metrics"],
            "baseline_combined": models["baseline_combined"]["metrics"],
            "best_baseline_auroc": best_baseline_auroc,
            "best_baseline_pr_auc": best_baseline_pr,
            "best_baseline_recall_at_precision_floor": best_baseline_recall,
            "theorem_auroc_lift": auroc_lift,
            "theorem_pr_auc_lift": pr_lift,
            "theorem_recall_margin_vs_best_baseline": recall_margin,
            "theorem_ci95": ci,
        },
        "acceptance": acceptance,
        "fixed_policies": {
            "delta_pct_formula": "100*(candidate-baseline)/baseline",
            "safe_label": f"all_trellis_off_size_delta_pct <= {args.safe_size_delta_pct}",
            "precision_floor": args.precision_floor,
            "noninferiority_margin": args.recall_noninferiority_margin,
            "noninferiority_margin_is_a_priori_fixed": True,
            "tau_b_frozen": float(theorem["model"].tau),
            "tau_b_score_direction": "higher score means more likely safe",
            "reproducibility_tolerances": {
                "exact_match_fields": [
                    "split_assignments",
                    "tau_b",
                    "tau_c",
                    "confusion_counts",
                ],
                "float_abs_tolerance_same_env": 1.0e-12,
                "float_abs_tolerance_cross_machine": 1.0e-8,
            },
        },
        "git": git_meta(pathlib.Path(__file__).resolve().parents[3]),
    }

    rows_csv = out_dir / "records.csv"
    write_rows_csv(
        rows_csv,
        [
            {
                "image": r["image"],
                "split": r["split"],
                "safe_label": r["safe_label"],
                "label_size_delta_pct": r["label_size_delta_pct"],
                "plain_runtime_ms": r["plain_runtime_ms"],
                "notrellis_runtime_ms": r["notrellis_runtime_ms"],
                "plain_bytes": r["plain_bytes"],
                "notrellis_bytes": r["notrellis_bytes"],
                "runtime_trellis_share": r["runtime_trellis_share"],
                **{f: r["features"][f] for f in THEOREM_FEATURES},
            }
            for r in records
        ],
        [
            "image",
            "split",
            "safe_label",
            "label_size_delta_pct",
            "plain_runtime_ms",
            "notrellis_runtime_ms",
            "plain_bytes",
            "notrellis_bytes",
            "runtime_trellis_share",
            *THEOREM_FEATURES,
        ],
    )

    model_path = out_dir / "model.json"
    summary_path = out_dir / "summary.json"
    dump_json(model_path, model_payload)
    dump_json(summary_path, summary)

    print(f"B records: {rows_csv}")
    print(f"B model: {model_path}")
    print(f"B summary: {summary_path}")
    print(f"B acceptance_all_pass: {acceptance['all_pass']}")

    return summary_path


def compare_off_vs_policy(rows: List[Dict[str, object]]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    converted = []
    for r in rows:
        rc = dict(r)
        if rc["variant"] == "policy":
            rc["variant"] = "guarded"
        converted.append(rc)
    return compare_off_vs_guarded(converted)


def runtime_trellis_share(
    bin_path: pathlib.Path,
    images: Sequence[pathlib.Path],
    quality: int,
    artifacts_dir: pathlib.Path,
) -> float:
    shares = []
    for image in images:
        m = measure_plain_notrellis(
            bin_path=bin_path,
            image=image,
            quality=quality,
            repeats=1,
            artifacts_dir=artifacts_dir,
        )
        shares.append(m["runtime_trellis_share"])
    return float(np.mean(shares)) if shares else 0.0


def tau_c_grid(min_v: float, max_v: float, step: float) -> List[float]:
    vals = []
    x = min_v
    while x <= max_v + 1.0e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def select_tau_c(
    bin_path: pathlib.Path,
    eligible_images: List[pathlib.Path],
    quality: int,
    qzig: np.ndarray,
    tau_candidates: List[float],
    artifacts_dir: pathlib.Path,
) -> Dict[str, object]:
    if not eligible_images:
        return {
            "selected_tau_c": None,
            "candidates": [],
            "reason": "no_eligible_images",
        }

    off_sizes = {}
    block_stats = {}

    for image in eligible_images:
        fb = compute_features_and_blocks(image, qzig)
        block_stats[str(image)] = fb
        out_off = artifacts_dir / f"val__{image.stem}__off.jpg"
        enc_off = run_encode(
            bin_path=bin_path,
            image_path=image,
            out_path=out_off,
            quality=quality,
            mode="off",
            gibbs_threshold=0.18,
            gibbs_cliff_min=1.0,
            gibbs_tail_min_flips=2,
            gibbs_tail_activity_max=3,
        )
        if enc_off["exit_code"] != 0:
            raise RuntimeError(f"off encode failed for tau_C calibration: {image}")
        off_sizes[str(image)] = float(enc_off["output_bytes"])
        if out_off.exists():
            out_off.unlink()

    candidate_rows = []
    selected_tau = None
    selected_s_block = None

    for tau in tau_candidates:
        size_deltas = []
        skipped_blocks = 0
        total_blocks = 0

        for image in eligible_images:
            out_guarded = artifacts_dir / f"val__{image.stem}__guarded_tau{tau:.4f}.jpg"
            enc = run_encode(
                bin_path=bin_path,
                image_path=image,
                out_path=out_guarded,
                quality=quality,
                mode="guarded",
                gibbs_threshold=float(tau),
                gibbs_cliff_min=1.0,
                gibbs_tail_min_flips=2,
                gibbs_tail_activity_max=3,
            )
            if enc["exit_code"] != 0:
                raise RuntimeError(f"guarded encode failed for tau_C calibration: {image}")
            d = delta_pct(float(enc["output_bytes"]), off_sizes[str(image)])
            if d is not None:
                size_deltas.append(float(d))

            bs = block_stats[str(image)]
            cond = bs["block_base_cond"] & (bs["block_score_c"] >= tau)
            skipped_blocks += int(np.sum(cond))
            total_blocks += int(bs["num_blocks"])

            if out_guarded.exists():
                out_guarded.unlink()

        mean_delta = float(np.mean(size_deltas)) if size_deltas else None
        p95_delta = float(np.percentile(size_deltas, 95.0)) if size_deltas else None
        s_block = (skipped_blocks / total_blocks) if total_blocks > 0 else 0.0

        row = {
            "tau_c": float(tau),
            "eligible_images": len(eligible_images),
            "size_delta_mean_pct": mean_delta,
            "size_delta_p95_pct": p95_delta,
            "s_block": float(s_block),
            "passes_size_constraints": bool(
                mean_delta is not None
                and p95_delta is not None
                and mean_delta <= 0.20
                and p95_delta <= 0.50
            ),
        }
        candidate_rows.append(row)

        # Lower tau means more aggressive because score >= tau gates skipping.
        if selected_tau is None and row["passes_size_constraints"]:
            selected_tau = float(tau)
            selected_s_block = float(s_block)

    return {
        "selected_tau_c": selected_tau,
        "selected_s_block": selected_s_block,
        "candidates": candidate_rows,
        "reason": None if selected_tau is not None else "no_tau_c_met_size_constraints",
    }


def run_stage_c(args) -> pathlib.Path:
    b_summary = json.loads(args.b_summary.read_text(encoding="utf-8"))
    model_path = args.b_summary.parent / "model.json"
    if not model_path.exists():
        raise SystemExit(f"missing model payload next to summary: {model_path}")
    b_model_payload = json.loads(model_path.read_text(encoding="utf-8"))

    theorem_model = model_from_json_obj(b_model_payload["models"]["theorem"])
    tau_b = float(theorem_model.tau)

    split_assignments = b_model_payload["split_assignments"]
    val_images = [pathlib.Path(p) for p, s in split_assignments.items() if s == "val"]
    if not val_images:
        raise SystemExit("no validation images in B split assignments")

    benchmark_images = read_manifest(args.benchmark_manifest)
    if not benchmark_images:
        raise SystemExit(f"benchmark manifest has no images: {args.benchmark_manifest}")

    run_tag = args.tag or time.strftime("protocol_v6_c_%Y%m%d_%H%M%S")
    out_dir = args.results_dir / run_tag
    artifacts_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    qzig = quant_table_for_quality(args.bin, benchmark_images[0], args.quality)

    val_features = {}
    eligible_val_images = []
    for image in val_images:
        fb = compute_features_and_blocks(image, qzig)
        score_b = score_image(fb["features"], theorem_model)
        eligible = score_b >= tau_b
        val_features[str(image)] = {
            "score_b": float(score_b),
            "eligible": bool(eligible),
            "num_blocks": int(fb["num_blocks"]),
            "features": fb["features"],
            "block_score_c": fb["block_score_c"],
            "block_base_cond": fb["block_base_cond"],
        }
        if eligible:
            eligible_val_images.append(image)

    f_eligible = len(eligible_val_images) / len(val_images)

    tau_candidates = tau_c_grid(args.tau_c_min, args.tau_c_max, args.tau_c_step)
    tau_c_cal = select_tau_c(
        bin_path=args.bin,
        eligible_images=eligible_val_images,
        quality=args.quality,
        qzig=qzig,
        tau_candidates=tau_candidates,
        artifacts_dir=artifacts_dir,
    )

    tau_c = tau_c_cal["selected_tau_c"]
    s_block = tau_c_cal["selected_s_block"] if tau_c is not None else 0.0

    r_trellis = runtime_trellis_share(
        bin_path=args.bin,
        images=benchmark_images,
        quality=args.quality,
        artifacts_dir=artifacts_dir,
    )

    projected_max_runtime_gain_pct = 100.0 * r_trellis * f_eligible * (s_block or 0.0)
    feasible = projected_max_runtime_gain_pct >= 3.0 and tau_c is not None

    rows = []
    decode_failures = []
    policy_image_map = {}

    if feasible:
        for image in benchmark_images:
            fb = compute_features_and_blocks(image, qzig)
            score_b = score_image(fb["features"], theorem_model)
            eligible = score_b >= tau_b
            policy_mode = "guarded" if eligible else "off"
            policy_image_map[str(image)] = {
                "score_b": float(score_b),
                "eligible": bool(eligible),
                "policy_mode": policy_mode,
            }

        for repeat in range(1, args.repeats + 1):
            for image in benchmark_images:
                key = image.stem

                out_off = artifacts_dir / f"bench__{key}__off__r{repeat}.jpg"
                out_pol = artifacts_dir / f"bench__{key}__policy__r{repeat}.jpg"

                enc_off = run_encode(
                    bin_path=args.bin,
                    image_path=image,
                    out_path=out_off,
                    quality=args.quality,
                    mode="off",
                    gibbs_threshold=0.18,
                    gibbs_cliff_min=1.0,
                    gibbs_tail_min_flips=2,
                    gibbs_tail_activity_max=3,
                )
                d_off = {"ok": False, "exit_code": -1, "stderr": ""}
                if enc_off["output_exists"] and enc_off["exit_code"] == 0:
                    d_off = validate_decode(args.djpeg, out_off)

                policy_mode = policy_image_map[str(image)]["policy_mode"]
                enc_pol = run_encode(
                    bin_path=args.bin,
                    image_path=image,
                    out_path=out_pol,
                    quality=args.quality,
                    mode=policy_mode,
                    gibbs_threshold=float(tau_c),
                    gibbs_cliff_min=1.0,
                    gibbs_tail_min_flips=2,
                    gibbs_tail_activity_max=3,
                )
                d_pol = {"ok": False, "exit_code": -1, "stderr": ""}
                if enc_pol["output_exists"] and enc_pol["exit_code"] == 0:
                    d_pol = validate_decode(args.djpeg, out_pol)

                rows.append({
                    "image": str(image),
                    "image_key": key,
                    "repeat": repeat,
                    "variant": "off",
                    "quality": args.quality,
                    "runtime_ms": enc_off["runtime_ms"],
                    "output_bytes": enc_off["output_bytes"],
                    "sha256": enc_off["sha256"],
                    "encode_exit_code": enc_off["exit_code"],
                    "encode_stderr": enc_off["stderr"],
                    "decode_ok": int(d_off["ok"]),
                    "decode_exit_code": d_off["exit_code"],
                    "decode_stderr": d_off["stderr"],
                    "output_path": str(out_off),
                })
                rows.append({
                    "image": str(image),
                    "image_key": key,
                    "repeat": repeat,
                    "variant": "policy",
                    "quality": args.quality,
                    "runtime_ms": enc_pol["runtime_ms"],
                    "output_bytes": enc_pol["output_bytes"],
                    "sha256": enc_pol["sha256"],
                    "encode_exit_code": enc_pol["exit_code"],
                    "encode_stderr": enc_pol["stderr"],
                    "decode_ok": int(d_pol["ok"]),
                    "decode_exit_code": d_pol["exit_code"],
                    "decode_stderr": d_pol["stderr"],
                    "output_path": str(out_pol),
                    "policy_mode": policy_mode,
                    "score_b": policy_image_map[str(image)]["score_b"],
                    "eligible": int(policy_image_map[str(image)]["eligible"]),
                })

                if enc_off["exit_code"] != 0 or not d_off["ok"]:
                    decode_failures.append({
                        "image": str(image),
                        "repeat": repeat,
                        "variant": "off",
                        "encode_exit_code": enc_off["exit_code"],
                        "encode_stderr": enc_off["stderr"],
                        "decode_exit_code": d_off["exit_code"],
                        "decode_stderr": d_off["stderr"],
                    })
                if enc_pol["exit_code"] != 0 or not d_pol["ok"]:
                    decode_failures.append({
                        "image": str(image),
                        "repeat": repeat,
                        "variant": "policy",
                        "encode_exit_code": enc_pol["exit_code"],
                        "encode_stderr": enc_pol["stderr"],
                        "decode_exit_code": d_pol["exit_code"],
                        "decode_stderr": d_pol["stderr"],
                    })

                if not args.keep_artifacts:
                    if out_off.exists():
                        out_off.unlink()
                    if out_pol.exists():
                        out_pol.unlink()

    metrics = {}
    per_image = []
    acceptance = {
        "runtime_median_improvement_ge_3pct": False,
        "runtime_p95_delta_le_1pct": False,
        "size_median_delta_le_0pct_hard": False,
        "size_mean_delta_le_0_20pct": False,
        "size_p95_delta_le_0_50pct": False,
        "decode_failures_zero": False,
        "all_pass": False,
    }

    if feasible:
        metrics, per_image = compare_off_vs_policy(rows)
        acceptance = evaluate_acceptance(metrics, decode_failures=len(decode_failures))
        size_med = metrics.get("size_delta_median_pct")
        acceptance["size_median_delta_le_0pct_hard"] = (
            size_med is not None and size_med <= 0.0
        )
        acceptance["all_pass"] = (
            acceptance["all_pass"] and acceptance["size_median_delta_le_0pct_hard"]
        )

    write_rows_csv(
        out_dir / "rows.csv",
        rows,
        [
            "image", "image_key", "repeat", "variant", "quality",
            "runtime_ms", "output_bytes", "sha256",
            "encode_exit_code", "encode_stderr",
            "decode_ok", "decode_exit_code", "decode_stderr",
            "output_path", "policy_mode", "score_b", "eligible",
        ],
    )

    summary = {
        "schema": "mozjpeg-gibbs-protocol-v6-c-summary-v1",
        "tag": run_tag,
        "benchmark_manifest": str(args.benchmark_manifest),
        "quality": args.quality,
        "repeats": args.repeats,
        "tau_b": float(tau_b),
        "tau_c": float(tau_c) if tau_c is not None else None,
        "score_direction": {
            "score_b": "higher means safer image",
            "score_c": "higher means more eligible block for skip",
        },
        "fixed_policy": {
            "routing": "score_b < tau_b => off; score_b >= tau_b and score_c >= tau_c => guarded skip",
            "single_seam": "huffman_only",
            "delta_pct_formula": "100*(candidate-baseline)/baseline",
            "reproducibility_tolerances": {
                "exact_match_fields": [
                    "split_assignments",
                    "tau_b",
                    "tau_c",
                    "confusion_counts",
                ],
                "float_abs_tolerance_same_env": 1.0e-12,
                "float_abs_tolerance_cross_machine": 1.0e-8,
            },
        },
        "feasibility": {
            "f_eligible": f_eligible,
            "s_block": s_block,
            "r_trellis": r_trellis,
            "projected_max_runtime_gain_pct": projected_max_runtime_gain_pct,
            "is_upper_bound_not_forecast": True,
            "threshold_runtime_gain_pct": 3.0,
            "feasible": feasible,
            "tau_c_reason": tau_c_cal["reason"],
        },
        "tau_c_calibration": {
            "grid": {
                "min": args.tau_c_min,
                "max": args.tau_c_max,
                "step": args.tau_c_step,
            },
            "eligible_validation_images": [str(p) for p in eligible_val_images],
            "selected_tau_c": tau_c,
            "selected_s_block": s_block,
            "candidates": tau_c_cal["candidates"],
            "size_constraints": {
                "mean_delta_pct_le": 0.20,
                "p95_delta_pct_le": 0.50,
            },
        },
        "metrics": metrics,
        "acceptance": acceptance,
        "decode_failures": len(decode_failures),
        "decode_failure_details": decode_failures,
        "per_image": per_image,
        "policy_image_map": policy_image_map,
        "git": git_meta(pathlib.Path(__file__).resolve().parents[3]),
    }

    dump_json(out_dir / "summary.json", summary)

    print(f"C rows: {out_dir / 'rows.csv'}")
    print(f"C summary: {out_dir / 'summary.json'}")
    print(f"feasible: {feasible}")
    print(f"acceptance_all_pass: {acceptance['all_pass']}")

    return out_dir / "summary.json"


def run_all(args):
    b_args = argparse.Namespace(
        bin=args.bin,
        manifest=args.b_manifest,
        quality=args.quality,
        results_dir=args.results_dir,
        tag=args.b_tag,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        label_repeats=args.label_repeats,
        safe_size_delta_pct=args.safe_size_delta_pct,
        precision_floor=args.precision_floor,
        recall_noninferiority_margin=args.recall_noninferiority_margin,
    )
    b_summary_path = run_stage_b(b_args)

    c_args = argparse.Namespace(
        bin=args.bin,
        djpeg=args.djpeg,
        b_summary=b_summary_path,
        benchmark_manifest=args.c_manifest,
        quality=args.quality,
        repeats=args.repeats,
        results_dir=args.results_dir,
        tag=args.c_tag,
        tau_c_min=args.tau_c_min,
        tau_c_max=args.tau_c_max,
        tau_c_step=args.tau_c_step,
        keep_artifacts=args.keep_artifacts,
    )
    run_stage_c(c_args)


def add_common_policy_args(ap: argparse.ArgumentParser):
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)


def main():
    parser = argparse.ArgumentParser(description="MozJPEG Gibbs Restart Protocol v6 runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("stage-b", help="Run protocol v6 Change B")
    b.add_argument("--bin", type=pathlib.Path, required=True)
    b.add_argument("--manifest", type=pathlib.Path, required=True)
    b.add_argument("--results-dir", type=pathlib.Path, required=True)
    b.add_argument("--tag", default="")
    b.add_argument("--quality", type=int, default=85)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    b.add_argument("--label-repeats", type=int, default=1)
    b.add_argument("--safe-size-delta-pct", type=float, default=DEFAULT_SAFE_SIZE_DELTA_PCT)
    b.add_argument("--precision-floor", type=float, default=DEFAULT_PRECISION_FLOOR)
    b.add_argument("--recall-noninferiority-margin", type=float, default=DEFAULT_RECALL_NONINFERIORITY_MARGIN)

    c = sub.add_parser("stage-c", help="Run protocol v6 Change C")
    c.add_argument("--bin", type=pathlib.Path, required=True)
    c.add_argument("--djpeg", type=pathlib.Path, required=True)
    c.add_argument("--b-summary", type=pathlib.Path, required=True)
    c.add_argument("--benchmark-manifest", type=pathlib.Path, required=True)
    c.add_argument("--results-dir", type=pathlib.Path, required=True)
    c.add_argument("--tag", default="")
    c.add_argument("--quality", type=int, default=85)
    c.add_argument("--repeats", type=int, default=7)
    c.add_argument("--tau-c-min", type=float, default=0.02)
    c.add_argument("--tau-c-max", type=float, default=0.60)
    c.add_argument("--tau-c-step", type=float, default=0.01)
    c.add_argument("--keep-artifacts", action="store_true")

    allp = sub.add_parser("run-all", help="Run B then C with fixed protocol settings")
    allp.add_argument("--bin", type=pathlib.Path, required=True)
    allp.add_argument("--djpeg", type=pathlib.Path, required=True)
    allp.add_argument("--b-manifest", type=pathlib.Path, required=True)
    allp.add_argument("--c-manifest", type=pathlib.Path, required=True)
    allp.add_argument("--results-dir", type=pathlib.Path, required=True)
    allp.add_argument("--quality", type=int, default=85)
    allp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    allp.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    allp.add_argument("--label-repeats", type=int, default=1)
    allp.add_argument("--safe-size-delta-pct", type=float, default=DEFAULT_SAFE_SIZE_DELTA_PCT)
    allp.add_argument("--precision-floor", type=float, default=DEFAULT_PRECISION_FLOOR)
    allp.add_argument("--recall-noninferiority-margin", type=float, default=DEFAULT_RECALL_NONINFERIORITY_MARGIN)
    allp.add_argument("--repeats", type=int, default=7)
    allp.add_argument("--tau-c-min", type=float, default=0.02)
    allp.add_argument("--tau-c-max", type=float, default=0.60)
    allp.add_argument("--tau-c-step", type=float, default=0.01)
    allp.add_argument("--b-tag", default="")
    allp.add_argument("--c-tag", default="")
    allp.add_argument("--keep-artifacts", action="store_true")

    args = parser.parse_args()

    if args.cmd == "stage-b":
        run_stage_b(args)
    elif args.cmd == "stage-c":
        run_stage_c(args)
    elif args.cmd == "run-all":
        run_all(args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
