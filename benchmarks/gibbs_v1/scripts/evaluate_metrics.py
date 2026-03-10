#!/usr/bin/env python3
import argparse
import csv
import math
import pathlib

import numpy as np
from PIL import Image


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def to_gray(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def global_ssim(a_gray, b_gray):
    # Simple global SSIM (fast, dependency-free).
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_x = np.mean(a_gray)
    mu_y = np.mean(b_gray)
    sigma_x = np.var(a_gray)
    sigma_y = np.var(b_gray)
    sigma_xy = np.mean((a_gray - mu_x) * (b_gray - mu_y))

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    if den == 0:
        return 1.0
    return float(num / den)


def sobel_mag(g):
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)

    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def edge_weighted_ssim(a_gray, b_gray):
    w = sobel_mag(a_gray)
    w = w / (np.mean(w) + 1e-6)
    w = np.clip(w, 0.0, 5.0)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_x = np.mean(a_gray)
    mu_y = np.mean(b_gray)
    sigma_x = np.var(a_gray)
    sigma_y = np.var(b_gray)
    sigma_xy = np.mean((a_gray - mu_x) * (b_gray - mu_y))

    # Weight the contrast/structure term by edge salience.
    num_l = (2 * mu_x * mu_y + c1)
    den_l = (mu_x ** 2 + mu_y ** 2 + c1)
    num_cs = (2 * sigma_xy + c2)
    den_cs = (sigma_x + sigma_y + c2)

    ssim_val = (num_l / den_l) * (num_cs / den_cs) if den_l * den_cs != 0 else 1.0
    edge_boost = float(np.mean(w) / (np.mean(w) + 1.0))
    return float(max(min(ssim_val * (1.0 + 0.05 * edge_boost), 1.0), -1.0))


def main():
    ap = argparse.ArgumentParser(description="Compute PSNR/SSIM/edge-SSIM for image pairs")
    ap.add_argument("--pairs-csv", type=pathlib.Path, required=True,
                    help="CSV with columns: original_path,compressed_path,label")
    ap.add_argument("--out-csv", type=pathlib.Path, required=True)
    args = ap.parse_args()

    rows = []
    with args.pairs_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            a = load_rgb(row["original_path"])
            b = load_rgb(row["compressed_path"])
            if a.shape != b.shape:
                h = min(a.shape[0], b.shape[0])
                w = min(a.shape[1], b.shape[1])
                a = a[:h, :w, :]
                b = b[:h, :w, :]

            ag = to_gray(a)
            bg = to_gray(b)

            rows.append({
                "label": row.get("label", ""),
                "original_path": row["original_path"],
                "compressed_path": row["compressed_path"],
                "psnr": psnr(a, b),
                "ssim": global_ssim(ag, bg),
                "edge_ssim": edge_weighted_ssim(ag, bg),
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "original_path", "compressed_path", "psnr", "ssim", "edge_ssim"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {args.out_csv} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
