#!/usr/bin/env python3
import argparse
import pathlib
import random

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def collect_images(root: pathlib.Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            files.append(p.resolve())
    files.sort()
    return files


def write_manifest(paths, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Create deterministic dataset manifests for Gibbs A/B benchmarks")
    ap.add_argument("--clic-train-dir", type=pathlib.Path, required=True)
    ap.add_argument("--kodak-dir", type=pathlib.Path, required=True)
    ap.add_argument("--tecnick-dir", type=pathlib.Path, required=True)
    ap.add_argument("--stress-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--dev-count", type=int, default=300)
    args = ap.parse_args()

    clic = collect_images(args.clic_train_dir)
    if len(clic) < args.dev_count:
        raise SystemExit(f"need at least {args.dev_count} images in CLIC train dir, found {len(clic)}")

    rng = random.Random(args.seed)
    dev = sorted(rng.sample(clic, args.dev_count))

    kodak = collect_images(args.kodak_dir)
    tecnick = collect_images(args.tecnick_dir)
    stress = collect_images(args.stress_dir)

    test = sorted(kodak + tecnick)

    write_manifest(dev, args.out_dir / "dev.txt")
    write_manifest(test, args.out_dir / "test.txt")
    write_manifest(stress, args.out_dir / "stress.txt")

    summary = args.out_dir / "dataset_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"dev={len(dev)}\n")
        f.write(f"test={len(test)}\n")
        f.write(f"stress={len(stress)}\n")
        f.write(f"kodak={len(kodak)}\n")
        f.write(f"tecnick={len(tecnick)}\n")

    print(f"Wrote {args.out_dir / 'dev.txt'} ({len(dev)} images)")
    print(f"Wrote {args.out_dir / 'test.txt'} ({len(test)} images)")
    print(f"Wrote {args.out_dir / 'stress.txt'} ({len(stress)} images)")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
