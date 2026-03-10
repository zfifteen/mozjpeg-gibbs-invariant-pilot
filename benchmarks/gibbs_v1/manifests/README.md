# Dataset Manifests

Expected files:

- `dev.txt` (300 sampled images from CLIC train)
- `test.txt` (Kodak24 + Tecnick100)
- `stress.txt` (200 deterministic synthetic line-art/text UI images)

Each file stores one absolute image path per line.

Generate with:

```bash
python3 benchmarks/gibbs_v1/scripts/make_dataset_manifest.py ...
```
