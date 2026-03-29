# Parallel Image Transforms for Gigapixel Images on Heterogeneous Compute

**CSE461 — Parallel & Distributed Computing | Group 5 | Milestone 1**

---

## Project Description

This project implements a high-performance, out-of-core image processing pipeline capable of transforming gigapixel TIFF images — images too large to fit in RAM — using parallel and heterogeneous computing techniques.

The core idea is that the image is never loaded into memory all at once. Instead, it is divided into small rectangular **tiles**, each tile is loaded independently with a surrounding **halo** (overlap border) for seamless filtering, processed by a pool of worker threads, and streamed back to disk. Peak RAM usage is bounded by the number of in-flight tiles multiplied by tile size — not by image dimensions. This makes the system capable of processing arbitrarily large images on commodity hardware.

Milestone 1 delivers the complete **CPU-only foundation**: a tiled reader, overlap/halo management, a composable image transform chain (identity, crop, resize, rotate, box blur), a producer-consumer thread pool, and a streaming output writer. A sequential single-thread mode is included for direct speedup comparison against the parallel pipeline.

---

## Group Members

| Name | ERP |
|---|---|
| Gehna Bhatia | 29054 |
| Muhammad Zayed Nauman | 29047 |
| Zainab Irfan | 29091 |
| Muhammad Anis Imran | 29017 |

---


## Generating Test Images

The `tools/gen_test_image.py` script generates synthetic checkerboard TIFF images of any size for testing.

```bash
cd Project

# Small image — fast correctness check (12 MB)
python3 tools/gen_test_image.py 2048 2048

# Medium image — visible speedup (~192 MB)
python3 tools/gen_test_image.py 8192 8192

# Gigapixel image — full speedup benchmark (~2.9 GB, takes 3-5 min to generate)
python3 tools/gen_test_image.py 32000 32000
```

> **Disk space:** The 32000×32000 image requires approximately more than 3 GB of free disk space plus 3 GB for the output file. Ensure at least 6 GB is available before generating.

---

## Running the Pipeline

### Basic usage (parallel mode, identity pass)

```bash
./build/milestone1 input.tiff output.tiff
```

### Available modes

| Flag | Description |
|---|---|
| `--mode parallel` | Run parallel pipeline only (default) |
| `--mode sequential` | Run single-threaded pipeline only |
| `--mode both` | Run both and print a speedup comparison table |

### Available transforms

| Transform | Command | Description |
|---|---|---|
| Identity | `--transform identity` | Copy pixels unchanged (correctness baseline) |
| Box blur | `--transform blur 8` | Separable box blur, radius 8 |
| Crop | `--transform crop 0 0 1024 1024` | Crop to rectangle (x0 y0 x1 y1) |
| Resize | `--transform resize 0.5 0.5` | Scale image by sx sy |
| Rotate | `--transform rotate 30` | Rotate counter-clockwise by degrees |

Transforms can be chained in any order:

```bash
./build/milestone1 --transform blur 4 --transform crop 0 0 1024 1024 input.tiff output.tiff
```

### Additional options

| Option | Default | Description |
|---|---|---|
| `--tile-size N` | 512 | Processing tile size in pixels |
| `--threads N` | hardware concurrency | Number of worker threads |
| `--halo N` | auto | Override halo width |
| `--border mode` | clamp | Border fill: `zero`, `clamp`, `reflect` |
| `--in-flight N` | 16 | Max tiles held in RAM at once |

---

## Quickstart

Run these commands from scratch on a clean machine. Each step must complete before the next, run the equivalent commands for Windows.

### Step 1 — Clone and build

```bash
git clone <https://github.com/zayed-nauman/Parallel-Image-Transforms-for-Gigapixel-Images>
cd Project
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu)   # macOS
cd ..
```

### Step 2 — Generate a small test image

```bash
python3 tools/gen_test_image.py 2048 2048
```

### Step 3 — Run the identity pass and verify correctness

```bash
./build/milestone1 test_2048x2048_rgb.tiff out.tiff
```

```bash
python3 -c "
import numpy as np, tifffile as tf
a = tf.imread('test_2048x2048_rgb.tiff')
b = tf.imread('out.tiff')
print('PASS' if np.array_equal(a,b) else 'FAIL')
"
```

Expected output: `PASS`

### Step 4 — Try individual transforms

```bash
# Box blur radius 8
./build/milestone1 --transform blur 8 test_2048x2048_rgb.tiff out_blur.tiff

# Crop
./build/milestone1 --transform crop 200 200 1800 1800 test_2048x2048_rgb.tiff out_crop.tiff

# Resize to 50%
./build/milestone1 --transform resize 0.5 0.5 test_2048x2048_rgb.tiff out_small.tiff

# Rotate 30 degrees
./build/milestone1 --transform rotate 30 test_2048x2048_rgb.tiff out_rotated.tiff
```

### Step 5 — Generate the gigapixel image

> Make sure you have at least 6 GB of free disk space before running this step.

```bash
df -h /                              # Check available space
python3 tools/gen_test_image.py 32000 32000   # Takes 3-5 minutes
```

### Step 6 — Run the speedup benchmark

```bash
./build/milestone1 --mode both test_32000x32000_rgb.tiff out_32k.tiff
```
