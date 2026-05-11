# Parallel Image Transforms for Gigapixel Images on Heterogeneous Compute

**CSE461 — Parallel & Distributed Computing | Group 5**

---

## Project Description

This project implements a high-performance, out-of-core image processing pipeline capable of transforming gigapixel TIFF images — images too large to fit in RAM — using parallel and heterogeneous computing techniques.

The core idea is that the image is never loaded into memory all at once. Instead, it is divided into small rectangular **tiles**, each tile is loaded independently with a surrounding **halo** (overlap border) for seamless filtering, processed by a pool of worker threads, and streamed back to disk. Peak RAM usage is bounded by the number of in-flight tiles multiplied by tile size — not by image dimensions. This makes the system capable of processing arbitrarily large images on commodity hardware.

- **Milestone 1** (`milestone1`) — CPU-only foundation: tiled reader/writer, overlap/halo management, composable transform chain, producer-consumer thread pool, and streaming output writer. The binary is built from the full shared source set (including M2 GPU/scheduler files compiled in) so all modes work on it.
- **Milestone 2** (`milestone2`) — same source set as M1 plus the heterogeneous CPU+GPU pipeline. Tiles are dynamically routed to CPU workers or GPU CUDA streams by the work scheduler. On machines without CUDA (e.g. Apple Silicon), `gpu_kernels_stub.cpp` satisfies the linker so the binary builds and runs identically — just without real GPU acceleration.
- **Milestone 3** (`milestone3`) — everything in M2, plus optional files compiled in when present: `pipeline_fusion.cpp`, `simd_kernels.cpp`, `performance_metrics.cpp`, `pipeline_report.cpp`. The `--mode analysis-sweep` CSV sweep is available on all three binaries since they share the same `main.cpp`.

---

## Group Members

| Name | ERP |
| --- | --- |
| Gehna Bhatia | 29054 |
| Muhammad Zayed Nauman | 29047 |
| Zainab Irfan | 29091 |
| Muhammad Anis Imran | 29017 |

---

## Building

### Prerequisites

- CMake ≥ 3.16
- C++17 compiler (AppleClang, GCC, MSVC)
- libtiff
  - macOS: `brew install libtiff`
  - Linux: `sudo apt install libtiff-dev`
- *(Optional)* CUDA toolkit for real GPU support on Linux/Windows

### macOS / Linux

```bash
git clone https://github.com/zayed-nauman/Parallel-Image-Transforms-for-Gigapixel-Images
cd Parallel-Image-Transforms-for-Gigapixel-Images
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu)   # macOS
make -j$(nproc)                      # Linux
```

This produces three executables in `build/`: `milestone1`, `milestone2`, `milestone3`.

### Windows (PowerShell)

```powershell
mkdir build_win_ps; cd build_win_ps
cmake ..
cmake --build . --config Release
```

---

## Generating Test Images

The `tools/gen_test_image.py` script generates synthetic checkerboard TIFF images of any size for testing.

```bash
# Small image — fast correctness check (12 MB)
python3 tools/gen_test_image.py 2048 2048

# Medium image — visible speedup (~192 MB)
python3 tools/gen_test_image.py 8192 8192

# Gigapixel image — full speedup benchmark (~2.9 GB, takes 3–5 min to generate)
python3 tools/gen_test_image.py 32000 32000
```

> **Disk space:** The 32000×32000 image requires approximately 3 GB plus 3 GB for the output. Ensure at least 6 GB is available.

---

## Milestone 1 — CPU Pipeline

> All modes listed under Milestones 2 and 3 are also available on the `milestone1` binary since all three share the same `main.cpp` and source set. The distinction is that M2 adds the heterogeneous scheduler and M3 adds the optional fusion/SIMD/metrics files.

### Basic usage

```bash
./build/milestone1 input.tiff output.tiff
```

### Modes

| Flag | Description |
| --- | --- |
| `--mode parallel` | Run parallel pipeline only (default) |
| `--mode sequential` | Run single-threaded pipeline only |
| `--mode both` | Run both and print a speedup comparison table |
| `--mode hetero` | Heterogeneous CPU+GPU pipeline |
| `--mode hetero-bench` | Hetero vs CPU-parallel speedup comparison |
| `--mode analysis-sweep` | Full M3 benchmark sweep, writes CSV |

### Transforms

| Transform | Command | Description |
| --- | --- | --- |
| Identity | `--transform identity` | Copy pixels unchanged (correctness baseline) |
| Box blur | `--transform blur 8` | Separable box blur, radius 8 |
| Crop | `--transform crop 0 0 1024 1024` | Crop to rectangle (x0 y0 x1 y1) |
| Resize | `--transform resize 0.5 0.5` | Scale image by sx sy |
| Rotate | `--transform rotate 30` | Rotate counter-clockwise by degrees |

Transforms can be chained in any order:

```bash
./build/milestone1 --transform blur 4 --transform crop 0 0 1024 1024 input.tiff output.tiff
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--tile-size N` | 512 | Processing tile size in pixels |
| `--threads N` | hardware concurrency | Number of worker threads |
| `--halo N` | auto | Override halo width |
| `--border mode` | clamp | Border fill: `zero`, `clamp`, `reflect` |
| `--in-flight N` | 16 | Max tiles held in RAM at once |

### Quickstart

```bash
# Step 1 — Generate a small test image
python3 tools/gen_test_image.py 2048 2048

# Step 2 — Run the identity pass and verify correctness
./build/milestone1 test_2048x2048_rgb.tiff out.tiff

python3 -c "
import numpy as np, tifffile as tf
a = tf.imread('test_2048x2048_rgb.tiff')
b = tf.imread('out.tiff')
print('PASS' if np.array_equal(a,b) else 'FAIL')
"

# Step 3 — Try individual transforms
./build/milestone1 --transform blur 8 test_2048x2048_rgb.tiff out_blur.tiff
./build/milestone1 --transform crop 200 200 1800 1800 test_2048x2048_rgb.tiff out_crop.tiff
./build/milestone1 --transform resize 0.5 0.5 test_2048x2048_rgb.tiff out_small.tiff
./build/milestone1 --transform rotate 30 test_2048x2048_rgb.tiff out_rotated.tiff

# Step 4 — Generate the gigapixel image and benchmark
df -h /   # check available space (need 6 GB)
python3 tools/gen_test_image.py 32000 32000
./build/milestone1 --mode both test_32000x32000_rgb.tiff out_32k.tiff
```

---

## Milestone 2 — Heterogeneous CPU+GPU Pipeline

Milestone 2 introduces `GpuTileProcessor`, a heterogeneous scheduler that dynamically routes tiles to CPU workers or GPU CUDA streams. On machines without CUDA (e.g. Apple Silicon Macs), `gpu_kernels_stub.cpp` provides CPU fallback stubs so the binary builds and runs identically — just without real GPU acceleration.

The `milestone2` binary adds `--mode hetero` and `--mode hetero-bench` on top of all M1 modes.

### GPU options

| Option | Default | Description |
| --- | --- | --- |
| `--gpu` | off | Enable GPU tile routing (requires CUDA build; without this all tiles go to CPU) |
| `--gpu-tile-size N` | 1024 | Tile size used for GPU workers |
| `--cpu-tile-size N` | 512 | Tile size used for CPU workers |
| `--streams N` | 3 | Number of CUDA streams |

### Examples

```bash
# Heterogeneous run with GPU enabled, box blur
./build/milestone2 --mode hetero --gpu --transform blur 8 input.tiff out_hetero.tiff

# Heterogeneous run without a CUDA GPU (CPU stub — all tiles go to CPU workers)
./build/milestone2 --mode hetero --transform blur 8 input.tiff out_hetero.tiff

# Benchmark hetero vs CPU-parallel and print speedup table
./build/milestone2 --mode hetero-bench --gpu --transform blur 8 input.tiff out_hetero.tiff

# Rotate with custom GPU and CPU tile sizes
./build/milestone2 --mode hetero --gpu --gpu-tile-size 2048 --cpu-tile-size 256 \
    --transform rotate 30 input.tiff out_rotated.tiff

# Hetero with multiple transforms chained
./build/milestone2 --mode hetero --gpu \
    --transform blur 4 --transform crop 0 0 4096 4096 \
    input.tiff out_hetero_crop.tiff

# All M1 modes still work on the milestone2 binary
./build/milestone2 --mode both input.tiff out_both.tiff
./build/milestone2 --mode sequential --transform resize 0.5 0.5 input.tiff out_seq.tiff
```

### Output (hetero mode)

The heterogeneous pipeline prints a detailed routing report:

```
Tiles total=256  cpu=180  gpu=76  skipped=0
Elapsed: 2.31s  412.5 Mpix/s
Sched: routed_gpu=76  routed_cpu=180  spill_from_gpu=0
GPU est. throughput: 890.2 Mpix/s  CPU est. throughput: 310.4 Mpix/s
```

---

## Milestone 3 — Pipeline Fusion, SIMD & Analysis Sweep

Milestone 3 compiles in optional files on top of M2 when they are present: `pipeline_fusion.cpp` (fuses adjacent transforms into single-pass kernels via `FusionOptimizer`), `simd_kernels.cpp` (SIMD-accelerated inner loops), `performance_metrics.cpp`, and `pipeline_report.cpp`. If any of these are absent the build still succeeds with a status message.

### New mode

| Flag | Description |
| --- | --- |
| `--mode analysis-sweep` | Run the full Milestone 3 benchmark sweep across tile sizes, halo depths, and image scales. Writes results to `<output>.milestone3_analysis.csv` |

### Benchmark image path options

These are used by the analysis sweep to test across image scales:

| Option | Description |
| --- | --- |
| `--bench-1gp <path>` | Path to a ~1 gigapixel test image |
| `--bench-10gp <path>` | Path to a ~10 gigapixel test image |
| `--bench-50gp <path>` | Path to a ~50 gigapixel test image |

### Examples

```bash
# Run the full Milestone 3 analysis sweep
./build/milestone3 --mode analysis-sweep \
    --bench-1gp test_32000x32000_rgb.tiff \
    --bench-10gp test_100000x100000_rgb.tiff \
    --bench-50gp test_224000x224000_rgb.tiff \
    --transform blur 8 \
    input.tiff results.tiff
# Results written to: results.tiff.milestone3_analysis.csv

# Pipeline fusion — all M2 modes also available on milestone3 binary
./build/milestone3 --mode hetero --gpu --transform blur 8 \
    --transform rotate 30 input.tiff out_fused.tiff

# Hetero-bench with SIMD and fusion active
./build/milestone3 --mode hetero-bench --gpu \
    --transform blur 4 --transform crop 0 0 8192 8192 \
    input.tiff out_m3.tiff
```

---

## Notes on CUDA / GPU Support

- **Apple Silicon (M1/M2/M3):** CUDA is not supported. The build uses `gpu_kernels_stub.cpp` which routes all GPU-bound tiles back to CPU workers transparently. All modes run correctly; GPU throughput stats will reflect CPU execution.
- **Linux with NVIDIA GPU:** Install the CUDA toolkit, then replace `gpu_kernels_stub.cpp` with the real `gpu_kernels.cu` in CMakeLists.txt and enable `find_package(CUDA)`. Pass `--gpu` at runtime to activate GPU routing.
- **Windows:** Same as Linux for CUDA support. Use the `build_win/` or `build_win_ps/` directories.
