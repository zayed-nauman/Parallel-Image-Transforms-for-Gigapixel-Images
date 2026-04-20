#pragma once
// =============================================================================
//  gpu_kernels.cuh   —  Milestone 2
//  Zainab Irfan Ansari  (29091)
//
//  Declares every kernel and host-side launcher used by gpu_tile_processor.cpp.
//
//  Two build modes
//  ───────────────
//  HAVE_CUDA defined   ->  real __global__ kernels + cudaStream_t launchers
//                          (compiled by nvcc from gpu_kernels.cu)
//  HAVE_CUDA not set   ->  launchers have identical signatures but no stream
//                          argument; bodies live in gpu_kernels_stub.cpp and
//                          fall back to plain C++ so the project links on any
//                          machine, including MSYS2 without a CUDA toolkit.
//
//  Kernel families
//  ───────────────
//  1.  Box-blur         -- separable H + V passes, shared-memory staging
//  2.  Colour convert   -- RGB<->Gray (BT.601), RGB<->HSV
//  3.  Geometric        -- bilinear resize, bilinear rotation (reverse-map)
// =============================================================================

#include <cstdint>

#ifdef HAVE_CUDA
#  include <cuda_runtime.h>

static constexpr int SMEM_TILE = 16;   // 16x16 thread block = 256 threads

// ---------------------------------------------------------------------------
//  1.  Box-blur
// ---------------------------------------------------------------------------
__global__ void kernel_box_blur_h(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int W, int H, int C, int radius);

__global__ void kernel_box_blur_v(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int W, int H, int C, int radius);

// ---------------------------------------------------------------------------
//  2.  Colour-space conversions
// ---------------------------------------------------------------------------
__global__ void kernel_rgb_to_gray(
    const uint8_t* __restrict__ src_rgb, uint8_t* __restrict__ dst_gray,
    int W, int H);

__global__ void kernel_gray_to_rgb(
    const uint8_t* __restrict__ src_gray, uint8_t* __restrict__ dst_rgb,
    int W, int H);

__global__ void kernel_rgb_to_hsv(
    const uint8_t* __restrict__ src_rgb, uint8_t* __restrict__ dst_hsv,
    int W, int H);

__global__ void kernel_hsv_to_rgb(
    const uint8_t* __restrict__ src_hsv, uint8_t* __restrict__ dst_rgb,
    int W, int H);

// ---------------------------------------------------------------------------
//  3.  Geometric transforms
// ---------------------------------------------------------------------------
__global__ void kernel_resize_bilinear(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int W_src, int H_src, int W_dst, int H_dst, int C,
    float scale_x, float scale_y);

__global__ void kernel_rotate_bilinear(
    const uint8_t* __restrict__ src_buf, uint8_t* __restrict__ dst_core,
    int W_buf, int H_buf, int W_core, int H_core, int C,
    float img_cx, float img_cy,
    float tile_gx, float tile_gy,
    float buf_gx,  float buf_gy,
    float cos_a,   float sin_a);

// ---------------------------------------------------------------------------
//  Host launchers  (CUDA build — all take an optional cudaStream_t)
// ---------------------------------------------------------------------------
void launch_box_blur(
    const uint8_t* d_src, uint8_t* d_dst, uint8_t* d_tmp,
    int W, int H, int C, int radius, cudaStream_t stream = 0);

void launch_rgb_to_gray(const uint8_t* d_src, uint8_t* d_dst,
                        int W, int H, cudaStream_t stream = 0);
void launch_gray_to_rgb(const uint8_t* d_src, uint8_t* d_dst,
                        int W, int H, cudaStream_t stream = 0);
void launch_rgb_to_hsv (const uint8_t* d_src, uint8_t* d_dst,
                        int W, int H, cudaStream_t stream = 0);
void launch_hsv_to_rgb (const uint8_t* d_src, uint8_t* d_dst,
                        int W, int H, cudaStream_t stream = 0);

void launch_resize(const uint8_t* d_src, uint8_t* d_dst,
                   int W_src, int H_src, int W_dst, int H_dst, int C,
                   float scale_x, float scale_y, cudaStream_t stream = 0);

void launch_rotate(const uint8_t* d_src_buf, uint8_t* d_dst_core,
                   int W_buf, int H_buf, int W_core, int H_core, int C,
                   float img_cx, float img_cy,
                   float tile_gx, float tile_gy,
                   float buf_gx,  float buf_gy,
                   float cos_a,   float sin_a,
                   cudaStream_t stream = 0);

#else  // !HAVE_CUDA  ---------------------------------------------------------
// ---------------------------------------------------------------------------
//  Host launchers  (CPU-only stub build — no stream argument)
//  Implementations are in gpu_kernels_stub.cpp
// ---------------------------------------------------------------------------

void launch_box_blur(
    const uint8_t* src, uint8_t* dst, uint8_t* tmp,
    int W, int H, int C, int radius);

void launch_rgb_to_gray(const uint8_t* src, uint8_t* dst, int W, int H);
void launch_gray_to_rgb(const uint8_t* src, uint8_t* dst, int W, int H);
void launch_rgb_to_hsv (const uint8_t* src, uint8_t* dst, int W, int H);
void launch_hsv_to_rgb (const uint8_t* src, uint8_t* dst, int W, int H);

void launch_resize(const uint8_t* src, uint8_t* dst,
                   int W_src, int H_src, int W_dst, int H_dst, int C,
                   float scale_x, float scale_y);

void launch_rotate(const uint8_t* src_buf, uint8_t* dst_core,
                   int W_buf, int H_buf, int W_core, int H_core, int C,
                   float img_cx, float img_cy,
                   float tile_gx, float tile_gy,
                   float buf_gx,  float buf_gy,
                   float cos_a,   float sin_a);

#endif  // HAVE_CUDA
