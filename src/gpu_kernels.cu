
/*#ifdef HAVE_CUDA

#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>

#define CK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
    } while(0)

// Helper function for clamping
__device__ __forceinline__
static int d_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Box blur horizontal pass
__global__ void kernel_box_blur_h(
    const uint8_t* __restrict__ src, 
    uint8_t* __restrict__ dst,
    int W, int H, int C, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;

    int cnt = 2 * radius + 1;
    int base_idx = (y * W + x) * C;
    
    for (int c = 0; c < C; ++c) {
        int sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            int cx = d_clamp(x + k, 0, W - 1);
            sum += src[(y * W + cx) * C + c];
        }
        dst[base_idx + c] = (uint8_t)((sum + cnt/2) / cnt);
    }
}

// Box blur vertical pass
__global__ void kernel_box_blur_v(
    const uint8_t* __restrict__ src, 
    uint8_t* __restrict__ dst,
    int W, int H, int C, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;

    int cnt = 2 * radius + 1;
    int base_idx = (y * W + x) * C;
    
    for (int c = 0; c < C; ++c) {
        int sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            int cy = d_clamp(y + k, 0, H - 1);
            sum += src[(cy * W + x) * C + c];
        }
        dst[base_idx + c] = (uint8_t)((sum + cnt/2) / cnt);
    }
}

// Simple copy kernel for identity transform
__global__ void kernel_copy(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int bytes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bytes) {
        dst[idx] = src[idx];
    }
}

// Main blur launcher
void launch_box_blur(
    const uint8_t* d_src, uint8_t* d_dst, uint8_t* d_tmp,
    int W, int H, int C, int radius, cudaStream_t stream)
{
    if (radius == 0) {
        // Just copy
        int bytes = W * H * C;
        int threads = 256;
        int blocks = (bytes + threads - 1) / threads;
        kernel_copy<<<blocks, threads, 0, stream>>>(d_src, d_dst, bytes);
        CK(cudaGetLastError());
        return;
    }
    
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    
    // Horizontal blur: src -> d_tmp
    kernel_box_blur_h<<<grid, block, 0, stream>>>(d_src, d_tmp, W, H, C, radius);
    CK(cudaGetLastError());
    
    // Vertical blur: d_tmp -> d_dst
    kernel_box_blur_v<<<grid, block, 0, stream>>>(d_tmp, d_dst, W, H, C, radius);
    CK(cudaGetLastError());
}

// Stub implementations for other transforms
void launch_rgb_to_gray(const uint8_t* d_src, uint8_t* d_dst, int W, int H, cudaStream_t stream) {}
void launch_gray_to_rgb(const uint8_t* d_src, uint8_t* d_dst, int W, int H, cudaStream_t stream) {}
void launch_rgb_to_hsv(const uint8_t* d_src, uint8_t* d_dst, int W, int H, cudaStream_t stream) {}
void launch_hsv_to_rgb(const uint8_t* d_src, uint8_t* d_dst, int W, int H, cudaStream_t stream) {}
void launch_resize(const uint8_t* d_src, uint8_t* d_dst, int W_src, int H_src, int W_dst, int H_dst, int C, float scale_x, float scale_y, cudaStream_t stream) {}
void launch_rotate(const uint8_t* d_src_buf, uint8_t* d_dst_core, int W_buf, int H_buf, int W_core, int H_core, int C, float img_cx, float img_cy, float tile_gx, float tile_gy, float buf_gx, float buf_gy, float cos_a, float sin_a, cudaStream_t stream) {}

#endif
*/

// =============================================================================
//  gpu_kernels.cu   —  Milestone 2
//
//  Real CUDA kernel implementations for every launcher declared in
//  gpu_kernels.cuh.  Compiled by nvcc only when HAVE_CUDA is defined.
//
//  Kernel families
//  ───────────────
//  1.  Box-blur         separable H + V passes, 16x16 thread blocks
//  2.  Colour convert   RGB<->Gray (BT.601), RGB<->HSV
//  3.  Geometric        bilinear resize, bilinear rotation (reverse-map)
//
//  Every __global__ kernel maps one output pixel per thread.
//  Thread layout: dim3 block(16,16), grid covers full (W x H) output.
//  All launchers are bit-exact with gpu_kernels_stub.cpp on the same input.
// =============================================================================

#ifdef HAVE_CUDA

#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(_e));                 \
    } while(0)

// ─────────────────────────────────────────────────────────────────────────────
//  Device helpers  (inlined into every kernel that needs them)
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
static int d_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Bilinear sample from a (W x H x C) row-major device buffer.
// Matches s_bilinear() in gpu_kernels_stub.cpp exactly.
__device__ __forceinline__
static void d_bilinear(const uint8_t* __restrict__ buf,
                       int W, int H, int C,
                       float fx, float fy,
                       uint8_t* __restrict__ out)
{
    fx = fmaxf(0.f, fminf(fx, (float)(W - 1)));
    fy = fmaxf(0.f, fminf(fy, (float)(H - 1)));
    int x0 = (int)fx,  y0 = (int)fy;
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    float dx = fx - x0, dy = fy - y0;
    for (int c = 0; c < C; ++c) {
        float v = (1.f - dy) * ((1.f - dx) * buf[(y0*W + x0)*C + c]
                              +       dx   * buf[(y0*W + x1)*C + c])
                +       dy   * ((1.f - dx) * buf[(y1*W + x0)*C + c]
                              +       dx   * buf[(y1*W + x1)*C + c]);
        out[c] = (uint8_t)fmaxf(0.f, fminf(v, 255.f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  1.  Box-blur  (separable H + V passes)
//      These two kernels were already correct in the original file.
//      Kept here verbatim so the file is self-contained.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_box_blur_h(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__       dst,
    int W, int H, int C, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int cnt      = 2 * radius + 1;
    int base_idx = (y * W + x) * C;
    for (int c = 0; c < C; ++c) {
        int sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            int cx = d_clamp(x + k, 0, W - 1);
            sum += src[(y * W + cx) * C + c];
        }
        dst[base_idx + c] = (uint8_t)((sum + cnt / 2) / cnt);
    }
}

__global__ void kernel_box_blur_v(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__       dst,
    int W, int H, int C, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int cnt      = 2 * radius + 1;
    int base_idx = (y * W + x) * C;
    for (int c = 0; c < C; ++c) {
        int sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            int cy = d_clamp(y + k, 0, H - 1);
            sum += src[(cy * W + x) * C + c];
        }
        dst[base_idx + c] = (uint8_t)((sum + cnt / 2) / cnt);
    }
}

// Simple byte-copy kernel (used by launch_box_blur when radius == 0)
__global__ void kernel_copy(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__       dst,
    int bytes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bytes) dst[idx] = src[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
//  2a.  RGB -> Grayscale   (BT.601 fixed-point, matches stub exactly)
//       Coefficients: R*306 + G*601 + B*117, >> 10  ≡  *0.299 *0.587 *0.114
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_rgb_to_gray(
    const uint8_t* __restrict__ src_rgb,
    uint8_t* __restrict__       dst_gray,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i      = y * W + x;
    int r      = src_rgb[i * 3    ];
    int g      = src_rgb[i * 3 + 1];
    int b      = src_rgb[i * 3 + 2];
    dst_gray[i] = (uint8_t)((306 * r + 601 * g + 117 * b) >> 10);
}

// ─────────────────────────────────────────────────────────────────────────────
//  2b.  Grayscale -> RGB   (replicate single channel into R, G, B)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_gray_to_rgb(
    const uint8_t* __restrict__ src_gray,
    uint8_t* __restrict__       dst_rgb,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = y * W + x;
    uint8_t v = src_gray[i];
    dst_rgb[i * 3    ] = v;
    dst_rgb[i * 3 + 1] = v;
    dst_rgb[i * 3 + 2] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
//  2c.  RGB -> HSV
//       H stored as H/2 in [0,179], S and V in [0,255].
//       Matches stub launch_rgb_to_hsv() exactly.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_rgb_to_hsv(
    const uint8_t* __restrict__ src_rgb,
    uint8_t* __restrict__       dst_hsv,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i   = y * W + x;
    float r = src_rgb[i * 3    ] / 255.f;
    float g = src_rgb[i * 3 + 1] / 255.f;
    float b = src_rgb[i * 3 + 2] / 255.f;

    float mx = fmaxf(r, fmaxf(g, b));
    float mn = fminf(r, fminf(g, b));
    float d  = mx - mn;

    float hue = 0.f;
    float sat = (mx > 1e-6f) ? d / mx : 0.f;
    float val = mx;

    if (d > 1e-6f) {
        if      (mx == r) hue = 60.f * fmodf((g - b) / d + 6.f, 6.f);
        else if (mx == g) hue = 60.f * ((b - r) / d + 2.f);
        else              hue = 60.f * ((r - g) / d + 4.f);
        // fmodf on device doesn't guarantee [0,360) for negative input,
        // so clamp the same way the stub does:
        if (hue < 0.f) hue += 360.f;
    }

    dst_hsv[i * 3    ] = (uint8_t)(hue * 0.5f);
    dst_hsv[i * 3 + 1] = (uint8_t)(sat * 255.f + 0.5f);
    dst_hsv[i * 3 + 2] = (uint8_t)(val * 255.f + 0.5f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  2d.  HSV -> RGB
//       Matches stub launch_hsv_to_rgb() exactly.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_hsv_to_rgb(
    const uint8_t* __restrict__ src_hsv,
    uint8_t* __restrict__       dst_rgb,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i   = y * W + x;
    float h = src_hsv[i * 3    ] * 2.f;          // [0, 360)
    float s = src_hsv[i * 3 + 1] / 255.f;
    float v = src_hsv[i * 3 + 2] / 255.f;

    float c  = v * s;
    float xc = c * (1.f - fabsf(fmodf(h / 60.f, 2.f) - 1.f));
    float m  = v - c;

    float r = 0.f, g = 0.f, b = 0.f;
    int sec = (int)(h / 60.f) % 6;
    switch (sec) {
        case 0: r = c; g = xc;       break;
        case 1: r = xc; g = c;       break;
        case 2:         g = c; b = xc; break;
        case 3:         g = xc; b = c; break;
        case 4: r = xc;        b = c; break;
        default:r = c;         b = xc; break;
    }

    dst_rgb[i * 3    ] = (uint8_t)((r + m) * 255.f + 0.5f);
    dst_rgb[i * 3 + 1] = (uint8_t)((g + m) * 255.f + 0.5f);
    dst_rgb[i * 3 + 2] = (uint8_t)((b + m) * 255.f + 0.5f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  3a.  Bilinear resize   (centre-aligned reverse mapping)
//       Each thread writes one output pixel (ox, oy).
//       Matches stub launch_resize() exactly.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_resize_bilinear(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__       dst,
    int W_src, int H_src,
    int W_dst, int H_dst,
    int C,
    float scale_x, float scale_y)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W_dst || oy >= H_dst) return;

    // Centre-aligned reverse map: same formula as the stub
    float gx = ((float)ox + 0.5f) / scale_x - 0.5f;
    float gy = ((float)oy + 0.5f) / scale_y - 0.5f;

    d_bilinear(src, W_src, H_src, C, gx, gy,
               dst + ((size_t)oy * W_dst + ox) * C);
}

// ─────────────────────────────────────────────────────────────────────────────
//  3b.  Bilinear rotation   (global-space reverse rotation around image centre)
//       Each thread writes one output core pixel (ox, oy).
//       Matches stub launch_rotate() exactly — same coordinate maths,
//       same out-of-bounds zeroing, same bilinear sample.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_rotate_bilinear(
    const uint8_t* __restrict__ src_buf,
    uint8_t* __restrict__       dst_core,
    int W_buf, int H_buf,
    int W_core, int H_core,
    int C,
    float img_cx,  float img_cy,
    float tile_gx, float tile_gy,
    float buf_gx,  float buf_gy,
    float cos_a,   float sin_a)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W_core || oy >= H_core) return;

    // Global position of this output pixel
    float gox = tile_gx + (float)ox;
    float goy = tile_gy + (float)oy;

    // Rotate backwards to find the source position in global space
    float dx  = gox - img_cx;
    float dy  = goy - img_cy;
    float rsx = img_cx + (cos_a * dx + sin_a * dy);
    float rsy = img_cy + (-sin_a * dx + cos_a * dy);

    // Convert from global to buffer-local coordinates
    float bx = rsx - buf_gx;
    float by = rsy - buf_gy;

    uint8_t* out = dst_core + ((size_t)oy * W_core + ox) * C;

    if (bx < 0.f || bx >= (float)(W_buf - 1) ||
        by < 0.f || by >= (float)(H_buf - 1)) {
        // Out of bounds — zero fill (same as stub memset(out, 0, C))
        for (int c = 0; c < C; ++c) out[c] = 0;
    } else {
        d_bilinear(src_buf, W_buf, H_buf, C, bx, by, out);
    }
}

// =============================================================================
//  Host launchers
//  These are the only functions called from gpu_tile_processor_fixed.cpp.
//  All previously empty — now fully implemented.
// =============================================================================

// ── 1.  Box-blur ─────────────────────────────────────────────────────────────

void launch_box_blur(
    const uint8_t* d_src, uint8_t* d_dst, uint8_t* d_tmp,
    int W, int H, int C, int radius, cudaStream_t stream)
{
    if (radius == 0) {
        int bytes   = W * H * C;
        int threads = 256;
        int blocks  = (bytes + threads - 1) / threads;
        kernel_copy<<<blocks, threads, 0, stream>>>(d_src, d_dst, bytes);
        CK(cudaGetLastError());
        return;
    }

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    // Horizontal pass: src -> d_tmp
    kernel_box_blur_h<<<grid, block, 0, stream>>>(d_src, d_tmp, W, H, C, radius);
    CK(cudaGetLastError());

    // Vertical pass: d_tmp -> d_dst
    kernel_box_blur_v<<<grid, block, 0, stream>>>(d_tmp, d_dst, W, H, C, radius);
    CK(cudaGetLastError());
}

// ── 2.  Colour-space conversions ─────────────────────────────────────────────

void launch_rgb_to_gray(
    const uint8_t* d_src, uint8_t* d_dst,
    int W, int H, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    kernel_rgb_to_gray<<<grid, block, 0, stream>>>(d_src, d_dst, W, H);
    CK(cudaGetLastError());
}

void launch_gray_to_rgb(
    const uint8_t* d_src, uint8_t* d_dst,
    int W, int H, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    kernel_gray_to_rgb<<<grid, block, 0, stream>>>(d_src, d_dst, W, H);
    CK(cudaGetLastError());
}

void launch_rgb_to_hsv(
    const uint8_t* d_src, uint8_t* d_dst,
    int W, int H, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    kernel_rgb_to_hsv<<<grid, block, 0, stream>>>(d_src, d_dst, W, H);
    CK(cudaGetLastError());
}

void launch_hsv_to_rgb(
    const uint8_t* d_src, uint8_t* d_dst,
    int W, int H, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    kernel_hsv_to_rgb<<<grid, block, 0, stream>>>(d_src, d_dst, W, H);
    CK(cudaGetLastError());
}

// ── 3.  Geometric transforms ─────────────────────────────────────────────────

void launch_resize(
    const uint8_t* d_src, uint8_t* d_dst,
    int W_src, int H_src, int W_dst, int H_dst, int C,
    float scale_x, float scale_y, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W_dst + block.x - 1) / block.x,
              (H_dst + block.y - 1) / block.y);
    kernel_resize_bilinear<<<grid, block, 0, stream>>>(
        d_src, d_dst,
        W_src, H_src, W_dst, H_dst, C,
        scale_x, scale_y);
    CK(cudaGetLastError());
}

void launch_rotate(
    const uint8_t* d_src_buf, uint8_t* d_dst_core,
    int W_buf, int H_buf, int W_core, int H_core, int C,
    float img_cx,  float img_cy,
    float tile_gx, float tile_gy,
    float buf_gx,  float buf_gy,
    float cos_a,   float sin_a,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W_core + block.x - 1) / block.x,
              (H_core + block.y - 1) / block.y);
    kernel_rotate_bilinear<<<grid, block, 0, stream>>>(
        d_src_buf, d_dst_core,
        W_buf, H_buf, W_core, H_core, C,
        img_cx,  img_cy,
        tile_gx, tile_gy,
        buf_gx,  buf_gy,
        cos_a,   sin_a);
    CK(cudaGetLastError());
}

#endif  // HAVE_CUDA
