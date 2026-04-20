
#ifdef HAVE_CUDA

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
