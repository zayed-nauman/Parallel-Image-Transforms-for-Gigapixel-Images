
#include "gpu_tile_processor.h"
#include "tile_reader.h"
#include "tile_writer.h"
#include "transforms.h"
#include <iostream>
#include <cuda_runtime.h>

// Simple kernel that just copies data
__global__ void copy_kernel(uint8_t* dst, const uint8_t* src, int bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bytes) {
        dst[idx] = src[idx];
    }
}

// Simplified GPU tile processing that should work
static ResultTile process_tile_on_gpu_simple(
    const RoutedTile& rt,
    const TileReader& reader,
    const TransformChain& chain,
    const ImageInfo& img_info,
    int halo,
    MemoryManager& mem_mgr,
    WorkScheduler& scheduler,
    cudaStream_t stream)
{
    auto t0 = std::chrono::steady_clock::now();
    
    // Read tile
    Tile raw = reader.read_tile(rt.coord.col, rt.coord.row, rt.tile_size, halo);
    
    int W_buf = raw.buf_w(), H_buf = raw.buf_h();
    int C = channels_of(raw.fmt);
    std::size_t buf_bytes = (std::size_t)W_buf * H_buf * C;
    std::size_t core_bytes = (std::size_t)raw.core_w * raw.core_h * C;
    
    std::cout << "[GPU] Processing tile " << rt.coord.col << "," << rt.coord.row 
              << " size=" << W_buf << "x" << H_buf << " bytes=" << buf_bytes << std::endl;
    
    // Allocate device memory
    uint8_t *d_src, *d_dst;
    cudaError_t err;
    
    err = cudaMalloc(&d_src, buf_bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for src: " << cudaGetErrorString(err) << std::endl;
        Tile result = chain.apply(std::move(raw), img_info);
        return { std::move(result), false };
    }
    
    err = cudaMalloc(&d_dst, core_bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dst: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_src);
        Tile result = chain.apply(std::move(raw), img_info);
        return { std::move(result), false };
    }
    
    // Copy H2D
    std::cout << "[GPU] H2D transfer: " << buf_bytes << " bytes" << std::endl;
    err = cudaMemcpyAsync(d_src, raw.data.data(), buf_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        Tile result = chain.apply(std::move(raw), img_info);
        return { std::move(result), false };
    }
    
    // Launch copy kernel (instead of blur)
    int threads = 256;
    int blocks = (core_bytes + threads - 1) / threads;
    copy_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, core_bytes);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy D2H
    Tile result;
    result.global_x = raw.global_x;
    result.global_y = raw.global_y;
    result.core_w = raw.core_w;
    result.core_h = raw.core_h;
    result.halo = 0;
    result.fmt = raw.fmt;
    result.data.resize(core_bytes);
    
    std::cout << "[GPU] D2H transfer: " << core_bytes << " bytes" << std::endl;
    err = cudaMemcpyAsync(result.data.data(), d_dst, core_bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaStreamSynchronize(stream);
    
    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);
    
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    scheduler.report_gpu_tile_time(elapsed, raw.core_w * raw.core_h);
    
    std::cout << "[GPU] Tile complete in " << elapsed << "s" << std::endl;
    
    return { std::move(result), false };
}
