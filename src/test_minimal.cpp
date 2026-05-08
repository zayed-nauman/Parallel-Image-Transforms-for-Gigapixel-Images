
#include "gpu_tile_processor.h"
#include "tile_reader.h"
#include "tile_writer.h"
#include "transforms.h"
#include <iostream>
#include <cuda_runtime.h>

int main() {
    try {
        std::cout << "=== GPU Initialization Test ===" << std::endl;
        
        // Initialize CUDA
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        std::cout << "Found " << deviceCount << " GPU devices" << std::endl;
        
        // Set device
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set device: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        
        // Create config with proper GPU settings
        HeterogeneousConfig hcfg;
        hcfg.gpu_available = true;
        hcfg.gpu_device = 0;
        hcfg.num_streams = 3;
        hcfg.cpu_tile_size = 256;
        hcfg.gpu_tile_size = 256;
        hcfg.mem_cfg.gpu_available = true;
        hcfg.mem_cfg.max_gpu_bytes = 512 * 1024 * 1024;  // 512 MB
        hcfg.sched_cfg.gpu_available = true;
        hcfg.sched_cfg.min_gpu_tile_pixels = 1;
        
        PipelineConfig base;
        base.num_threads = 2;
        base.max_in_flight = 4;
        base.halo_size = 16;
        hcfg.base = base;
        
        std::cout << "\n=== Opening image ===" << std::endl;
        // Open reader
        TileReader reader("test_image.tiff", BorderMode::CLAMP);
        
        // Create writer
        auto& info = reader.info();
        std::cout << "Image size: " << info.width << "x" << info.height << std::endl;
        TileWriter writer("test_gpu_out.tiff", info.width, info.height, info.fmt, 256);
        
        // Create transform chain
        TransformChain chain;
        chain.add(std::make_unique<BoxBlurTransform>(4));
        
        std::cout << "\n=== Running GPU Processor ===" << std::endl;
        // Run GPU processor
        GpuTileProcessor processor(hcfg, reader, writer, chain);
        auto stats = processor.run();
        
        std::cout << "\n=== RESULTS ===" << std::endl;
        std::cout << "Tiles total: " << stats.tiles_total << std::endl;
        std::cout << "Tiles on GPU: " << stats.tiles_on_gpu << std::endl;
        std::cout << "Tiles on CPU: " << stats.tiles_on_cpu << std::endl;
        std::cout << "Tiles skipped: " << stats.tiles_skipped << std::endl;
        std::cout << "Elapsed: " << stats.elapsed_sec << "s" << std::endl;
        std::cout << "Throughput: " << stats.mpix_per_sec << " Mpix/s" << std::endl;
        
        writer.close();
        std::cout << "\n✅ SUCCESS!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
