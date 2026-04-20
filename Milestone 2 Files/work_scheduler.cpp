#include "work_scheduler.h"
#include <algorithm>
#include <sstream>
#include <iostream>

WorkScheduler::WorkScheduler(const SchedulerConfig& cfg)
    : cfg_(cfg)
    , gpu_mpix_sec_(cfg.gpu_blur_pixels_per_sec / 1e6)
    , cpu_mpix_sec_(100.0)
{}

void WorkScheduler::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    gpu_queue_occ_ = 0.f;
    cpu_queue_occ_ = 0.f;
    cnt_gpu_ = 0;
    cnt_cpu_ = 0;
    spill_gpu_ = 0;
    spill_cpu_ = 0;
}

OperationClass WorkScheduler::classify(const TransformChain& chain) {
    int h = chain.max_halo();
    
    // REAL CLASSIFICATION based on actual transforms
    if (h > 0) {
        // Blur (radius>0), Resize (needs 1), Rotate (needs 256) -> GPU
        return OperationClass::GPU_PREFERRED;
    }
    // Identity or crop -> CPU
    return OperationClass::CPU_PREFERRED;
}

double WorkScheduler::transfer_time(std::size_t bytes) const {
    return (double)(bytes * 2) / cfg_.pcie_bandwidth_bytes_per_sec;
}

double WorkScheduler::kernel_time(int tile_pixels, OperationClass op) const {
    // Different estimates for different operations
    // Blur is fast on GPU, Rotate is slower due to bilinear interpolation
    if (op == OperationClass::GPU_PREFERRED) {
        // For now, use a conservative estimate for rotate (slower than blur)
        // In production, you'd track this per operation type
        double estimate = gpu_mpix_sec_ * 0.3;  // Rotate is ~3x slower than blur
        return (double)tile_pixels / (estimate * 1e6);
    }
    return (double)tile_pixels / (cpu_mpix_sec_ * 1e6);
}

RoutingDecision WorkScheduler::route_tile(const Tile& tile, const TransformChain& chain) {
    static int call_count = 0;
    call_count++;
    
    // Step 1: No GPU available
    if (!cfg_.gpu_available) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        return { Device::CPU, "No GPU available" };
    }
    
    // Step 2: Classify operation type
    OperationClass op = classify(chain);
    
    // Step 3: CPU-only operations go to CPU immediately
    if (op == OperationClass::CPU_PREFERRED) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        if (call_count <= 5) {
            std::cout << "[SCHEDULER] CPU: Identity/Crop (halo=0)" << std::endl;
        }
        return { Device::CPU, "CPU preferred for identity/crop" };
    }
    
    // Step 4: Tile size check - small tiles not worth GPU overhead
    int tile_pixels = tile.core_w * tile.core_h;
    if (tile_pixels < cfg_.min_gpu_tile_pixels) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        if (call_count <= 5) {
            std::cout << "[SCHEDULER] CPU: Tile too small (" << tile_pixels << " px)" << std::endl;
        }
        return { Device::CPU, "Tile too small for GPU" };
    }
    
    // Step 5: Compute-to-transfer ratio
    std::size_t tile_bytes = (std::size_t)tile.buf_w() * tile.buf_h() * channels_of(tile.fmt);
    double t_transfer = transfer_time(tile_bytes);
    double t_kernel = kernel_time(tile_pixels, op);
    double ratio = t_kernel / (t_transfer + 1e-12);
    
    if (ratio < 1.0) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        if (call_count <= 5) {
            std::cout << "[SCHEDULER] CPU: PCIe ratio=" << ratio << " < 1.0" << std::endl;
        }
        return { Device::CPU, "PCIe transfer dominates" };
    }
    
    // Step 6: Load balancing - check queue occupancy
    float gpu_occ;
    {
        std::lock_guard<std::mutex> lk(mu_);
        gpu_occ = gpu_queue_occ_;
    }
    
    if (gpu_occ >= cfg_.gpu_queue_spill_threshold) {
        spill_gpu_.fetch_add(1, std::memory_order_relaxed);
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        if (call_count <= 5) {
            std::cout << "[SCHEDULER] CPU: GPU queue full (" << gpu_occ << ")" << std::endl;
        }
        return { Device::CPU, "GPU queue full" };
    }
    
    // Step 7: Route to GPU
    cnt_gpu_.fetch_add(1, std::memory_order_relaxed);
    if (call_count <= 5) {
        std::cout << "[SCHEDULER] GPU: Compute-heavy (ratio=" << ratio << ", " << tile_pixels << "px)" << std::endl;
    }
    return { Device::GPU, "Compute-heavy transform" };
}

void WorkScheduler::set_gpu_queue_occupancy(float occ) {
    std::lock_guard<std::mutex> lk(mu_);
    gpu_queue_occ_ = std::clamp(occ, 0.f, 1.f);
}

void WorkScheduler::set_cpu_queue_occupancy(float occ) {
    std::lock_guard<std::mutex> lk(mu_);
    cpu_queue_occ_ = std::clamp(occ, 0.f, 1.f);
}

void WorkScheduler::report_gpu_tile_time(double seconds, int tile_pixels) {
    if (seconds <= 0 || tile_pixels <= 0) return;
    double mpix_sec = (double)tile_pixels / (seconds * 1e6);
    std::lock_guard<std::mutex> lk(mu_);
    gpu_mpix_sec_ = 0.9 * gpu_mpix_sec_ + 0.1 * mpix_sec;
}

void WorkScheduler::report_cpu_tile_time(double seconds, int tile_pixels) {
    if (seconds <= 0 || tile_pixels <= 0) return;
    double mpix_sec = (double)tile_pixels / (seconds * 1e6);
    std::lock_guard<std::mutex> lk(mu_);
    cpu_mpix_sec_ = 0.9 * cpu_mpix_sec_ + 0.1 * mpix_sec;
}

WorkScheduler::Stats WorkScheduler::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return {
        cnt_gpu_.load(),
        cnt_cpu_.load(),
        spill_gpu_.load(),
        spill_cpu_.load(),
        gpu_mpix_sec_,
        cpu_mpix_sec_
    };
}
