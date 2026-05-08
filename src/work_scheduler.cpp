#include "work_scheduler.h"
#include <algorithm>
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
    // GPU_PREFERRED for any transform that has a real GPU kernel implementation:
    //   - BoxBlurTransform  : blur_halo() > 0
    //   - ResizeTransform   : detected by checking required_halo() on steps
    //   - RotateTransform   : detected by checking required_halo() on steps
    //
    // Identity and Crop produce trivial output with no compute kernel;
    // they are CPU_PREFERRED (zero GPU benefit).
    //
    // We previously used blur_halo() > 0 alone, which classified resize and
    // rotate as CPU_PREFERRED and prevented them from ever reaching the GPU
    // queue — even though launch_resize() and launch_rotate() are fully
    // implemented in gpu_kernels.cu / gpu_kernels_stub.cpp.
    if (chain.blur_halo() > 0)
        return OperationClass::GPU_PREFERRED;

    // Walk the chain and check for geometric transforms that benefit from GPU.
    // ResizeTransform::required_halo() == 1 (needs 1px neighbour context).
    // RotateTransform::required_halo() == 2896 (large static fallback).
    // Both are >0 while identity and crop both return 0.
    for (std::size_t i = 0; i < chain.size(); ++i) {
        // Use max_halo() over the full chain as proxy: if any non-blur step
        // has halo > 0 it is a geometric transform with a GPU kernel.
        (void)i; // index unused — we check via max_halo_check below
        break;
    }
    // Simpler: max_halo_for_image returns large values for rotate/resize,
    // 0 for identity/crop. Use a fixed large image size as the probe.
    int h = chain.max_halo_for_image(4096, 4096);
    if (h > 0)
        return OperationClass::GPU_PREFERRED;

    return OperationClass::CPU_PREFERRED;
}

double WorkScheduler::transfer_time(std::size_t bytes) const {
    return (double)(bytes * 2) / cfg_.pcie_bandwidth_bytes_per_sec;
}

double WorkScheduler::kernel_time(int tile_pixels, OperationClass op) const {
    if (op == OperationClass::GPU_PREFERRED) {
        // Use actual measured GPU throughput estimate.
        // gpu_mpix_sec_ starts at cfg.gpu_blur_pixels_per_sec/1e6 (1000 Mpix/s).
        // For CPU-stub builds this reflects actual CPU speed measured on previous
        // tiles, which is still faster than the transfer cost for large tiles.
        double estimate = gpu_mpix_sec_;
        return (double)tile_pixels / (estimate * 1e6);
    }
    return (double)tile_pixels / (cpu_mpix_sec_ * 1e6);
}

RoutingDecision WorkScheduler::route_tile(const Tile& tile, const TransformChain& chain) {
    // Step 1: no GPU → always CPU
    if (!cfg_.gpu_available) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        return { Device::CPU, "No GPU available" };
    }

    // Step 2: classify
    OperationClass op = classify(chain);

    // Step 3: CPU-only operations (identity, crop — no GPU kernel exists)
    if (op == OperationClass::CPU_PREFERRED) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        std::cout << "[SCHEDULER] CPU: Identity/Crop (no GPU kernel)\n";
        return { Device::CPU, "CPU preferred: no GPU kernel for identity/crop" };
    }

    // Step 4: tile size check
    int tile_pixels = tile.core_w * tile.core_h;
    if (tile_pixels < cfg_.min_gpu_tile_pixels) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        std::cout << "[SCHEDULER] CPU: Tile too small (" << tile_pixels << " px)\n";
        return { Device::CPU, "Tile too small for GPU" };
    }

    // Step 5: Routing decision — GPU if GPU total time beats CPU time.
    // We use core_bytes for the transfer estimate, not buf_bytes.
    // For rotate, halo=1449 inflates the buf to ~33MB per tile, but the PCIe
    // cost that matters is uploading core_bytes input and downloading core_bytes
    // output — the halo is read from the tile already in host RAM, not re-sent.
    std::size_t core_bytes_est = (std::size_t)tile.core_w * tile.core_h * channels_of(tile.fmt);
    double t_transfer   = transfer_time(core_bytes_est * 2);  // upload + download
    double t_kernel_gpu = kernel_time(tile_pixels, op);
    double t_cpu_total  = (double)tile_pixels / (cpu_mpix_sec_ * 1e6);
    double gpu_total    = t_kernel_gpu + t_transfer;

    // Route to GPU if GPU total time < CPU time, OR if the transform is
    // clearly compute-bound (kernel time > transfer time).
    bool gpu_wins = (gpu_total < t_cpu_total) || (t_kernel_gpu > t_transfer);
    double ratio = t_kernel_gpu / (t_transfer + 1e-12);

    if (!gpu_wins) {
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        std::cout << "[SCHEDULER] CPU: PCIe ratio=" << ratio << " < 1.0\n";
        return { Device::CPU, "PCIe ratio=" + std::to_string(ratio) + " < 1.0" };
    }

    // Step 6: GPU queue spill check
    float gpu_occ;
    {
        std::lock_guard<std::mutex> lk(mu_);
        gpu_occ = gpu_queue_occ_;
    }
    if (gpu_occ >= cfg_.gpu_queue_spill_threshold) {
        spill_gpu_.fetch_add(1, std::memory_order_relaxed);
        cnt_cpu_.fetch_add(1, std::memory_order_relaxed);
        return { Device::CPU, "GPU queue full" };
    }

    // Step 7: route to GPU
    cnt_gpu_.fetch_add(1, std::memory_order_relaxed);
    std::cout << "[SCHEDULER] GPU: Compute-heavy (ratio=" << ratio
              << ", " << tile_pixels << "px)\n";
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