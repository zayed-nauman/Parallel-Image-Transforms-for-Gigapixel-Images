
#pragma once

#include "common.h"
#include "transforms.h"
#include <atomic>
#include <mutex>
#include <string>

enum class OperationClass {
    GPU_PREFERRED,
    CPU_PREFERRED,
    NEUTRAL
};

enum class Device { CPU, GPU };

struct RoutingDecision {
    Device device;
    std::string reason;
};

struct SchedulerConfig {
    bool gpu_available = true;  // Force GPU available
    int min_gpu_tile_pixels = 1;  // Allow any tile size
    float gpu_queue_spill_threshold = 0.99f;
    float cpu_queue_spill_threshold = 0.99f;
    double pcie_bandwidth_bytes_per_sec = 8.0e9;
    double gpu_blur_pixels_per_sec = 1000e6;
    int num_cpu_workers = 4;
};

class WorkScheduler {
public:
    explicit WorkScheduler(const SchedulerConfig& cfg = {});
    
    RoutingDecision route_tile(const Tile& tile, const TransformChain& chain);
    static OperationClass classify(const TransformChain& chain);
    
    void set_gpu_queue_occupancy(float occupancy);
    void set_cpu_queue_occupancy(float occupancy);
    void report_gpu_tile_time(double seconds, int tile_pixels);
    void report_cpu_tile_time(double seconds, int tile_pixels);
    
    struct Stats {
        uint64_t routed_to_gpu = 0;
        uint64_t routed_to_cpu = 0;
        uint64_t spilled_from_gpu = 0;
        uint64_t spilled_from_cpu = 0;
        double gpu_mpix_per_sec = 0.0;
        double cpu_mpix_per_sec = 0.0;
    };
    Stats stats() const;
    void reset();
    
private:
    SchedulerConfig cfg_;
    mutable std::mutex mu_;
    float gpu_queue_occ_ = 0.f;
    float cpu_queue_occ_ = 0.f;
    double gpu_mpix_sec_ = 0.0;
    double cpu_mpix_sec_ = 0.0;
    std::atomic<uint64_t> cnt_gpu_{0};
    std::atomic<uint64_t> cnt_cpu_{0};
    std::atomic<uint64_t> spill_gpu_{0};
    std::atomic<uint64_t> spill_cpu_{0};
    
    double transfer_time(std::size_t bytes) const;
    double kernel_time(int tile_pixels, OperationClass op) const;
};
