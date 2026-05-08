#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  gpu_tile_processor.h   — Milestone 2
//
//
//  Extends the Milestone 1 TileProcessor with GPU acceleration:
//
//    • A three-buffer CUDA stream pipeline overlaps:
//        Buffer A: H→D transfer of tile N+2
//        Buffer B: GPU kernel execution on tile N+1
//        Buffer C: D→H transfer of result tile N
//      This keeps the GPU continuously occupied.
//
//    • The WorkScheduler decides per-tile whether to use CPU or GPU.
//      CPU-bound tiles go directly to the M1 thread pool.
//      GPU-bound tiles enter the CUDA stream pipeline.
//
//    • Separate tile sizes for CPU and GPU are honoured:
//        cpu_tile_size  (L3-cache optimised, default 512)
//        gpu_tile_size  (occupancy optimised, default 1024)
//
//    • The MemoryManager controls all VRAM and pinned-memory allocation.
// ─────────────────────────────────────────────────────────────────────────────
 
#include "common.h"
#include "transforms.h"
#include "work_scheduler.h"
#include "memory_manager.h"
#include <memory>
#include <string>
 
class TileReader;
class TileWriter;
 
// ─────────────────────────────────────────────────────────────────────────────
//  HeterogeneousConfig  — extends PipelineConfig for M2
// ─────────────────────────────────────────────────────────────────────────────
 
struct HeterogeneousConfig {
    // Base pipeline settings (paths, threads, in-flight limit)
    PipelineConfig base;
 
    // GPU tile size — larger tiles improve occupancy (power of two)
    int gpu_tile_size  = 1024;
 
    // CPU tile size — smaller for cache fit (overrides base.tile_size)
    int cpu_tile_size  = 512;
 
    // Number of CUDA streams for triple-buffering
    int num_streams    = 3;
 
    // Whether a CUDA-capable GPU is available
    bool gpu_available = false;
 
    // GPU device index (default 0)
    int gpu_device     = 0;
 
    // Memory manager settings
    MemoryManagerConfig mem_cfg;
 
    // Scheduler settings
    SchedulerConfig sched_cfg;
};
 
// ─────────────────────────────────────────────────────────────────────────────
//  GpuTileProcessor
// ─────────────────────────────────────────────────────────────────────────────
 
class GpuTileProcessor {
public:
    struct Stats {
        uint64_t tiles_total       = 0;
        uint64_t tiles_on_cpu      = 0;
        uint64_t tiles_on_gpu      = 0;
        uint64_t tiles_skipped     = 0;
        uint64_t tiles_written     = 0;
        double   elapsed_sec       = 0.0;
        double   mpix_per_sec      = 0.0;
        WorkScheduler::Stats sched_stats;
    };
 
    GpuTileProcessor(const HeterogeneousConfig&  cfg,
                     const TileReader&            reader,
                     TileWriter&                  writer,
                     const TransformChain&        chain);
    ~GpuTileProcessor();
 
    // Run full pipeline.  Blocks until all tiles are processed and written.
    Stats run();
 
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};