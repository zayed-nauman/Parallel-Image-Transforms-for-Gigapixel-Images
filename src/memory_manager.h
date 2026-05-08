#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  memory_manager.h   — Milestone 2
//
//
//  Manages three memory tiers:
//    1. CPU RAM      — regular heap tiles (std::vector<uint8_t>)
//    2. Pinned RAM   — page-locked buffers for async PCIe DMA (cudaMallocHost)
//    3. GPU VRAM     — device buffers (cudaMalloc)
//
//  Each tier has a configurable capacity limit (bytes).  Callers request
//  a buffer via acquire_*() and return it via release_*(). When VRAM is
//  exhausted, tiles are spilled to a compressed disk cache (via zlib/LZ4)
//  for graceful degradation rather than a hard failure.
//
//  Thread safety: all public methods are mutex-protected.
// ─────────────────────────────────────────────────────────────────────────────
 
#include "common.h"
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>
#include <functional>
#include <condition_variable>
 
// ─────────────────────────────────────────────────────────────────────────────
//  MemoryManagerConfig
// ─────────────────────────────────────────────────────────────────────────────
 
struct MemoryManagerConfig {
    // Maximum bytes to hold as pinned (page-locked) host memory.
    // Rule of thumb: 2–4 × (max tile size in bytes) × num_streams
    std::size_t max_pinned_bytes  = 256ULL * 1024 * 1024;   // 256 MB
 
    // Maximum bytes to hold on GPU VRAM at any time.
    // Should be < (available VRAM) - kernel overhead.
    std::size_t max_gpu_bytes     = 512ULL * 1024 * 1024;   // 512 MB
 
    // Maximum bytes allowed in the CPU heap pool.
    std::size_t max_cpu_bytes     = 1ULL * 1024 * 1024 * 1024; // 1 GB
 
    // Disk spill directory (used only when VRAM is exhausted).
    std::string spill_dir         = "/tmp/m2_spill";
 
    // Whether GPU (CUDA) is actually available.
    bool        gpu_available     = false;
};
 
// ─────────────────────────────────────────────────────────────────────────────
//  PinnedBuffer — RAII wrapper for cudaMallocHost / cudaFreeHost
// ─────────────────────────────────────────────────────────────────────────────
 
struct PinnedBuffer {
    void*       ptr   = nullptr;
    std::size_t bytes = 0;
 
    PinnedBuffer() = default;
    PinnedBuffer(std::size_t n);
    ~PinnedBuffer();
 
    // Non-copyable, movable
    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept;
};
 
// ─────────────────────────────────────────────────────────────────────────────
//  DeviceBuffer — RAII wrapper for cudaMalloc / cudaFree
// ─────────────────────────────────────────────────────────────────────────────
 
struct DeviceBuffer {
    void*       ptr   = nullptr;
    std::size_t bytes = 0;
 
    DeviceBuffer() = default;
    DeviceBuffer(std::size_t n);
    ~DeviceBuffer();
 
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
};
 
// ─────────────────────────────────────────────────────────────────────────────
//  MemoryManager
// ─────────────────────────────────────────────────────────────────────────────
 
class MemoryManager {
public:
    explicit MemoryManager(const MemoryManagerConfig& cfg = {});
    ~MemoryManager();
 
    // Non-copyable
    MemoryManager(const MemoryManager&)            = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
 
    // ── CPU heap allocation ───────────────────────────────────────────────
    // Returns a CPU Tile with its data buffer allocated on the heap.
    // Blocks (back-pressures) if max_cpu_bytes would be exceeded.
    Tile alloc_cpu_tile(int32_t core_w, int32_t core_h, int32_t halo, PixelFormat fmt);
    void free_cpu_tile(Tile& t);     // zeroes t.data, decrements accounting
 
    // ── Pinned memory (for async DMA) ─────────────────────────────────────
    // Acquires a PinnedBuffer of at least `bytes` bytes.
    // Returns nullptr (not blocked) if pinned pool is exhausted — caller
    // should fall back to regular malloc.
    std::unique_ptr<PinnedBuffer> acquire_pinned(std::size_t bytes);
    void release_pinned(std::unique_ptr<PinnedBuffer> buf);
 
    // ── GPU VRAM ──────────────────────────────────────────────────────────
    // Acquires a DeviceBuffer of exactly `bytes` bytes.
    // Returns nullptr if VRAM is exhausted (caller must spill or use CPU).
    // Non-blocking (returns immediately with nullptr on failure).
    std::unique_ptr<DeviceBuffer> try_acquire_gpu(std::size_t bytes);
    void release_gpu(std::unique_ptr<DeviceBuffer> buf);
 
    // ── Spill to disk (compressed) ────────────────────────────────────────
    // Serialise a Tile to a temp file in spill_dir.  Returns a handle (path).
    std::string spill_tile(const Tile& t);
    // Restore a previously spilled tile and delete the spill file.
    Tile restore_tile(const std::string& spill_path);
 
    // ── Statistics ────────────────────────────────────────────────────────
    struct Stats {
        std::size_t cpu_bytes_in_use    = 0;
        std::size_t pinned_bytes_in_use = 0;
        std::size_t gpu_bytes_in_use    = 0;
        uint64_t    spills              = 0;
        uint64_t    restores            = 0;
    };
    Stats stats() const;
 
private:
    MemoryManagerConfig cfg_;
    mutable std::mutex  mu_;
 
    // Accounting
    std::size_t cpu_bytes_used_    = 0;
    std::size_t pinned_bytes_used_ = 0;
    std::size_t gpu_bytes_used_    = 0;
    uint64_t    spill_count_       = 0;
    uint64_t    restore_count_     = 0;
 
    // Pinned buffer free-list (reuse blocks of the same size)
    std::vector<std::unique_ptr<PinnedBuffer>> pinned_free_list_;
    // GPU buffer free-list
    std::vector<std::unique_ptr<DeviceBuffer>> gpu_free_list_;
 
    // Back-pressure condition for CPU pool
    std::condition_variable cpu_cv_;
 
    // Spill counter (for unique file names)
    uint64_t spill_seq_ = 0;
 
    // Internal helpers
    std::unique_ptr<PinnedBuffer> alloc_pinned_internal(std::size_t bytes);
    std::unique_ptr<DeviceBuffer> alloc_gpu_internal(std::size_t bytes);
};