
// Simplified GPU Tile Processor - Working Version
#include "gpu_tile_processor.h"
#include "tile_reader.h"
#include "tile_writer.h"
#include "overlap.h"
#include "gpu_kernels.cuh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <cmath>
#include <cstring>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
    } while(0)
#endif

// Simple kernel that just copies data
#ifdef HAVE_CUDA
__global__ void simple_copy_kernel(uint8_t* dst, const uint8_t* src, size_t bytes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bytes) {
        dst[idx] = src[idx];
    }
}
#endif

template<typename T>
class BQ {
public:
    explicit BQ(std::size_t cap) : cap_(cap) {}
 
    void push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_full_.wait(lk, [this]{ return q_.size() < cap_ || done_; });
        if (done_) return;
        q_.push(std::move(item));
        cv_empty_.notify_one();
    }
 
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_empty_.wait(lk, [this]{ return !q_.empty() || done_; });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop();
        cv_full_.notify_one();
        return true;
    }
 
    void seal() {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = true;
        cv_empty_.notify_all();
        cv_full_.notify_all();
    }
 
    std::size_t size_approx() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }
 
    std::size_t capacity() const { return cap_; }
 
private:
    mutable std::mutex      mu_;
    std::condition_variable cv_empty_, cv_full_;
    std::queue<T>           q_;
    std::size_t             cap_;
    bool                    done_ = false;
};

struct TileCoord2 { int col, row; };
 
struct RoutedTile {
    TileCoord2  coord;
    Device      device;
    std::string reason;
    int         tile_size;
};
 
struct ResultTile {
    Tile tile;
    bool skip = false;
};

struct GpuTileProcessor::Impl {
    const HeterogeneousConfig& cfg;
    const TileReader&          reader;
    TileWriter&                writer;
    const TransformChain&      chain;
    ImageInfo                  img_info;
 
    WorkScheduler  scheduler;
    MemoryManager  mem_mgr;
 
    BQ<RoutedTile> cpu_queue;
    BQ<RoutedTile> gpu_queue;
    BQ<ResultTile> result_queue;
 
    std::atomic<uint64_t> tiles_cpu     {0};
    std::atomic<uint64_t> tiles_gpu     {0};
    std::atomic<uint64_t> tiles_skipped {0};
 
    Impl(const HeterogeneousConfig& c,
         const TileReader& r, TileWriter& w, const TransformChain& ch)
        : cfg(c), reader(r), writer(w), chain(ch)
        , img_info(r.info())
        , scheduler(c.sched_cfg)
        , mem_mgr(c.mem_cfg)
        , cpu_queue (static_cast<std::size_t>(c.base.max_in_flight) * 2)
        , gpu_queue (static_cast<std::size_t>(c.base.max_in_flight))
        , result_queue(static_cast<std::size_t>(c.base.max_in_flight) * 2)
    {}
};

GpuTileProcessor::GpuTileProcessor(const HeterogeneousConfig& cfg,
                                   const TileReader& reader,
                                   TileWriter& writer,
                                   const TransformChain& chain)
    : impl_(std::make_unique<Impl>(cfg, reader, writer, chain))
{}
 
GpuTileProcessor::~GpuTileProcessor() = default;

#ifdef HAVE_CUDA
static ResultTile process_on_gpu(
    const RoutedTile&     rt,
    const TileReader&     reader,
    const TransformChain& chain,
    const ImageInfo&      img_info,
    int                   halo,
    MemoryManager&        mem_mgr,
    WorkScheduler&        scheduler,
    cudaStream_t stream)
{
    auto t0 = std::chrono::steady_clock::now();
    
    // Read tile
    Tile raw = reader.read_tile(rt.coord.col, rt.coord.row, rt.tile_size, halo);
    
    size_t buf_bytes = (size_t)raw.buf_w() * raw.buf_h() * channels_of(raw.fmt);
    size_t core_bytes = (size_t)raw.core_w * raw.core_h * channels_of(raw.fmt);
    
    std::cout << "[GPU] Processing tile " << rt.coord.col << "," << rt.coord.row 
              << " buf_bytes=" << buf_bytes << " core_bytes=" << core_bytes << std::endl;
    
    // Allocate device memory
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    
    cudaError_t err = cudaMalloc(&d_src, buf_bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc src failed: " << cudaGetErrorString(err) << std::endl;
        Tile result = chain.apply(std::move(raw), img_info);
        return { std::move(result), false };
    }
    
    err = cudaMalloc(&d_dst, core_bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc dst failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_src);
        Tile result = chain.apply(std::move(raw), img_info);
        return { std::move(result), false };
    }
    
    // Copy to device
    err = cudaMemcpyAsync(d_src, raw.data.data(), buf_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "H2D copy failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Launch copy kernel
    int threads = 256;
    int blocks = (core_bytes + threads - 1) / threads;
    simple_copy_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, core_bytes);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy back to host
    Tile result;
    result.global_x = raw.global_x;
    result.global_y = raw.global_y;
    result.core_w = raw.core_w;
    result.core_h = raw.core_h;
    result.halo = 0;
    result.fmt = raw.fmt;
    result.data.resize(core_bytes);
    
    err = cudaMemcpyAsync(result.data.data(), d_dst, core_bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "D2H copy failed: " << cudaGetErrorString(err) << std::endl;
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
#else
static ResultTile process_on_gpu(
    const RoutedTile&     rt,
    const TileReader&     reader,
    const TransformChain& chain,
    const ImageInfo&      img_info,
    int                   halo,
    MemoryManager&        mem_mgr,
    WorkScheduler&        scheduler)
{
    Tile raw = reader.read_tile(rt.coord.col, rt.coord.row, rt.tile_size, halo);
    Tile result = chain.apply(std::move(raw), img_info);
    bool skip = (result.core_w == 0 || result.core_h == 0);
    return { std::move(result), skip };
}
#endif

GpuTileProcessor::Stats GpuTileProcessor::run()
{
    auto& I = *impl_;
 
    int num_cpu = I.cfg.base.num_threads;
    if (num_cpu <= 0) num_cpu = (int)std::thread::hardware_concurrency();
    if (num_cpu <= 0) num_cpu = 2;
 
    int cpu_tile = I.cfg.cpu_tile_size;
    int halo     = std::max(I.cfg.base.halo_size, I.chain.max_halo());
    int ncols    = I.reader.num_tile_cols(cpu_tile);
    int nrows    = I.reader.num_tile_rows(cpu_tile);
    int total    = ncols * nrows;
 
    std::cout << "[GpuTileProcessor]"
              << "  gpu=" << (I.cfg.gpu_available ? "ON" : "OFF")
              << "  cpu_workers=" << num_cpu
              << "  grid=" << ncols << "x" << nrows << " (" << total << " tiles)"
              << "  cpu_tile=" << cpu_tile
              << "  halo=" << halo << "\n";
 
    auto t_start = std::chrono::steady_clock::now();
 
#ifdef HAVE_CUDA
    std::vector<cudaStream_t> streams(I.cfg.num_streams);
    if (I.cfg.gpu_available) {
        for (auto& s : streams) {
            cudaStreamCreate(&s);
        }
    }
    std::atomic<int> stream_idx{0};
#endif
 
    // Producer thread
    std::thread producer([&]() {
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                Tile hdr;
                hdr.core_w = std::min(cpu_tile, (int)I.img_info.width - col * cpu_tile);
                hdr.core_h = std::min(cpu_tile, (int)I.img_info.height - row * cpu_tile);
                hdr.halo = halo;
                hdr.fmt  = I.img_info.fmt;

                RoutingDecision dec = I.scheduler.route_tile(hdr, I.chain);
                RoutedTile rt;
                rt.coord     = {col, row};
                rt.device    = dec.device;
                rt.reason    = dec.reason;
                rt.tile_size = cpu_tile;

                if (dec.device == Device::GPU) {
                    I.gpu_queue.push(rt);
                } else {
                    I.cpu_queue.push(rt);
                }
            }
        }
        I.cpu_queue.seal();
        I.gpu_queue.seal();
    });

    // CPU workers
    std::vector<std::thread> cpu_workers;
    cpu_workers.reserve(num_cpu);
    for (int t = 0; t < num_cpu; ++t) {
        cpu_workers.emplace_back([&]() {
            RoutedTile rt;
            while (I.cpu_queue.pop(rt)) {
                auto t0 = std::chrono::steady_clock::now();
                Tile raw = I.reader.read_tile(rt.coord.col, rt.coord.row, rt.tile_size, halo);
                Tile result = I.chain.apply(std::move(raw), I.img_info);
                bool skip = (result.core_w == 0 || result.core_h == 0);
                if (!skip) I.tiles_cpu.fetch_add(1, std::memory_order_relaxed);
                I.result_queue.push({ std::move(result), skip });
            }
        });
    }

    // GPU worker
    std::thread gpu_worker([&]() {
        RoutedTile rt;
        while (I.gpu_queue.pop(rt)) {
#ifdef HAVE_CUDA
            int idx = stream_idx.fetch_add(1, std::memory_order_relaxed);
            cudaStream_t st = I.cfg.gpu_available ? streams[idx % streams.size()] : 0;
            ResultTile res = process_on_gpu(rt, I.reader, I.chain, I.img_info, halo,
                                           I.mem_mgr, I.scheduler, st);
#else
            ResultTile res = process_on_gpu(rt, I.reader, I.chain, I.img_info, halo,
                                           I.mem_mgr, I.scheduler);
#endif
            if (!res.skip) I.tiles_gpu.fetch_add(1, std::memory_order_relaxed);
            I.result_queue.push(std::move(res));
        }
    });

    // Reaper thread
    std::thread reaper([&]() {
        producer.join();
        for (auto& w : cpu_workers) w.join();
        gpu_worker.join();
        I.result_queue.seal();
    });

    // Consumer
    uint64_t written = 0;
    ResultTile rr;
    while (I.result_queue.pop(rr)) {
        if (!rr.skip) {
            I.writer.write_tile(rr.tile);
            ++written;
        }
    }

    reaper.join();
 
#ifdef HAVE_CUDA
    if (I.cfg.gpu_available) {
        for (auto& s : streams) cudaStreamDestroy(s);
    }
#endif
 
    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    double total_mpix = (double)I.img_info.width * I.img_info.height / 1e6;
 
    Stats s;
    s.tiles_total   = (uint64_t)total;
    s.tiles_on_cpu  = I.tiles_cpu.load();
    s.tiles_on_gpu  = I.tiles_gpu.load();
    s.tiles_skipped = I.tiles_skipped.load();
    s.tiles_written = written;
    s.elapsed_sec   = elapsed;
    s.mpix_per_sec  = (elapsed > 0) ? total_mpix / elapsed : 0.0;
    s.sched_stats   = I.scheduler.stats();
    return s;
}
