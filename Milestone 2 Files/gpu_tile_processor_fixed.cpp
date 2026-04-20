// ─────────────────────────────────────────────────────────────────────────────
//  gpu_tile_processor.cpp   — Milestone 2 (FIXED)
// ─────────────────────────────────────────────────────────────────────────────

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
#  include <cuda_runtime.h>
#  define CUDA_CHECK_GTP(call)                                               \
     do { cudaError_t _e = (call);                                           \
          if (_e != cudaSuccess)                                              \
              throw std::runtime_error(std::string("CUDA: ") +                \
                                       cudaGetErrorString(_e)); } while(0)
#endif

// Bounded queue (same as before)
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

static ResultTile process_on_gpu(
    const RoutedTile&     rt,
    const TileReader&     reader,
    const TransformChain& chain,
    const ImageInfo&      img_info,
    int                   halo,
    MemoryManager&        mem_mgr,
    WorkScheduler&        scheduler
#ifdef HAVE_CUDA
    , cudaStream_t stream
#endif
)
{
    auto t0 = std::chrono::steady_clock::now();
 
    Tile raw = reader.read_tile(rt.coord.col, rt.coord.row, rt.tile_size, halo);
 
    int W_buf  = raw.buf_w(), H_buf  = raw.buf_h();
    int W_core = raw.core_w, H_core = raw.core_h;
    int C      = channels_of(raw.fmt);
    std::size_t buf_bytes  = (std::size_t)W_buf  * H_buf  * C;
    std::size_t core_bytes = (std::size_t)W_core * H_core * C;
 
    int kern_radius = chain.max_halo();
 
#ifdef HAVE_CUDA
    auto d_src = mem_mgr.try_acquire_gpu(buf_bytes);
    auto d_dst = mem_mgr.try_acquire_gpu(core_bytes);
    auto d_tmp = mem_mgr.try_acquire_gpu(buf_bytes);
 
    if (!d_src || !d_dst || !d_tmp) {
        if (d_src) mem_mgr.release_gpu(std::move(d_src));
        if (d_dst) mem_mgr.release_gpu(std::move(d_dst));
        if (d_tmp) mem_mgr.release_gpu(std::move(d_tmp));
        Tile result = chain.apply(std::move(raw), img_info);
        bool skip = (result.core_w == 0 || result.core_h == 0);
        return { std::move(result), skip };
    }
 
    uint8_t* d_src_p = static_cast<uint8_t*>(d_src->ptr);
    uint8_t* d_dst_p = static_cast<uint8_t*>(d_dst->ptr);
    uint8_t* d_tmp_p = static_cast<uint8_t*>(d_tmp->ptr);
 
    std::cout << "[GPU] Transferring to device..." << std::endl;
    std::cout << "[GPU] Copying " << buf_bytes << " bytes H2D" << std::endl;\n    CUDA_CHECK_GTP(cudaMemcpyAsync(d_src_p, raw.data.data(), buf_bytes,
                                   cudaMemcpyHostToDevice, stream));
 
    if (kern_radius > 0) {
        launch_box_blur(d_src_p, d_dst_p, d_tmp_p,
                        W_buf, H_buf, C, kern_radius, stream);
        // Check bounds\n    if (halo < 0 || W_core <= 0 || H_core <= 0) {\n        std::cerr << "Invalid parameters for cudaMemcpy2D" << std::endl;\n        throw std::runtime_error("Invalid memcpy parameters");\n    }\n    CUDA_CHECK_GTP(cudaMemcpy2DAsync(
            d_tmp_p,
            (std::size_t)W_core * C,
            d_dst_p + (std::size_t)halo * W_buf * C + halo * C,
            (std::size_t)W_buf * C,
            (std::size_t)W_core * C,
            (std::size_t)H_core,
            cudaMemcpyDeviceToDevice, stream));
        std::swap(d_tmp_p, d_dst_p);
    } else {
        // Check bounds\n    if (halo < 0 || W_core <= 0 || H_core <= 0) {\n        std::cerr << "Invalid parameters for cudaMemcpy2D" << std::endl;\n        throw std::runtime_error("Invalid memcpy parameters");\n    }\n    CUDA_CHECK_GTP(cudaMemcpy2DAsync(
            d_dst_p,
            (std::size_t)W_core * C,
            d_src_p + (std::size_t)halo * W_buf * C + halo * C,
            (std::size_t)W_buf * C,
            (std::size_t)W_core * C,
            (std::size_t)H_core,
            cudaMemcpyDeviceToDevice, stream));
    }
 
    Tile result;
    result.global_x = raw.global_x; result.global_y = raw.global_y;
    result.core_w   = W_core;       result.core_h   = H_core;
    result.halo     = 0;            result.fmt      = raw.fmt;
    result.allocate();
 
    std::cout << "[GPU] Transferring to device..." << std::endl;
    CUDA_CHECK_GTP(cudaMemcpyAsync(result.data.data(), d_dst_p, core_bytes,
                                   cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_GTP(cudaStreamSynchronize(stream));
 
    mem_mgr.release_gpu(std::move(d_src));
    mem_mgr.release_gpu(std::move(d_dst));
    mem_mgr.release_gpu(std::move(d_tmp));
 
#else
    std::vector<uint8_t> buf_src(buf_bytes);
    std::vector<uint8_t> buf_tmp(buf_bytes);
 
    std::memcpy(buf_src.data(), raw.data.data(), buf_bytes);
 
    Tile result;
    result.global_x = raw.global_x; result.global_y = raw.global_y;
    result.core_w   = W_core;       result.core_h   = H_core;
    result.halo     = 0;            result.fmt      = raw.fmt;
    result.allocate();
 
    if (kern_radius > 0) {
        std::vector<uint8_t> buf_scratch(buf_bytes);
        launch_box_blur(buf_src.data(), buf_tmp.data(), buf_scratch.data(),
                        W_buf, H_buf, C, kern_radius);
        for (int row = 0; row < H_core; ++row) {
            const uint8_t* src_row = buf_tmp.data() +
                (std::size_t)(row + halo) * W_buf * C + (std::size_t)halo * C;
            std::memcpy(result.data.data() + (std::size_t)row * W_core * C,
                        src_row, (std::size_t)W_core * C);
        }
    } else {
        for (int row = 0; row < H_core; ++row) {
            const uint8_t* src_row = buf_src.data() +
                (std::size_t)(row + halo) * W_buf * C + (std::size_t)halo * C;
            std::memcpy(result.data.data() + (std::size_t)row * W_core * C,
                        src_row, (std::size_t)W_core * C);
        }
    }
#endif
 
    auto t1 = std::chrono::steady_clock::now();
    scheduler.report_gpu_tile_time(
        std::chrono::duration<double>(t1 - t0).count(), W_core * H_core);
 
    bool skip = (result.core_w == 0 || result.core_h == 0);
    return { std::move(result), skip };
}

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
              << "  gpu=" << (I.cfg.gpu_available ? "ON" : "OFF(CPU-stub)")
              << "  cpu_workers=" << num_cpu
              << "  grid=" << ncols << "x" << nrows << " (" << total << " tiles)"
              << "  cpu_tile=" << cpu_tile
              << "  gpu_tile=" << I.cfg.gpu_tile_size
              << "  halo=" << halo << "\n";
 
    auto t_start = std::chrono::steady_clock::now();
 
#ifdef HAVE_CUDA
    std::vector<cudaStream_t> streams(I.cfg.num_streams);
    if (I.cfg.gpu_available)
        for (auto& s : streams) CUDA_CHECK_GTP(cudaStreamCreate(&s));
    std::atomic<int> stream_idx{0};
#endif
 
    // Producer thread
    std::thread producer([&]() {
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                Tile hdr;
                hdr.core_w = std::min(cpu_tile,
                    (int)I.img_info.width  - col * cpu_tile);
                hdr.core_h = std::min(cpu_tile,
                    (int)I.img_info.height - row * cpu_tile);
                hdr.halo = halo;
                hdr.fmt  = I.img_info.fmt;

                float gpu_occ = (float)I.gpu_queue.size_approx() /
                                (float)std::max(std::size_t(1), I.gpu_queue.capacity());
                float cpu_occ = (float)I.cpu_queue.size_approx() /
                                (float)std::max(std::size_t(1), I.cpu_queue.capacity());
                I.scheduler.set_gpu_queue_occupancy(gpu_occ);
                I.scheduler.set_cpu_queue_occupancy(cpu_occ);

                RoutingDecision dec = I.scheduler.route_tile(hdr, I.chain);
                RoutedTile rt;
                rt.coord     = {col, row};
                rt.device    = dec.device;
                rt.reason    = dec.reason;
                rt.tile_size = cpu_tile;

                if (dec.device == Device::GPU) I.gpu_queue.push(rt);
                else                           I.cpu_queue.push(rt);
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
                Tile raw    = I.reader.read_tile(rt.coord.col, rt.coord.row,
                                                  rt.tile_size, halo);
                Tile result = I.chain.apply(std::move(raw), I.img_info);
                bool skip   = (result.core_w == 0 || result.core_h == 0);
                auto t1 = std::chrono::steady_clock::now();
                I.scheduler.report_cpu_tile_time(
                    std::chrono::duration<double>(t1-t0).count(),
                    result.core_w * result.core_h);
                if (!skip) I.tiles_cpu.fetch_add(1, std::memory_order_relaxed);
                else       I.tiles_skipped.fetch_add(1, std::memory_order_relaxed);
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
            cudaStream_t st = I.cfg.gpu_available
                              ? streams[idx % (int)streams.size()] : 0;
            ResultTile res = process_on_gpu(
                rt, I.reader, I.chain, I.img_info, halo,
                I.mem_mgr, I.scheduler, st);
#else
            ResultTile res = process_on_gpu(
                rt, I.reader, I.chain, I.img_info, halo,
                I.mem_mgr, I.scheduler);
#endif
            if (!res.skip) I.tiles_gpu.fetch_add(1, std::memory_order_relaxed);
            else           I.tiles_skipped.fetch_add(1, std::memory_order_relaxed);
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

    // Consumer (main thread)
    uint64_t written = 0;
    ResultTile rr;
    while (I.result_queue.pop(rr)) {
        if (!rr.skip) {
            I.writer.write_tile(rr.tile);
            ++written;
            if (written % 100 == 0)
                std::cout << "  wrote " << written << "/" << total
                          << " tiles\r" << std::flush;
        }
    }
    std::cout << "\n";

    reaper.join();
 
#ifdef HAVE_CUDA
    if (I.cfg.gpu_available)
        for (auto& s : streams) cudaStreamDestroy(s);
#endif
 
    auto t_end = std::chrono::steady_clock::now();
    double elapsed    = std::chrono::duration<double>(t_end - t_start).count();
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