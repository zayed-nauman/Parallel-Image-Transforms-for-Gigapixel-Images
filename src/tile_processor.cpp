#include "tile_processor.h"
#include "tile_reader.h"
#include "tile_writer.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
//  Thread-safe bounded queue
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : cap_(capacity) {}

    // Push an item; blocks if the queue is at capacity.
    void push(T item) {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [this]{ return items_.size() < cap_ || done_; });
        if (done_) return;
        items_.push(std::move(item));
        not_empty_.notify_one();
    }

    // Pop an item; blocks until one is available or the queue is sealed.
    // Returns false if the queue is empty AND sealed (no more items ever).
    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [this]{ return !items_.empty() || done_; });
        if (items_.empty()) return false;
        out = std::move(items_.front());
        items_.pop();
        not_full_.notify_one();
        return true;
    }

    // Seal the queue: no more items will be pushed.
    // Unblocks all waiting consumers.
    void seal() {
        std::lock_guard<std::mutex> lock(mu_);
        done_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    std::size_t size_approx() const {
        std::lock_guard<std::mutex> lock(mu_);
        return items_.size();
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T>           items_;
    std::size_t             cap_;
    bool                    done_ = false;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Job types
// ─────────────────────────────────────────────────────────────────────────────

struct TileCoord {
    int col, row;
};

struct ProcessedTile {
    Tile     tile;
    bool     skip = false;  // true if transform produced an empty tile
};

// ─────────────────────────────────────────────────────────────────────────────
//  Pimpl
// ─────────────────────────────────────────────────────────────────────────────

struct TileProcessor::Impl {
    const PipelineConfig& cfg;
    const TileReader&     reader;
    TileWriter&           writer;
    const TransformChain& chain;
    ImageInfo             img_info;

    // Queues
    BoundedQueue<TileCoord>    work_queue;    // producer → workers
    BoundedQueue<ProcessedTile> result_queue; // workers  → consumer

    // Shared stats
    std::atomic<uint64_t> tiles_read      {0};
    std::atomic<uint64_t> tiles_processed {0};
    std::atomic<uint64_t> tiles_skipped   {0};

    Impl(const PipelineConfig& c,
         const TileReader&     r,
         TileWriter&           w,
         const TransformChain& ch)
        : cfg(c), reader(r), writer(w), chain(ch),
          img_info(r.info()),
          // work_queue capacity = a few tiles ahead of worker count
          work_queue(static_cast<std::size_t>(c.max_in_flight) * 2),
          result_queue(static_cast<std::size_t>(c.max_in_flight))
    {}
};

// ─────────────────────────────────────────────────────────────────────────────
//  Constructor / destructor
// ─────────────────────────────────────────────────────────────────────────────

TileProcessor::TileProcessor(const PipelineConfig& cfg,
                             const TileReader&     reader,
                             TileWriter&           writer,
                             const TransformChain& chain)
    : impl_(std::make_unique<Impl>(cfg, reader, writer, chain))
{}

TileProcessor::~TileProcessor() = default;

// ─────────────────────────────────────────────────────────────────────────────
//  run()
// ─────────────────────────────────────────────────────────────────────────────

TileProcessor::Stats TileProcessor::run() {
    auto& I = *impl_;

    int num_threads = I.cfg.num_threads;
    if (num_threads <= 0)
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 2;

    int tile_size = I.cfg.tile_size;
    int halo      = std::max(I.cfg.halo_size, I.chain.max_halo());
    int ncols     = I.reader.num_tile_cols(tile_size);
    int nrows     = I.reader.num_tile_rows(tile_size);
    int total     = ncols * nrows;

    std::cout << "[TileProcessor] " << num_threads << " worker threads, "
              << ncols << "x" << nrows << " tile grid ("
              << total << " tiles), halo=" << halo << "\n";

    auto t_start = std::chrono::steady_clock::now();

    // ── Producer thread ───────────────────────────────────────────────────
    std::thread producer([&]() {
        for (int row = 0; row < nrows; ++row)
            for (int col = 0; col < ncols; ++col)
                I.work_queue.push({col, row});
        I.work_queue.seal();
    });

    // ── Worker threads ────────────────────────────────────────────────────
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            TileCoord coord;
            while (I.work_queue.pop(coord)) {
                // 1. Read tile from disk (includes halo).
                Tile raw = I.reader.read_tile(coord.col, coord.row,
                                              tile_size, halo);
                I.tiles_read.fetch_add(1, std::memory_order_relaxed);

                // 2. Apply the transform chain.
                Tile result = I.chain.apply(std::move(raw), I.img_info);

                bool skip = (result.core_w == 0 || result.core_h == 0);
                if (skip)
                    I.tiles_skipped.fetch_add(1, std::memory_order_relaxed);
                else
                    I.tiles_processed.fetch_add(1, std::memory_order_relaxed);

                // 3. Enqueue result for the consumer.
                I.result_queue.push({std::move(result), skip});
            }
        });
    }

    // ── Consumer thread ───────────────────────────────────────────────────
    uint64_t written = 0;

    // We seal the result queue once all workers have finished.
    // We do this in a separate "reaper" thread so the consumer can drain it
    // concurrently with the workers running.
    std::thread reaper([&]() {
        for (auto& w : workers) w.join();
        I.result_queue.seal();
    });

    ProcessedTile pt;
    while (I.result_queue.pop(pt)) {
        if (!pt.skip) {
            I.writer.write_tile(pt.tile);
            ++written;

            // Progress report every 100 tiles.
            if (written % 100 == 0)
                std::cout << "  wrote " << written << "/" << total << " tiles\r"
                          << std::flush;
        }
        // pt goes out of scope here → Tile::data freed → back-pressure relieved
    }
    std::cout << "\n";

    producer.join();
    reaper.join();

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    double total_mpix = static_cast<double>(I.img_info.width) *
                        I.img_info.height / 1e6;

    TileProcessor::Stats s;
    s.tiles_read      = I.tiles_read.load();
    s.tiles_processed = I.tiles_processed.load();
    s.tiles_skipped   = I.tiles_skipped.load();
    s.tiles_written   = written;
    s.elapsed_sec     = elapsed;
    s.mpix_per_sec    = (elapsed > 0) ? total_mpix / elapsed : 0.0;

    return s;
}