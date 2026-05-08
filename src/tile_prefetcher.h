#pragma once
// =============================================================================
//  tile_prefetcher.h   — Milestone 3
//
//  Overlaps disk I/O for tile N+1 with compute work on tile N.
//
//  Usage (inside tile_processor.cpp or gpu_tile_processor_fixed.cpp):
//
//    TilePrefetcher pf(reader, tile_size, halo, /*depth=*/2);
//    pf.schedule(0, 0);   // kick off first tile
//    pf.schedule(0, 1);   // kick off second tile
//
//    for each (col, row) in grid:
//        pf.schedule(next_col, next_row);   // always one step ahead
//        Tile t = pf.next();                // blocks only if I/O is slow
//        ... process t ...
// =============================================================================

#include "common.h"
#include "tile_reader.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <cassert>

// ---------------------------------------------------------------------------
//  Stats reported after the pipeline completes
// ---------------------------------------------------------------------------
struct PrefetchStats {
    uint64_t tiles_scheduled      = 0;
    uint64_t tiles_ready_on_get   = 0;  // consumer found tile already loaded
    uint64_t tiles_stalled        = 0;  // consumer had to wait for I/O
    double   io_overlap_ratio     = 0.0;// tiles_ready_on_get / tiles_scheduled
};

// ---------------------------------------------------------------------------
//  TilePrefetcher
// ---------------------------------------------------------------------------
class TilePrefetcher {
public:
    // depth: how many tiles to keep pre-loaded in the ready queue.
    //        2 is enough for most SSDs; increase for spinning disks.
    TilePrefetcher(const TileReader& reader,
                   int tile_size, int halo,
                   int depth = 2)
        : reader_(reader)
        , tile_size_(tile_size)
        , halo_(halo)
        , depth_(std::max(1, depth))
    {
        worker_ = std::thread([this]{ worker_loop(); });
    }

    ~TilePrefetcher() {
        // Send poison pill and wait for the worker to exit cleanly
        {
            std::lock_guard<std::mutex> lk(req_mu_);
            req_queue_.push({-1, -1});   // sentinel
            req_cv_.notify_one();
        }
        worker_.join();
    }

    // Non-copyable
    TilePrefetcher(const TilePrefetcher&)            = delete;
    TilePrefetcher& operator=(const TilePrefetcher&) = delete;

    // Schedule a tile to be prefetched.  Returns immediately.
    // Call this as early as possible — ideally while the previous tile is
    // still being processed.
    void schedule(int col, int row) {
        std::unique_lock<std::mutex> lk(req_mu_);
        // Back-pressure: don't queue more than depth_ pending requests
        req_not_full_.wait(lk, [this]{
            return (int)req_queue_.size() < depth_ + 1;
        });
        req_queue_.push({col, row});
        req_cv_.notify_one();
        ++stats_.tiles_scheduled;
    }

    // Retrieve the next prefetched tile (in the order schedule() was called).
    // Blocks only if the background I/O hasn't finished yet.
    Tile next() {
        std::unique_lock<std::mutex> lk(tile_mu_);
        if (tile_queue_.empty()) {
            ++stats_.tiles_stalled;
            tile_cv_.wait(lk, [this]{ return !tile_queue_.empty(); });
        } else {
            ++stats_.tiles_ready_on_get;
        }
        Tile t = std::move(tile_queue_.front());
        tile_queue_.pop();
        // Signal the request side that one slot opened up
        req_not_full_.notify_one();
        return t;
    }

    PrefetchStats stats() {
        PrefetchStats s = stats_;
        if (s.tiles_scheduled > 0)
            s.io_overlap_ratio =
                (double)s.tiles_ready_on_get / s.tiles_scheduled;
        return s;
    }

private:
    struct Req { int col, row; };

    void worker_loop() {
        while (true) {
            Req r;
            {
                std::unique_lock<std::mutex> lk(req_mu_);
                req_cv_.wait(lk, [this]{ return !req_queue_.empty(); });
                r = req_queue_.front();
                req_queue_.pop();
                req_not_full_.notify_one();
            }
            if (r.col < 0) break;  // sentinel — exit

            Tile tile = reader_.read_tile(r.col, r.row, tile_size_, halo_);

            {
                std::lock_guard<std::mutex> lk(tile_mu_);
                tile_queue_.push(std::move(tile));
                tile_cv_.notify_one();
            }
        }
    }

    const TileReader& reader_;
    int               tile_size_;
    int               halo_;
    int               depth_;

    // Request queue  (caller → worker)
    std::mutex              req_mu_;
    std::condition_variable req_cv_;
    std::condition_variable req_not_full_;
    std::queue<Req>         req_queue_;

    // Ready-tile queue  (worker → caller)
    std::mutex              tile_mu_;
    std::condition_variable tile_cv_;
    std::queue<Tile>        tile_queue_;

    std::thread  worker_;
    PrefetchStats stats_{};
};
