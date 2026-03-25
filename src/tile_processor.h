#pragma once
#include "common.h"
#include "transforms.h"
#include <functional>
#include <memory>

// ─────────────────────────────────────────────────────────────────────────────
//  TileProcessor
//
//  A thread-pool producer-consumer pipeline:
//
//    Producer thread
//      └─ reads tile coordinates from the grid
//      └─ enqueues (TileJob) into the input queue
//
//    N Worker threads
//      └─ dequeue TileJob from input queue
//      └─ call reader.read_tile()
//      └─ apply TransformChain
//      └─ enqueue result into output queue
//
//    Consumer thread
//      └─ dequeue result from output queue
//      └─ call writer.write_tile()
//      └─ free tile memory
//
//  Back-pressure: if more than max_in_flight tiles are in flight, the
//  producer blocks until a tile has been consumed.
// ─────────────────────────────────────────────────────────────────────────────

class TileReader;
class TileWriter;

class TileProcessor {
public:
    struct Stats {
        uint64_t tiles_read      = 0;
        uint64_t tiles_processed = 0;
        uint64_t tiles_skipped   = 0;   // e.g. outside crop rect
        uint64_t tiles_written   = 0;
        double   elapsed_sec     = 0.0;
        double   mpix_per_sec    = 0.0;
    };

    TileProcessor(const PipelineConfig& cfg,
                  const TileReader&     reader,
                  TileWriter&           writer,
                  const TransformChain& chain);
    ~TileProcessor();

    // Run the full pipeline.  Blocks until all tiles are processed.
    Stats run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};