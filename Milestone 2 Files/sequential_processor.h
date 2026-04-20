#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  sequential_processor.h
//
//  A single-threaded, no-queue image processor that mirrors the parallel
//  TileProcessor interface exactly.  Used to measure the sequential baseline
//  for speedup analysis.
//
//  Usage:
//      SequentialProcessor seq(cfg, reader, writer, chain);
//      auto stats = seq.run();
//      std::cout << "Sequential: " << stats.elapsed_sec << "s\n";
// ─────────────────────────────────────────────────────────────────────────────

#include "common.h"
#include "transforms.h"

class TileReader;
class TileWriter;

class SequentialProcessor {
public:
    struct Stats {
        uint64_t tiles_read      = 0;
        uint64_t tiles_processed = 0;
        uint64_t tiles_skipped   = 0;
        uint64_t tiles_written   = 0;
        double   elapsed_sec     = 0.0;
        double   mpix_per_sec    = 0.0;
    };

    SequentialProcessor(const PipelineConfig& cfg,
                        const TileReader&     reader,
                        TileWriter&           writer,
                        const TransformChain& chain)
        : cfg_(cfg), reader_(reader), writer_(writer), chain_(chain) {}

    Stats run();

private:
    const PipelineConfig& cfg_;
    const TileReader&     reader_;
    TileWriter&           writer_;
    const TransformChain& chain_;
};