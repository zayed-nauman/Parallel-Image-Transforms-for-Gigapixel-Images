#include "sequential_processor.h"
#include "tile_reader.h"
#include "tile_writer.h"
#include "overlap.h"

#include <chrono>
#include <iostream>

SequentialProcessor::Stats SequentialProcessor::run() {
    int tile_size = cfg_.tile_size;
    int halo      = std::max(cfg_.halo_size, chain_.max_halo());
    int ncols     = reader_.num_tile_cols(tile_size);
    int nrows     = reader_.num_tile_rows(tile_size);
    int total     = ncols * nrows;

    std::cout << "[SequentialProcessor] 1 thread, "
              << ncols << "x" << nrows << " tile grid ("
              << total << " tiles), halo=" << halo << "\n";

    Stats s;
    auto t_start = std::chrono::steady_clock::now();

    // Process every tile in raster order (top-left → bottom-right).
    // No threads, no queues — just a straight loop.
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {

            // 1. Read tile + halo from disk
            Tile raw = reader_.read_tile(col, row, tile_size, halo);
            ++s.tiles_read;

            // 2. Apply the transform chain
            Tile result = chain_.apply(std::move(raw), reader_.info());

            bool skip = (result.core_w == 0 || result.core_h == 0);
            if (skip) {
                ++s.tiles_skipped;
            } else {
                // 3. Write immediately — no queue needed
                writer_.write_tile(result);
                ++s.tiles_written;
                ++s.tiles_processed;
            }

            // Progress every 50 tiles
            if ((s.tiles_read % 50) == 0)
                std::cout << "  [seq] " << s.tiles_read << "/" << total
                          << " tiles\r" << std::flush;
        }
    }
    std::cout << "\n";

    auto t_end   = std::chrono::steady_clock::now();
    s.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    double total_mpix = static_cast<double>(reader_.info().width) *
                        reader_.info().height / 1e6;
    s.mpix_per_sec = (s.elapsed_sec > 0) ? total_mpix / s.elapsed_sec : 0.0;

    return s;
}