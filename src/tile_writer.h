#pragma once
#include "common.h"
#include <string>
#include <memory>
#include <mutex>

// ─────────────────────────────────────────────────────────────────────────────
//  TileWriter
//
//  Writes processed tiles to an output (Big)TIFF file tile-by-tile.
//  The output image is NEVER fully materialised in RAM: each tile is written
//  to disk immediately after it arrives from the thread pool, then its
//  memory is freed.
//
//  Thread safety: write_tile() is protected by an internal mutex so multiple
//  worker threads can call it concurrently.
// ─────────────────────────────────────────────────────────────────────────────

class TileWriter {
public:
    // Create (or overwrite) an output tiled TIFF.
    //   path        : output file path
    //   width/height: full output image dimensions
    //   fmt         : pixel format
    //   tile_size   : desired native tile size in the output file (e.g. 256)
    TileWriter(const std::string& path,
               uint32_t   width,
               uint32_t   height,
               PixelFormat fmt,
               uint32_t   tile_size = 256);

    ~TileWriter();

    // Non-copyable
    TileWriter(const TileWriter&)            = delete;
    TileWriter& operator=(const TileWriter&) = delete;

    // Write a single tile.  The tile must have halo == 0.
    // global_x / global_y set the position in the output image.
    // Thread-safe.
    void write_tile(const Tile& tile);

    // Flush and close the file.  Called automatically by the destructor.
    void close();

    // Statistics
    uint64_t tiles_written() const { return tiles_written_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::mutex            mu_;
    uint64_t              tiles_written_ = 0;
    bool                  closed_        = false;
};