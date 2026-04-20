#pragma once
#include "common.h"
#include <string>
#include <memory>

// ─────────────────────────────────────────────────────────────────────────────
//  TileReader
//
//  Opens a (Big)TIFF file and provides random-access tile loading.
//  Each call to read_tile() loads the requested core region PLUS a halo of
//  `halo` pixels on every side.  Boundary halos are filled according to
//  the chosen BorderMode.
// ─────────────────────────────────────────────────────────────────────────────

enum class BorderMode {
    ZERO,       // Fill out-of-bounds pixels with 0
    CLAMP,      // Repeat the nearest in-bounds pixel
    REFLECT     // Mirror pixels at the boundary (best for smooth filters)
};

class TileReader {
public:
    explicit TileReader(const std::string& path, BorderMode border = BorderMode::CLAMP);
    ~TileReader();

    // Non-copyable (owns a TIFF handle)
    TileReader(const TileReader&)            = delete;
    TileReader& operator=(const TileReader&) = delete;

    // Image-level metadata
    const ImageInfo& info() const { return info_; }

    // Load one tile.
    //   tile_col, tile_row : zero-based index into the processing tile grid
    //   tile_size          : desired core size (e.g. 512)
    //   halo               : extra pixels to load on every side
    //
    // Returns a fully allocated Tile with buf_w = tile_size+2*halo, etc.
    // The function clips against image boundaries automatically.
    Tile read_tile(int tile_col, int tile_row, int tile_size, int halo) const;

    // Convenience: total number of tile columns / rows for a given tile_size.
    int num_tile_cols(int tile_size) const;
    int num_tile_rows(int tile_size) const;

private:
    // Read a rectangle [gx, gx+w) x [gy, gy+h) from the image into dst.
    // dst must point to w*h*channels bytes.
    // Out-of-bounds pixels are handled via the border mode.
    void read_region(int32_t gx, int32_t gy,
                     int32_t w,  int32_t h,
                     uint8_t* dst) const;

    // Read one native TIFF tile that contains pixel (px, py) and cache it.
    // Returns pointer into native_cache_.
    const uint8_t* fetch_native_tile(int32_t px, int32_t py) const;

    // Apply border mode for a single out-of-bounds query.
    void apply_border(int32_t& x, int32_t& y) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    ImageInfo             info_;
    BorderMode            border_;

    // Simple LRU cache of native TIFF tiles (keyed by (col, row)).
    mutable struct NativeCache {
        static constexpr int CAPACITY = 64;
        struct Entry {
            int32_t              col = -1, row = -1;
            std::vector<uint8_t> data;
        };
        Entry              slots[CAPACITY];
        int                next_evict = 0;

        const uint8_t* find(int32_t col, int32_t row) const;
        uint8_t*       insert(int32_t col, int32_t row, std::size_t bytes);
    } cache_;
};