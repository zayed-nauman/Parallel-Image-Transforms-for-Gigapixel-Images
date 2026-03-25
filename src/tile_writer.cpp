#include "tile_writer.h"
#include <tiffio.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Pimpl
// ─────────────────────────────────────────────────────────────────────────────

struct TileWriter::Impl {
    TIFF*    tif       = nullptr;
    uint32_t img_w     = 0;
    uint32_t img_h     = 0;
    uint32_t tile_size = 256;
    int      channels  = 3;

    // Full image scanline buffer — we accumulate all tiles here then flush.
    // This guarantees correct output regardless of tile arrival order,
    // and avoids libtiff's out-of-order write restrictions.
    std::vector<uint8_t> scanlines;   // img_w * img_h * channels

    ~Impl() { if (tif) { TIFFClose(tif); tif = nullptr; } }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Constructor
// ─────────────────────────────────────────────────────────────────────────────

TileWriter::TileWriter(const std::string& path,
                       uint32_t   width,
                       uint32_t   height,
                       PixelFormat fmt,
                       uint32_t   tile_size)
    : impl_(std::make_unique<Impl>())
{
    TIFFSetWarningHandler(nullptr);

    uint64_t bytes = static_cast<uint64_t>(width) * height * channels_of(fmt);
    const char* mode = (bytes > (uint64_t)3 * 1024 * 1024 * 1024) ? "w8" : "w";

    impl_->tif = TIFFOpen(path.c_str(), mode);
    M1_CHECK(impl_->tif, "Cannot create output TIFF: " + path);

    int ch = channels_of(fmt);
    impl_->img_w     = width;
    impl_->img_h     = height;
    impl_->tile_size = tile_size;
    impl_->channels  = ch;

    // Allocate the full-image accumulation buffer (zero-initialised).
    impl_->scanlines.assign(static_cast<std::size_t>(width) * height * ch, 0);

    // Set TIFF tags — we will write scanlines (not tiles) at close() time.
    TIFF* tif = impl_->tif;
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   8);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, ch);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,
                 ch == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,     COMPRESSION_NONE);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,    1);
    TIFFSetField(tif, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);

    if (ch == 4) {
        uint16_t extra = EXTRASAMPLE_UNASSALPHA;
        TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, 1, &extra);
    }
}

TileWriter::~TileWriter() {
    if (!closed_) try { close(); } catch (...) {}
}

// ─────────────────────────────────────────────────────────────────────────────
//  write_tile  — copies pixels into the in-memory accumulation buffer.
//  Thread-safe. No libtiff calls here; actual disk write happens in close().
// ─────────────────────────────────────────────────────────────────────────────

void TileWriter::write_tile(const Tile& tile) {
    M1_CHECK(tile.halo == 0,    "write_tile called with halo != 0.");
    M1_CHECK(tile.core_w > 0 && tile.core_h > 0, "write_tile: empty tile.");

    int      ch  = impl_->channels;
    uint32_t img_w = impl_->img_w;

    M1_CHECK(ch == channels_of(tile.fmt), "write_tile: channel count mismatch.");
    M1_CHECK(tile.global_x + tile.core_w <= (int32_t)impl_->img_w &&
             tile.global_y + tile.core_h <= (int32_t)impl_->img_h,
             "write_tile: tile extends beyond image bounds.");

    std::lock_guard<std::mutex> lock(mu_);

    for (int32_t row = 0; row < tile.core_h; ++row) {
        // Source: row in the tile's packed data (halo=0, so data is contiguous).
        const uint8_t* src = tile.data.data() +
            static_cast<std::size_t>(row) * tile.core_w * ch;

        // Destination: correct position in the full-image scanline buffer.
        uint8_t* dst = impl_->scanlines.data() +
            (static_cast<std::size_t>(tile.global_y + row) * img_w +
             tile.global_x) * ch;

        std::memcpy(dst, src, static_cast<std::size_t>(tile.core_w) * ch);
    }

    ++tiles_written_;
}

// ─────────────────────────────────────────────────────────────────────────────
//  close  — flush the accumulation buffer to disk row by row, then close.
// ─────────────────────────────────────────────────────────────────────────────

void TileWriter::close() {
    if (closed_) return;

    TIFF*    tif   = impl_->tif;
    uint32_t img_w = impl_->img_w;
    uint32_t img_h = impl_->img_h;
    int      ch    = impl_->channels;

    if (tif && !impl_->scanlines.empty()) {
        for (uint32_t row = 0; row < img_h; ++row) {
            uint8_t* row_ptr = impl_->scanlines.data() +
                static_cast<std::size_t>(row) * img_w * ch;
            tmsize_t ret = TIFFWriteScanline(tif, row_ptr,
                                             static_cast<uint32_t>(row), 0);
            M1_CHECK(ret != static_cast<tmsize_t>(-1),
                     "TIFFWriteScanline failed at row " + std::to_string(row));
        }
        TIFFWriteDirectory(tif);
        TIFFClose(tif);
        impl_->tif = nullptr;
    }

    closed_ = true;
}