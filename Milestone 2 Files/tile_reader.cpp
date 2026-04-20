#include "tile_reader.h"
#include <tiffio.h>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <vector>

struct TileReader::Impl {
    TIFF*      tif = nullptr;
    std::mutex mu;
    ~Impl() { if (tif) { TIFFClose(tif); tif = nullptr; } }
};

const uint8_t* TileReader::NativeCache::find(int32_t col, int32_t row) const {
    for (const auto& e : slots)
        if (e.col == col && e.row == row && !e.data.empty())
            return e.data.data();
    return nullptr;
}

uint8_t* TileReader::NativeCache::insert(int32_t col, int32_t row, std::size_t bytes) {
    Entry& e   = slots[next_evict % CAPACITY];
    next_evict = (next_evict + 1) % CAPACITY;
    e.col = col; e.row = row;
    e.data.resize(bytes);
    return e.data.data();
}

TileReader::TileReader(const std::string& path, BorderMode border)
    : impl_(std::make_unique<Impl>()), border_(border)
{
    TIFFSetWarningHandler(nullptr);
    impl_->tif = TIFFOpen(path.c_str(), "r");
    M1_CHECK(impl_->tif, "Cannot open TIFF: " + path);
    TIFF* tif = impl_->tif;

    uint32_t w = 0, h = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,  &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    M1_CHECK(w > 0 && h > 0, "TIFF has zero dimensions: " + path);
    info_.width  = w;
    info_.height = h;

    uint16_t spp = 1, bps = 8;
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE,   &bps);
    M1_CHECK(bps == 8, "Only 8-bit-per-sample TIFFs are supported.");

    if      (spp == 1) info_.fmt = PixelFormat::GRAY8;
    else if (spp == 3) info_.fmt = PixelFormat::RGB8;
    else if (spp == 4) info_.fmt = PixelFormat::RGBA8;
    else M1_CHECK(false, "Unsupported samples-per-pixel: " + std::to_string(spp));

    if (TIFFIsTiled(tif)) {
        uint32_t tw = 0, th = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH,  &tw);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &th);
        info_.tile_width  = tw;
        info_.tile_height = th;
    } else {
        info_.tile_width = w;
        uint32_t rps = 1;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rps);
        info_.tile_height = rps;
    }
}

TileReader::~TileReader() = default;

int TileReader::num_tile_cols(int tile_size) const {
    return (static_cast<int>(info_.width)  + tile_size - 1) / tile_size;
}
int TileReader::num_tile_rows(int tile_size) const {
    return (static_cast<int>(info_.height) + tile_size - 1) / tile_size;
}

void TileReader::apply_border(int32_t& x, int32_t& y) const {
    int32_t W = static_cast<int32_t>(info_.width);
    int32_t H = static_cast<int32_t>(info_.height);
    switch (border_) {
        case BorderMode::ZERO: break;
        case BorderMode::CLAMP:
            x = std::clamp(x, 0, W - 1);
            y = std::clamp(y, 0, H - 1);
            break;
        case BorderMode::REFLECT: {
            auto refl = [](int32_t v, int32_t N) -> int32_t {
                if (N == 1) return 0;
                v = std::abs(v) % (2 * (N - 1));
                if (v >= N) v = 2 * (N - 1) - v;
                return v;
            };
            x = refl(x, W);
            y = refl(y, H);
            break;
        }
    }
}

const uint8_t* TileReader::fetch_native_tile(int32_t px, int32_t py) const {
    TIFF*   tif = impl_->tif;
    int32_t tw  = static_cast<int32_t>(info_.tile_width);
    int32_t th  = static_cast<int32_t>(info_.tile_height);
    int32_t col = px / tw;
    int32_t row = py / th;
    int     ch  = channels_of(info_.fmt);

    // NOTE on thread-safety:
    // `read_region()` is executed by multiple worker threads concurrently.
    // The original shared cache used for native TIFF tiles was not fully
    // synchronized (lookup/eviction + returned pointer could race), which can
    // corrupt halo pixels and cause tile-boundary seams for transforms that
    // use interpolation (rotate/resize).
    //
    // Fix: use a thread-local cache so each worker has independent cache
    // storage. libtiff itself is still not thread-safe, so we keep the
    // TIFFReadTile() mutex.
    thread_local NativeCache tls_cache;

    const uint8_t* cached = tls_cache.find(col, row);
    if (cached) return cached;

    std::size_t bytes = static_cast<std::size_t>(tw) * th * ch;
    uint8_t*    dst   = tls_cache.insert(col, row, bytes);
    std::memset(dst, 0, bytes);

    std::lock_guard<std::mutex> lock(impl_->mu);
    TIFFReadTile(tif, dst,
                 static_cast<uint32_t>(col * tw),
                 static_cast<uint32_t>(row * th),
                 0, 0);
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
//  read_region
//
//  For stripped TIFFs: read one scanline at a time under a mutex (libtiff is
//  not thread-safe). No cache. Each row is read fresh — simple and correct.
//
//  For tiled TIFFs: pixel-by-pixel with the slot cache.
// ─────────────────────────────────────────────────────────────────────────────

void TileReader::read_region(int32_t gx, int32_t gy,
                             int32_t w,  int32_t h,
                             uint8_t* dst) const
{
    int     ch  = channels_of(info_.fmt);
    int32_t IW  = static_cast<int32_t>(info_.width);
    int32_t IH  = static_cast<int32_t>(info_.height);

    if (!TIFFIsTiled(impl_->tif)) {
        // ── Stripped TIFF ────────────────────────────────────────────────────
        // One full-width scanline buffer per read_region call (not shared).
        int32_t tw = static_cast<int32_t>(info_.tile_width);  // == IW
        std::vector<uint8_t> scanline(static_cast<std::size_t>(tw) * ch);

        for (int32_t dy = 0; dy < h; ++dy) {
            int32_t sy       = gy + dy;
            uint8_t* dst_row = dst + static_cast<std::size_t>(dy) * w * ch;

            // Map sy to a valid image row.
            int32_t read_sy = sy;
            if (sy < 0 || sy >= IH) {
                if (border_ == BorderMode::ZERO) {
                    std::memset(dst_row, 0, static_cast<std::size_t>(w) * ch);
                    continue;
                }
                int32_t dummy_x = 0;
                apply_border(dummy_x, read_sy);
            }

            // Read the scanline (serialised — libtiff is not thread-safe).
            {
                std::lock_guard<std::mutex> lock(impl_->mu);
                TIFFReadScanline(impl_->tif, scanline.data(),
                                 static_cast<uint32_t>(read_sy), 0);
            }

            // Copy each output pixel from the scanline, handling x borders.
            for (int32_t dx = 0; dx < w; ++dx) {
                int32_t sx      = gx + dx;
                uint8_t* out_px = dst_row + static_cast<std::size_t>(dx) * ch;

                if (sx < 0 || sx >= IW) {
                    if (border_ == BorderMode::ZERO) {
                        std::memset(out_px, 0, ch);
                    } else {
                        int32_t mapped_x = sx;
                        int32_t mapped_y = read_sy;
                        apply_border(mapped_x, mapped_y);
                        if (mapped_y == read_sy) {
                            // Same row already loaded — just offset x.
                            std::memcpy(out_px,
                                        scanline.data() +
                                        static_cast<std::size_t>(mapped_x) * ch,
                                        ch);
                        } else {
                            // Different row needed (corner pixel reflect).
                            std::vector<uint8_t> tmp(tw * ch);
                            {
                                std::lock_guard<std::mutex> lock(impl_->mu);
                                TIFFReadScanline(impl_->tif, tmp.data(),
                                                 static_cast<uint32_t>(mapped_y), 0);
                            }
                            std::memcpy(out_px,
                                        tmp.data() +
                                        static_cast<std::size_t>(mapped_x) * ch,
                                        ch);
                        }
                    }
                } else {
                    // Normal in-bounds pixel.
                    std::memcpy(out_px,
                                scanline.data() + static_cast<std::size_t>(sx) * ch,
                                ch);
                }
            }
        }
        return;
    }

    // ── Tiled TIFF ───────────────────────────────────────────────────────────
    int32_t tw = static_cast<int32_t>(info_.tile_width);
    int32_t th = static_cast<int32_t>(info_.tile_height);

    for (int32_t dy = 0; dy < h; ++dy) {
        for (int32_t dx = 0; dx < w; ++dx) {
            int32_t sx = gx + dx;
            int32_t sy = gy + dy;
            uint8_t* out_px = dst + (static_cast<std::size_t>(dy) * w + dx) * ch;

            bool oob = (sx < 0 || sx >= IW || sy < 0 || sy >= IH);
            if (oob) {
                if (border_ == BorderMode::ZERO) { std::memset(out_px, 0, ch); continue; }
                apply_border(sx, sy);
            }
            const uint8_t* ntile = fetch_native_tile(sx, sy);
            int32_t lx = sx % tw, ly = sy % th;
            std::memcpy(out_px,
                        ntile + (static_cast<std::size_t>(ly) * tw + lx) * ch,
                        ch);
        }
    }
}

Tile TileReader::read_tile(int tile_col, int tile_row,
                           int tile_size, int halo) const
{
    M1_CHECK(tile_size > 0, "tile_size must be positive");
    M1_CHECK(halo >= 0,     "halo must be non-negative");

    int32_t core_x = tile_col * tile_size;
    int32_t core_y = tile_row * tile_size;
    int32_t core_w = std::min(tile_size, static_cast<int>(info_.width)  - core_x);
    int32_t core_h = std::min(tile_size, static_cast<int>(info_.height) - core_y);

    M1_CHECK(core_w > 0 && core_h > 0,
             "Tile (" + std::to_string(tile_col) + "," +
             std::to_string(tile_row) + ") outside image bounds.");

    Tile t;
    t.global_x = core_x;
    t.global_y = core_y;
    t.core_w   = core_w;
    t.core_h   = core_h;
    t.halo     = halo;
    t.fmt      = info_.fmt;
    t.allocate();

    read_region(core_x - halo, core_y - halo, t.buf_w(), t.buf_h(), t.data.data());
    return t;
}