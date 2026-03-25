#include "overlap.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace overlap {

// ─────────────────────────────────────────────────────────────────────────────
//  check_halo
// ─────────────────────────────────────────────────────────────────────────────

void check_halo(const Tile& tile, int kernel_radius) {
    M1_CHECK(tile.halo >= kernel_radius,
             "Tile halo (" + std::to_string(tile.halo) +
             ") is smaller than kernel radius (" +
             std::to_string(kernel_radius) + "). "
             "Re-read the tile with a larger halo.");
}

// ─────────────────────────────────────────────────────────────────────────────
//  strip_halo
// ─────────────────────────────────────────────────────────────────────────────

Tile strip_halo(const Tile& src) {
    if (src.halo == 0) return src;  // nothing to strip

    Tile dst;
    dst.global_x = src.global_x;
    dst.global_y = src.global_y;
    dst.core_w   = src.core_w;
    dst.core_h   = src.core_h;
    dst.halo     = 0;
    dst.fmt      = src.fmt;
    dst.allocate();   // buf_w = core_w, buf_h = core_h

    int ch = channels_of(src.fmt);
    int row_bytes = src.core_w * ch;

    for (int row = 0; row < src.core_h; ++row) {
        const uint8_t* src_row = src.core_px(0, row);     // halo-offset applied
        uint8_t*       dst_row = dst.data.data() + static_cast<std::size_t>(row) * row_bytes;
        std::memcpy(dst_row, src_row, row_bytes);
    }

    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
//  fill_halo
//  Fills the four halo bands using the chosen border mode, reading from the
//  CORE pixels that are already present in the tile buffer.
// ─────────────────────────────────────────────────────────────────────────────

void fill_halo(Tile& tile, BorderMode mode) {
    if (tile.halo == 0) return;

    int ch   = channels_of(tile.fmt);
    int bw   = tile.buf_w();
    int bh   = tile.buf_h();
    int h    = tile.halo;
    int cw   = tile.core_w;
    int coh  = tile.core_h;

    // Helper lambdas that return a pointer to the nearest CORE pixel for a
    // given BUFFER coordinate (bx, by), after applying the border mode.
    auto clamp_core = [&](int bx, int by) -> const uint8_t* {
        // Convert to core coords
        int cx = bx - h;
        int cy = by - h;

        switch (mode) {
            case BorderMode::ZERO:
                // The caller should zero the halo first (tile.allocate() does this).
                return nullptr;
            case BorderMode::CLAMP:
                cx = std::clamp(cx, 0, cw  - 1);
                cy = std::clamp(cy, 0, coh - 1);
                break;
            case BorderMode::REFLECT: {
                auto refl = [](int v, int N) {
                    if (N == 1) return 0;
                    v = std::abs(v) % (2 * (N - 1));
                    if (v >= N) v = 2 * (N - 1) - v;
                    return v;
                };
                cx = refl(cx, cw);
                cy = refl(cy, coh);
                break;
            }
        }
        return tile.core_px(cx, cy);
    };

    // Iterate over every pixel in the four halo bands (top, bottom, left, right).
    // We iterate over the full buffer and only fill pixels outside the core.
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            // Is this pixel in the core region?
            bool in_core = (bx >= h && bx < h + cw &&
                            by >= h && by < h + coh);
            if (in_core) continue;

            uint8_t* dst = tile.px(bx, by);

            if (mode == BorderMode::ZERO) {
                std::memset(dst, 0, ch);
                continue;
            }

            const uint8_t* src = clamp_core(bx, by);
            if (src) std::memcpy(dst, src, ch);
            else     std::memset(dst, 0, ch);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  debug_visualise_halo
// ─────────────────────────────────────────────────────────────────────────────

void debug_visualise_halo(const Tile& tile, uint8_t* dst_rgba) {
    int bw = tile.buf_w();
    int bh = tile.buf_h();
    int h  = tile.halo;
    int cw = tile.core_w;
    int ch_c = tile.core_h;
    int ch = channels_of(tile.fmt);

    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            bool in_core = (bx >= h && bx < h + cw &&
                            by >= h && by < h + ch_c);

            const uint8_t* src = tile.px(bx, by);
            uint8_t* dst       = dst_rgba + (static_cast<std::size_t>(by) * bw + bx) * 4;

            if (in_core) {
                // Copy pixel, convert to RGBA.
                if      (ch == 1) { dst[0]=dst[1]=dst[2]=src[0]; dst[3]=255; }
                else if (ch == 3) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=255; }
                else              { std::memcpy(dst, src, 4); }
            } else {
                // Tint halo red so it's visually distinct.
                uint8_t luma = (ch >= 1) ? src[0] : 128;
                dst[0] = std::min(255, (int)luma + 80);  // red boost
                dst[1] = luma / 3;
                dst[2] = luma / 3;
                dst[3] = 200;
            }
        }
    }
}

} // namespace overlap