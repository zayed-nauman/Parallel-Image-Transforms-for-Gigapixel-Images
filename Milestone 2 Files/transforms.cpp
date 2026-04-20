#include "transforms.h"
#include "overlap.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector> // FIX 1: Added missing header

// ─────────────────────────────────────────────────────────────────────────────
//  Bilinear sampling helper
// ─────────────────────────────────────────────────────────────────────────────
static void bilinear_sample(const Tile& t, double bx, double by, uint8_t* out) {
    int ch  = channels_of(t.fmt);
    int bw  = t.buf_w();
    int bh  = t.buf_h();

    // Use double consistently for precision (FIX 5)
    double bx_d = std::clamp(bx, 0.0, static_cast<double>(bw - 1));
    double by_d = std::clamp(by, 0.0, static_cast<double>(bh - 1));

    int x0 = static_cast<int>(std::floor(bx_d));
    int y0 = static_cast<int>(std::floor(by_d));
    
    // FIX 6: Relaxed boundary check to allow sampling at the very edge
    int x1 = std::min(x0 + 1, bw - 1);
    int y1 = std::min(y0 + 1, bh - 1);

    double fx = bx_d - static_cast<double>(x0);
    double fy = by_d - static_cast<double>(y0);

    const uint8_t* p00 = t.px(x0, y0);
    const uint8_t* p10 = t.px(x1, y0);
    const uint8_t* p01 = t.px(x0, y1);
    const uint8_t* p11 = t.px(x1, y1);

    for (int c = 0; c < ch; ++c) {
        double v = (1.0 - fy) * ((1.0 - fx) * p00[c] + fx * p10[c])
                 +       fy  * ((1.0 - fx) * p01[c] + fx * p11[c]);
        out[c] = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  IdentityTransform
// ─────────────────────────────────────────────────────────────────────────────
Tile IdentityTransform::apply(const Tile& in, const ImageInfo&) const {
    return overlap::strip_halo(in);
}

// ─────────────────────────────────────────────────────────────────────────────
//  CropTransform
// ─────────────────────────────────────────────────────────────────────────────
Tile CropTransform::apply(const Tile& in, const ImageInfo&) const {
    int32_t ix0 = std::max(in.global_x, x0);
    int32_t iy0 = std::max(in.global_y, y0);
    int32_t ix1 = std::min(in.global_x + in.core_w, x1);
    int32_t iy1 = std::min(in.global_y + in.core_h, y1);

    if (ix0 >= ix1 || iy0 >= iy1) {
        Tile empty; empty.core_w = 0; empty.core_h = 0; return empty;
    }

    Tile out;
    // FIX 2: Relative coordinates (Option A). The cropped image top-left is (0,0).
    out.global_x = ix0 - x0;
    out.global_y = iy0 - y0;
    out.core_w   = ix1 - ix0;
    out.core_h   = iy1 - iy0;
    out.halo     = 0;
    out.fmt      = in.fmt;
    out.allocate();

    int ch = channels_of(in.fmt);
    int row_bytes = out.core_w * ch;
    int src_cx_off = ix0 - in.global_x;
    int src_cy_off = iy0 - in.global_y;

    for (int row = 0; row < out.core_h; ++row) {
        // FIX 3: Safe pointer arithmetic using core_px() for each row
        const uint8_t* src = in.core_px(src_cx_off, src_cy_off + row);
        uint8_t* dst = out.data.data() + static_cast<std::size_t>(row) * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ResizeTransform
// ─────────────────────────────────────────────────────────────────────────────
Tile ResizeTransform::apply(const Tile& in, const ImageInfo& img) const {
    int ch = channels_of(in.fmt);

    int32_t out_x0 = static_cast<int32_t>(std::floor(in.global_x * scale_x));
    int32_t out_y0 = static_cast<int32_t>(std::floor(in.global_y * scale_y));
    int32_t out_x1 = static_cast<int32_t>(std::ceil((in.global_x + in.core_w) * scale_x));
    int32_t out_y1 = static_cast<int32_t>(std::ceil((in.global_y + in.core_h) * scale_y));

    Tile out;
    out.global_x = out_x0;
    out.global_y = out_y0;
    out.core_w   = out_x1 - out_x0;
    out.core_h   = out_y1 - out_y0;
    out.halo     = 0;
    out.fmt      = in.fmt;
    out.allocate();

    double inv_sx = 1.0 / static_cast<double>(scale_x);
    double inv_sy = 1.0 / static_cast<double>(scale_y);

    for (int oy = 0; oy < out.core_h; ++oy) {
        for (int ox = 0; ox < out.core_w; ++ox) {
            double gx = static_cast<double>(out.global_x + ox) + 0.5;
            double gy = static_cast<double>(out.global_y + oy) + 0.5;

            double gsx = gx * inv_sx - 0.5;
            double gsy = gy * inv_sy - 0.5;

            double bx = gsx - (static_cast<double>(in.global_x) - in.halo);
            double by = gsy - (static_cast<double>(in.global_y) - in.halo);

            uint8_t* dst = &out.data[(static_cast<size_t>(oy) * out.core_w + ox) * ch];
            bilinear_sample(in, bx, by, dst);
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  RotateTransform
// ─────────────────────────────────────────────────────────────────────────────
Tile RotateTransform::apply(const Tile& in, const ImageInfo& img) const {
    const double PI = 3.14159265358979323846;
    double rad = static_cast<double>(angle_deg) * PI / 180.0;
    double cos_a = std::cos(rad);
    double sin_a = std::sin(rad);

    // Pivot around the GLOBAL image center (2048x2048)
    double global_cx = (static_cast<double>(img.width) - 1.0) * 0.5;
    double global_cy = (static_cast<double>(img.height) - 1.0) * 0.5;

    Tile out;
    out.global_x = in.global_x; 
    out.global_y = in.global_y;
    out.core_w = in.core_w; 
    out.core_h = in.core_h;
    out.halo = 0; 
    out.fmt = in.fmt;
    out.allocate();

    int ch = channels_of(in.fmt);

    for (int oy = 0; oy < out.core_h; ++oy) {
        for (int ox = 0; ox < out.core_w; ++ox) {
            
            // MAP TO GLOBAL SPACE: This is the most important line.
            double absolute_gx = static_cast<double>(out.global_x) + static_cast<double>(ox);
            double absolute_gy = static_cast<double>(out.global_y) + static_cast<double>(oy);

            double dx = absolute_gx - global_cx;
            double dy = absolute_gy - global_cy;

            // Inverse rotation to find source
            double rsx = global_cx + (cos_a * dx + sin_a * dy);
            double rsy = global_cy + (-sin_a * dx + cos_a * dy);

            // Map global source back to local buffer (subtract global_x and halo)
            double bx = rsx - (static_cast<double>(in.global_x) - static_cast<double>(in.halo));
            double by = rsy - (static_cast<double>(in.global_y) - static_cast<double>(in.halo));

            uint8_t* dst = &out.data[(static_cast<size_t>(oy) * out.core_w + ox) * ch];

            // Boundary Check
            if (bx < 0 || bx >= in.buf_w() - 1 || by < 0 || by >= in.buf_h() - 1) {
                std::memset(dst, 0, ch);
            } else {
                bilinear_sample(in, bx, by, dst);
            }
        }
    }
    return out;
}
// ─────────────────────────────────────────────────────────────────────────────
//  BoxBlurTransform
// ─────────────────────────────────────────────────────────────────────────────
Tile BoxBlurTransform::apply(const Tile& in, const ImageInfo&) const {
    overlap::check_halo(in, radius);
    int ch = channels_of(in.fmt), bw = in.buf_w(), bh = in.buf_h();
    std::vector<uint8_t> tmp(static_cast<std::size_t>(bw) * bh * ch, 0);

    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            for (int c = 0; c < ch; ++c) {
                int sum = 0, cnt = 0;
                for (int k = -radius; k <= radius; ++k) {
                    sum += in.px(std::clamp(bx + k, 0, bw - 1), by)[c];
                    ++cnt;
                }
                // FIX 4: Rounded integer division
                tmp[(static_cast<std::size_t>(by) * bw + bx) * ch + c] = static_cast<uint8_t>((sum + cnt/2) / cnt);
            }
        }
    }

    Tile out;
    out.global_x = in.global_x; out.global_y = in.global_y;
    out.core_w = in.core_w; out.core_h = in.core_h;
    out.halo = 0; out.fmt = in.fmt;
    out.allocate();

    for (int cy = 0; cy < in.core_h; ++cy) {
        int by = cy + in.halo;
        for (int cx = 0; cx < in.core_w; ++cx) {
            int bx = cx + in.halo;
            uint8_t* dst = out.data.data() + (static_cast<std::size_t>(cy) * out.core_w + cx) * ch;
            for (int c = 0; c < ch; ++c) {
                int sum = 0, cnt = 0;
                for (int k = -radius; k <= radius; ++k) {
                    sum += tmp[(static_cast<std::size_t>(std::clamp(by + k, 0, bh - 1)) * bw + bx) * ch + c];
                    ++cnt;
                }
                // FIX 4: Rounded integer division
                dst[c] = static_cast<uint8_t>((sum + cnt/2) / cnt);
            }
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  TransformChain Implementation
// ─────────────────────────────────────────────────────────────────────────────

// FIX: Ensure the signature matches 'Tile TransformChain::apply(Tile, const ImageInfo&) const'
Tile TransformChain::apply(Tile tile, const ImageInfo& img) const {
    for (const auto& step : steps_) {
        tile = step->apply(tile, img);
        // If a crop or transform results in an empty tile, stop processing
        if (tile.core_w == 0 || tile.core_h == 0) return tile;
    }
    return tile;
}

// FIX: Ensure 'const' is present at the end
int TransformChain::max_halo() const {
    int m = 0;
    for (const auto& s : steps_) {
        int h = s->required_halo();
        if (h > m) m = h;
    }
    return m;
}

// FIX: Use 'unsigned int' specifically as requested by your MinGW linker error
void TransformChain::compute_output_size(unsigned int in_w, unsigned int in_h, unsigned int& out_w, unsigned int& out_h) const {
    out_w = in_w; 
    out_h = in_h;
    for (const auto& step : steps_) {
        unsigned int next_w, next_h;
        step->output_size(out_w, out_h, next_w, next_h);
        out_w = next_w;
        out_h = next_h;
    }
}