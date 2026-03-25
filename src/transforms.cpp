#include "transforms.h"
#include "overlap.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
//  Bilinear sampling helper (shared between resize & rotate)
//
//  bx, by are in buffer-space (i.e. 0 = top-left of the halo region).
//  Returns 0-filled if the coordinate falls outside the buffer.
// ─────────────────────────────────────────────────────────────────────────────

static void bilinear_sample(const Tile& t, double bx, double by, uint8_t* out) {
    int ch  = channels_of(t.fmt);
    int bw  = t.buf_w();
    int bh  = t.buf_h();

    // Clamp coordinates first, then compute indices/weights.
    // This avoids incorrect interpolation weights when bx/by are outside the buffer
    // due to float edge conditions near tile boundaries.
    double bx_d = std::clamp(bx, 0.0, static_cast<double>(bw - 1));
    double by_d = std::clamp(by, 0.0, static_cast<double>(bh - 1));

    int x0 = static_cast<int>(std::floor(bx_d));
    int y0 = static_cast<int>(std::floor(by_d));
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
                 +        fy  * ((1.0 - fx) * p01[c] + fx * p11[c]);
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
    // Intersection of tile core with the crop rectangle.
    int32_t ix0 = std::max(in.global_x, x0);
    int32_t iy0 = std::max(in.global_y, y0);
    int32_t ix1 = std::min(in.global_x + in.core_w, x1);
    int32_t iy1 = std::min(in.global_y + in.core_h, y1);

    if (ix0 >= ix1 || iy0 >= iy1) {
        // No overlap — return empty tile signalling "skip".
        Tile empty;
        empty.core_w = 0; empty.core_h = 0;
        return empty;
    }

    Tile out;
    out.global_x = ix0;
    out.global_y = iy0;
    out.core_w   = ix1 - ix0;
    out.core_h   = iy1 - iy0;
    out.halo     = 0;
    out.fmt      = in.fmt;
    out.allocate();

    int ch          = channels_of(in.fmt);
    int row_bytes   = out.core_w * ch;

    // Offset within the source's core region.
    int src_cx_off  = ix0 - in.global_x;
    int src_cy_off  = iy0 - in.global_y;

    for (int row = 0; row < out.core_h; ++row) {
        const uint8_t* src = in.core_px(src_cx_off, src_cy_off + row);
        uint8_t*       dst = out.data.data() + static_cast<std::size_t>(row) * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ResizeTransform  (bilinear, output-to-input reverse mapping)
// ─────────────────────────────────────────────────────────────────────────────

Tile ResizeTransform::apply(const Tile& in, const ImageInfo& img) const {
    int ch = channels_of(in.fmt);

    // Output image dimensions.
    int32_t out_img_w = static_cast<int32_t>(std::round(img.width  * scale_x));
    int32_t out_img_h = static_cast<int32_t>(std::round(img.height * scale_y));
    if (out_img_w < 1) out_img_w = 1;
    if (out_img_h < 1) out_img_h = 1;

    // This tile's output position and size.
    // global_x / global_y refer to the CORE region of the input tile.
    // The corresponding output tile starts at (global_x * scale_x, ...).
    int32_t out_x0 = static_cast<int32_t>(std::floor(in.global_x * scale_x));
    int32_t out_y0 = static_cast<int32_t>(std::floor(in.global_y * scale_y));
    int32_t out_x1 = static_cast<int32_t>(std::ceil((in.global_x + in.core_w) * scale_x));
    int32_t out_y1 = static_cast<int32_t>(std::ceil((in.global_y + in.core_h) * scale_y));

    // Clip to output image bounds.
    out_x0 = std::clamp(out_x0, 0, out_img_w);
    out_y0 = std::clamp(out_y0, 0, out_img_h);
    out_x1 = std::clamp(out_x1, 0, out_img_w);
    out_y1 = std::clamp(out_y1, 0, out_img_h);

    if (out_x0 >= out_x1 || out_y0 >= out_y1) {
        Tile empty; empty.core_w = 0; empty.core_h = 0; return empty;
    }

    Tile out;
    out.global_x = out_x0;
    out.global_y = out_y0;
    out.core_w   = out_x1 - out_x0;
    out.core_h   = out_y1 - out_y0;
    out.halo     = 0;
    out.fmt      = in.fmt;
    out.allocate();

    // Inverse-map: for each output pixel (ox, oy) find source pixel in global space,
    // then convert to buffer space.
    float inv_sx = 1.0f / scale_x;
    float inv_sy = 1.0f / scale_y;

    for (int oy = 0; oy < out.core_h; ++oy) {
        for (int ox = 0; ox < out.core_w; ++ox) {
            // Global output pixel
            float gox = out_x0 + ox;
            float goy = out_y0 + oy;

            // Corresponding global source pixel (centre of output pixel mapped back)
            float gsx = gox * inv_sx;
            float gsy = goy * inv_sy;

            // Convert to buffer space (halo included at offset `in.halo`)
            float bsx = gsx - (in.global_x - in.halo);
            float bsy = gsy - (in.global_y - in.halo);

            uint8_t* dst = out.data.data() +
                           (static_cast<std::size_t>(oy) * out.core_w + ox) * ch;
            bilinear_sample(in, static_cast<double>(bsx), static_cast<double>(bsy), dst);
        }
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  RotateTransform (counter-clockwise, around image centre, bilinear)
// ─────────────────────────────────────────────────────────────────────────────

Tile RotateTransform::apply(const Tile& in, const ImageInfo& img) const {
    const float PI  = 3.14159265358979f;
    double rad       = static_cast<double>(angle_deg) * static_cast<double>(PI) / 180.0;
    double cos_a     =  std::cos(rad);
    double sin_a     =  std::sin(rad);

    // Image centre in global pixel coordinates.
    double cx = (img.width  - 1) * 0.5;
    double cy = (img.height - 1) * 0.5;

    int ch = channels_of(in.fmt);

    // Output tile has the same core size and position as the input core.
    // (For a full pipeline you would compute the rotated bounding box; here
    //  we keep tiles at the same grid positions and fill from the rotated source.)
    Tile out;
    out.global_x = in.global_x;
    out.global_y = in.global_y;
    out.core_w   = in.core_w;
    out.core_h   = in.core_h;
    out.halo     = 0;
    out.fmt      = in.fmt;
    out.allocate();

    for (int oy = 0; oy < out.core_h; ++oy) {
        for (int ox = 0; ox < out.core_w; ++ox) {
            // Global output pixel (centre).
            double gox = static_cast<double>(in.global_x + ox);
            double goy = static_cast<double>(in.global_y + oy);

            // Translate to image centre.
            double dx = gox - cx;
            double dy = goy - cy;

            // Inverse rotation (rotate back by -angle to find source).
            // Counter-clockwise rotation: forward = (cos,-sin; sin,cos)
            // Inverse = transpose = (cos,sin; -sin,cos)
            double src_dx =  cos_a * dx + sin_a * dy;
            double src_dy = -sin_a * dx + cos_a * dy;

            // Global source coordinates.
            double gsx = cx + src_dx;
            double gsy = cy + src_dy;

            // Convert to buffer space.
            double bsx = gsx - (static_cast<double>(in.global_x) - static_cast<double>(in.halo));
            double bsy = gsy - (static_cast<double>(in.global_y) - static_cast<double>(in.halo));

            uint8_t* dst = out.data.data() +
                           (static_cast<std::size_t>(oy) * out.core_w + ox) * ch;
            // Always bilinear-sample. `bilinear_sample()` clamps indices to the
            // tile buffer bounds, preventing seam artefacts from float edge
            // conditions near tile boundaries.
            bilinear_sample(in, bsx, bsy, dst);
        }
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BoxBlurTransform  (separable H-pass then V-pass)
// ─────────────────────────────────────────────────────────────────────────────

Tile BoxBlurTransform::apply(const Tile& in, const ImageInfo&) const {
    overlap::check_halo(in, radius);

    int ch  = channels_of(in.fmt);
    int bw  = in.buf_w();
    int bh  = in.buf_h();
    int diam = 2 * radius + 1;

    // Intermediate buffer for the horizontal pass (same size as input buffer).
    std::vector<uint8_t> tmp(static_cast<std::size_t>(bw) * bh * ch, 0);

    // ── Horizontal pass ──────────────────────────────────────────────────────
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            for (int c = 0; c < ch; ++c) {
                int sum = 0;
                int cnt = 0;
                for (int k = -radius; k <= radius; ++k) {
                    int sx = std::clamp(bx + k, 0, bw - 1);
                    sum += in.px(sx, by)[c];
                    ++cnt;
                }
                tmp[(static_cast<std::size_t>(by) * bw + bx) * ch + c] =
                    static_cast<uint8_t>(sum / cnt);
            }
        }
    }

    // ── Vertical pass (reads from tmp, writes to output core) ────────────────
    Tile out;
    out.global_x = in.global_x;
    out.global_y = in.global_y;
    out.core_w   = in.core_w;
    out.core_h   = in.core_h;
    out.halo     = 0;
    out.fmt      = in.fmt;
    out.allocate();

    // Only write the core region (halo region discarded).
    for (int cy = 0; cy < in.core_h; ++cy) {
        int by = cy + in.halo;   // buffer row for this core row
        for (int cx = 0; cx < in.core_w; ++cx) {
            int bx = cx + in.halo;   // buffer col
            uint8_t* dst = out.data.data() +
                           (static_cast<std::size_t>(cy) * out.core_w + cx) * ch;

            for (int c = 0; c < ch; ++c) {
                int sum = 0;
                int cnt = 0;
                for (int k = -radius; k <= radius; ++k) {
                    int sy = std::clamp(by + k, 0, bh - 1);
                    sum += tmp[(static_cast<std::size_t>(sy) * bw + bx) * ch + c];
                    ++cnt;
                }
                dst[c] = static_cast<uint8_t>(sum / cnt);
            }
        }
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  TransformChain
// ─────────────────────────────────────────────────────────────────────────────

Tile TransformChain::apply(Tile tile, const ImageInfo& img) const {
    for (const auto& step : steps_) {
        tile = step->apply(tile, img);
        // If a step signals "skip this tile" by zeroing dimensions, propagate.
        if (tile.core_w == 0 || tile.core_h == 0) return tile;
    }
    return tile;
}

int TransformChain::max_halo() const {
    int m = 0;
    for (const auto& s : steps_)
        m = std::max(m, s->required_halo());
    return m;
}