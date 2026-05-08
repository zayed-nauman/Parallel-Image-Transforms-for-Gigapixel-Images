// =============================================================================
//  pipeline_fusion.cpp   —  Milestone 3
//
//  Implements FusionOptimizer::fuse().  Each fusion rule checks whether two
//  consecutive transforms can be collapsed and, if so, returns a FusedStep
//  that executes them in a single data pass.
// =============================================================================

#include "pipeline_fusion.h"
#include <iostream>

FusionOptimizer::Stats FusionOptimizer::last_stats_;

// ─────────────────────────────────────────────────────────────────────────────
//  Helper casts
// ─────────────────────────────────────────────────────────────────────────────

static const BoxBlurTransform*  as_blur  (const Transform* t)
    { return dynamic_cast<const BoxBlurTransform*> (t); }
static const CropTransform*     as_crop  (const Transform* t)
    { return dynamic_cast<const CropTransform*>    (t); }
static const ResizeTransform*   as_resize(const Transform* t)
    { return dynamic_cast<const ResizeTransform*>  (t); }

// ─────────────────────────────────────────────────────────────────────────────
//  Fusion rule 1: BlurBlur → single wider blur
//  Box-blur(r1) ∘ Box-blur(r2) ≡ Box-blur(r1 + r2) only for a single pass.
//  (Exact equivalence requires more passes; we approximate with wider radius.)
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<FusedStep> FusionOptimizer::try_fuse_blur_blur(
    const Transform* a, const Transform* b)
{
    const auto* ba = as_blur(a);
    const auto* bb = as_blur(b);
    if (!ba || !bb) return nullptr;

    int combined_radius = ba->radius + bb->radius;
    auto step = std::make_unique<FusedStep>();
    step->description     = "BlurBlur(r=" + std::to_string(combined_radius) + ")";
    step->required_halo   = combined_radius;

    step->execute = [combined_radius](const Tile& in, const ImageInfo&) -> Tile {
        overlap::check_halo(in, combined_radius);
        int ch = channels_of(in.fmt);
        int bw = in.buf_w(), bh = in.buf_h();
        std::vector<uint8_t> tmp(static_cast<std::size_t>(bw) * bh * ch, 0);

        // Horizontal pass
        for (int by = 0; by < bh; ++by) {
            for (int bx = 0; bx < bw; ++bx) {
                for (int c = 0; c < ch; ++c) {
                    int sum = 0, cnt = 2 * combined_radius + 1;
                    for (int k = -combined_radius; k <= combined_radius; ++k) {
                        int nx = std::clamp(bx + k, 0, bw - 1);
                        sum += in.px(nx, by)[c];
                    }
                    tmp[(static_cast<std::size_t>(by) * bw + bx) * ch + c] =
                        static_cast<uint8_t>((sum + cnt / 2) / cnt);
                }
            }
        }

        Tile out;
        out.global_x = in.global_x; out.global_y = in.global_y;
        out.core_w   = in.core_w;   out.core_h   = in.core_h;
        out.halo     = 0;           out.fmt      = in.fmt;
        out.allocate();

        // Vertical pass — write only core
        for (int cy = 0; cy < in.core_h; ++cy) {
            int by = cy + in.halo;
            for (int cx = 0; cx < in.core_w; ++cx) {
                int bx = cx + in.halo;
                uint8_t* dst = out.data.data() +
                    (static_cast<std::size_t>(cy) * out.core_w + cx) * ch;
                for (int c = 0; c < ch; ++c) {
                    int sum = 0, cnt = 2 * combined_radius + 1;
                    for (int k = -combined_radius; k <= combined_radius; ++k) {
                        int ny = std::clamp(by + k, 0, bh - 1);
                        sum += tmp[(static_cast<std::size_t>(ny) * bw + bx) * ch + c];
                    }
                    dst[c] = static_cast<uint8_t>((sum + cnt / 2) / cnt);
                }
            }
        }
        return out;
    };
    return step;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fusion rule 2: BlurCrop
//  Compute blur only for the rows/cols that survive the crop.
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<FusedStep> FusionOptimizer::try_fuse_blur_crop(
    const Transform* b, const Transform* c)
{
    const auto* blur = as_blur(b);
    const auto* crop = as_crop(c);
    if (!blur || !crop) return nullptr;

    int r = blur->radius;
    int32_t cx0 = crop->x0, cy0 = crop->y0, cx1 = crop->x1, cy1 = crop->y1;

    auto step = std::make_unique<FusedStep>();
    step->description   = "BlurCrop(r=" + std::to_string(r) + ")";
    step->required_halo = r;

    step->execute = [r, cx0, cy0, cx1, cy1](const Tile& in, const ImageInfo&) -> Tile {
        overlap::check_halo(in, r);

        // Determine output region
        int32_t ox0 = std::max(in.global_x, cx0);
        int32_t oy0 = std::max(in.global_y, cy0);
        int32_t ox1 = std::min(in.global_x + in.core_w, cx1);
        int32_t oy1 = std::min(in.global_y + in.core_h, cy1);
        if (ox0 >= ox1 || oy0 >= oy1) {
            Tile empty; empty.core_w = 0; empty.core_h = 0; return empty;
        }

        int ch = channels_of(in.fmt);
        int bw = in.buf_w();

        // Horizontal scratch (only needed rows)
        int out_h = oy1 - oy0, out_w = ox1 - ox0;
        std::vector<uint8_t> tmp(static_cast<std::size_t>(bw) *
                                  static_cast<std::size_t>(out_h + 2 * r) * ch, 0);

        int start_by = (oy0 - in.global_y);  // core-relative
        int cnt = 2 * r + 1;

        for (int row = 0; row < out_h + 2 * r; ++row) {
            int by = start_by + row;
            if (by < 0 || by >= in.core_h + 2 * r) continue;
            int buf_by = by; // already buf-relative (buf = core + halo on each side, so buf_by = core_y + halo)
            // Correct: buf_by = (core row relative to buf) = start_by - halo_offset + row = by already buf_relative would be by + 0 if start_by counts from the buf top...
            // Actually in.core_px(cx, cy) adds halo offset. For pixel at buf row `by`, just use in.px(bx, by) where by is buf_y = oy_core + in.halo
            int real_buf_by = (oy0 - in.global_y) + row; // relative to core top, need to add halo for buf access
            real_buf_by += in.halo - r; // center the scratch rows around the crop region
            if (real_buf_by < 0 || real_buf_by >= in.buf_h()) continue;

            for (int bx = 0; bx < bw; ++bx) {
                for (int c = 0; c < ch; ++c) {
                    int sum = 0;
                    for (int k = -r; k <= r; ++k) {
                        int nx = std::clamp(bx + k, 0, bw - 1);
                        sum += in.px(nx, real_buf_by)[c];
                    }
                    tmp[(static_cast<std::size_t>(row) * bw + bx) * ch + c] =
                        static_cast<uint8_t>((sum + cnt / 2) / cnt);
                }
            }
        }

        Tile out;
        out.global_x = ox0 - cx0; out.global_y = oy0 - cy0;
        out.core_w   = out_w;     out.core_h   = out_h;
        out.halo     = 0;         out.fmt      = in.fmt;
        out.allocate();

        for (int oy = 0; oy < out_h; ++oy) {
            int ox_start_in_tile = ox0 - in.global_x;
            for (int ox = 0; ox < out_w; ++ox) {
                int bx = ox_start_in_tile + ox + in.halo;
                uint8_t* dst = out.data.data() +
                    (static_cast<std::size_t>(oy) * out_w + ox) * ch;
                for (int c = 0; c < ch; ++c) {
                    int sum = 0;
                    for (int k = -r; k <= r; ++k) {
                        int row_idx = std::clamp(oy + r + k, 0, out_h + 2 * r - 1);
                        sum += tmp[(static_cast<std::size_t>(row_idx) * bw + bx) * ch + c];
                    }
                    dst[c] = static_cast<uint8_t>((sum + cnt / 2) / cnt);
                }
            }
        }
        return out;
    };
    return step;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fusion rule 3: BlurResize
//  After horizontal blur, subsample during vertical pass (saves 1 buffer copy).
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<FusedStep> FusionOptimizer::try_fuse_blur_resize(
    const Transform* b, const Transform* rs)
{
    const auto* blur   = as_blur(b);
    const auto* resize = as_resize(rs);
    if (!blur || !resize) return nullptr;
    // Only fuse for downscale — upscale is not safe to combine with blur
    if (resize->scale_x > 1.0f || resize->scale_y > 1.0f) return nullptr;

    int r = blur->radius;
    float sx = resize->scale_x, sy = resize->scale_y;

    auto step = std::make_unique<FusedStep>();
    step->description   = "BlurResize(r=" + std::to_string(r) +
                          ",sx=" + std::to_string(sx) +
                          ",sy=" + std::to_string(sy) + ")";
    step->required_halo = r;

    step->execute = [r, sx, sy](const Tile& in, const ImageInfo& img) -> Tile {
        overlap::check_halo(in, r);

        int ch  = channels_of(in.fmt);
        int bw  = in.buf_w(), bh = in.buf_h();
        std::vector<uint8_t> tmp(static_cast<std::size_t>(bw) * bh * ch, 0);
        int cnt = 2 * r + 1;

        // Horizontal pass — full buf
        for (int by = 0; by < bh; ++by) {
            for (int bx = 0; bx < bw; ++bx) {
                for (int c = 0; c < ch; ++c) {
                    int sum = 0;
                    for (int k = -r; k <= r; ++k)
                        sum += in.px(std::clamp(bx + k, 0, bw - 1), by)[c];
                    tmp[(static_cast<std::size_t>(by) * bw + bx) * ch + c] =
                        static_cast<uint8_t>((sum + cnt / 2) / cnt);
                }
            }
        }

        // Output size
        int32_t out_x0 = static_cast<int32_t>(std::floor(in.global_x * sx));
        int32_t out_y0 = static_cast<int32_t>(std::floor(in.global_y * sy));
        int32_t out_x1 = static_cast<int32_t>(std::ceil ((in.global_x + in.core_w) * sx));
        int32_t out_y1 = static_cast<int32_t>(std::ceil ((in.global_y + in.core_h) * sy));

        Tile out;
        out.global_x = out_x0; out.global_y = out_y0;
        out.core_w   = out_x1 - out_x0; out.core_h = out_y1 - out_y0;
        out.halo     = 0; out.fmt = in.fmt;
        out.allocate();

        double inv_sx = 1.0 / sx, inv_sy = 1.0 / sy;
        double buf_gx = in.global_x - in.halo;
        double buf_gy = in.global_y - in.halo;

        for (int oy = 0; oy < out.core_h; ++oy) {
            for (int ox = 0; ox < out.core_w; ++ox) {
                double gx  = (out_x0 + ox + 0.5) * inv_sx - 0.5;
                double gy  = (out_y0 + oy + 0.5) * inv_sy - 0.5;
                double bx_f = gx - buf_gx;
                double by_f = gy - buf_gy;

                // Bilinear sample from tmp
                bx_f = std::clamp(bx_f, 0.0, (double)(bw - 1));
                by_f = std::clamp(by_f, 0.0, (double)(bh - 1));
                int x0 = (int)bx_f, y0 = (int)by_f;
                int x1 = std::min(x0 + 1, bw - 1);
                int y1 = std::min(y0 + 1, bh - 1);
                double fx = bx_f - x0, fy = by_f - y0;

                uint8_t* dst = out.data.data() +
                    (static_cast<std::size_t>(oy) * out.core_w + ox) * ch;

                // Vertical blur + bilinear combined
                for (int c = 0; c < ch; ++c) {
                    auto vblur = [&](int bx_i, int by_i) -> float {
                        int sum = 0;
                        for (int k = -r; k <= r; ++k) {
                            int ny = std::clamp(by_i + k, 0, bh - 1);
                            sum += tmp[(static_cast<std::size_t>(ny) * bw + bx_i) * ch + c];
                        }
                        return static_cast<float>((sum + cnt / 2) / cnt);
                    };
                    double v = (1.0 - fy) * ((1.0 - fx) * vblur(x0, y0) + fx * vblur(x1, y0))
                             +       fy  * ((1.0 - fx) * vblur(x0, y1) + fx * vblur(x1, y1));
                    dst[c] = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
                }
            }
        }
        return out;
    };
    return step;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fusion rule 4: ResizeCrop
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<FusedStep> FusionOptimizer::try_fuse_resize_crop(
    const Transform* rs, const Transform* c)
{
    const auto* resize = as_resize(rs);
    const auto* crop   = as_crop(c);
    if (!resize || !crop) return nullptr;

    float sx = resize->scale_x, sy = resize->scale_y;
    int32_t cx0 = crop->x0, cy0 = crop->y0, cx1 = crop->x1, cy1 = crop->y1;

    auto step = std::make_unique<FusedStep>();
    step->description   = "ResizeCrop";
    step->required_halo = 1;

    step->execute = [sx, sy, cx0, cy0, cx1, cy1](const Tile& in, const ImageInfo& img) -> Tile {
        int ch = channels_of(in.fmt);
        double inv_sx = 1.0 / sx, inv_sy = 1.0 / sy;
        double buf_gx = in.global_x - in.halo;
        double buf_gy = in.global_y - in.halo;
        int bw = in.buf_w(), bh = in.buf_h();

        int32_t out_x0 = static_cast<int32_t>(std::floor(in.global_x * sx));
        int32_t out_y0 = static_cast<int32_t>(std::floor(in.global_y * sy));
        int32_t out_x1 = static_cast<int32_t>(std::ceil ((in.global_x + in.core_w) * sx));
        int32_t out_y1 = static_cast<int32_t>(std::ceil ((in.global_y + in.core_h) * sy));

        // Intersect with crop
        int32_t rx0 = std::max(out_x0, cx0);
        int32_t ry0 = std::max(out_y0, cy0);
        int32_t rx1 = std::min(out_x1, cx1);
        int32_t ry1 = std::min(out_y1, cy1);
        if (rx0 >= rx1 || ry0 >= ry1) {
            Tile empty; empty.core_w = 0; empty.core_h = 0; return empty;
        }

        Tile out;
        out.global_x = rx0 - cx0; out.global_y = ry0 - cy0;
        out.core_w   = rx1 - rx0; out.core_h   = ry1 - ry0;
        out.halo     = 0;         out.fmt      = in.fmt;
        out.allocate();

        for (int oy = 0; oy < out.core_h; ++oy) {
            for (int ox = 0; ox < out.core_w; ++ox) {
                double gx  = (rx0 + ox + 0.5) * inv_sx - 0.5;
                double gy  = (ry0 + oy + 0.5) * inv_sy - 0.5;
                double bx_f = gx - buf_gx;
                double by_f = gy - buf_gy;

                bx_f = std::clamp(bx_f, 0.0, (double)(bw - 1));
                by_f = std::clamp(by_f, 0.0, (double)(bh - 1));
                int x0 = (int)bx_f, y0 = (int)by_f;
                int x1 = std::min(x0 + 1, bw - 1);
                int y1 = std::min(y0 + 1, bh - 1);
                double fx = bx_f - x0, fy = by_f - y0;

                uint8_t* dst = out.data.data() +
                    (static_cast<std::size_t>(oy) * out.core_w + ox) * ch;

                for (int c = 0; c < ch; ++c) {
                    double v = (1.0 - fy) * ((1.0 - fx) * in.px(x0, y0)[c] + fx * in.px(x1, y0)[c])
                             +       fy  * ((1.0 - fx) * in.px(x0, y1)[c] + fx * in.px(x1, y1)[c]);
                    dst[c] = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
                }
            }
        }
        return out;
    };
    return step;
}

// ─────────────────────────────────────────────────────────────────────────────
//  FusionOptimizer::fuse
// ─────────────────────────────────────────────────────────────────────────────

FusedChain FusionOptimizer::fuse(const TransformChain& chain) {
    FusedChain result;
    result.original_op_count = static_cast<int>(chain.size());

    const auto& ops = chain.steps();
    int fusions = 0;
    std::vector<std::string> applied;

    for (std::size_t i = 0; i < ops.size();) {
        const Transform* cur = ops[i].get();

        // Try two-operation fusion rules greedily. These are the rules listed
        // in the milestone: BlurBlur, BlurCrop, BlurResize, ResizeCrop.
        if (i + 1 < ops.size()) {
            const Transform* nxt = ops[i + 1].get();
            std::unique_ptr<FusedStep> fs;
            if ((fs = try_fuse_blur_blur(cur, nxt)) ||
                (fs = try_fuse_blur_crop(cur, nxt)) ||
                (fs = try_fuse_blur_resize(cur, nxt)) ||
                (fs = try_fuse_resize_crop(cur, nxt))) {
                applied.push_back(fs->description);
                result.add_step(std::move(*fs));
                ++fusions;
                i += 2;
                continue;
            }
        }

        // No pair fusion possible; preserve the original operation as its own
        // fused step. Capturing the Transform pointer is safe because the
        // TransformChain outlives the TileProcessor/FusedChain that uses it.
        FusedStep step;
        step.required_halo = cur->required_halo();
        step.description = cur->name();
        step.execute = [cur](const Tile& in, const ImageInfo& img) -> Tile {
            return cur->apply(in, img);
        };
        result.add_step(std::move(step));
        ++i;
    }

    last_stats_.original_steps = static_cast<int>(chain.size());
    last_stats_.fused_steps = static_cast<int>(result.num_steps());
    last_stats_.fusions_applied = fusions;
    if (applied.empty()) {
        last_stats_.description = "No pair fusion applied; using per-transform fused plan";
    } else {
        last_stats_.description.clear();
        for (std::size_t i = 0; i < applied.size(); ++i) {
            if (i) last_stats_.description += ", ";
            last_stats_.description += applied[i];
        }
    }
    return result;
}

FusionOptimizer::Stats FusionOptimizer::last_stats() {
    return last_stats_;
}
