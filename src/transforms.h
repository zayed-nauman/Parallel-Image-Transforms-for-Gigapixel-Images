#pragma once
#include "common.h"
#include <cmath>
#include <functional>
#include <memory>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
//  Transform — base class for all per-tile operations
// ─────────────────────────────────────────────────────────────────────────────

struct Transform {
    virtual ~Transform() = default;

    virtual Tile apply(const Tile& in, const ImageInfo& img) const = 0;
    virtual std::string name() const = 0;
    virtual int required_halo() const { return 0; }

    virtual void output_size(uint32_t in_w, uint32_t in_h,
                             uint32_t& out_w, uint32_t& out_h) const {
        out_w = in_w;
        out_h = in_h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Concrete transforms
// ─────────────────────────────────────────────────────────────────────────────

struct IdentityTransform : Transform {
    Tile        apply(const Tile& in, const ImageInfo&) const override;
    std::string name()  const override { return "identity"; }
};

struct CropTransform : Transform {
    int32_t x0, y0, x1, y1;

    CropTransform(int32_t x0, int32_t y0, int32_t x1, int32_t y1)
        : x0(x0), y0(y0), x1(x1), y1(y1) {}

    void output_size(uint32_t, uint32_t,
                     uint32_t& out_w, uint32_t& out_h) const override {
        out_w = static_cast<uint32_t>(std::max(0, x1 - x0));
        out_h = static_cast<uint32_t>(std::max(0, y1 - y0));
    }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "crop"; }
};

struct ResizeTransform : Transform {
    float scale_x, scale_y;

    ResizeTransform(float sx, float sy) : scale_x(sx), scale_y(sy) {}

    int required_halo() const override { return 1; }

    void output_size(uint32_t in_w, uint32_t in_h,
                     uint32_t& out_w, uint32_t& out_h) const override {
        out_w = static_cast<uint32_t>(std::max(1, static_cast<int>(
                    std::round(static_cast<float>(in_w) * scale_x))));
        out_h = static_cast<uint32_t>(std::max(1, static_cast<int>(
                    std::round(static_cast<float>(in_h) * scale_y))));
    }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "resize"; }
};

struct RotateTransform : Transform {
    float angle_deg;

    explicit RotateTransform(float deg) : angle_deg(deg) {}

    // Static fallback — returns half the diagonal of a 4096×4096 image (2896px).
    // Safe upper bound when image dimensions are not yet known.
    // Always use required_halo_for_image() when ImageInfo is available.
    int required_halo() const override {
        return 2896;
    }

    // Compute the exact halo needed for this rotation angle on this image.
    //
    // Why the fixed 256 was wrong:
    //   For a 90° rotation on a 2048×2048 image, the corner tile (0,0) needs
    //   source pixels up to 1535px away — halo=256 covers only ~15° at this size.
    //
    // Formula: the worst-case source pixel for any output pixel in any tile is
    // bounded by the half-diagonal of the image (the furthest any rotated point
    // can travel from its original position is the full diagonal distance from
    // the image centre to a corner).
    //   halo = ceil(sqrt((W/2)² + (H/2)²))
    //
    // This is angle-independent and always safe: a point exactly at the corner
    // is at distance half-diagonal from the centre, so after any rotation its
    // source is at most half-diagonal away from any tile boundary.
    int required_halo_for_image(uint32_t img_w, uint32_t img_h) const {
        double half_diag = std::sqrt(
            (static_cast<double>(img_w) * 0.5) * (static_cast<double>(img_w) * 0.5) +
            (static_cast<double>(img_h) * 0.5) * (static_cast<double>(img_h) * 0.5));
        return static_cast<int>(std::ceil(half_diag));
    }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "rotate"; }
};

struct BoxBlurTransform : Transform {
    int radius;

    explicit BoxBlurTransform(int r) : radius(r) {}

    int required_halo() const override { return radius; }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "box_blur(r=" + std::to_string(radius) + ")"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  TransformChain
// ─────────────────────────────────────────────────────────────────────────────

class TransformChain {
public:
    void add(std::unique_ptr<Transform> t) { steps_.push_back(std::move(t)); }

    // Apply all steps in order.
    Tile apply(Tile tile, const ImageInfo& img) const;

    // Apply only steps from index `start_idx` onward.
    // Used by the GPU path to skip transforms already executed on the GPU
    // (e.g. blur), then run the remaining ones (rotate, resize, crop) on CPU.
    Tile apply_from(std::size_t start_idx, Tile tile, const ImageInfo& img) const;

    // Maximum required_halo() across all steps.
    // Used as a fallback when image dimensions are not yet known.
    int max_halo() const;

    // Image-aware halo: like max_halo() but asks each transform for its
    // halo given the actual image dimensions. Use this everywhere ImageInfo
    // is available — it returns the correct large halo for rotation instead
    // of the conservative static fallback.
    int max_halo_for_image(uint32_t img_w, uint32_t img_h) const;

    // Blur-only halo: the maximum BoxBlurTransform::radius in the chain.
    // Returns 0 if there is no blur step.
    //
    // THE GPU PATH MUST USE THIS, NOT max_halo().
    // Geometric transforms (rotate, resize) have required_halo() > 0 for
    // pixel context, not for a blur kernel radius. If max_halo() is used
    // as kern_radius, the GPU will run a meaningless box blur on rotate/resize
    // tiles (e.g. a 256-wide blur on every rotate tile), corrupting output.
    int blur_halo() const;

    // Index of the first step that is NOT a BoxBlurTransform.
    // Returns steps_.size() if every step is a blur.
    // Used by the GPU path to know where to resume apply_from() after
    // the GPU blur completes.
    std::size_t first_non_blur_step() const;

    std::size_t size() const { return steps_.size(); }

    // Milestone 3: expose read-only transform list so FusionOptimizer can
    // build a real fused execution plan instead of wrapping the whole chain.
    const std::vector<std::unique_ptr<Transform>>& steps() const { return steps_; }

    void compute_output_size(uint32_t in_w, uint32_t in_h,
                             uint32_t& out_w, uint32_t& out_h) const;

private:
    std::vector<std::unique_ptr<Transform>> steps_;
};