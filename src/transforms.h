#pragma once
#include "common.h"
#include <cmath>
#include <functional>
#include <memory>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
//  Transform — base class for all per-tile operations
//
//  Each Transform receives a Tile with its full buffer (core + halo) and
//  returns a new Tile.  All operations work in global image coordinate space;
//  the Tile carries global_x/global_y so every transform can resolve
//  coordinates relative to the full image.
// ─────────────────────────────────────────────────────────────────────────────

struct Transform {
    virtual ~Transform() = default;

    // Apply the transform to `in` and return the result.
    virtual Tile apply(const Tile& in, const ImageInfo& img) const = 0;

    // Human-readable name for logging.
    virtual std::string name() const = 0;

    // Minimum halo this transform needs from the reader.
    virtual int required_halo() const { return 0; }

    // Compute the output image dimensions this transform produces.
    // Default: same size as input (identity, rotate, blur).
    // Override for transforms that change image dimensions (crop, resize).
    virtual void output_size(uint32_t in_w, uint32_t in_h,
                             uint32_t& out_w, uint32_t& out_h) const {
        out_w = in_w;
        out_h = in_h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Concrete transforms
// ─────────────────────────────────────────────────────────────────────────────

// Identity — copies core pixels unchanged.
struct IdentityTransform : Transform {
    Tile        apply(const Tile& in, const ImageInfo&) const override;
    std::string name()  const override { return "identity"; }
};

// ── Crop ─────────────────────────────────────────────────────────────────────
struct CropTransform : Transform {
    int32_t x0, y0, x1, y1;   // crop rect in global image coords (exclusive end)

    CropTransform(int32_t x0, int32_t y0, int32_t x1, int32_t y1)
        : x0(x0), y0(y0), x1(x1), y1(y1) {}

    // Output image is exactly (x1-x0) x (y1-y0) pixels.
    void output_size(uint32_t /*in_w*/, uint32_t /*in_h*/,
                     uint32_t& out_w, uint32_t& out_h) const override {
        out_w = static_cast<uint32_t>(std::max(0, x1 - x0));
        out_h = static_cast<uint32_t>(std::max(0, y1 - y0));
    }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "crop"; }
};

// ── Resize (bilinear) ─────────────────────────────────────────────────────────
struct ResizeTransform : Transform {
    float scale_x, scale_y;

    ResizeTransform(float sx, float sy) : scale_x(sx), scale_y(sy) {}

    int required_halo() const override { return 1; }

    // Output image dimensions scaled by (scale_x, scale_y).
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

// ── Rotation (bilinear, around image centre) ──────────────────────────────────
struct RotateTransform : Transform {
    float angle_deg;

    explicit RotateTransform(float deg) : angle_deg(deg) {}

    // Rotation needs a large halo: output pixels can map to source pixels
    // far from the tile core. 256 is sufficient for typical angles on
    // the project's test images.

    virtual int required_halo() const override { return 256; }
    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "rotate"; }
};

// ── Box Blur (separable) ──────────────────────────────────────────────────────
struct BoxBlurTransform : Transform {
    int radius;

    explicit BoxBlurTransform(int r) : radius(r) {}

    int required_halo() const override { return radius; }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "box_blur(r=" + std::to_string(radius) + ")"; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  TransformChain — applies a sequence of transforms in order
// ─────────────────────────────────────────────────────────────────────────────

class TransformChain {
public:
    void add(std::unique_ptr<Transform> t) { steps_.push_back(std::move(t)); }

    // Apply all steps in order.
    Tile apply(Tile tile, const ImageInfo& img) const;

    // Maximum required_halo() across all steps.
    int max_halo() const;

    std::size_t size() const { return steps_.size(); }

    // Compute the output image dimensions produced by running the full chain.
    // Each transform's output_size() feeds as the input to the next.
    void compute_output_size(uint32_t in_w, uint32_t in_h,
                             uint32_t& out_w, uint32_t& out_h) const;

private:
    std::vector<std::unique_ptr<Transform>> steps_;
};