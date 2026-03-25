#pragma once
#include "common.h"
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
    // The returned tile must have:
    //   halo = 0  (strip_halo is called by the pipeline after transforms)
    //   global_x, global_y set to the correct OUTPUT position.
    virtual Tile apply(const Tile& in, const ImageInfo& img) const = 0;

    // Human-readable name for logging.
    virtual std::string name() const = 0;

    // Minimum halo this transform needs from the reader.
    virtual int required_halo() const { return 0; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Concrete transforms
// ─────────────────────────────────────────────────────────────────────────────

// Identity — copies core pixels unchanged.  Useful for baseline testing.
struct IdentityTransform : Transform {
    Tile        apply(const Tile& in, const ImageInfo&) const override;
    std::string name()  const override { return "identity"; }
};

// ── Crop ─────────────────────────────────────────────────────────────────────
// Returns the intersection of the tile with the crop rectangle.
// If the tile is entirely outside the rectangle, returns an empty Tile
// (core_w == 0 || core_h == 0).  The pipeline skips empty tiles.
struct CropTransform : Transform {
    int32_t x0, y0, x1, y1;   // crop rect in global image coords (exclusive end)

    CropTransform(int32_t x0, int32_t y0, int32_t x1, int32_t y1)
        : x0(x0), y0(y0), x1(x1), y1(y1) {}

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "crop"; }
};

// ── Resize (bilinear) ─────────────────────────────────────────────────────────
// Scales the entire image by (scale_x, scale_y).
// Per tile: each output pixel reverse-maps to a source pixel in the input tile.
// The tile already carries the halo needed for bilinear interpolation.
struct ResizeTransform : Transform {
    float scale_x, scale_y;

    ResizeTransform(float sx, float sy) : scale_x(sx), scale_y(sy) {}

    // The resize transform needs a halo of 1 for bilinear interpolation.
    int required_halo() const override { return 1; }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "resize"; }
};

// ── Rotation (bilinear, around image centre) ──────────────────────────────────
// Rotates the image by `angle_deg` degrees counter-clockwise around the
// image centre.  Uses inverse-mapping: for each output pixel, compute the
// corresponding input pixel and bilinear-interpolate.
struct RotateTransform : Transform {
    float angle_deg;

    explicit RotateTransform(float deg) : angle_deg(deg) {}

    // Rotation is a geometric transform: output pixels within a tile can map
    // to source pixels far away from the tile core. To ensure tile-wise
    // correctness (no seams when tiling changes), we must load a large halo.
    //
    // The value 256 is sufficient for the project's milestone test image and
    // rotation angles used for verification.
    int required_halo() const override { return 256; }

    Tile        apply(const Tile& in, const ImageInfo& img) const override;
    std::string name()  const override { return "rotate"; }
};

// ── Box Blur (separable) ──────────────────────────────────────────────────────
// A simple N×N box blur using a separable pass (H then V).
// Demonstrates correct halo usage: requires halo >= radius.
struct BoxBlurTransform : Transform {
    int radius;   // kernel half-size; full kernel = (2*radius+1)²

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

    // Apply all steps; strip_halo is called once after the last step.
    Tile apply(Tile tile, const ImageInfo& img) const;

    // Maximum required_halo() across all steps.
    int max_halo() const;

    std::size_t size() const { return steps_.size(); }

private:
    std::vector<std::unique_ptr<Transform>> steps_;
};