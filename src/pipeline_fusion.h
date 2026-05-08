#pragma once
// =============================================================================
//  pipeline_fusion.h   —  Milestone 3
//
//  Operation fusion: collapse consecutive compatible transforms into a single
//  fused kernel pass to eliminate intermediate tile materialisation.
//
//  Fused operation families
//  ────────────────────────
//  1. BlurBlur        : sequential box-blur passes → single wider blur
//  2. BlurCrop        : blur result clipped to crop rect in one loop
//  3. BlurResize      : blur + downsample combined (saves one buffer)
//  4. ResizeCrop      : resize then crop output region
//  5. GenericFusion   : arbitrary chain collapsed to row-callback lambdas
//
//  Usage
//  ─────
//  FusedChain fused = FusionOptimizer::fuse(chain);
//  Tile result      = fused.apply(tile, img_info);
//
//  If no fusion is possible the FusedChain simply wraps the original chain.
// =============================================================================

#include "common.h"
#include "transforms.h"
#include "overlap.h"
#include "memory_manager.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  FusedStep — a single logical kernel that may embody 1-N original transforms
// ─────────────────────────────────────────────────────────────────────────────

struct FusedStep {
    // Human-readable description of what was fused
    std::string description;

    // How many bytes of intermediate scratch this step needs (per pixel row)
    std::size_t scratch_bytes_per_row = 0;

    // Required halo for this fused step
    int required_halo = 0;

    // Execute the fused step.  `in` has halo already attached.
    // Returns a tile with halo == 0.
    std::function<Tile(const Tile&, const ImageInfo&)> execute;
};

// ─────────────────────────────────────────────────────────────────────────────
//  FusedChain
// ─────────────────────────────────────────────────────────────────────────────

class FusedChain {
public:
    void add_step(FusedStep step) { steps_.push_back(std::move(step)); }

    Tile apply(Tile tile, const ImageInfo& img) const {
        for (const auto& step : steps_) {
            tile = step.execute(tile, img);
            if (tile.core_w == 0 || tile.core_h == 0) return tile;
        }
        return tile;
    }

    // Milestone 3: run with compressed intermediate storage. After every
    // non-final fused stage, the tile is RLE-spilled by MemoryManager and
    // restored before the next stage. This demonstrates the required
    // intermediate-result compression path in the normal pipeline.
    Tile apply_with_spill(Tile tile, const ImageInfo& img, MemoryManager& mm) const {
        for (std::size_t i = 0; i < steps_.size(); ++i) {
            tile = steps_[i].execute(tile, img);
            if (tile.core_w == 0 || tile.core_h == 0) return tile;
            if (i + 1 < steps_.size()) {
                std::string handle = mm.spill_tile(tile);
                tile.data.clear();
                tile.data.shrink_to_fit();
                tile = mm.restore_tile(handle);
            }
        }
        return tile;
    }

    int max_halo() const {
        int m = 0;
        for (const auto& s : steps_) m = std::max(m, s.required_halo);
        return m;
    }

    std::size_t num_steps() const { return steps_.size(); }

    // Report fusion summary
    std::string summary() const {
        std::string s;
        for (std::size_t i = 0; i < steps_.size(); ++i) {
            if (i > 0) s += " → ";
            s += steps_[i].description;
        }
        return s;
    }

    // Count of fused operations (original_count - fused_count = savings)
    int original_op_count = 0;

private:
    std::vector<FusedStep> steps_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  FusionOptimizer
// ─────────────────────────────────────────────────────────────────────────────

class FusionOptimizer {
public:
    // Analyse the chain and produce a FusedChain with optimal step count.
    static FusedChain fuse(const TransformChain& chain);

    // Statistics from last fuse() call
    struct Stats {
        int original_steps = 0;
        int fused_steps    = 0;
        int fusions_applied = 0;
        std::string description;
    };
    static Stats last_stats();

private:
    static Stats last_stats_;

    // Individual fusion rules (return nullptr if not applicable)
    static std::unique_ptr<FusedStep> try_fuse_blur_blur(
        const Transform* a, const Transform* b);
    static std::unique_ptr<FusedStep> try_fuse_blur_crop(
        const Transform* blur, const Transform* crop);
    static std::unique_ptr<FusedStep> try_fuse_blur_resize(
        const Transform* blur, const Transform* resize);
    static std::unique_ptr<FusedStep> try_fuse_resize_crop(
        const Transform* resize, const Transform* crop);
};
