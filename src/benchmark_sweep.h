#pragma once
// =============================================================================
// benchmark_sweep.h — Milestone 3
//
// Automated parameter sweep for the final analysis requirement:
//   - tile size impact
//   - overlap/halo impact
//   - pipeline depth impact
//   - real 1GP / 10GP / 50GP benchmark image runs when file paths are supplied
//     (otherwise rows are explicitly marked as projected)
// =============================================================================

#include "common.h"
#include "transforms.h"
#include <string>

class TileReader;

struct BenchmarkImagePaths {
    std::string gp1;
    std::string gp10;
    std::string gp50;
};

void run_milestone3_analysis_sweep(const PipelineConfig& base_cfg,
                                   const TileReader& reader,
                                   const TransformChain& chain,
                                   uint32_t out_w,
                                   uint32_t out_h,
                                   const std::string& csv_path,
                                   const BenchmarkImagePaths& bench_paths = {});
