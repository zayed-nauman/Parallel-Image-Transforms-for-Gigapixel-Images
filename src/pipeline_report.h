#pragma once
// =============================================================================
// pipeline_report.h — Milestone 3
//
// Lightweight reporting helpers for printing pipeline optimization decisions.
// =============================================================================

#include <string>
#include <vector>

struct PipelineOptimizationReport {
    bool fusion_enabled = false;
    bool compression_enabled = false;
    int prefetch_readers = 0;
    int block_size = 64;
    int pipeline_depth = 0;
    std::vector<std::string> transforms;

    std::string summary() const;
};
