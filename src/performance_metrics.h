#pragma once
// =============================================================================
// performance_metrics.h — Milestone 3
//
// Small utility for recording end-to-end pipeline performance.  It is separate
// from TileProcessor::Stats so the project can write repeatable benchmark rows
// for different tile sizes, halo sizes, operation chains, and pipeline depths.
// =============================================================================

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

struct PipelineRunMetrics {
    std::string run_name;
    std::string operation_chain;
    std::string mode = "cpu";

    uint32_t image_w = 0;
    uint32_t image_h = 0;
    int tile_size = 0;
    int halo = 0;
    int pipeline_depth = 0;
    int threads = 1;

    uint64_t tiles_read = 0;
    uint64_t tiles_processed = 0;
    uint64_t tiles_written = 0;

    double elapsed_sec = 0.0;
    double throughput_mpix_sec = 0.0;
    double speedup_vs_baseline = 0.0;

    std::string to_csv_header() const;
    std::string to_csv_row() const;
};

class ScopedTimer {
public:
    ScopedTimer();
    double elapsed_seconds() const;
private:
    std::chrono::high_resolution_clock::time_point start_;
};

class MetricsCsvWriter {
public:
    explicit MetricsCsvWriter(const std::string& path);
    void append(const PipelineRunMetrics& m);
private:
    std::string path_;
};

inline double mpix_per_second(uint64_t pixels, double seconds) {
    return seconds > 0.0 ? (static_cast<double>(pixels) / 1.0e6) / seconds : 0.0;
}
