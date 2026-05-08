#include "performance_metrics.h"
#include <filesystem>

std::string PipelineRunMetrics::to_csv_header() const {
    return "run_name,operation_chain,mode,image_w,image_h,tile_size,halo,"
           "pipeline_depth,threads,tiles_read,tiles_processed,tiles_written,"
           "elapsed_sec,throughput_mpix_sec,speedup_vs_baseline\n";
}

static std::string clean_csv(std::string s) {
    for (char& ch : s) {
        if (ch == ',' || ch == '\n' || ch == '\r') ch = ' ';
    }
    return s;
}

std::string PipelineRunMetrics::to_csv_row() const {
    std::ostringstream out;
    out << clean_csv(run_name) << ','
        << clean_csv(operation_chain) << ','
        << clean_csv(mode) << ','
        << image_w << ',' << image_h << ','
        << tile_size << ',' << halo << ','
        << pipeline_depth << ',' << threads << ','
        << tiles_read << ',' << tiles_processed << ',' << tiles_written << ','
        << std::fixed << std::setprecision(6) << elapsed_sec << ','
        << std::fixed << std::setprecision(3) << throughput_mpix_sec << ','
        << std::fixed << std::setprecision(3) << speedup_vs_baseline << '\n';
    return out.str();
}

ScopedTimer::ScopedTimer()
    : start_(std::chrono::high_resolution_clock::now()) {}

double ScopedTimer::elapsed_seconds() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start_).count();
}

MetricsCsvWriter::MetricsCsvWriter(const std::string& path)
    : path_(path) {}

void MetricsCsvWriter::append(const PipelineRunMetrics& m) {
    const bool exists = std::filesystem::exists(path_);
    std::ofstream file(path_, std::ios::app);
    if (!file) return;
    if (!exists) file << m.to_csv_header();
    file << m.to_csv_row();
}
