#include "pipeline_report.h"
#include <sstream>

std::string PipelineOptimizationReport::summary() const {
    std::ostringstream out;
    out << "Milestone 3 pipeline report\n";
    out << "  fusion: " << (fusion_enabled ? "enabled" : "disabled") << "\n";
    out << "  compression: " << (compression_enabled ? "enabled" : "disabled") << "\n";
    out << "  prefetch readers: " << prefetch_readers << "\n";
    out << "  cache block size: " << block_size << "\n";
    out << "  pipeline depth: " << pipeline_depth << "\n";
    out << "  transforms: ";
    if (transforms.empty()) {
        out << "identity";
    } else {
        for (std::size_t i = 0; i < transforms.size(); ++i) {
            if (i) out << " -> ";
            out << transforms[i];
        }
    }
    out << "\n";
    return out.str();
}
