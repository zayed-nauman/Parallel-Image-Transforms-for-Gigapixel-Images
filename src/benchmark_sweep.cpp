#include "benchmark_sweep.h"
#include "tile_reader.h"
#include "tile_writer.h"
#include "tile_processor.h"
#include "transforms.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct SweepRow {
    std::string experiment;
    int tile_size = 0;
    int halo = 0;
    int pipeline_depth = 0;
    double target_gpix = 0.0;
    double elapsed_sec = 0.0;
    double mpix_per_sec = 0.0;
    uint64_t tiles_read = 0;
    uint64_t tiles_written = 0;
    double io_overlap_ratio = 0.0;
    std::string data_source = "measured";
    std::string input_file;
};

void write_row(std::ofstream& out, const SweepRow& r) {
    out << r.experiment << ','
        << r.tile_size << ','
        << r.halo << ','
        << r.pipeline_depth << ','
        << std::fixed << std::setprecision(3) << r.target_gpix << ','
        << std::fixed << std::setprecision(6) << r.elapsed_sec << ','
        << std::fixed << std::setprecision(3) << r.mpix_per_sec << ','
        << r.tiles_read << ','
        << r.tiles_written << ','
        << std::fixed << std::setprecision(4) << r.io_overlap_ratio << ','
        << r.data_source << ','
        << r.input_file << '\n';
}

TileProcessor::Stats run_once(const PipelineConfig& cfg,
                              const TileReader& reader,
                              const TransformChain& chain,
                              uint32_t out_w,
                              uint32_t out_h,
                              const std::string& out_path)
{
    TileWriter writer(out_path, out_w, out_h, reader.info().fmt, 256);
    TileProcessor proc(cfg, reader, writer, chain);
    auto s = proc.run();
    writer.close();
    return s;
}
}

void run_milestone3_analysis_sweep(const PipelineConfig& base_cfg,
                                   const TileReader& reader,
                                   const TransformChain& chain,
                                   uint32_t out_w,
                                   uint32_t out_h,
                                   const std::string& csv_path,
                                   const BenchmarkImagePaths& bench_paths)
{
    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("Could not open analysis CSV: " + csv_path);

    csv << "experiment,tile_size,halo,pipeline_depth,target_gpix,elapsed_sec,mpix_per_sec,"
           "tiles_read,tiles_written,io_overlap_ratio,data_source,input_file\n";

    const int base_halo = std::max(base_cfg.halo_size,
        chain.max_halo_for_image(reader.info().width, reader.info().height));

    std::cout << "\n── Running MILESTONE 3 ANALYSIS SWEEP ─────────────\n";
    std::cout << "Writing CSV: " << csv_path << "\n";

    // 1) Tile-size impact.
    for (int ts : {128, 256, 512, 1024}) {
        PipelineConfig cfg = base_cfg;
        cfg.tile_size = ts;
        cfg.halo_size = base_halo;
        std::string out = base_cfg.output_path + ".sweep_tile_" + std::to_string(ts) + ".tiff";
        auto s = run_once(cfg, reader, chain, out_w, out_h, out);
        write_row(csv, {"tile_size", ts, cfg.halo_size, static_cast<int>(chain.size()), 0.0,
                        s.elapsed_sec, s.mpix_per_sec, s.tiles_read, s.tiles_written,
                        s.io_overlap_ratio});
    }

    // 2) Overlap/halo impact.
    for (int h : {0, base_halo, base_halo * 2}) {
        PipelineConfig cfg = base_cfg;
        cfg.halo_size = std::max(0, h);
        std::string out = base_cfg.output_path + ".sweep_halo_" + std::to_string(cfg.halo_size) + ".tiff";
        auto s = run_once(cfg, reader, chain, out_w, out_h, out);
        write_row(csv, {"halo", cfg.tile_size, cfg.halo_size, static_cast<int>(chain.size()), 0.0,
                        s.elapsed_sec, s.mpix_per_sec, s.tiles_read, s.tiles_written,
                        s.io_overlap_ratio});
    }

    // 3) Pipeline-depth impact using identity stages. This isolates framework
    // overhead for chain length without changing the output dimensions.
    for (int depth : {1, 2, 4, 8}) {
        TransformChain depth_chain;
        for (int i = 0; i < depth; ++i)
            depth_chain.add(std::make_unique<IdentityTransform>());
        PipelineConfig cfg = base_cfg;
        cfg.halo_size = 0;
        std::string out = base_cfg.output_path + ".sweep_depth_" + std::to_string(depth) + ".tiff";
        auto s = run_once(cfg, reader, depth_chain, reader.info().width, reader.info().height, out);
        write_row(csv, {"pipeline_depth", cfg.tile_size, cfg.halo_size, depth, 0.0,
                        s.elapsed_sec, s.mpix_per_sec, s.tiles_read, s.tiles_written,
                        s.io_overlap_ratio});
    }

    // 4) 1GP / 10GP / 50GP image-size benchmarks. If real image paths are
    // supplied through --bench-1gp/--bench-10gp/--bench-50gp, the sweep runs
    // those files and records measured throughput. If a path is missing, the
    // CSV row is explicitly marked projected so it cannot be mistaken for a
    // real gigapixel run.
    PipelineConfig cfg = base_cfg;
    auto measured = run_once(cfg, reader, chain, out_w, out_h, base_cfg.output_path + ".sweep_measured.tiff");

    auto run_size_case = [&](double gpix, const std::string& path, const std::string& label) {
        if (!path.empty()) {
            std::cout << "Running real " << label << " benchmark: " << path << "\n";
            TileReader size_reader(path);
            uint32_t sw = 0, sh = 0;
            chain.compute_output_size(size_reader.info().width, size_reader.info().height, sw, sh);
            PipelineConfig scfg = base_cfg;
            scfg.input_path = path;
            scfg.output_path = base_cfg.output_path + ".real_" + label + ".tiff";
            auto ss = run_once(scfg, size_reader, chain, sw, sh, scfg.output_path);
            write_row(csv, {"real_image_size", scfg.tile_size, scfg.halo_size,
                            static_cast<int>(chain.size()), gpix, ss.elapsed_sec,
                            ss.mpix_per_sec, ss.tiles_read, ss.tiles_written,
                            ss.io_overlap_ratio, "real", path});
        } else {
            double mpix = gpix * 1000.0;
            double projected_sec = measured.mpix_per_sec > 0.0 ? mpix / measured.mpix_per_sec : 0.0;
            write_row(csv, {"projected_image_size", cfg.tile_size, cfg.halo_size,
                            static_cast<int>(chain.size()), gpix, projected_sec,
                            measured.mpix_per_sec, measured.tiles_read, measured.tiles_written,
                            measured.io_overlap_ratio, "projected", "not_provided"});
        }
    };

    run_size_case(1.0,  bench_paths.gp1,  "1GP");
    run_size_case(10.0, bench_paths.gp10, "10GP");
    run_size_case(50.0, bench_paths.gp50, "50GP");

    std::cout << "Milestone 3 analysis sweep complete. CSV saved to " << csv_path << "\n";
}
