#include "common.h"
#include "tile_reader.h"
#include "tile_writer.h"
#include "tile_processor.h"
#include "sequential_processor.h"
#include "transforms.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iomanip>
#include <thread>

// ─────────────────────────────────────────────────────────────────────────────
//  CLI usage
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options] <input.tiff> <output.tiff>\n"
        "\n"
        "Mode:\n"
        "  --mode parallel      Run parallel pipeline (default)\n"
        "  --mode sequential    Run sequential single-thread pipeline\n"
        "  --mode both          Run BOTH and print speedup comparison\n"
        "\n"
        "Options:\n"
        "  --transform <name>   Add a transform (can be repeated).\n"
        "                       Names: identity | crop | resize | rotate | blur\n"
        "  --tile-size  <N>     Processing tile size in pixels (default 512)\n"
        "  --halo       <N>     Halo override; default = max required by transforms\n"
        "  --threads    <N>     Worker threads (default = hardware concurrency)\n"
        "  --in-flight  <N>     Max tiles in RAM at once (default 16)\n"
        "  --border     <mode>  zero | clamp | reflect (default clamp)\n"
        "\n"
        "Transform parameters (follow --transform <name>):\n"
        "  identity              No-op, copies pixels unchanged\n"
        "  blur   <radius>       Box blur, e.g.:  --transform blur 8\n"
        "  crop   <x0 y0 x1 y1> Crop rect,  e.g.: --transform crop 100 100 900 900\n"
        "  resize <sx sy>        Scale,      e.g.: --transform resize 0.5 0.5\n"
        "  rotate <degrees>      CCW rotate, e.g.: --transform rotate 45\n"
        "\n"
        "Examples:\n"
        "  " << prog << " input.tiff output.tiff\n"
        "  " << prog << " --mode both input.tiff output.tiff\n"
        "  " << prog << " --transform blur 4 input.tiff output.tiff\n"
        "  " << prog << " --transform rotate 30 --transform resize 0.5 0.5 in.tiff out.tiff\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Argument parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

static int   parse_int(const char* s)   { return std::stoi(s); }
static float parse_float(const char* s) { return std::stof(s); }

// ─────────────────────────────────────────────────────────────────────────────
//  Speedup table
// ─────────────────────────────────────────────────────────────────────────────

static void print_speedup_table(
    double par_elapsed,  double par_mpix,
    double seq_elapsed,  double seq_mpix,
    int    num_threads,
    uint32_t img_w,      uint32_t img_h)
{
    double speedup    = (par_elapsed > 0) ? seq_elapsed / par_elapsed : 0.0;
    double efficiency = speedup / num_threads * 100.0;

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║           SPEEDUP ANALYSIS REPORT                   ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Image : " << img_w << " x " << img_h
              << "  (" << std::fixed << std::setprecision(1)
              << (double)img_w * img_h / 1e6 << " Mpix)\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Mode            Threads    Time(s)    Mpix/s       ║\n";
    std::cout << "║  ─────────────  ────────   ────────   ────────      ║\n";
    std::cout << "║  Sequential          1    "
              << std::setw(8) << std::fixed << std::setprecision(4) << seq_elapsed
              << "   "
              << std::setw(8) << std::fixed << std::setprecision(2) << seq_mpix
              << "      ║\n";
    std::cout << "║  Parallel     "
              << std::setw(8) << num_threads << "    "
              << std::setw(8) << std::fixed << std::setprecision(4) << par_elapsed
              << "   "
              << std::setw(8) << std::fixed << std::setprecision(2) << par_mpix
              << "      ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Speedup    : " << std::fixed << std::setprecision(2)
              << speedup << "x  (ideal: " << num_threads << "x)              ║\n";
    std::cout << "║  Efficiency : " << std::fixed << std::setprecision(1)
              << efficiency << "% per thread                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    PipelineConfig cfg;
    TransformChain chain;
    BorderMode     border = BorderMode::CLAMP;
    std::string    mode   = "parallel";
    std::string    input_path;
    std::string    output_path;

    // ── Parse arguments ───────────────────────────────────────────────────
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            if (mode != "parallel" && mode != "sequential" && mode != "both") {
                std::cerr << "Unknown mode: " << mode
                          << " (use: parallel | sequential | both)\n";
                return 1;
            }
        } else if (arg == "--tile-size" && i + 1 < argc) {
            cfg.tile_size = parse_int(argv[++i]);
        } else if (arg == "--halo" && i + 1 < argc) {
            cfg.halo_size = parse_int(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.num_threads = parse_int(argv[++i]);
        } else if (arg == "--in-flight" && i + 1 < argc) {
            cfg.max_in_flight = parse_int(argv[++i]);
        } else if (arg == "--border" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "zero")    border = BorderMode::ZERO;
            else if (m == "clamp")   border = BorderMode::CLAMP;
            else if (m == "reflect") border = BorderMode::REFLECT;
            else { std::cerr << "Unknown border mode: " << m << "\n"; return 1; }

        } else if (arg == "--transform" && i + 1 < argc) {
            std::string tname = argv[++i];

            if (tname == "identity") {
                chain.add(std::make_unique<IdentityTransform>());

            } else if (tname == "blur" && i + 1 < argc) {
                int r = parse_int(argv[++i]);
                chain.add(std::make_unique<BoxBlurTransform>(r));

            } else if (tname == "crop" && i + 4 < argc) {
                int32_t x0 = parse_int(argv[++i]);
                int32_t y0 = parse_int(argv[++i]);
                int32_t x1 = parse_int(argv[++i]);
                int32_t y1 = parse_int(argv[++i]);
                chain.add(std::make_unique<CropTransform>(x0, y0, x1, y1));

            } else if (tname == "resize" && i + 2 < argc) {
                float sx = parse_float(argv[++i]);
                float sy = parse_float(argv[++i]);
                chain.add(std::make_unique<ResizeTransform>(sx, sy));

            } else if (tname == "rotate" && i + 1 < argc) {
                float deg = parse_float(argv[++i]);
                chain.add(std::make_unique<RotateTransform>(deg));

            } else {
                std::cerr << "Unknown or incomplete transform: " << tname << "\n";
                return 1;
            }

        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;

        } else {
            if      (input_path.empty())  input_path  = arg;
            else if (output_path.empty()) output_path = arg;
            else { std::cerr << "Unexpected argument: " << arg << "\n"; return 1; }
        }

        ++i;
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: input and output paths are required.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    if (chain.size() == 0)
        chain.add(std::make_unique<IdentityTransform>());

    cfg.input_path  = input_path;
    cfg.output_path = output_path;

    // Resolve actual thread count now so we can report it correctly.
    int actual_threads = cfg.num_threads;
    if (actual_threads <= 0)
        actual_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (actual_threads <= 0) actual_threads = 2;

    // ── Open reader ───────────────────────────────────────────────────────
    try {
        TileReader reader(input_path, border);
        const ImageInfo& info = reader.info();

        std::cout << "Input:  " << input_path << "\n"
                  << "        " << info.width << " x " << info.height
                  << "  channels=" << channels_of(info.fmt) << "\n"
                  << "Output: " << output_path << "\n"
                  << "Config: tile=" << cfg.tile_size
                  << "  halo=" << std::max(cfg.halo_size, chain.max_halo())
                  << "  threads=" << actual_threads
                  << "  mode=" << mode << "\n";

        double par_elapsed = 0, par_mpix = 0;
        double seq_elapsed = 0, seq_mpix = 0;

        // ── Sequential run ────────────────────────────────────────────────
        if (mode == "sequential" || mode == "both") {
            std::string seq_out = (mode == "both")
                ? output_path + ".seq.tiff"
                : output_path;

            std::cout << "\n── Running SEQUENTIAL ────────────────────────────\n";
            TileWriter seq_writer(seq_out, info.width, info.height, info.fmt, 256);
            SequentialProcessor seq(cfg, reader, seq_writer, chain);
            auto s = seq.run();
            seq_writer.close();

            seq_elapsed = s.elapsed_sec;
            seq_mpix    = s.mpix_per_sec;

            std::cout << "\n── Sequential Results ────────────────────────────\n"
                      << "  Tiles read:      " << s.tiles_read      << "\n"
                      << "  Tiles processed: " << s.tiles_processed  << "\n"
                      << "  Tiles skipped:   " << s.tiles_skipped    << "\n"
                      << "  Tiles written:   " << s.tiles_written    << "\n"
                      << "  Elapsed:         " << std::fixed << std::setprecision(4)
                      << s.elapsed_sec << " s\n"
                      << "  Throughput:      " << std::fixed << std::setprecision(2)
                      << s.mpix_per_sec << " Mpix/s\n";
        }

        // ── Parallel run ──────────────────────────────────────────────────
        if (mode == "parallel" || mode == "both") {
            std::cout << "\n── Running PARALLEL ──────────────────────────────\n";
            TileWriter par_writer(output_path, info.width, info.height, info.fmt, 256);
            TileProcessor proc(cfg, reader, par_writer, chain);
            auto s = proc.run();
            par_writer.close();

            par_elapsed = s.elapsed_sec;
            par_mpix    = s.mpix_per_sec;

            std::cout << "\n── Parallel Results ──────────────────────────────\n"
                      << "  Tiles read:      " << s.tiles_read      << "\n"
                      << "  Tiles processed: " << s.tiles_processed  << "\n"
                      << "  Tiles skipped:   " << s.tiles_skipped    << "\n"
                      << "  Tiles written:   " << s.tiles_written    << "\n"
                      << "  Elapsed:         " << std::fixed << std::setprecision(4)
                      << s.elapsed_sec << " s\n"
                      << "  Throughput:      " << std::fixed << std::setprecision(2)
                      << s.mpix_per_sec << " Mpix/s\n";
        }

        // ── Speedup table (only when both ran) ────────────────────────────
        if (mode == "both") {
            print_speedup_table(par_elapsed, par_mpix,
                                seq_elapsed, seq_mpix,
                                actual_threads,
                                info.width, info.height);
        }

        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "\nFatal error: " << ex.what() << "\n";
        return 2;
    }
}