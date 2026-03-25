#include "common.h"
#include "tile_reader.h"
#include "tile_writer.h"
#include "tile_processor.h"
#include "transforms.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
//  CLI usage
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options] <input.tiff> <output.tiff>\n"
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
        "  " << prog << " --transform blur 4 input.tiff output.tiff\n"
        "  " << prog << " --transform rotate 30 --transform resize 0.5 0.5 in.tiff out.tiff\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Argument parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

static int parse_int(const char* s) {
    return std::stoi(s);
}
static float parse_float(const char* s) {
    return std::stof(s);
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    PipelineConfig cfg;
    TransformChain chain;
    BorderMode border = BorderMode::CLAMP;

    // Input and output paths are the last two positional arguments.
    std::string input_path;
    std::string output_path;

    // ── Parse arguments ───────────────────────────────────────────────────
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];

        if (arg == "--tile-size" && i + 1 < argc) {
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
            // Positional: first = input, second = output.
            if (input_path.empty())       input_path  = arg;
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

    // If no transform was specified, add identity so the pipeline has work.
    if (chain.size() == 0)
        chain.add(std::make_unique<IdentityTransform>());

    cfg.input_path  = input_path;
    cfg.output_path = output_path;

    // ── Open reader ────────────────────────────────────────────────────────
    try {
        TileReader reader(input_path, border);
        const ImageInfo& info = reader.info();

        std::cout << "Input:  " << input_path << "\n"
                  << "        " << info.width << " x " << info.height
                  << "  channels=" << channels_of(info.fmt) << "\n"
                  << "Output: " << output_path << "\n"
                  << "Config: tile=" << cfg.tile_size
                  << "  halo=" << std::max(cfg.halo_size, chain.max_halo())
                  << "  threads=" << cfg.num_threads
                  << "\n";

        // Output image has the same dimensions as input unless a resize was added.
        // (For a resize, the processor writes to the scaled-down coordinates.)
        // For simplicity, we use the input dimensions for the output file.
        // A production system would compute the output dimensions from the chain.
        uint32_t out_w = info.width;
        uint32_t out_h = info.height;

        TileWriter writer(output_path, out_w, out_h, info.fmt, 256);
        TileProcessor proc(cfg, reader, writer, chain);

        TileProcessor::Stats stats = proc.run();

        std::cout << "\n── Results ─────────────────────────────────────\n"
                  << "  Tiles read:      " << stats.tiles_read      << "\n"
                  << "  Tiles processed: " << stats.tiles_processed  << "\n"
                  << "  Tiles skipped:   " << stats.tiles_skipped    << "\n"
                  << "  Tiles written:   " << stats.tiles_written     << "\n"
                  << "  Elapsed:         " << stats.elapsed_sec      << " s\n"
                  << "  Throughput:      " << stats.mpix_per_sec     << " Mpix/s\n";

        writer.close();
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "\nFatal error: " << ex.what() << "\n";
        return 2;
    }
}