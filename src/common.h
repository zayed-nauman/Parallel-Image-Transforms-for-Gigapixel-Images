#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>  // std::clamp (C++17), std::min, std::max

// Polyfill for compilers that have <algorithm> but not std::clamp
#if defined(__apple_build_version__) && __cplusplus < 201703L
namespace std {
    template<typename T>
    const T& clamp(const T& v, const T& lo, const T& hi) {
        return (v < lo) ? lo : (v > hi) ? hi : v;
    }
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Pixel format
// ─────────────────────────────────────────────────────────────────────────────

enum class PixelFormat { GRAY8, RGB8, RGBA8 };

inline int channels_of(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::GRAY8:  return 1;
        case PixelFormat::RGB8:   return 3;
        case PixelFormat::RGBA8:  return 4;
    }
    return 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tile  — the fundamental unit of work
// ─────────────────────────────────────────────────────────────────────────────

struct Tile {
    // Position in the global image (top-left corner of the CORE region, i.e.
    // excluding halo).
    int32_t global_x = 0;
    int32_t global_y = 0;

    // Dimensions of the CORE region (what will be written to output).
    int32_t core_w   = 0;
    int32_t core_h   = 0;

    // Halo width on each side (same value used for all four sides).
    int32_t halo     = 0;

    // Full buffer dimensions including halo on all sides.
    int32_t buf_w()  const { return core_w + 2 * halo; }
    int32_t buf_h()  const { return core_h + 2 * halo; }

    PixelFormat fmt  = PixelFormat::RGB8;

    // Pixel data: row-major, buf_w * buf_h * channels bytes.
    std::vector<uint8_t> data;

    // Convenience: pointer to first byte of pixel (bx, by) in buffer coords.
    uint8_t* px(int bx, int by) {
        int ch = channels_of(fmt);
        return data.data() + (by * buf_w() + bx) * ch;
    }
    const uint8_t* px(int bx, int by) const {
        int ch = channels_of(fmt);
        return data.data() + (by * buf_w() + bx) * ch;
    }

    // Pointer to first byte of pixel (cx, cy) in CORE coords (halo-offset applied).
    uint8_t* core_px(int cx, int cy) { return px(cx + halo, cy + halo); }
    const uint8_t* core_px(int cx, int cy) const { return px(cx + halo, cy + halo); }

    void allocate() {
        data.assign(buf_w() * buf_h() * channels_of(fmt), 0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Image metadata
// ─────────────────────────────────────────────────────────────────────────────

struct ImageInfo {
    uint32_t    width       = 0;
    uint32_t    height      = 0;
    PixelFormat fmt         = PixelFormat::RGB8;
    uint32_t    tile_width  = 256;  // native tile size stored in the file
    uint32_t    tile_height = 256;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Pipeline configuration
// ─────────────────────────────────────────────────────────────────────────────

struct PipelineConfig {
    std::string input_path;
    std::string output_path;

    // Processing tile size (power of two; fits tile+halo in L3 cache).
    int tile_size     = 512;

    // Halo width = kernel radius for the largest filter in the chain.
    int halo_size     = 16;

    // Number of CPU worker threads (0 = auto-detect).
    int num_threads   = 0;

    // Maximum tiles held in RAM at once (back-pressure limit).
    int max_in_flight = 16;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Error helpers
// ─────────────────────────────────────────────────────────────────────────────

#define M1_CHECK(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(std::string("milestone1: ") + (msg)); } while(0)