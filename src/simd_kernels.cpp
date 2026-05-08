#include "simd_kernels.h"
#include <algorithm>
#include <cstring>

namespace simd_kernels {

void copy_u8(const uint8_t* src, uint8_t* dst, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void colour_affine_u8(const uint8_t* src,
                      uint8_t* dst,
                      std::size_t pixels,
                      int channels,
                      const float scale[4],
                      const float offset[4]) {
    const int active = std::min(channels, 4);

    for (std::size_t i = 0; i < pixels; ++i) {
        const std::size_t base = i * static_cast<std::size_t>(channels);
#if defined(_OPENMP)
#pragma omp simd simdlen(4)
#endif
        for (int c = 0; c < active; ++c) {
            float v = scale[c] * static_cast<float>(src[base + c]) + offset[c];
            dst[base + c] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
        }
        for (int c = active; c < channels; ++c) {
            dst[base + c] = src[base + c];
        }
    }
}

} // namespace simd_kernels
