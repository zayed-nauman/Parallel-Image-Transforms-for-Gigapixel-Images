#pragma once
// =============================================================================
// simd_kernels.h — Milestone 3
//
// SIMD-friendly CPU kernels.  The functions are written as tight, contiguous
// loops so compilers can auto-vectorize them.  AVX2/NEON-specific intrinsics can
// be added later behind the same interface without changing transform code.
// =============================================================================

#include <cstdint>
#include <cstddef>

namespace simd_kernels {

void colour_affine_u8(const uint8_t* src,
                      uint8_t* dst,
                      std::size_t pixels,
                      int channels,
                      const float scale[4],
                      const float offset[4]);

void copy_u8(const uint8_t* src, uint8_t* dst, std::size_t bytes);

} // namespace simd_kernels
