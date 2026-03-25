#pragma once
#include "common.h"
#include "tile_reader.h"   // defines BorderMode

// ─────────────────────────────────────────────────────────────────────────────
//  Overlap / halo utilities
//
//  After a kernel-based filter has been applied to a Tile (which was loaded
//  with a halo), only the CORE region is valid output.  These helpers:
//    1. Verify the halo was set correctly for a given kernel radius.
//    2. Strip the halo from a processed Tile, returning a new Tile whose
//       buf_w == core_w and buf_h == core_h (halo = 0).
//    3. Provide a utility to blend two tiles at a seam (for debugging).
// ─────────────────────────────────────────────────────────────────────────────

namespace overlap {

// Return the minimum halo size needed for a kernel of the given radius.
// For a box/Gaussian kernel of radius r, every edge pixel needs r neighbours,
// so halo >= r.
inline int required_halo(int kernel_radius) { return kernel_radius; }

// Verify that a tile has enough halo for the requested kernel radius.
// Throws if insufficient.
void check_halo(const Tile& tile, int kernel_radius);

// Strip the halo: copy only the core region into a new Tile with halo = 0.
// The returned tile has:
//   buf_w = core_w,  buf_h = core_h,  halo = 0
//   global_x / global_y unchanged (still refers to the core's position)
Tile strip_halo(const Tile& src);

// Fill the halo region of `tile` using the specified border mode.
// Useful when you want to manually construct a tile buffer and then apply
// a filter without loading halo data from disk.
void fill_halo(Tile& tile, BorderMode mode);

// Debug utility: copy tile into an RGBA image buffer (dst must be at least
// buf_w * buf_h * 4 bytes).  The halo pixels are tinted red so you can
// visually inspect boundary handling.
void debug_visualise_halo(const Tile& tile, uint8_t* dst_rgba);

} // namespace overlap