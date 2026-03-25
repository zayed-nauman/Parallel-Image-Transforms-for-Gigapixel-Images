#!/usr/bin/env python3
"""
gen_test_image.py - Generate synthetic tiled TIFF images for Milestone 1 testing.

Install dependency:  pip install tifffile numpy
Usage:
    python3 tools/gen_test_image.py               # 4096x4096 RGB test image
    python3 tools/gen_test_image.py 10000 10000   # 10 Kx10 K (100 Mpix)
    python3 tools/gen_test_image.py 1000 1000 --gray
"""

import sys
import argparse
import numpy as np  # type: ignore[import]

try:
    import tifffile  # type: ignore[import]
except ImportError:
    print("Install tifffile:  pip install tifffile numpy")
    sys.exit(1)


def make_checkerboard(h: int, w: int, channels: int, block: int = 128) -> np.ndarray:
    """Checkerboard pattern with a colour gradient so tiles are visually distinct."""
    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)
    grid_r = (rows // block).astype(np.int32) % 2
    grid_c = (cols // block).astype(np.int32) % 2
    checker = (grid_r[:, None] ^ grid_c[None, :]).astype(np.uint8)  # (H, W)

    if channels == 1:
        # Grayscale: bright/dark squares with a gentle gradient
        base  = checker * 180 + 30
        grad  = ((rows / h * 50)[:, None] + (cols / w * 50)[None, :]).astype(np.uint8)
        img   = np.clip(base + grad, 0, 255).astype(np.uint8)
        return img[:, :, None]  # (H, W, 1)

    # RGB: red channel encodes column, green encodes row, blue encodes checker
    r = (cols / w * 200 + 30).astype(np.uint8)[None, :]   # (1, W)
    g = (rows / h * 200 + 30).astype(np.uint8)[:, None]   # (H, 1)
    b = checker * 200 + 20
    r = np.broadcast_to(r, (h, w))
    g = np.broadcast_to(g, (h, w))
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test TIFF")
    parser.add_argument("width",  type=int, nargs="?", default=4096)
    parser.add_argument("height", type=int, nargs="?", default=4096)
    parser.add_argument("--gray",       action="store_true", help="Grayscale output")
    parser.add_argument("--tile",       type=int, default=256, help="TIFF tile size")
    parser.add_argument("--out",        type=str, default="", help="Output path")
    parser.add_argument("--block",      type=int, default=128, help="Checkerboard block size")
    args = parser.parse_args()

    channels  = 1 if args.gray else 3
    chan_str  = "gray" if args.gray else "rgb"
    out_path  = args.out or f"test_{args.width}x{args.height}_{chan_str}.tiff"

    print(f"Generating {args.width}x{args.height} {chan_str.upper()} image → {out_path}")
    print(f"  Native tile size: {args.tile}x{args.tile}")

    # Allocate the image in tiles to avoid a single giant allocation.
    tile_h = args.tile
    tile_w = args.tile

    # tifffile can write tiled TIFFs iteratively.
    with tifffile.TiffWriter(out_path, bigtiff=(args.width * args.height * channels > 2**31)) as tif:
        options = dict(
            photometric="minisblack" if channels == 1 else "rgb",
            tile=(tile_h, tile_w),
            compression=None,
            dtype=np.uint8,
        )
        # Build the full image in memory (for images <= ~1 GB).
        # For truly gigapixel images, write in strips to keep RAM low.
        if args.width * args.height * channels <= 1024 * 1024 * 1024:
            img = make_checkerboard(args.height, args.width, channels, args.block)
            if channels == 1:
                img = img[:, :, 0]   # tifffile wants (H, W) for grayscale
            tif.write(img, **options)
        else:
            # Iterative strip write (reduces peak RAM at cost of multiple passes).
            strip_rows = max(tile_h, 4096)
            print("  Large image — writing in strips to conserve RAM...")
            for y0 in range(0, args.height, strip_rows):
                y1 = min(y0 + strip_rows, args.height)
                strip = make_checkerboard(y1 - y0, args.width, channels, args.block)
                if channels == 1:
                    strip = strip[:, :, 0]
                if y0 == 0:
                    tif.write(strip, **options, contiguous=False)
                else:
                    tif.write(strip, contiguous=True)
                print(f"    rows {y0}–{y1}  ({100 * y1 / args.height:.0f}%)", end="\r")
            print()

    import os
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done — {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()