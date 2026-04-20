// =============================================================================
//  gpu_kernels_stub.cpp   —  Milestone 2
//  Zainab Irfan Ansari  (29091)
//
//  CPU fallback implementations for every launcher declared in gpu_kernels.cuh.
//  This file is compiled ONLY when HAVE_CUDA is NOT defined (i.e. the normal
//  case on MSYS2 / Windows without a CUDA toolkit installed).
//
//  Each function performs the identical operation as its CUDA counterpart,
//  just on plain CPU memory so the full project builds and runs correctly
//  without any GPU hardware or driver.
//
//  When you later install the CUDA toolkit and rebuild with -DENABLE_CUDA=ON,
//  this file is automatically excluded by CMakeLists.txt and the real .cu
//  kernels are used instead.
// =============================================================================

#ifndef HAVE_CUDA   // entire file is a no-op in a CUDA build

#include "gpu_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------

static inline int s_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Bilinear sample from a (W x H x C) row-major buffer.
static void s_bilinear(const uint8_t* buf, int W, int H, int C,
                       float fx, float fy, uint8_t* out)
{
    fx = std::max(0.f, std::min(fx, (float)(W - 1)));
    fy = std::max(0.f, std::min(fy, (float)(H - 1)));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = std::min(x0 + 1, W - 1);
    int y1 = std::min(y0 + 1, H - 1);
    float dx = fx - x0, dy = fy - y0;
    for (int c = 0; c < C; ++c) {
        float v = (1.f - dy) * ((1.f - dx) * buf[(y0*W + x0)*C + c]
                              +       dx  * buf[(y0*W + x1)*C + c])
                +       dy  * ((1.f - dx) * buf[(y1*W + x0)*C + c]
                              +       dx  * buf[(y1*W + x1)*C + c]);
        out[c] = (uint8_t)std::max(0.f, std::min(v, 255.f));
    }
}

// =============================================================================
//  1.  Box-blur  (separable H + V)
//
//  Matches launch_box_blur() CUDA behaviour exactly:
//  two-pass separable box filter of radius `radius`, clamp border mode,
//  rounded integer division.
// =============================================================================

void launch_box_blur(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                     int W, int H, int C, int radius)
{
    int cnt = 2 * radius + 1;

    // Horizontal pass: src -> tmp
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                int sum = 0;
                for (int k = -radius; k <= radius; ++k)
                    sum += src[(y * W + s_clamp(x + k, 0, W - 1)) * C + c];
                tmp[(y * W + x) * C + c] =
                    (uint8_t)((sum + cnt / 2) / cnt);
            }
        }
    }

    // Vertical pass: tmp -> dst
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                int sum = 0;
                for (int k = -radius; k <= radius; ++k)
                    sum += tmp[(s_clamp(y + k, 0, H - 1) * W + x) * C + c];
                dst[(y * W + x) * C + c] =
                    (uint8_t)((sum + cnt / 2) / cnt);
            }
        }
    }
}

// =============================================================================
//  2.  Colour-space conversions
// =============================================================================

// RGB -> Grayscale  (BT.601 fixed-point)
void launch_rgb_to_gray(const uint8_t* src, uint8_t* dst, int W, int H)
{
    int n = W * H;
    for (int i = 0; i < n; ++i)
        dst[i] = (uint8_t)(
            (306 * (int)src[i*3]
           + 601 * (int)src[i*3 + 1]
           + 117 * (int)src[i*3 + 2]) >> 10);
}

// Grayscale -> RGB  (replicate channel)
void launch_gray_to_rgb(const uint8_t* src, uint8_t* dst, int W, int H)
{
    int n = W * H;
    for (int i = 0; i < n; ++i)
        dst[i*3] = dst[i*3 + 1] = dst[i*3 + 2] = src[i];
}

// RGB -> HSV  (H stored as H/2 in [0,179], S and V in [0,255])
void launch_rgb_to_hsv(const uint8_t* src, uint8_t* dst, int W, int H)
{
    int n = W * H;
    for (int i = 0; i < n; ++i) {
        float r = src[i*3]   / 255.f;
        float g = src[i*3+1] / 255.f;
        float b = src[i*3+2] / 255.f;
        float mx = std::max({r, g, b});
        float mn = std::min({r, g, b});
        float d  = mx - mn;
        float h = 0.f, s = (mx > 1e-6f) ? d / mx : 0.f, v = mx;
        if (d > 1e-6f) {
            if      (mx == r) h = 60.f * std::fmod((g - b) / d, 6.f);
            else if (mx == g) h = 60.f * ((b - r) / d + 2.f);
            else              h = 60.f * ((r - g) / d + 4.f);
            if (h < 0.f) h += 360.f;
        }
        dst[i*3]   = (uint8_t)(h * 0.5f);
        dst[i*3+1] = (uint8_t)(s * 255.f + 0.5f);
        dst[i*3+2] = (uint8_t)(v * 255.f + 0.5f);
    }
}

// HSV -> RGB
void launch_hsv_to_rgb(const uint8_t* src, uint8_t* dst, int W, int H)
{
    int n = W * H;
    for (int i = 0; i < n; ++i) {
        float h = src[i*3]   * 2.f;
        float s = src[i*3+1] / 255.f;
        float v = src[i*3+2] / 255.f;
        float c = v * s;
        float x = c * (1.f - std::abs(std::fmod(h / 60.f, 2.f) - 1.f));
        float m = v - c;
        float r = 0.f, g = 0.f, b = 0.f;
        int sec = (int)(h / 60.f) % 6;
        switch (sec) {
            case 0: r=c; g=x;       break;
            case 1: r=x; g=c;       break;
            case 2:      g=c; b=x;  break;
            case 3:      g=x; b=c;  break;
            case 4: r=x;      b=c;  break;
            default:r=c;      b=x;  break;
        }
        dst[i*3]   = (uint8_t)((r + m) * 255.f + 0.5f);
        dst[i*3+1] = (uint8_t)((g + m) * 255.f + 0.5f);
        dst[i*3+2] = (uint8_t)((b + m) * 255.f + 0.5f);
    }
}

// =============================================================================
//  3.  Geometric transforms
// =============================================================================

// Bilinear resize — centre-aligned reverse mapping
void launch_resize(const uint8_t* src, uint8_t* dst,
                   int W_src, int H_src, int W_dst, int H_dst, int C,
                   float scale_x, float scale_y)
{
    for (int oy = 0; oy < H_dst; ++oy) {
        for (int ox = 0; ox < W_dst; ++ox) {
            float gx = ((float)ox + 0.5f) / scale_x - 0.5f;
            float gy = ((float)oy + 0.5f) / scale_y - 0.5f;
            s_bilinear(src, W_src, H_src, C, gx, gy,
                       dst + ((size_t)oy * W_dst + ox) * C);
        }
    }
}

// Bilinear rotation — global-space reverse rotation around (img_cx, img_cy)
void launch_rotate(const uint8_t* src_buf, uint8_t* dst_core,
                   int W_buf, int H_buf, int W_core, int H_core, int C,
                   float img_cx, float img_cy,
                   float tile_gx, float tile_gy,
                   float buf_gx,  float buf_gy,
                   float cos_a,   float sin_a)
{
    for (int oy = 0; oy < H_core; ++oy) {
        for (int ox = 0; ox < W_core; ++ox) {
            float gox = tile_gx + (float)ox;
            float goy = tile_gy + (float)oy;
            float dx  = gox - img_cx;
            float dy  = goy - img_cy;
            float rsx = img_cx + (cos_a * dx + sin_a * dy);
            float rsy = img_cy + (-sin_a * dx + cos_a * dy);
            float bx  = rsx - buf_gx;
            float by  = rsy - buf_gy;

            uint8_t* out = dst_core + ((size_t)oy * W_core + ox) * C;
            if (bx < 0.f || bx >= (float)(W_buf - 1) ||
                by < 0.f || by >= (float)(H_buf - 1))
                std::memset(out, 0, C);
            else
                s_bilinear(src_buf, W_buf, H_buf, C, bx, by, out);
        }
    }
}

#endif  // !HAVE_CUDA
