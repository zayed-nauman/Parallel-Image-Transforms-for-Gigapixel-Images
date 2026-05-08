
#include "memory_manager.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdio>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

PinnedBuffer::PinnedBuffer(std::size_t n) : bytes(n) {
#ifdef HAVE_CUDA
    cudaMallocHost(&ptr, n);
#else
    ptr = ::operator new(n);
#endif
}

PinnedBuffer::~PinnedBuffer() {
    if (ptr) {
#ifdef HAVE_CUDA
        cudaFreeHost(ptr);
#else
        ::operator delete(ptr);
#endif
        ptr = nullptr;
    }
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& o) noexcept
    : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
        this->~PinnedBuffer();
        ptr = o.ptr; bytes = o.bytes;
        o.ptr = nullptr; o.bytes = 0;
    }
    return *this;
}

DeviceBuffer::DeviceBuffer(std::size_t n) : bytes(n) {
#ifdef HAVE_CUDA
    cudaMalloc(&ptr, n);
#else
    ptr = ::operator new(n);
#endif
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr) {
#ifdef HAVE_CUDA
        cudaFree(ptr);
#else
        ::operator delete(ptr);
#endif
        ptr = nullptr;
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept
    : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
        this->~DeviceBuffer();
        ptr = o.ptr; bytes = o.bytes;
        o.ptr = nullptr; o.bytes = 0;
    }
    return *this;
}

MemoryManager::MemoryManager(const MemoryManagerConfig& cfg) : cfg_(cfg) {
    if (cfg_.gpu_available) {
        std::filesystem::create_directories(cfg_.spill_dir);
    }
}

MemoryManager::~MemoryManager() {}

Tile MemoryManager::alloc_cpu_tile(int32_t core_w, int32_t core_h, int32_t halo, PixelFormat fmt) {
    Tile t;
    t.core_w = core_w; t.core_h = core_h; t.halo = halo; t.fmt = fmt;
    std::size_t bytes = (std::size_t)t.buf_w() * t.buf_h() * channels_of(fmt);
    {
        std::unique_lock<std::mutex> lk(mu_);
        cpu_cv_.wait(lk, [&]{ return cpu_bytes_used_ + bytes <= cfg_.max_cpu_bytes; });
        cpu_bytes_used_ += bytes;
    }
    t.data.resize(bytes, 0);
    return t;
}

void MemoryManager::free_cpu_tile(Tile& t) {
    std::size_t bytes = t.data.size();
    t.data.clear(); t.data.shrink_to_fit();
    {
        std::lock_guard<std::mutex> lk(mu_);
        cpu_bytes_used_ = (bytes <= cpu_bytes_used_) ? cpu_bytes_used_ - bytes : 0;
    }
    cpu_cv_.notify_one();
}

std::unique_ptr<PinnedBuffer> MemoryManager::acquire_pinned(std::size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    if (pinned_bytes_used_ + bytes > cfg_.max_pinned_bytes) return nullptr;
    return std::make_unique<PinnedBuffer>(bytes);
}

void MemoryManager::release_pinned(std::unique_ptr<PinnedBuffer> buf) {
    if (!buf) return;
    std::lock_guard<std::mutex> lk(mu_);
    pinned_bytes_used_ -= std::min(buf->bytes, pinned_bytes_used_);
}

std::unique_ptr<DeviceBuffer> MemoryManager::try_acquire_gpu(std::size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!cfg_.gpu_available) return nullptr;
    if (gpu_bytes_used_ + bytes > cfg_.max_gpu_bytes) return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    if (!buf) return nullptr;
    gpu_bytes_used_ += bytes;
    return buf;
}

void MemoryManager::release_gpu(std::unique_ptr<DeviceBuffer> buf) {
    if (!buf) return;
    std::lock_guard<std::mutex> lk(mu_);
    gpu_bytes_used_ -= std::min(buf->bytes, gpu_bytes_used_);
}

namespace {
struct SpillHeader {
    char magic[8] = {'M','3','S','P','I','L','L','1'};
    int32_t global_x = 0, global_y = 0, core_w = 0, core_h = 0, halo = 0;
    int32_t fmt = 0;
    uint64_t raw_bytes = 0;
    uint64_t rle_bytes = 0;
};

static std::vector<uint8_t> rle_compress(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out;
    out.reserve(in.size() / 2 + 16);
    for (size_t i = 0; i < in.size();) {
        uint8_t v = in[i];
        size_t run = 1;
        while (i + run < in.size() && in[i + run] == v && run < 255) ++run;
        out.push_back(static_cast<uint8_t>(run));
        out.push_back(v);
        i += run;
    }
    return out;
}

static std::vector<uint8_t> rle_decompress(const std::vector<uint8_t>& in, size_t expected) {
    std::vector<uint8_t> out;
    out.reserve(expected);
    for (size_t i = 0; i + 1 < in.size(); i += 2) {
        uint8_t run = in[i], v = in[i + 1];
        out.insert(out.end(), run, v);
    }
    if (out.size() != expected)
        throw std::runtime_error("spill restore failed: decompressed byte count mismatch");
    return out;
}
}

std::string MemoryManager::spill_tile(const Tile& t) {
    std::filesystem::create_directories(cfg_.spill_dir);

    SpillHeader h;
    h.global_x = t.global_x; h.global_y = t.global_y;
    h.core_w = t.core_w; h.core_h = t.core_h; h.halo = t.halo;
    h.fmt = static_cast<int32_t>(t.fmt);
    h.raw_bytes = static_cast<uint64_t>(t.data.size());

    std::vector<uint8_t> payload = rle_compress(t.data);
    h.rle_bytes = static_cast<uint64_t>(payload.size());

    uint64_t seq;
    {
        std::lock_guard<std::mutex> lk(mu_);
        seq = spill_seq_++;
        ++spill_count_;
    }

    std::filesystem::path path = std::filesystem::path(cfg_.spill_dir) /
        ("tile_" + std::to_string(seq) + ".m3rle");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open spill file for writing: " + path.string());
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return path.string();
}

Tile MemoryManager::restore_tile(const std::string& spill_path) {
    std::ifstream in(spill_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open spill file for reading: " + spill_path);

    SpillHeader h;
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in || std::string(h.magic, h.magic + 7) != "M3SPILL")
        throw std::runtime_error("invalid spill file header: " + spill_path);

    std::vector<uint8_t> payload(static_cast<size_t>(h.rle_bytes));
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!in) throw std::runtime_error("truncated spill file: " + spill_path);

    Tile t;
    t.global_x = h.global_x; t.global_y = h.global_y;
    t.core_w = h.core_w; t.core_h = h.core_h; t.halo = h.halo;
    t.fmt = static_cast<PixelFormat>(h.fmt);
    t.data = rle_decompress(payload, static_cast<size_t>(h.raw_bytes));

    {
        std::lock_guard<std::mutex> lk(mu_);
        ++restore_count_;
    }
    std::error_code ec;
    std::filesystem::remove(spill_path, ec);
    return t;
}

MemoryManager::Stats MemoryManager::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return { cpu_bytes_used_, pinned_bytes_used_, gpu_bytes_used_, spill_count_, restore_count_ };
}
