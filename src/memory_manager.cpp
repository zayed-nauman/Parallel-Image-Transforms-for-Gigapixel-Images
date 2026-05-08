
#include "memory_manager.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

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

std::string MemoryManager::spill_tile(const Tile& t) { return ""; }
Tile MemoryManager::restore_tile(const std::string& spill_path) { return Tile(); }

MemoryManager::Stats MemoryManager::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return { cpu_bytes_used_, pinned_bytes_used_, gpu_bytes_used_, spill_count_, restore_count_ };
}
