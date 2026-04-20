
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    
    // Test simple allocation
    void* ptr;
    err = cudaMalloc(&ptr, 1024);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "cudaMalloc successful" << std::endl;
    cudaFree(ptr);
    
    return 0;
}
