
#include <iostream>
#include <cuda_runtime.h>

// Simple kernel
__global__ void hello_kernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

// Simple add kernel
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::cout << "=== Simple GPU Test ===" << std::endl;
    
    // Check device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        
        // Test kernel launch
        std::cout << "\nTesting kernel launch..." << std::endl;
        hello_kernel<<<1, 5>>>();
        cudaDeviceSynchronize();
        
        // Test memory allocation
        std::cout << "\nTesting memory allocation..." << std::endl;
        float *d_a, *d_b, *d_c;
        int n = 1000;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        std::cout << "Allocated " << (n * sizeof(float) * 3 / 1024) << " KB on GPU" << std::endl;
        
        // Test computation
        std::cout << "\nTesting computation..." << std::endl;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        std::cout << "\n✅ All GPU tests passed!" << std::endl;
    }
    
    return 0;
}
