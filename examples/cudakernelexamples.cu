#include "shape.hpp"
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "dtypes/complex32.hpp"
#include "file_loaders/safetensors.hpp"
// #include "file_loaders/gltf.hpp"
#include <string>


__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}


int main(){


    

    auto weight = Tensor<float>({1024,1024}, DeviceType::kCPU);
    weight = 0.5f;
    auto weightcuda = weight.to(DeviceType::kCUDA);


    auto input = Tensor<float>({64,1024}, DeviceType::kCPU);
    input = 2.0f;
    auto inputcuda = input.to(DeviceType::kCUDA);
    // input = 2.0f;
    auto output = Tensor<float>({64,1024}, DeviceType::kCUDA);
    // output = 0.0f;

    int M = 64;
    int N = 1024;
    int K = 1024;
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    matmul_kernel<<<gridSize, blockSize>>>(input.data, weight.data, output.data, M, N, K);
    cudaDeviceSynchronize();

    std::cout << "Output Tensor:" << std::endl;
    std::cout << output.to(kCPU) << std::endl;

    return 0;
}