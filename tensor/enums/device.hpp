#include <string.h>
#include "file_loaders/json.hpp"
#ifndef DEVICE_TYPE
#define DEVICE_TYPE
enum DeviceType
{
    kCPU,
    kCUDA,
    kHIP
};

std::ostream &operator<<(std::ostream &os, const DeviceType &dtype)
{
    std::string s;
    switch (dtype)
    {
    case DeviceType::kCPU:
        s = "CPU";
        break;
    case DeviceType::kCUDA:
        s = "CUDA";
        break;
    case DeviceType::kHIP:
        s = "HIP";
        break;
    }
    os << s;
    return os;
}

NLOHMANN_JSON_SERIALIZE_ENUM(DeviceType, {
                                             {kCPU, "CPU"},
                                                {kCUDA, "CUDA"},
                                                {kHIP, "HIP"}
                                         })

template <DeviceType T>
struct DeviceAllocator{
    static constexpr DeviceType device_type = T;
    static void* allocate(size_t size){
        std::cout << "DeviceAllocator not implemented for this device type" << std::endl;
        throw std::runtime_error("DeviceAllocator not implemented for this device type");
    }
};

#if defined(__CUDACC__)
template <>
struct DeviceAllocator<DeviceType::kCUDA>{
    static constexpr DeviceType device_type = DeviceType::kCUDA;
    static void* allocate(size_t size){
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
};
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
template <>
struct DeviceAllocator<DeviceType::kHIP>{
    static constexpr DeviceType device_type = DeviceType::kHIP;
    static void* allocate(size_t size){
        void* ptr;
        hipMalloc(&ptr, size);
        return ptr;
    }
};
#endif

template <>
struct DeviceAllocator<DeviceType::kCPU>{
    static constexpr DeviceType device_type = DeviceType::kCPU;
    static void* allocate(size_t size){
        return malloc(size);
    }
};

#endif