#include <string.h>
#include "file_loaders/json.hpp"
#include <iostream>
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
struct DeviceAllocator {
    static constexpr DeviceType device_type = T;
    static void* allocate(size_t size){
        std::cout << "DeviceAllocator not implemented for this device type" << std::endl;
        throw std::runtime_error("DeviceAllocator not implemented for this device type");
    }

    template <typename U>
    static void memset(U* ptr, U value, size_t values){
        std::cout << "DeviceAllocator memset not implemented for this device type" << std::endl;
        throw std::runtime_error("DeviceAllocator memset not implemented for this device type");
    }

    static void deallocate(void* ptr){
        std::cout << "DeviceAllocator deallocate not implemented for this device type" << std::endl;
        throw std::runtime_error("DeviceAllocator deallocate not implemented for this device type");
    }
};


template <DeviceType T>
struct AllocationMapper{
    std::map<void*, int> allocation_counts;
    void register_allocation(void* ptr) {

        allocation_counts[ptr]++;
    }
    bool unregister_allocation(void* ptr) {

        if (allocation_counts.find(ptr) != allocation_counts.end()) {
            allocation_counts[ptr]--;
            if (allocation_counts[ptr] <= 0) {
                allocation_counts.erase(ptr);
                return true;
            }
        }
        return false;
    }
};

std::map<DeviceType, void*> allocation_mappers;

template <DeviceType device>
AllocationMapper<device>* get_allocation_mapper() {
    if (allocation_mappers.find(device) == allocation_mappers.end()) {
        allocation_mappers[device] = new AllocationMapper<device>();
    }
    return static_cast<AllocationMapper<device>*>(allocation_mappers[device]);
}

void register_allocation(void* ptr,DeviceType device) {
    switch (device) {
        case DeviceType::kCPU:
            get_allocation_mapper<DeviceType::kCPU>()->register_allocation(ptr);
            break;
        case DeviceType::kCUDA:
            get_allocation_mapper<DeviceType::kCUDA>()->register_allocation(ptr);
            break;
        case DeviceType::kHIP:
            get_allocation_mapper<DeviceType::kHIP>()->register_allocation(ptr);
            break;
    }
}


#if defined(__CUDACC__)
template <>
struct DeviceAllocator<DeviceType::kCUDA>{
    static constexpr DeviceType device_type = DeviceType::kCUDA;
    static void* allocate(size_t size){
        void* ptr;
        cudaMalloc(&ptr, size);
        get_allocation_mapper<DeviceType::kCUDA>()->register_allocation(ptr);
        return ptr;
    }

    template <typename U>
    static void memset(U* ptr, U value, size_t values){
        cudaMemcpy(ptr, &value, sizeof(U)*values, cudaMemcpyHostToDevice);
    }

    static void deallocate(void* ptr){
        if (get_allocation_mapper<DeviceType::kCUDA>()->unregister_allocation(ptr)) {
            cudaFree(ptr);
        }
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
        get_allocation_mapper<DeviceType::kHIP>()->register_allocation(ptr);
        return ptr;
    }

    template <typename U>
    static void memset(U* ptr, U value, size_t values){
        hipMemset(ptr, value, sizeof(U)*values);
    }

    static void deallocate(void* ptr){
        if (get_allocation_mapper<DeviceType::kHIP>()->unregister_allocation(ptr)) {
            hipFree(ptr);
        }
    }
};
#endif

template <>
struct DeviceAllocator<DeviceType::kCPU>{
    static constexpr DeviceType device_type = DeviceType::kCPU;
    static void* allocate(size_t size){
        void* ptr = malloc(size);
        get_allocation_mapper<DeviceType::kCPU>()->register_allocation(ptr);
        return ptr;
    }

    template <typename U>
    static void memset(U* ptr, U value, size_t values){
        for (size_t i = 0; i < values; i++){
            ptr[i] = value;
        }
    }

    static void deallocate(void* ptr){
        if (get_allocation_mapper<DeviceType::kCPU>()->unregister_allocation(ptr)) {
            free(ptr);
        }
    }
};


#endif