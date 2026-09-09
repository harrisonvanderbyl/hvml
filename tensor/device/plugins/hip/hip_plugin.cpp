// hip_plugin.cpp — HIP backend plugin
//
// Built with:  hipcc -std=c++20 -fPIC -shared \
//                  -I.. -I../../tensor \
//                  -o plugins/hip/libhip_plugin.so hip_plugin.cpp
//
// Priority 40 — loaded after CPU/Disk/CUDA.

#include "plugin.hpp"

#include <hip/hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_gl_interop.h>

#define HIP_CHECK(__call)                                                      \
    do {                                                                       \
        hipError_t __err = __call;                                             \
        if (__err != hipSuccess) {                                             \
            std::cerr << "HIP error: " << hipGetErrorString(__err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  HIP AllocationMap
// ---------------------------------------------------------------------------

static AllocationMap* create_hip_mapper(int device_id) {
    AllocationMap* mapper = new AllocationMap();
    mapper->device_id = device_id;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    mapper->default_compute_type = ComputeType::kHIP;
    mapper->default_allocator_type = ComputeType::kHIP;
    mapper->supports_compute_device[ComputeType::kHIP] = true;

    mapper->compute_device_allocators[ComputeType::kHIP] = [device_id](AllocationMetadata meta, void* existing_data) {
        void* ptr;
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMalloc(&ptr, meta.byte_size));
        if (existing_data) {
            HIP_CHECK(hipMemcpy(ptr, existing_data, meta.byte_size, hipMemcpyHostToDevice));
        }
        return new BaseMemoryAllocation(meta, ptr);
    };

    mapper->compute_device_deallocators[ComputeType::kHIP] = [device_id](void* ptr) {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipFree(ptr));
    };

    mapper->memory_type_converters[MemoryType::kDDR] = [device_id](void* ptr, AllocationMetadata meta) {
        auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        auto host_ptr = host_device.allocate(meta);
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMemcpy((char*)host_ptr->data, (char*)ptr, meta.byte_size, hipMemcpyDeviceToHost));
        return host_ptr;
    };

    mapper->memory_type_converters[MemoryType::kCUDA_VRAM] = [device_id](void* ptr, AllocationMetadata meta) {
        std::cerr << "Conversion from HIP_VRAM to CUDA_VRAM not implemented" << std::endl;
        return (BaseMemoryAllocation*)nullptr;
    };

    // OpenGL buffer → HIP interop
    mapper->compute_type_converters[{ComputeType::kOPENGL, ComputeType::kHIP}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
        hipGraphicsResource_t m = nullptr;
        auto err = hipGraphicsGLRegisterBuffer(&m, (GLuint)(long long)ptr, hipGraphicsRegisterFlagsNone);
        if (err != hipSuccess) {
            if (err == 999) {
                std::cout << "HIP-GL interop registration failed (USING wrong gpu for openGL)" << std::endl;
            }
            throw std::runtime_error("Failed to register HIP-GL interop: " + std::string(hipGetErrorString(err)));
        }
        auto errMap = hipGraphicsMapResources(1, &m);
        if (errMap != hipSuccess) {
            throw std::runtime_error("Failed to map HIP-GL resources: " + std::string(hipGetErrorString(errMap)));
        }
        void* temp;
        size_t size;
        auto hipError = hipGraphicsResourceGetMappedPointer(&temp, &size, m);
        if (hipError != hipSuccess) {
            throw std::runtime_error("Failed to get mapped pointer from HIP-GL resource: " + std::string(hipGetErrorString(hipError)));
        }
        return temp;
    };

    mapper->synchronize_function = [device_id]() {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipDeviceSynchronize());
    };

    // Register converters on CPU + Disk devices
    try {
        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.memory_type_converters[MemoryType::kHIP_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr);
        };
    } catch (...) {}

    try {
        auto& mem_device_disk = global_device_manager.get_device(MemoryType::kDISK, 0);
        mem_device_disk.memory_type_converters[MemoryType::kHIP_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr);
        };
    } catch (...) {}

    mapper->this_device_type = MemoryType::kHIP_VRAM;
    return mapper;
}

// ---------------------------------------------------------------------------
//  HIP ComputeDeviceBase
// ---------------------------------------------------------------------------

static ComputeDeviceBase* create_hip_compute_device(int device_id) {
    ComputeDeviceBase* device = new ComputeDeviceBase();
    device->supports_memory_location[MemoryType::kHIP_VRAM] = true;
    device->default_memory_type = MemoryType::kHIP_VRAM;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    device->compute_units = prop.multiProcessorCount;
    device->shared_memory_size = prop.sharedMemPerBlock;

    if (prop.canMapHostMemory) {
        std::cout << "[hip] Device " << device_id << " supports mapping host memory, enabling zero-copy" << std::endl;
        device->supports_memory_location[MemoryType::kDDR] = true;

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.supports_compute_device[ComputeType::kHIP] = true;

        mem_device.compute_device_allocators[ComputeType::kHIP] = [device_id](AllocationMetadata meta, void* existing_data) {
            void* ptr;
            HIP_CHECK(hipSetDevice(device_id));
            HIP_CHECK(hipMallocManaged(&ptr, meta.byte_size, hipMemAttachGlobal));
            if (existing_data) {
                HIP_CHECK(hipMemcpy((char*)ptr, (char*)existing_data, meta.byte_size, hipMemcpyHostToDevice));
            }
            return new BaseMemoryAllocation(meta, ptr);
        };

        mem_device.compute_type_converters[{ComputeType::kCPU, ComputeType::kHIP}] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            if (original->metadata.compute_device == ComputeType::kHIP) return ptr;
            if (original->metadata.compute_device == ComputeType::kCPU) {
                HIP_CHECK(hipHostRegister(ptr, metadata.byte_size, hipHostRegisterMapped));
                void* device_ptr;
                HIP_CHECK(hipSetDevice(device_id));
                HIP_CHECK(hipHostGetDevicePointer(&device_ptr, ptr, 0));
                return device_ptr;
            }
            throw std::runtime_error("Unsupported compute device for conversion to HIP");
        };

        mem_device.compute_type_converters[{ComputeType::kHIP, ComputeType::kCPU}] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            if (original->metadata.compute_device == ComputeType::kCPU) return ptr;
            if (original->metadata.compute_device == ComputeType::kHIP) return ptr;
            throw std::runtime_error("Unsupported compute device for conversion to CPU");
        };

        mem_device.compute_mapping_deallocators[ComputeType::kHIP] = [device_id](void* ptr, BaseMemoryAllocation* original) {};
        mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [device_id](void* ptr, BaseMemoryAllocation* original) {};

        mem_device.compute_device_deallocators[ComputeType::kHIP] = [device_id](void* ptr) {
            HIP_CHECK(hipSetDevice(device_id));
            HIP_CHECK(hipFree(ptr));
        };
    }

    return device;
}

// ---------------------------------------------------------------------------
//  Plugin C ABI
// ---------------------------------------------------------------------------

extern "C" const char* plugin_name() {
    return "hip";
}

extern "C" int plugin_priority() {
    return 40;
}

extern "C" void plugin_register(DeviceManager* dm) {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        std::cerr << "[hip] hipGetDeviceCount failed: " << hipGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "[hip] HIP Device Count: " << count << std::endl;
    if (count == 0) return;

    for (int i = 0; i < count; i++) {
        AllocationMap* mapper = create_hip_mapper(i);
        dm->register_memory_device(MemoryType::kHIP_VRAM, i, mapper);
    }
    for (int i = 0; i < count; i++) {
        ComputeDeviceBase* dev = create_hip_compute_device(i);
        dm->register_compute_device(ComputeType::kHIP, i, dev);
    }
}
