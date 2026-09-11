// cuda_plugin.cpp — CUDA backend plugin
//
// Built with:  nvcc -std=c++20 -Xcompiler -fPIC -shared \
//                  -I.. -I../../tensor \
//                  -o plugins/cuda/libcuda_plugin.so cuda_plugin.cpp \
//                  -lcudart
//
// Priority 30 — loaded after CPU/Disk so it can register converters on the
// CPU AllocationMap.

#include "plugin.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <unistd.h>  // dup()

// Note: GL constants are needed for the OpenGL-interop converter.  We define
// them here so the plugin does not require GL headers at compile time; the
// actual values come from the OpenGL spec and never change.
#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D 0x0DE1
#endif

#define CUDA_CHECK(__call)                                                     \
    do {                                                                       \
        cudaError_t __err = __call;                                            \
        if (__err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(__err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  CUDA AllocationMap
// ---------------------------------------------------------------------------

static AllocationMap* create_cuda_mapper(int device_id) {
    AllocationMap* mapper = new AllocationMap();
    mapper->device_id = device_id;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    mapper->default_compute_type = ComputeType::kCUDA;
    mapper->default_allocator_type = ComputeType::kCUDA;
    mapper->supports_compute_device[ComputeType::kCUDA] = true;

    mapper->compute_device_allocators[ComputeType::kCUDA] = [device_id](AllocationMetadata metadata, void* existing_data) {
        void* ptr;
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaMalloc(&ptr, metadata.byte_size));
        if (existing_data) {
            CUDA_CHECK(cudaMemcpy(ptr, existing_data, metadata.byte_size, cudaMemcpyHostToDevice));
        }
        return new BaseMemoryAllocation(metadata, ptr);
    };

    mapper->compute_device_deallocators[ComputeType::kCUDA] = [device_id](void* ptr) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaFree(ptr));
    };

    mapper->memory_type_converters[MemoryType::kDDR] = [device_id](void* ptr, AllocationMetadata meta) {
        auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        auto host_ptr = host_device.allocate(meta);
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaMemcpy((char*)host_ptr->data, (char*)ptr, meta.byte_size, cudaMemcpyDeviceToHost));
        return host_ptr;
    };

    // OpenGL buffer → CUDA interop
    mapper->compute_type_converters[{ComputeType::kOPENGL, ComputeType::kCUDA}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
        cudaGraphicsResource* m = nullptr;
        cudaGraphicsResource_t* resource = &m;
        auto err = cudaGraphicsGLRegisterBuffer(resource, (GLuint)(size_t)ptr, cudaGraphicsRegisterFlagsNone);
        if (err != cudaSuccess) {
            std::string errorMsg = "Failed to register CUDA-GL interop: " + std::string(cudaGetErrorString(err));
            throw std::runtime_error(errorMsg);
        }
        auto errMap = cudaGraphicsMapResources(1, resource);
        if (errMap != cudaSuccess) {
            throw std::runtime_error("Failed to map CUDA-GL resources: " + std::string(cudaGetErrorString(errMap)));
        }
        void* temp;
        size_t size;
        auto cudaError = cudaGraphicsResourceGetMappedPointer(&temp, &size, resource[0]);
        if (cudaError != cudaSuccess) {
            throw std::runtime_error("Failed to get mapped pointer from CUDA-GL resource: " + std::string(cudaGetErrorString(cudaError)));
        }
        return temp;
    };

    // OpenGL texture → CUDA interop
    mapper->compute_type_converters[{ComputeType::kOPENGLTEXTURE, ComputeType::kCUDA}] = [](void* ptra, BaseMemoryAllocation* original, AllocationMetadata metadata) {
        cudaGraphicsResource* m = nullptr;
        cudaGraphicsResource_t* resource = &m;
        GLuint ptr = (GLuint)(size_t)original->data;
        if (original->metadata.format != 0) {
            return (void*)nullptr;
        }
        auto err = cudaGraphicsGLRegisterImage(resource, ptr, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to register CUDA-GL texture interop: " + std::string(cudaGetErrorString(err)));
        }
        auto errMap = cudaGraphicsMapResources(1, resource);
        if (errMap != cudaSuccess) {
            throw std::runtime_error("Failed to map CUDA-GL texture resources: " + std::string(cudaGetErrorString(errMap)));
        }
        cudaArray_t* array = new cudaArray_t[1];
        auto cudaError = cudaGraphicsSubResourceGetMappedArray(array, resource[0], 0, 0);
        if (cudaError != cudaSuccess) {
            throw std::runtime_error("Failed to get mapped array from CUDA-GL texture resource: " + std::string(cudaGetErrorString(cudaError)));
        }
        return (void*)array;
    };

    mapper->compute_mapping_deallocators[ComputeType::kCUDA] = [device_id](void* ptr, BaseMemoryAllocation* original) {
        if (original->metadata.compute_device == ComputeType::kOPENGL) {
            // nothing
        } else if (original->metadata.compute_device == ComputeType::kOPENGLTEXTURE) {
            // nothing
        } else if (original->metadata.compute_device == ComputeType::kVULKAN) {
            // No-op — memory owned by VulkanBufferHandle
        } else {
            throw std::runtime_error("No CUDA mapping deallocator found for original compute device");
        }
    };

    mapper->synchronize_function = [device_id]() {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Register converters on CPU + Disk devices (created earlier)

    AllocationMap* mem_device = &global_device_manager.get_device(MemoryType::kDDR, 0);
    mem_device->memory_type_converters[MemoryType::kCUDA_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
        return mapper->allocate(meta, ptr);
    };
    std::cout << "[cuda] Registered CUDA_VRAM converter for DDR memory" << std::endl;
    std::cout << "[cuda] Mem device pointer: " << mem_device << std::endl;

    AllocationMap* mem_device_disk = &global_device_manager.get_device(MemoryType::kDISK, 0);
    mem_device_disk->memory_type_converters[MemoryType::kCUDA_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
        return mapper->allocate(meta, ptr);
    };
    std::cout << "[cuda] Registered CUDA_VRAM converter for DISK memory" << std::endl;

    // -----------------------------------------------------------------
    //  Vulkan → CUDA external memory interop
    //
    //  When a tensor is allocated with kVULKAN on this memory device,
    //  the vulkan plugin stores a VulkanBufferHandle* in
    //  BaseMemoryAllocation::data.  The Tensor constructor auto-calls
    //  get_massaged_pointer(default_compute_type=kCUDA), which looks up
    //  this {kVULKAN, kCUDA} converter.
    //
    //  We import the VkBuffer's exported fd into CUDA via
    //  cudaImportExternalMemory + cudaExternalMemoryGetMappedBuffer,
    //  returning a CUDA device pointer that CUDA kernels can read/write
    //  directly.  The underlying memory is the same VkDeviceMemory —
    //  no copy.
    // -----------------------------------------------------------------
    mapper->supports_compute_device[ComputeType::kVULKAN] = true;

    mapper->compute_type_converters[{ComputeType::kVULKAN, ComputeType::kCUDA}] =
        [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) -> void* {
            VulkanBufferHandle* handle = (VulkanBufferHandle*)ptr;
            if (!handle || handle->fd < 0) {
                std::cerr << "[cuda] No fd for Vulkan→CUDA interop" << std::endl;
                return nullptr;
            }

            CUDA_CHECK(cudaSetDevice(device_id));

            int dup_fd = dup(handle->fd);
            if (dup_fd < 0) {
                std::cerr << "[cuda] dup(fd) failed for Vulkan→CUDA interop" << std::endl;
                return nullptr;
            }

            cudaExternalMemoryHandleDesc extMemDesc{};
            extMemDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
            extMemDesc.handle.fd = dup_fd;
            extMemDesc.size = handle->alloc_size;
            extMemDesc.flags = 0;

            cudaExternalMemory_t extMem;
            cudaError_t err = cudaImportExternalMemory(&extMem, &extMemDesc);
            if (err != cudaSuccess) {
                std::cerr << "[cuda] cudaImportExternalMemory failed: " << cudaGetErrorString(err) << std::endl;
                return nullptr;
            }

            cudaExternalMemoryBufferDesc bufDesc{};
            bufDesc.offset = 0;
            bufDesc.size = metadata.byte_size;
            bufDesc.flags = 0;

            void* devPtr = nullptr;
            err = cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);
            if (err != cudaSuccess) {
                std::cerr << "[cuda] cudaExternalMemoryGetMappedBuffer failed: " << cudaGetErrorString(err) << std::endl;
                return nullptr;
            }

            return devPtr;
        };

    mapper->this_device_type = MemoryType::kCUDA_VRAM;
    return mapper;
}

// ---------------------------------------------------------------------------
//  CUDA ComputeDeviceBase
// ---------------------------------------------------------------------------

static ComputeDeviceBase* create_cuda_compute_device(int device_id) {
    ComputeDeviceBase* device = new ComputeDeviceBase();
    device->supports_memory_location[MemoryType::kCUDA_VRAM] = true;
    device->default_memory_type = MemoryType::kCUDA_VRAM;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    device->compute_units = prop.multiProcessorCount;
    device->shared_memory_size = prop.sharedMemPerBlock;

    if (prop.canMapHostMemory) {
        std::cout << "[cuda] Device " << device_id << " supports mapping host memory, enabling zero-copy" << std::endl;
        device->supports_memory_location[MemoryType::kDDR] = true;

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        std::cout << "[cuda] Mapping host memory to CUDA device " << device_id << std::endl;
        mem_device.supports_compute_device[ComputeType::kCUDA] = true;

        mem_device.compute_device_allocators[ComputeType::kCUDA] = [device_id](AllocationMetadata meta, void* existing_data) {
            void* ptr;
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaMallocManaged(&ptr, meta.byte_size, cudaMemAttachGlobal));
            if (existing_data) {
                CUDA_CHECK(cudaMemcpy((char*)ptr, (char*)existing_data, meta.byte_size, cudaMemcpyHostToDevice));
            }
            return new BaseMemoryAllocation(meta, ptr);
        };

        mem_device.compute_type_converters[{ComputeType::kCPU, ComputeType::kCUDA}] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            if (original->metadata.compute_device == ComputeType::kCUDA) return ptr;
            if (original->metadata.compute_device == ComputeType::kCPU) {
                cudaHostRegister(ptr, metadata.byte_size, cudaHostRegisterMapped);
                void* device_ptr;
                CUDA_CHECK(cudaSetDevice(device_id));
                CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr, ptr, 0));
                return device_ptr;
            }
            throw std::runtime_error("Unsupported compute device for conversion to CUDA");
        };

        mem_device.compute_type_converters[{ComputeType::kCUDA, ComputeType::kCPU}] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            if (original->metadata.compute_device == ComputeType::kCPU) return ptr;
            if (original->metadata.compute_device == ComputeType::kCUDA) return ptr;
            throw std::runtime_error("Unsupported compute device for conversion to CPU");
        };

        mem_device.compute_mapping_deallocators[ComputeType::kCUDA] = [device_id](void* ptr, BaseMemoryAllocation* original) {};
        mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [device_id](void* ptr, BaseMemoryAllocation* original) {};

        mem_device.compute_device_deallocators[ComputeType::kCUDA] = [device_id](void* ptr) {
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaFree(ptr));
        };
    }

    return device;
}

// ---------------------------------------------------------------------------
//  Plugin C ABI
// ---------------------------------------------------------------------------

extern "C" const char* plugin_name() {
    return "cuda";
}

extern "C" int plugin_priority() {
    return 30;
}

extern "C" void plugin_register(DeviceManager* dm) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "[cuda] cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "[cuda] CUDA Device Count: " << count << std::endl;
    if (count == 0) return;

    for (int i = 0; i < count; i++) {
        AllocationMap* mapper = create_cuda_mapper(i);
        dm->register_memory_device(MemoryType::kCUDA_VRAM, i, mapper);
    }
    for (int i = 0; i < count; i++) {
        ComputeDeviceBase* dev = create_cuda_compute_device(i);
        dm->register_compute_device(ComputeType::kCUDA, i, dev);
    }
}
