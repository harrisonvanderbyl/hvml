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
#include <map>
#include <mutex>
#include "vulkan_interop.hpp"

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
//  Vulkan → CUDA external memory interop
//
//  A Vulkan allocation in CUDA memory exports its VkDeviceMemory as an opaque
//  fd (vulkan plugin).  CUDA imports it and maps:
//    buffers → a device pointer  (cudaExternalMemoryGetMappedBuffer)
//    images  → a cudaArray_t     (cudaExternalMemoryGetMappedMipmappedArray)
//  over the same memory — no copy.  Each import is recorded so the mapping
//  deallocator can unmap it and destroy the external memory when the
//  allocation is freed.
// ---------------------------------------------------------------------------

struct VulkanImport {
    cudaExternalMemory_t memory = nullptr;
    cudaMipmappedArray_t mipmap = nullptr;   // images only
};

static std::mutex g_vulkan_imports_mutex;
static std::map<void*, VulkanImport> g_vulkan_imports;   // mapped pointer / array → import

static cudaExternalMemory_t import_vulkan_memory(const VulkanResource* r, int device_id) {
    if (!r || r->fd < 0) {
        std::cerr << "[cuda] Vulkan memory not exported (no fd) — no CUDA view" << std::endl;
        return nullptr;
    }
    CUDA_CHECK(cudaSetDevice(device_id));
    int fd = dup(r->fd);   // CUDA owns the fd once the import succeeds
    if (fd < 0) {
        std::cerr << "[cuda] dup(fd) failed for Vulkan→CUDA interop" << std::endl;
        return nullptr;
    }
    cudaExternalMemoryHandleDesc desc{};
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = r->alloc_size;
    desc.flags = r->dedicated ? cudaExternalMemoryDedicated : 0;
    cudaExternalMemory_t memory = nullptr;
    cudaError_t err = cudaImportExternalMemory(&memory, &desc);
    if (err != cudaSuccess) {
        close(fd);
        std::cerr << "[cuda] cudaImportExternalMemory failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return memory;
}

static void* import_vulkan_buffer(const VulkanResource* r, size_t bytes, int device_id) {
    cudaExternalMemory_t memory = import_vulkan_memory(r, device_id);
    if (!memory) return nullptr;
    cudaExternalMemoryBufferDesc desc{};
    desc.offset = 0;
    desc.size = bytes;
    void* ptr = nullptr;
    cudaError_t err = cudaExternalMemoryGetMappedBuffer(&ptr, memory, &desc);
    if (err != cudaSuccess) {
        std::cerr << "[cuda] cudaExternalMemoryGetMappedBuffer failed: " << cudaGetErrorString(err) << std::endl;
        cudaDestroyExternalMemory(memory);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_vulkan_imports_mutex);
    g_vulkan_imports[ptr] = VulkanImport{memory, nullptr};
    return ptr;
}

static void* import_vulkan_image(const VulkanResource* r, int device_id) {
    if (!r || !r->image) return nullptr;
    VulkanFormatChannels ch = vulkan_format_channels(r->format);
    if (!ch.ok) {
        std::cerr << "[cuda] Vulkan image format " << r->format << " has no CUDA array equivalent" << std::endl;
        return nullptr;
    }
    cudaExternalMemory_t memory = import_vulkan_memory(r, device_id);
    if (!memory) return nullptr;

    cudaChannelFormatKind kind = ch.kind == VulkanFormatChannels::kFloat  ? cudaChannelFormatKindFloat
                               : ch.kind == VulkanFormatChannels::kSigned ? cudaChannelFormatKindSigned
                                                                          : cudaChannelFormatKindUnsigned;
    cudaExternalMemoryMipmappedArrayDesc desc{};
    desc.offset = 0;
    desc.formatDesc = cudaCreateChannelDesc(ch.x, ch.y, ch.z, ch.w, kind);
    desc.extent = make_cudaExtent(r->width, r->height, 0);
    desc.flags = 0;
    if (r->image_usage & kVkImageUsageStorage) desc.flags |= cudaArraySurfaceLoadStore;
    if (r->image_usage & kVkImageUsageColorAttachment) desc.flags |= cudaArrayColorAttachment;
    desc.numLevels = 1;

    cudaMipmappedArray_t mipmap = nullptr;
    cudaError_t err = cudaExternalMemoryGetMappedMipmappedArray(&mipmap, memory, &desc);
    if (err != cudaSuccess) {
        std::cerr << "[cuda] cudaExternalMemoryGetMappedMipmappedArray failed: " << cudaGetErrorString(err) << std::endl;
        cudaDestroyExternalMemory(memory);
        return nullptr;
    }
    cudaArray_t level0 = nullptr;
    err = cudaGetMipmappedArrayLevel(&level0, mipmap, 0);
    if (err != cudaSuccess) {
        std::cerr << "[cuda] cudaGetMipmappedArrayLevel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFreeMipmappedArray(mipmap);
        cudaDestroyExternalMemory(memory);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_vulkan_imports_mutex);
    g_vulkan_imports[(void*)level0] = VulkanImport{memory, mipmap};
    return (void*)level0;
}

// Unmap a pointer / array returned by the imports above (no-op for anything
// else, e.g. a handle returned when the memory was not exported).
static void release_vulkan_import(void* mapped) {
    VulkanImport imp;
    {
        std::lock_guard<std::mutex> lock(g_vulkan_imports_mutex);
        auto it = g_vulkan_imports.find(mapped);
        if (it == g_vulkan_imports.end()) return;
        imp = it->second;
        g_vulkan_imports.erase(it);
    }
    if (imp.mipmap) CUDA_CHECK(cudaFreeMipmappedArray(imp.mipmap));
    else CUDA_CHECK(cudaFree(mapped));
    CUDA_CHECK(cudaDestroyExternalMemory(imp.memory));
}

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
    mapper->pci_domain = prop.pciDomainID;
    mapper->pci_bus = prop.pciBusID;
    mapper->pci_device = prop.pciDeviceID;

    // Stream-ordered pool allocation when the device supports it: cudaFree
    // synchronises the whole device, and every temporary tensor ends in a
    // free, so a pool keeps kernels queued back to back.  Freed blocks stay
    // in the pool for reuse.
    int pools_supported = 0;
    cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, device_id);
    bool use_pool = pools_supported != 0;
    if (use_pool) {
        cudaMemPool_t pool;
        CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device_id));
        uint64_t keep = UINT64_MAX;
        CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep));
    }

    mapper->compute_device_allocators[ComputeType::kCUDA] = [device_id, use_pool](AllocationMetadata metadata, void* existing_data) {
        void* ptr;
        CUDA_CHECK(cudaSetDevice(device_id));
        if (use_pool) CUDA_CHECK(cudaMallocAsync(&ptr, metadata.byte_size, 0));
        else CUDA_CHECK(cudaMalloc(&ptr, metadata.byte_size));
        if (existing_data) {
            CUDA_CHECK(cudaMemcpy(ptr, existing_data, metadata.byte_size, cudaMemcpyHostToDevice));
        }
        return new BaseMemoryAllocation(metadata, ptr);
    };

    mapper->compute_device_deallocators[ComputeType::kCUDA] = [device_id, use_pool](void* ptr) {
        CUDA_CHECK(cudaSetDevice(device_id));
        if (use_pool) CUDA_CHECK(cudaFreeAsync(ptr, 0));
        else CUDA_CHECK(cudaFree(ptr));
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
        } else if (original->metadata.compute_device == ComputeType::kVULKAN ||
                   original->metadata.compute_device == ComputeType::kVULKANTEXTURE) {
            // Unmap the import; the memory itself is owned by the VulkanResource.
            std::cout << "[cuda] Releasing Vulkan→CUDA interop mapping for device " << device_id << std::endl;
            CUDA_CHECK(cudaSetDevice(device_id));
            release_vulkan_import(ptr);
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

    // Vulkan allocations in this GPU's memory, viewed as CUDA in place
    // (import_vulkan_buffer / import_vulkan_image above).
    mapper->supports_compute_device[ComputeType::kVULKAN] = true;

    mapper->supports_compute_device[ComputeType::kVULKANTEXTURE] = true;

    // buffer (kVULKAN, or a kLINEAR texture's buffer) → device pointer
    mapper->compute_type_converters[{ComputeType::kVULKAN, ComputeType::kCUDA}] =
        [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) -> void* {
            return import_vulkan_buffer((VulkanResource*)ptr, metadata.byte_size, device_id);
        };

    // optimal-tiled image (kVULKANTEXTURE) → cudaArray_t (level 0), for
    // surface / texture objects — the counterpart of the OpenGL texture
    // interop above.  The vulkan plugin sends buffer-backed textures to the
    // buffer import instead.
    mapper->compute_type_converters[{ComputeType::kVULKANTEXTURE, ComputeType::kCUDA}] =
        [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) -> void* {
            VulkanResource* r = (VulkanResource*)ptr;
            if (r && r->buffer) return import_vulkan_buffer(r, metadata.byte_size, device_id);
            return import_vulkan_image(r, device_id);
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
    device->default_memory_device_id = device_id;

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
