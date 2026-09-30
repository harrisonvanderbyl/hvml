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
#include <unistd.h>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include "vulkan_interop.hpp"

#define HIP_CHECK(__call)                                                      \
    do {                                                                       \
        hipError_t __err = __call;                                             \
        if (__err != hipSuccess) {                                             \
            std::cerr << "HIP error: " << hipGetErrorString(__err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  Vulkan → HIP external memory interop
//
//  A Vulkan allocation in HIP memory exports its VkDeviceMemory as an opaque
//  fd (vulkan plugin).  HIP imports it and maps:
//    buffers → a device pointer (hipExternalMemoryGetMappedBuffer)
//    images  → a hipArray_t     (hipExternalMemoryGetMappedMipmappedArray, ROCm 6+)
//  over the same memory — no copy.  Each import is recorded so the mapping
//  deallocator can unmap it and destroy the external memory when the
//  allocation is freed.
// ---------------------------------------------------------------------------

#if defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
#define HVML_HIP_VULKAN_IMAGES 1
#endif

struct VulkanImport {
    hipExternalMemory_t memory = nullptr;
#ifdef HVML_HIP_VULKAN_IMAGES
    hipMipmappedArray_t mipmap = nullptr;   // images only
#endif
};

static std::mutex g_vulkan_imports_mutex;
static std::map<void*, VulkanImport> g_vulkan_imports;   // mapped pointer / array → import

static hipExternalMemory_t import_vulkan_memory(const VulkanResource* r, int device_id) {
    if (!r || r->fd < 0) {
        std::cerr << "[hip] Vulkan memory not exported (no fd) — no HIP view" << std::endl;
        return nullptr;
    }
    HIP_CHECK(hipSetDevice(device_id));
    int fd = dup(r->fd);   // HIP owns the fd once the import succeeds
    if (fd < 0) {
        std::cerr << "[hip] dup(fd) failed for Vulkan→HIP interop" << std::endl;
        return nullptr;
    }
    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = r->alloc_size;
    desc.flags = r->dedicated ? hipExternalMemoryDedicated : 0;
    hipExternalMemory_t memory = nullptr;
    hipError_t err = hipImportExternalMemory(&memory, &desc);
    if (err != hipSuccess) {
        close(fd);
        std::cerr << "[hip] hipImportExternalMemory failed: " << hipGetErrorString(err) << std::endl;
        return nullptr;
    }
    return memory;
}

static void* import_vulkan_buffer(const VulkanResource* r, size_t bytes, int device_id) {
    hipExternalMemory_t memory = import_vulkan_memory(r, device_id);
    if (!memory) return nullptr;
    hipExternalMemoryBufferDesc desc{};
    desc.offset = 0;
    desc.size = bytes;
    void* ptr = nullptr;
    hipError_t err = hipExternalMemoryGetMappedBuffer(&ptr, memory, &desc);
    if (err != hipSuccess) {
        std::cerr << "[hip] hipExternalMemoryGetMappedBuffer failed: " << hipGetErrorString(err) << std::endl;
        hipDestroyExternalMemory(memory);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_vulkan_imports_mutex);
    g_vulkan_imports[ptr] = VulkanImport{memory};
    return ptr;
}

static void* import_vulkan_image(const VulkanResource* r, int device_id) {
#ifdef HVML_HIP_VULKAN_IMAGES
    if (!r || !r->image) return nullptr;
    VulkanFormatChannels ch = vulkan_format_channels(r->format);
    if (!ch.ok) {
        std::cerr << "[hip] Vulkan image format " << r->format << " has no HIP array equivalent" << std::endl;
        return nullptr;
    }
    hipExternalMemory_t memory = import_vulkan_memory(r, device_id);
    if (!memory) return nullptr;

    hipChannelFormatKind kind = ch.kind == VulkanFormatChannels::kFloat  ? hipChannelFormatKindFloat
                              : ch.kind == VulkanFormatChannels::kSigned ? hipChannelFormatKindSigned
                                                                         : hipChannelFormatKindUnsigned;
    hipExternalMemoryMipmappedArrayDesc desc{};
    desc.offset = 0;
    desc.formatDesc = hipCreateChannelDesc(ch.x, ch.y, ch.z, ch.w, kind);
    desc.extent = make_hipExtent(r->width, r->height, 0);
    desc.flags = 0;
    if (r->image_usage & kVkImageUsageStorage) desc.flags |= hipArraySurfaceLoadStore;
#ifdef hipArrayColorAttachment
    if (r->image_usage & kVkImageUsageColorAttachment) desc.flags |= hipArrayColorAttachment;
#endif
    desc.numLevels = 1;

    // Declared in the headers but not exported by libamdhip64 on Linux
    // (hip_hcc.map lacks it as of ROCm 7.0), so look it up at run time: a
    // runtime that exports it gets image import, others report it missing.
    using MappedMipmappedArrayFn = hipError_t (*)(hipMipmappedArray_t*, hipExternalMemory_t,
                                                  const hipExternalMemoryMipmappedArrayDesc*);
    static auto mapped_mipmapped_array =
        (MappedMipmappedArrayFn)dlsym(RTLD_DEFAULT, "hipExternalMemoryGetMappedMipmappedArray");
    if (!mapped_mipmapped_array) {
        std::cerr << "[hip] this HIP runtime does not export hipExternalMemoryGetMappedMipmappedArray — "
                     "optimal-tiled Vulkan images have no HIP view (allocate them kLINEAR for a HIP pointer)"
                  << std::endl;
        hipDestroyExternalMemory(memory);
        return nullptr;
    }

    hipMipmappedArray_t mipmap = nullptr;
    hipError_t err = mapped_mipmapped_array(&mipmap, memory, &desc);
    if (err != hipSuccess) {
        std::cerr << "[hip] hipExternalMemoryGetMappedMipmappedArray failed: " << hipGetErrorString(err) << std::endl;
        hipDestroyExternalMemory(memory);
        return nullptr;
    }
    hipArray_t level0 = nullptr;
    err = hipGetMipmappedArrayLevel(&level0, mipmap, 0);
    if (err != hipSuccess) {
        std::cerr << "[hip] hipGetMipmappedArrayLevel failed: " << hipGetErrorString(err) << std::endl;
        hipFreeMipmappedArray(mipmap);
        hipDestroyExternalMemory(memory);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_vulkan_imports_mutex);
    VulkanImport imp;
    imp.memory = memory;
    imp.mipmap = mipmap;
    g_vulkan_imports[(void*)level0] = imp;
    return (void*)level0;
#else
    std::cerr << "[hip] Vulkan image import needs ROCm 6 or newer" << std::endl;
    return nullptr;
#endif
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
#ifdef HVML_HIP_VULKAN_IMAGES
    if (imp.mipmap) HIP_CHECK(hipFreeMipmappedArray(imp.mipmap));
    else
#endif
    HIP_CHECK(hipFree(mapped));
    HIP_CHECK(hipDestroyExternalMemory(imp.memory));
}

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
    mapper->pci_domain = prop.pciDomainID;
    mapper->pci_bus = prop.pciBusID;
    mapper->pci_device = prop.pciDeviceID;

    // Stream-ordered pool allocation when the device supports it (hipFree
    // synchronises the device on every temporary tensor).
    int pools_supported = 0;
    hipDeviceGetAttribute(&pools_supported, hipDeviceAttributeMemoryPoolsSupported, device_id);
    bool use_pool = pools_supported != 0;
    if (use_pool) {
        hipMemPool_t pool;
        HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, device_id));
        uint64_t keep = UINT64_MAX;
        HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &keep));
    }

    mapper->compute_device_allocators[ComputeType::kHIP] = [device_id, use_pool](AllocationMetadata meta, void* existing_data) {
        void* ptr;
        HIP_CHECK(hipSetDevice(device_id));
        if (use_pool) HIP_CHECK(hipMallocAsync(&ptr, meta.byte_size, 0));
        else HIP_CHECK(hipMalloc(&ptr, meta.byte_size));
        if (existing_data) {
            HIP_CHECK(hipMemcpy(ptr, existing_data, meta.byte_size, hipMemcpyHostToDevice));
        }
        return new BaseMemoryAllocation(meta, ptr);
    };

    mapper->compute_device_deallocators[ComputeType::kHIP] = [device_id, use_pool](void* ptr) {
        HIP_CHECK(hipSetDevice(device_id));
        if (use_pool) HIP_CHECK(hipFreeAsync(ptr, 0));
        else HIP_CHECK(hipFree(ptr));
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
    device->default_memory_device_id = device_id;

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

    // Vulkan allocations in this GPU's memory, viewed as HIP in place
    // (import_vulkan_buffer / import_vulkan_image above).
    auto& hip_mem_device = global_device_manager.get_device(MemoryType::kHIP_VRAM, device_id);
    hip_mem_device.supports_compute_device[ComputeType::kVULKAN] = true;
    hip_mem_device.supports_compute_device[ComputeType::kVULKANTEXTURE] = true;

    // buffer (kVULKAN, or a kLINEAR texture's buffer) → device pointer
    hip_mem_device.compute_type_converters[{ComputeType::kVULKAN, ComputeType::kHIP}] =
        [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) -> void* {
            return import_vulkan_buffer((VulkanResource*)ptr, metadata.byte_size, device_id);
        };

    // optimal-tiled image (kVULKANTEXTURE) → hipArray_t (level 0), for
    // surface / texture objects.  The vulkan plugin sends buffer-backed
    // textures to the buffer import instead.
    hip_mem_device.compute_type_converters[{ComputeType::kVULKANTEXTURE, ComputeType::kHIP}] =
        [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) -> void* {
            VulkanResource* r = (VulkanResource*)ptr;
            if (r && r->buffer) return import_vulkan_buffer(r, metadata.byte_size, device_id);
            return import_vulkan_image(r, device_id);
        };

    // Unmap Vulkan imports; other HIP views of this memory (OpenGL interop)
    // own nothing.  The memory itself belongs to the Vulkan / GL resource.
    hip_mem_device.compute_mapping_deallocators[ComputeType::kHIP] =
        [device_id](void* ptr, BaseMemoryAllocation* original) {
            if (original->metadata.compute_device == ComputeType::kVULKAN ||
                original->metadata.compute_device == ComputeType::kVULKANTEXTURE) {
                HIP_CHECK(hipSetDevice(device_id));
                release_vulkan_import(ptr);
            }
        };

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
