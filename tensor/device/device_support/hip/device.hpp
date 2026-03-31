#include "device/common.hpp"
#include <hip/hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_gl_interop.h>
#include "shape.hpp"

void HIP_ERROR_CHECK(hipError_t err){
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("HIP error");
    }
}

AllocationMap* create_hip_mapper(int device_id){
    // Create HIP AllocationMap
    AllocationMap* mapper = new AllocationMap();
        hipDeviceProp_t prop;
        HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, device_id));
        mapper->default_compute_type = ComputeType::kHIP;
        mapper->default_allocator_type = ComputeType::kHIP;
        
        mapper->supports_compute_device[ComputeType::kHIP] = true;
        mapper->compute_device_allocators[ComputeType::kHIP] = [device_id](AllocationMetadata meta, void* existing_data) {
            void* ptr;
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipMalloc(&ptr, meta.byte_size));
            if (existing_data != nullptr){
                HIP_ERROR_CHECK(hipMemcpy(ptr, existing_data, meta.byte_size, hipMemcpyHostToDevice));
            }
            
            return new BaseMemoryAllocation(meta, ptr);
        };
        mapper->compute_device_deallocators[ComputeType::kHIP] = [device_id](void* ptr) {
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipFree(ptr));
        };

        mapper->memory_type_converters[MemoryType::kDDR] = [device_id](void* ptr, AllocationMetadata meta) {
            auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
            auto host_ptr = host_device.allocate(meta); // dont pass existing data here, as it doesnt know what to do with it
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipMemcpy((char*)host_ptr->data, (char*)ptr, meta.byte_size, hipMemcpyDeviceToHost));
            
            return host_ptr;
        };

        // mapper->memory_type_converters[MemoryType::kHIP_VRAM] = [device_id](void* ptr, AllocationMetadata meta) {
        //     return ptr; // No conversion needed
        // };

        mapper->memory_type_converters[MemoryType::kCUDA_VRAM] = [device_id](void* ptr, AllocationMetadata meta) {
            std::cerr << "Conversion from HIP_VRAM to CUDA_VRAM not implemented" << std::endl;
            return nullptr;
        };

        mapper->compute_type_converters[std::tuple<ComputeType,ComputeType>({ComputeType::kOPENGL,ComputeType::kHIP})] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            std::cout << "Registering OpenGL buffer with HIP for interop, buffer ID: " << ptr << std::endl;
            hipGraphicsResource_t m = nullptr;
            auto err = hipGraphicsGLRegisterBuffer(&m, (GLuint)(long long)ptr, hipGraphicsRegisterFlagsNone);
            if (err != hipSuccess) {
                if(err == 999){
                    std::cout << "HIP–GL interop registration failed with error code: " << err << " (USING wrong gpu for openGL)" << std::endl;
                }
                std::cout << "HIP–GL interop registration failed with error code: " << err << std::endl;
                std::string errorMsg = "Failed to register HIP-GL interop: " + 
                                    std::string(hipGetErrorString(err));
                throw std::runtime_error(errorMsg);
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

        // mapper->compute_type_converters[{ComputeType::kOPENGLTEXTURE, ComputeType::kHIP}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
        //     hipGraphicsResource* m = nullptr;
        //     hipGraphicsResource_t* resource = &m;
        //     // read and writable 2d array
        //     auto err = hipGraphicsGLRegisterImage(resource, (((GLuint)(size_t)ptr)), GL_TEXTURE_2D, hipGraphicsRegisterFlagsSurfaceLoadStore);
        //     if (err != hipSuccess) {
        //         if (err == 999) {
        //             std::cout << "HIP–GL interop registration failed with error code: " << err << " (USING wrong gpu for openGL)" << std::endl;
        //         }
        //         std::cout << "HIP–GL interop registration failed with error code: " << err << std::endl;
        //         std::string errorMsg = "Failed to register HIP-GL texture interop: " + 
        //             std::string(hipGetErrorString(err));
        //         throw std::runtime_error(errorMsg);   
        //     }
            
        //     auto errMap = hipGraphicsMapResources(1, resource);
        //     if (errMap != hipSuccess) {
        //         throw std::runtime_error("Failed to map HIP-GL texture resources: " + std::string(hipGetErrorString(errMap)));
        //     }
            
        //     hipArray_t* array = new hipArray_t[1];
        //     auto hipError = hipGraphicsSubResourceGetMappedArray(array, resource[0], 0, 0);
        //     if (hipError != hipSuccess) {
        //         throw std::runtime_error("Failed to get mapped array from HIP-GL texture resource: " + std::string(hipGetErrorString(hipError)));
        //     }
            
        //     return array;
        // };

        mapper->synchronize_function = [device_id]() {
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipDeviceSynchronize());
        };
        

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.memory_type_converters[MemoryType::kHIP_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr); // pass the host pointer as existing data to the HIP allocator, which will copy it to the device
        };

        auto& mem_device_disk = global_device_manager.get_device(MemoryType::kDISK, 0);
        mem_device_disk.memory_type_converters[MemoryType::kHIP_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr); // pass the host pointer as existing data to the HIP allocator, which will copy it to the device
        };

        mapper->this_device_type = MemoryType::kHIP_VRAM;
    return mapper;
}



ComputeDeviceBase* create_hip_compute_device(int device_id){
    ComputeDeviceBase* device = new ComputeDeviceBase();
    device->supports_memory_location[MemoryType::kHIP_VRAM] = true;
    device->default_memory_type = MemoryType::kHIP_VRAM;

    hipDeviceProp_t prop;
    HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, device_id));
    
    device->compute_units = prop.multiProcessorCount;
    device->shared_memory_size = prop.sharedMemPerBlock;

    if(prop.canMapHostMemory){
                        std::cout << "Device " << device_id << " supports mapping host memory, enabling zero-copy access" << std::endl;
                device->supports_memory_location[MemoryType::kDDR] = true;
                // device->default_memory_type = MemoryType::kDDR;

                auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
                mem_device.supports_compute_device[ComputeType::kHIP] = true;
                // mem_device.default_compute_type = ComputeType::kHIP;
                // mem_device.default_allocator_type = ComputeType::kHIP;
                mem_device.compute_device_allocators[ComputeType::kHIP] = [device_id](AllocationMetadata meta, void* existing_data) {
                    void* ptr;
                    HIP_ERROR_CHECK(hipSetDevice(device_id));
                    HIP_ERROR_CHECK(hipMallocManaged(&ptr, meta.byte_size, hipMemAttachGlobal));
                    if (existing_data != nullptr){
                        HIP_ERROR_CHECK(hipMemcpy((char*)ptr, (char*)existing_data, meta.byte_size, hipMemcpyHostToDevice));
                    }
                    
                    return new BaseMemoryAllocation(meta, ptr);
                };

                mem_device.compute_type_converters[std::tuple<ComputeType,ComputeType>({ComputeType::kCPU, ComputeType::kHIP})] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                    
                    if (original->metadata.compute_device == ComputeType::kHIP) {
                        return ptr; // No conversion needed, already in hip memory
                    }

                    if (original->metadata.compute_device == ComputeType::kCPU) {
                        // map the existing host pointer into hip address space using hipMallocManaged with hipMemAttachGlobal, which allows it to be accessed from both CPU and hip without explicit copying
                        HIP_ERROR_CHECK(hipHostRegister(ptr, metadata.byte_size, hipHostRegisterMapped));
                        void* device_ptr;
                        HIP_ERROR_CHECK(hipSetDevice(device_id));
                        HIP_ERROR_CHECK(hipHostGetDevicePointer(&device_ptr, ptr, 0));
                        return device_ptr;
                    }

                    throw std::runtime_error("Unsupported compute device for conversion to hip");
                };

                mem_device.compute_type_converters[std::tuple<ComputeType,ComputeType>({ComputeType::kHIP, ComputeType::kCPU})] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                    if (original->metadata.compute_device == ComputeType::kCPU) {
                        return ptr; // No conversion needed, already in CPU memory
                    }

                    if (original->metadata.compute_device == ComputeType::kHIP) {
                        // Since the memory is allocated with hipMallocManaged and hipMemAttachGlobal, it can be accessed directly from the CPU without explicit copying. Just return the original pointer.
                        return ptr;
                    }

                    throw std::runtime_error("Unsupported compute device for conversion to CPU");
                };

                mem_device.compute_mapping_deallocators[ComputeType::kHIP] = [device_id](void* ptr, BaseMemoryAllocation* original) {
                    // No deallocation needed, since HIP can directly access host memory if the device supports it. The original host allocation will be deallocated by the CPU allocator's deallocator.
                };

                mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [device_id](void* ptr, BaseMemoryAllocation* original) {
                    // No deallocation needed, since the memory is shared and can be accessed from both CPU and hip. The original allocation will be deallocated by its respective allocator's deallocator.
                };

                mem_device.compute_device_deallocators[ComputeType::kHIP] = [device_id](void* ptr) {
                    HIP_ERROR_CHECK(hipSetDevice(device_id));
                    HIP_ERROR_CHECK(hipFree(ptr));
                };
    }
    
    return device;
}

int count_hip_devices(){
    int count;
    HIP_ERROR_CHECK(hipGetDeviceCount(&count));
    
    std::cout << "HIP Device Count: " << count << std::endl;
    return count;
}