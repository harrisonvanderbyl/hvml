#include "device/common.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(__call)                                      \
    do {                                                   \
        __call;                                               \
    } while (0)

AllocationMap* create_cuda_mapper(int device_id){
   

        AllocationMap* mapper = new AllocationMap();
        cudaDeviceProp prop;
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, device_id));
        mapper->default_compute_type = ComputeType::kCUDA;
        mapper->default_allocator_type = ComputeType::kCUDA;
        
        mapper->supports_compute_device[ComputeType::kCUDA] = true;
        mapper->compute_device_allocators[ComputeType::kCUDA] = [device_id](AllocationMetadata metadata, void* existing_data) {
            void* ptr;
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaMalloc(&ptr, metadata.byte_size));
            if (existing_data != nullptr){
                CUDA_ERROR_CHECK(cudaMemcpy(ptr, existing_data, metadata.byte_size, cudaMemcpyHostToDevice));
            }
            
            return new BaseMemoryAllocation(metadata, ptr);
        };
        mapper->compute_device_deallocators[ComputeType::kCUDA] = [device_id](void* ptr) {
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaFree(ptr));
            
        };

        mapper->memory_type_converters[MemoryType::kDDR] = [device_id](void* ptr, AllocationMetadata meta)
        {
            auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
            auto host_ptr = host_device.allocate(meta); // dont pass existing data to host allocator, it doesnt know how to handle that
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaMemcpy((char*)host_ptr->data, (char*)ptr, meta.byte_size, cudaMemcpyDeviceToHost));
            
            return host_ptr;
        };

        // mapper->memory_type_converters[MemoryType::kHIP_VRAM] = [device_id](void* ptr, AllocationMetadata meta)
        // {
        //     auto& hip_device = global_device_manager.get_device(MemoryType::kHIP_VRAM, 0);
        //     // copy to new cpu memory first
        //     uint8_t* host_ptr = (uint8_t*)malloc(meta.byte_size);
        //     CUDA_ERROR_CHECK(cudaSetDevice(device_id));
        //     CUDA_ERROR_CHECK(cudaMemcpy(host_ptr, (uint8_t*)ptr, meta.byte_size, cudaMemcpyDeviceToHost));
        //     CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        //     void* hip_ptr = hip_device.allocate(meta, host_ptr); // allocate new memory on hip device
        //     hip_device.synchronize_function();
        //     free(host_ptr); // free temporary host memory
        //     return hip_ptr;
        // };

       
        mapper->compute_type_converters[{ComputeType::kOPENGL,ComputeType::kCUDA}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            cudaGraphicsResource* m = nullptr;
            cudaGraphicsResource_t* resource = &m;
            auto err = cudaGraphicsGLRegisterBuffer(resource, (GLuint)(size_t)ptr, cudaGraphicsRegisterFlagsNone);
            if (err != cudaSuccess) {
                if(err == 999){
                    std::cout << "CUDA–GL interop registration failed with error code: " << err << " (USING wrong gpu for openGL)" << std::endl;
                }
                std::cout << "CUDA–GL interop registration failed with error code: " << err << std::endl;
                std::string errorMsg = "Failed to register CUDA-GL interop: " + 
                                    std::string(cudaGetErrorString(err));
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

        mapper->compute_type_converters[{ComputeType::kOPENGLTEXTURE, ComputeType::kCUDA}] = [](void* ptra, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            cudaGraphicsResource* m = nullptr;
            cudaGraphicsResource_t* resource = &m;
            // read and writable 2d array
            GLuint ptr = (GLuint)(size_t)original->data;
            GLuint texture_type = GL_TEXTURE_2D;
            GLuint flags = cudaGraphicsRegisterFlagsSurfaceLoadStore;
            if(original->metadata.format != 0){
                return (void*)nullptr; // unsupported format
            }

            
            auto err = cudaGraphicsGLRegisterImage(resource, ptr, texture_type, flags);
            if (err != cudaSuccess) {
                if (err == 999) {
                    std::cout << "CUDA–GL interop registration failed with error code: " << err << " (USING wrong gpu for openGL)" << std::endl;
                }
                std::cout << "CUDA–GL interop registration failed with error code: " << err << std::endl;
                std::string errorMsg = "Failed to register CUDA-GL texture interop: " + 
                    std::string(cudaGetErrorString(err));
                std::cout << "Data pointer: " << original->data << std::endl;
                std::cout << metadata << std::endl;
                std::cout << original->metadata << std::endl;
                throw std::runtime_error(errorMsg);   
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
            // For CUDA-OpenGL interop, we need to unmap and unregister the resource instead of freeing memory directly
            std::cout << "Deallocating CUDA mapped, original compute device: " << original->metadata.compute_device << std::endl;
            if (original->metadata.compute_device == ComputeType::kOPENGL) {
                
            } else if (original->metadata.compute_device == ComputeType::kOPENGLTEXTURE) {
               std::cout << "Unregistering CUDA-GL texture resource for original compute device " << original->metadata.compute_device << std::endl;
               std::cout << "Original data pointer: " << original->data << std::endl;
               std::cout << "Original metadata: " << original->metadata << std::endl;
            } else {
                std::cerr << "No CUDA mapping deallocator found for original compute device " << original->metadata.compute_device << "{" << int(original->metadata.compute_device) << "} on device " << ComputeType::kCUDA << std::endl;
                throw std::runtime_error("No CUDA mapping deallocator found for original compute device");
            }
        };

        mapper->synchronize_function = [device_id]() {
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        };

        

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.memory_type_converters[MemoryType::kCUDA_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr); // pass the host pointer as existing data to the CUDA allocator, which will copy it to the device
        };

        auto& mem_device_disk = global_device_manager.get_device(MemoryType::kDISK, 0);
        mem_device_disk.memory_type_converters[MemoryType::kCUDA_VRAM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr); // pass the host pointer as existing data to the CUDA allocator, which will copy it to the device
        };

        mapper->this_device_type = MemoryType::kCUDA_VRAM;
        return mapper;        
    }


    
ComputeDeviceBase* create_cuda_compute_device(int device_id){
    ComputeDeviceBase* device = new ComputeDeviceBase();
    
        device->supports_memory_location[MemoryType::kCUDA_VRAM] = true;
        device->default_memory_type = MemoryType::kCUDA_VRAM;

        cudaDeviceProp prop;
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, device_id));


        device->compute_units = prop.multiProcessorCount;
        device->shared_memory_size = prop.sharedMemPerBlock;
        

        if(prop.canMapHostMemory){
                std::cout << "Device " << device_id << " supports mapping host memory, enabling zero-copy access" << std::endl;
                device->supports_memory_location[MemoryType::kDDR] = true;
                // device->default_memory_type = MemoryType::kDDR;

                auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
                mem_device.supports_compute_device[ComputeType::kCUDA] = true;
                // mem_device.default_compute_type = ComputeType::kCUDA;
                // mem_device.default_allocator_type = ComputeType::kCUDA;
                mem_device.compute_device_allocators[ComputeType::kCUDA] = [device_id](AllocationMetadata meta, void* existing_data) {
                    void* ptr;
                    CUDA_ERROR_CHECK(cudaSetDevice(device_id));
                    CUDA_ERROR_CHECK(cudaMallocManaged(&ptr, meta.byte_size, cudaMemAttachGlobal));
                    if (existing_data != nullptr){
                        CUDA_ERROR_CHECK(cudaMemcpy((char*)ptr, (char*)existing_data, meta.byte_size, cudaMemcpyHostToDevice));
                    }
                    
                    return new BaseMemoryAllocation(meta, ptr);
                };

                mem_device.compute_type_converters[std::tuple<ComputeType,ComputeType>({ComputeType::kCPU, ComputeType::kCUDA})] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                    
                    if (original->metadata.compute_device == ComputeType::kCUDA) {
                        return ptr; // No conversion needed, already in CUDA memory
                    }

                    if (original->metadata.compute_device == ComputeType::kCPU) {
                        // map the existing host pointer into CUDA address space using cudaMallocManaged with cudaMemAttachGlobal, which allows it to be accessed from both CPU and CUDA without explicit copying
                        cudaHostRegister(ptr, metadata.byte_size, cudaHostRegisterMapped);
                        void* device_ptr;
                        CUDA_ERROR_CHECK(cudaSetDevice(device_id));
                        CUDA_ERROR_CHECK(cudaHostGetDevicePointer(&device_ptr, ptr, 0));
                        return device_ptr;
                    }

                    throw std::runtime_error("Unsupported compute device for conversion to CUDA");
                };

                mem_device.compute_type_converters[std::tuple<ComputeType,ComputeType>({ComputeType::kCUDA, ComputeType::kCPU})] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                    if (original->metadata.compute_device == ComputeType::kCPU) {
                        return ptr; // No conversion needed, already in CPU memory
                    }

                    if (original->metadata.compute_device == ComputeType::kCUDA) {
                        // Since the memory is allocated with cudaMallocManaged and cudaMemAttachGlobal, it can be accessed directly from the CPU without explicit copying. Just return the original pointer.
                        return ptr;
                    }

                    throw std::runtime_error("Unsupported compute device for conversion to CPU");
                };

                mem_device.compute_mapping_deallocators[ComputeType::kCUDA] = [device_id](void* ptr, BaseMemoryAllocation* original) {
                    // No deallocation needed, since HIP can directly access host memory if the device supports it. The original host allocation will be deallocated by the CPU allocator's deallocator.
                };

                mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [device_id](void* ptr, BaseMemoryAllocation* original) {
                    // No deallocation needed, since the memory is shared and can be accessed from both CPU and CUDA. The original allocation will be deallocated by its respective allocator's deallocator.
                };

                mem_device.compute_device_deallocators[ComputeType::kCUDA] = [device_id](void* ptr) {
                    CUDA_ERROR_CHECK(cudaSetDevice(device_id));
                    CUDA_ERROR_CHECK(cudaFree(ptr));
                };
        }

    return device;
    }



int count_cuda_devices(){
    int count;
    CUDA_ERROR_CHECK(cudaGetDeviceCount(&count));
    
    std::cout << "Cuda Device Count: " << count << std::endl;
    return count;
}

