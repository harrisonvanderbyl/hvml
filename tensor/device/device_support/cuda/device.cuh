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
        
        mapper->supports_compute_device[ComputeType::kCUDA] = true;
        mapper->compute_device_allocators[ComputeType::kCUDA] = [device_id](Shape<-1> size, size_t bitsize, void* existing_data) {
            void* ptr;
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaMalloc(&ptr, size.total_size() * bitsize));
            if (existing_data != nullptr){
                CUDA_ERROR_CHECK(cudaMemcpy(ptr, existing_data, size.total_size() * bitsize, cudaMemcpyHostToDevice));
            }
            
            return ptr;
        };
        mapper->compute_device_deallocators[ComputeType::kCUDA] = [device_id](void* ptr) {
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaFree(ptr));
            
        };

        mapper->memory_type_converters[MemoryType::kDDR] = [device_id](Shape<-1> size, size_t bitsize, ComputeType compute_type, void* ptr) {
            auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
            void* host_ptr = host_device.allocate(size, bitsize, compute_type); // dont pass existing data to host allocator, it doesnt know how to handle that
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaMemcpy((char*)host_ptr, (char*)ptr, size.total_size()*bitsize, cudaMemcpyDeviceToHost));
            
            return host_ptr;
        };

       
        mapper->compute_type_converters[{ComputeType::kOPENGL,ComputeType::kCUDA}] = [](void* ptr) {
            cudaGraphicsResource* m = nullptr;
            cudaGraphicsResource_t* resource = &m;
            auto err = cudaGraphicsGLRegisterBuffer(resource, (GLuint)(unsigned long long)ptr, cudaGraphicsRegisterFlagsNone);
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

        mapper->compute_type_converters[{ComputeType::kOPENGLTEXTURE, ComputeType::kCUDA}] = [](void* ptr) {
            cudaGraphicsResource* m = nullptr;
            cudaGraphicsResource_t* resource = &m;
            // read and writable 2d array
            auto err = cudaGraphicsGLRegisterImage(resource, (((GLuint)(unsigned long long)ptr) - 0x10000), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
            if (err != cudaSuccess) {
                if (err == 999) {
                    std::cout << "CUDA–GL interop registration failed with error code: " << err << " (USING wrong gpu for openGL)" << std::endl;
                }
                std::cout << "CUDA–GL interop registration failed with error code: " << err << std::endl;
                std::string errorMsg = "Failed to register CUDA-GL texture interop: " + 
                    std::string(cudaGetErrorString(err));
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
            
            return array;
        };

        mapper->image_copy_function = [device_id,mapper](void* dst, Shape<-1> size, void* src) {
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            // for cudaArray, we need to use cudaMemcpy2DToArray (not cudaMemcpy or cudaMemcpy2D)
            cudaArray_t dst_array = static_cast<cudaArray_t*>(dst)[0];
            
            CUDA_ERROR_CHECK(cudaMemcpyToArray(dst_array, 0, 0, (void*)src, size.total_size() * sizeof(char), cudaMemcpyHostToDevice));
            mapper->synchronize_function();
        };

        mapper->synchronize_function = [device_id]() {
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        };

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.memory_type_converters[MemoryType::kCUDA_VRAM] = [mapper](Shape<-1> size, size_t bitsize, ComputeType targetct, void* ptr) {
            return mapper->allocate(size, bitsize, targetct, ptr);
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
        

        // if(prop.canMapHostMemory){
        //     supports_memory_location[MemoryType::kDDR] = true;
        //     default_memory_type = MemoryType::kDDR;

        //     auto& mem_device = get_device(MemoryType::kDDR, 0);
        //     mem_device.supports_compute_device[ComputeType::kCUDA] = true;
        //     mem_device.default_compute_type = ComputeType::kCUDA;
        //     mem_device.compute_device_allocators[ComputeType::kCUDA] = [device](size_t size) {
        //         void* ptr;
        //         CUDA_ERROR_CHECK(cudaSetDevice(device));
        //         CUDA_ERROR_CHECK(cudaMallocManaged(&ptr, size));
                
        //         return ptr;
        //     };
        //     mem_device.compute_device_deallocators[ComputeType::kCUDA] = [device](void* ptr) {
        //         CUDA_ERROR_CHECK(cudaSetDevice(device));
        //         CUDA_ERROR_CHECK(cudaFree(ptr));
        //     };
        // }
    return device;
    }



int count_cuda_devices(){
    int count;
    CUDA_ERROR_CHECK(cudaGetDeviceCount(&count));
    
    std::cout << "Cuda Device Count: " << count << std::endl;
    return count;
}

