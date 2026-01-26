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
        mapper->compute_device_allocators[ComputeType::kCUDA] = [device_id](size_t size) {
            void* ptr;
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaMalloc(&ptr, size));
            
            return ptr;
        };
        mapper->compute_device_deallocators[ComputeType::kCUDA] = [device_id](void* ptr) {
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaFree(ptr));
            
        };

        mapper->memory_type_converters[MemoryType::kDDR] = [device_id](void* ptr, size_t size) {
            auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
            void* host_ptr = host_device.allocate(size);
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaMemcpy(host_ptr, ptr, size, cudaMemcpyDeviceToHost));
            
            return host_ptr;
        };

        mapper->memory_type_converters[MemoryType::kCUDA_VRAM] = [device_id](void* ptr, size_t size) {
            return ptr; // No conversion needed
        };

        mapper->memory_type_converters[MemoryType::kCUDA_VRAM] = [device_id](void* ptr, size_t size) {
            std::cerr << "Conversion from Cuda_VRAM to CUDA_VRAM not implemented" << std::endl;
            return nullptr;
        };
       
        mapper->compute_device_massagers[ComputeType::kOPENGL] = [](unsigned int ptr) {
            cudaGraphicsResource* m = nullptr;
            cudaGraphicsResource_t* resource = &m;
            auto err = cudaGraphicsGLRegisterBuffer(resource, (GLuint)ptr, cudaGraphicsRegisterFlagsNone);
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

        mapper->synchronize_function = [device_id]() {
            CUDA_ERROR_CHECK(cudaSetDevice(device_id));
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        };

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.memory_type_converters[MemoryType::kCUDA_VRAM] = [mapper](void* ptr, size_t size) {
            void* device_ptr = mapper->allocate(size);
            CUDA_ERROR_CHECK(cudaMemcpy(device_ptr, ptr, size, cudaMemcpyHostToDevice));
            return device_ptr;
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

