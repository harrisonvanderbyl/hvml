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
        
        mapper->supports_compute_device[ComputeType::kHIP] = true;
        mapper->compute_device_allocators[ComputeType::kHIP] = [device_id](Shape<-1> size, size_t bitsize, void* existing_data) {
            void* ptr;
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipMalloc(&ptr, size.total_size() * bitsize));
            if (existing_data != nullptr){
                HIP_ERROR_CHECK(hipMemcpy(ptr, existing_data, size.total_size() * bitsize, hipMemcpyHostToDevice));
            }
            
            return ptr;
        };
        mapper->compute_device_deallocators[ComputeType::kHIP] = [device_id](void* ptr) {
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipFree(ptr));
        };

        mapper->memory_type_converters[MemoryType::kDDR] = [device_id](Shape<-1> size, size_t bitsize, ComputeType targetct, void* ptr) {
            auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
            void* host_ptr = host_device.allocate(size, bitsize, targetct); // dont pass existing data here, as it doesnt know what to do with it
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipMemcpy((char*)host_ptr, (char*)ptr, size.total_size()*bitsize, hipMemcpyDeviceToHost));
            
            return host_ptr;
        };

        mapper->memory_type_converters[MemoryType::kHIP_VRAM] = [device_id](Shape<-1> size, size_t bitsize, ComputeType targetct, void* ptr) {
            return ptr; // No conversion needed
        };

        mapper->memory_type_converters[MemoryType::kCUDA_VRAM] = [device_id](Shape<-1> size, size_t bitsize, ComputeType targetct, void* ptr) {
            std::cerr << "Conversion from HIP_VRAM to CUDA_VRAM not implemented" << std::endl;
            return nullptr;
        };

        mapper->compute_type_converters[std::tuple<ComputeType,ComputeType>({ComputeType::kOPENGL,ComputeType::kHIP})] = [](void* ptr) {
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

        mapper->synchronize_function = [device_id]() {
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipDeviceSynchronize());
        };
        

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.memory_type_converters[MemoryType::kHIP_VRAM] = [mapper](Shape<-1> size, size_t bitsize, ComputeType targetct, void* ptr) {
            return mapper->allocate(size, bitsize, targetct, ptr);
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

    if(prop.canMapHostMemory){
        device->supports_memory_location[MemoryType::kDDR] = true;
        device->default_memory_type = MemoryType::kDDR;

        auto& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.supports_compute_device[ComputeType::kHIP] = true;
        mem_device.default_compute_type = ComputeType::kHIP;
        mem_device.compute_device_allocators[ComputeType::kHIP] = [device_id](Shape<-1> size, size_t bitsize, void* existing_data) {
            void* ptr;
            HIP_ERROR_CHECK(hipSetDevice(device_id));
            HIP_ERROR_CHECK(hipMallocManaged(&ptr, size.total_size() * bitsize, hipMemAttachGlobal));
            if (existing_data != nullptr){
                HIP_ERROR_CHECK(hipMemcpy((char*)ptr, (char*)existing_data, size.total_size() * bitsize, hipMemcpyHostToDevice));
            }
            
            return ptr;
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