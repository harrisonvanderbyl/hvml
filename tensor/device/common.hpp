#include <iostream>
#include <map>
#include <functional>
#include "enums/device.hpp"
// #include "tensor/enums/device_support/x86/device.hpp"
#ifndef DEVICE_HPP
#define DEVICE_HPP


struct AllocationMap{
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

    

    public:
    MemoryType this_device_type = MemoryType::kUnknown_MEM;

    operator MemoryType() const {
        return this_device_type;
    }
    
    // map of supported compute devices
    std::map<ComputeType, bool> supports_compute_device;
    
    // map to compute device malloc lambdas
    std::map<ComputeType, std::function<void*(size_t)>> compute_device_allocators;
    std::map<ComputeType, std::function<void(void*)>> compute_device_deallocators;
    std::map<MemoryType, std::function<void*(void*, size_t)>> memory_type_converters;
    std::map<ComputeType, std::function<void*(unsigned int)>> compute_device_massagers;
    std::function<void()> synchronize_function = []() {};
    
    ComputeType default_compute_type = ComputeType::kUnknown;

    int device_id = 0;

    void* allocate(size_t size){
        void* pointer = nullptr;
        if (compute_device_allocators.find(default_compute_type) != compute_device_allocators.end()){
            pointer = compute_device_allocators[default_compute_type](size);
            register_allocation(pointer);
            return pointer;
        }else{
            std::cerr << "No allocator found for default compute type " << default_compute_type << std::endl;
            throw std::runtime_error("No allocator found for default compute type");
        }
    }

    void deallocate(void* ptr){
        if (compute_device_deallocators.find(default_compute_type) != compute_device_deallocators.end()){
            if (unregister_allocation(ptr)) {
                compute_device_deallocators[default_compute_type](ptr);
            }
        }else{
            std::cerr << "No deallocator found for default compute type " << default_compute_type << std::endl;
            throw std::runtime_error("No deallocator found for default compute type");
        }
    }

    void* convert_memory_type(void* ptr, MemoryType target_type, size_t size){
        if (memory_type_converters.find(target_type) != memory_type_converters.end()){
            return memory_type_converters[target_type](ptr, size);
        }else{
            std::cerr << "No memory type converter found for target type " << target_type << std::endl;
            throw std::runtime_error("No memory type converter found for target type");
        }
    }

};


struct ComputeDeviceBase{

    std::map<MemoryType, bool> supports_memory_location = {
        {MemoryType::kDDR, false},
        {MemoryType::kCUDA_VRAM, false},
        {MemoryType::kHIP_VRAM, false},
        {MemoryType::kUnknown_MEM, false}
    };
    
    MemoryType default_memory_type = MemoryType::kUnknown_MEM;
    int compute_units = 0;
    
    ComputeDeviceBase(){
    }

    
};

int count_cuda_devices();
AllocationMap* create_cuda_mapper(int device_id);
ComputeDeviceBase* create_cuda_compute_device(int device_id);

#if !defined(__CUDACC__)
    __weak int count_cuda_devices(){
        return 0;
    };


    __weak AllocationMap* create_cuda_mapper(int device_id){
        return nullptr;
    };


    __weak ComputeDeviceBase* create_cuda_compute_device(int device_id){
        return nullptr;
    }
#endif


int count_hip_devices();
AllocationMap* create_hip_mapper(int device_id);
ComputeDeviceBase* create_hip_compute_device(int device_id);

#if !defined(__HIPCC__)
__weak int count_hip_devices(){
    return 0;
};

__weak AllocationMap* create_hip_mapper(int device_id){
    return nullptr;
};

__weak ComputeDeviceBase* create_hip_compute_device(int device_id){
    return nullptr;
};
#endif


__weak int count_vulkan_devices();
__weak ComputeDeviceBase* create_vulkan_compute_device(int device_id);

__weak int count_opengl_devices();
__weak ComputeDeviceBase* create_opengl_compute_device(int device_id);

__weak int count_cpu_devices(){
    return 1;
}


__weak AllocationMap* create_cpu_mapper(int device_id){
        // No special properties for CPU
    AllocationMap* mapper = new AllocationMap();
    mapper->default_compute_type = ComputeType::kCPU;
    mapper->supports_compute_device[ComputeType::kCPU] = true;
    mapper->compute_device_allocators[ComputeType::kCPU] = [](size_t size) {
        return malloc(size);
    };
    mapper->compute_device_deallocators[ComputeType::kCPU] = [](void* ptr) {
        free(ptr);
    };
    mapper->memory_type_converters[MemoryType::kDDR] = [](void* ptr, size_t size) {
        return ptr; // No conversion needed
    };


    mapper->this_device_type = MemoryType::kDDR;

    return mapper;
}

__weak ComputeDeviceBase* create_cpu_compute_device(int device_id){
    ComputeDeviceBase* device = new ComputeDeviceBase();
    device->supports_memory_location[MemoryType::kDDR] = true;
    device->default_memory_type = MemoryType::kDDR;

    device->compute_units = 16;// get thread mapper later

    return device;
}





struct DeviceManager{
    public:

    std::map<MemoryType, AllocationMap*> memory_devices;
    std::map<MemoryType, int> device_counts;
    std::map<ComputeType, ComputeDeviceBase*> compute_devices;
    std::map<ComputeType, int> compute_device_counts;


    

    DeviceManager(){
        initialize_all_devices();
    }




    void initialize_all_devices(){
        std::cout << "Initializing all devices..." << std::endl;
        std::cout << "Counting devices..." << std::endl;
        device_counts[MemoryType::kDDR] = count_cpu_devices();
        device_counts[MemoryType::kCUDA_VRAM] = count_cuda_devices();
        device_counts[MemoryType::kHIP_VRAM] = count_hip_devices();
        int nym_vulkan_devices = count_vulkan_devices();
        int nym_opengl_devices = count_opengl_devices();
        
        memory_devices[MemoryType::kDDR] = new AllocationMap[device_counts[MemoryType::kDDR]];
        memory_devices[MemoryType::kCUDA_VRAM] = new AllocationMap[device_counts[MemoryType::kCUDA_VRAM]];
        memory_devices[MemoryType::kHIP_VRAM] = new AllocationMap[device_counts[MemoryType::kHIP_VRAM]];

        compute_devices[ComputeType::kCPU] = new ComputeDeviceBase[device_counts[MemoryType::kDDR]];
        compute_devices[ComputeType::kCUDA] = new ComputeDeviceBase[device_counts[MemoryType::kCUDA_VRAM]];
        compute_devices[ComputeType::kHIP] = new ComputeDeviceBase[device_counts[MemoryType::kHIP_VRAM]];

        compute_device_counts[ComputeType::kCPU] = device_counts[MemoryType::kDDR];
        compute_device_counts[ComputeType::kCUDA] = device_counts[MemoryType::kCUDA_VRAM];
        compute_device_counts[ComputeType::kHIP] = device_counts[MemoryType::kHIP_VRAM];
        compute_device_counts[ComputeType::kVULKAN] = nym_vulkan_devices;
        compute_device_counts[ComputeType::kOPENGL] = nym_opengl_devices;

        for (int i = 0; i < device_counts[MemoryType::kDDR]; i++){
            AllocationMap* mapper = create_cpu_mapper(i);
            ((AllocationMap*)memory_devices[MemoryType::kDDR])[i] = *mapper;
            ComputeDeviceBase* device = create_cpu_compute_device(i);
            ((ComputeDeviceBase*)compute_devices[ComputeType::kCPU])[i] = *device;
        }

        for (int i = 0; i < device_counts[MemoryType::kCUDA_VRAM]; i++){
            AllocationMap* mapper = create_cuda_mapper(i);
            ((AllocationMap*)memory_devices[MemoryType::kCUDA_VRAM])[i] = *mapper;
        }

        for (int i = 0; i < device_counts[MemoryType::kHIP_VRAM]; i++){
            AllocationMap* mapper = create_hip_mapper(i);
            ((AllocationMap*)memory_devices[MemoryType::kHIP_VRAM])[i] = *mapper;
        }

        for (int i = 0; i < nym_opengl_devices; i++){
        }

        for (int i = 0; i < device_counts[MemoryType::kCUDA_VRAM]; i++){
            ComputeDeviceBase* device = create_cuda_compute_device(i);
            ((ComputeDeviceBase*)compute_devices[ComputeType::kCUDA])[i] = *device;
        }

        for (int i = 0; i < device_counts[MemoryType::kHIP_VRAM]; i++){
            ComputeDeviceBase* device = create_hip_compute_device(i);
            ((ComputeDeviceBase*)compute_devices[ComputeType::kHIP])[i] = *device;
        }

        for (int i = 0; i < nym_vulkan_devices; i++){
            ComputeDeviceBase* device = create_vulkan_compute_device(i);
        }

    }

    AllocationMap& get_device(MemoryType device, int device_id = 0){
        if (memory_devices.find(device) == memory_devices.end()){
            initialize_all_devices();
        }
        int device_count = device_counts[device];
        if (device_id < 0 || device_id >= device_count){
            std::cerr << "Invalid device id " << device_id << " for device type " << device << std::endl;
            throw std::runtime_error("Invalid device id");
        }
        return ((AllocationMap*)memory_devices[device])[device_id];
    }




    ComputeDeviceBase& get_compute_device(ComputeType device_type, int device_id = 0){
        if (compute_devices.find(device_type) == compute_devices.end()){
            initialize_all_devices();
        }
        int device_count = compute_device_counts[device_type];
        if (device_id < 0 || device_id >= device_count){
            std::cerr << "Invalid device id " << device_id << " for compute device type " << device_type << std::endl;
            throw std::runtime_error("Invalid device id");
        }
        return ((ComputeDeviceBase*)compute_devices[device_type])[device_id];
    }
};

__weak DeviceManager global_device_manager;


struct MemoryLocation {
    int device_id;
    MemoryType memory_type;
    AllocationMap* allocation_map;
    MemoryLocation(MemoryType memory_type = MemoryType::kDDR, int device_id = 0) : memory_type(memory_type), device_id(device_id) {
        allocation_map = &global_device_manager.get_device(memory_type, device_id);
    }
    MemoryLocation(AllocationMap& allocation_map) : allocation_map(&allocation_map) {
        memory_type = allocation_map.this_device_type;
        device_id = allocation_map.device_id;
    }
    operator AllocationMap&() {
        return *allocation_map;
    }
};





#endif // DEVICE_TYPE