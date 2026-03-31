#include <iostream>
#include <map>
#include <functional>
#include "enums/device.hpp"
#include <fstream>
#include "shape.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
// #include "tensor/enums/device_support/x86/device.hpp"
#ifndef DEVICE_HPP
#define DEVICE_HPP

template <ComputeType compute_type, typename T>
struct ComputeDevicePointerType{
    using type = T*;
};

// template <typename T>
// struct ComputeDevicePointerType<kOPENGLTEXTURE, T>{
//     using type = uint;
// };

template <typename T>
struct ComputeDevicePointerType<kOPENGL, T>{
    using type = uint;
};



template <ComputeType CT, typename T>
struct ComputePointer{
    ComputeDevicePointerType<CT, T> storage_tensor;
};

// used for memory allocation information that is only relevant to compute type

struct AllocationMetadata{
    MemoryType storage_device;
    AllocationFlags rwstatus;
    ComputeType compute_device;
    size_t byte_size; // size in bytes of allocated data
    size_t type_size=0; // size in bytes of single entry
    int format=0; // used for type specific stuff
    Shape<-1> shape={}; // used for texture allocations
    int device_id=0;

    template<typename T>
    static AllocationMetadata create(
        const Shape<-1>& shapein,
        MemoryType mt=kDDR,
        ComputeType ct=kCPU,
        int informat = 0,
        AllocationFlags rwstatusin = AllocationFlags::kRW,
        int deviceid=0
    ){
        AllocationMetadata tocreate;
        tocreate.storage_device = mt;
        tocreate.compute_device = ct;
        tocreate.rwstatus = rwstatusin;
        tocreate.type_size = sizeof(T);
        tocreate.shape = shapein;
        tocreate.byte_size = shapein.total_size()*tocreate.type_size;
        tocreate.format = informat;
        tocreate.device_id = deviceid;

        return tocreate;
    }

    friend std::ostream &operator<<(std::ostream &os, AllocationMetadata& inp)
    {
        os << "AllocMeta(";
        os << "storage:" << inp.storage_device << ",";
        os << "compute:" << inp.compute_device << ",";
        os << "shape:" << inp.shape << ",";
        os << "format:" << inp.format << ",";
        os << "device_id:" << inp.device_id << ",";
        os<<")";
        return os;
    };
        

    auto hash() const{
        return std::tuple<int,ComputeType,int>(storage_device,compute_device,rwstatus);
    }
};

template <typename T = void>
struct MemoryWithMetadata{
    AllocationMetadata metadata;
    T* data;

    MemoryWithMetadata(AllocationMetadata meta,
    T* da):metadata(meta),data(da){};
};

struct BaseMemoryAllocation: public MemoryWithMetadata<void>{ // this always on cpu;
    using MemoryWithMetadata<void>::MemoryWithMetadata;
    
    std::map<std::tuple<int,ComputeType,int>, void*> cached_massaged_pointers = std::map<std::tuple<int,ComputeType,int>, void*>();
    size_t allocation_counts = 1;

   

    void alloc(){
        allocation_counts++;
        // std::cout << this << ": " << allocation_counts - 1 << ">" << allocation_counts << "\n";
    }

    bool dealloc(){
        if(allocation_counts==0){
            throw(std::runtime_error("trying to double deallocate"));
        }
        allocation_counts--;

        // std::cout << this << ": " << allocation_counts + 1 << ">" << allocation_counts << "\n";
    

        return allocation_counts==0;
    }

    BaseMemoryAllocation& operator=(const BaseMemoryAllocation&) = delete;
}; // always a pointer to this, this be a singleton

template <typename T>
struct MassagedMemory: public MemoryWithMetadata<T>{
    BaseMemoryAllocation* base_memory;

    MassagedMemory(): MemoryWithMetadata<T>(AllocationMetadata(), nullptr), base_memory(nullptr) {};

    operator T*(){
        return this->data;
    };

    operator const T*(){
        return this->data;
    };

    operator void*(){
        return (void*)this->data;
     };

     operator const void*(){
        return (const void*)this->data;
     };

     MassagedMemory(AllocationMetadata meta, T* da, BaseMemoryAllocation* base):MemoryWithMetadata<T>(meta, da), base_memory(base){};

    MassagedMemory operator += (size_t offset){
        this->data += offset;
        return *this;
     }

     MassagedMemory operator+(size_t offset){
        return MassagedMemory<T>(this->metadata, this->data + offset, this->base_memory);
     }

     MassagedMemory operator-(size_t offset){
        return MassagedMemory<T>(this->metadata, this->data - offset, this->base_memory);
     }

     bool operator==(const MassagedMemory& other) const {
        return this->data == other.data;
     }

     bool operator == ( const T* other) const {
        return this->data == other;
     }


};


struct AllocationMap{
    std::string device_name;

    void register_allocation(BaseMemoryAllocation* ptr) {
        ptr->alloc();
    }

    

    // delete copy constructors to avoid accidental copies
    AllocationMap& operator=(const AllocationMap&) = delete;


    

    public:
    MemoryType this_device_type = MemoryType::kUnknown_MEM;

    operator MemoryType() const {
        return this_device_type;
    }
    
    // map of supported compute devices
    std::map<ComputeType, bool> supports_compute_device;
    
    // map to compute device malloc lambdas
    std::map<ComputeType, std::function<BaseMemoryAllocation*(AllocationMetadata, void*)>> compute_device_allocators;
    std::map<ComputeType, std::function<void(void*)>> compute_device_deallocators;
    std::map<ComputeType, std::function<void(void*, BaseMemoryAllocation*)>> compute_mapping_deallocators;
    std::map<MemoryType, std::function<BaseMemoryAllocation*(void*, AllocationMetadata)>> memory_type_converters;
    std::map<std::tuple<ComputeType,ComputeType>, std::function<void*(void*, BaseMemoryAllocation*, AllocationMetadata)>> compute_type_converters;
    
    std::function<void()> synchronize_function = []() {};
    
    ComputeType default_compute_type = ComputeType::kUnknown;
    ComputeType default_allocator_type = ComputeType::kUnknown;

    int device_id = 0;

    BaseMemoryAllocation* allocate(AllocationMetadata meta, void* existing_data = nullptr) {
        ComputeType compute_type = meta.compute_device;
        auto allocation_compute_type = compute_type == ComputeType::kUnknown ? default_compute_type : compute_type;

        if (compute_device_allocators.find(allocation_compute_type) != compute_device_allocators.end()){
            return compute_device_allocators[allocation_compute_type](meta, existing_data);
            
        }else{
            std::cerr << "No allocator found for default compute type " << allocation_compute_type << "{" << int(allocation_compute_type) << "} on device " << this_device_type << std::endl;
            throw std::runtime_error("No allocator found for default compute type");
        }
    }

    void deallocate(BaseMemoryAllocation* ptr){
        auto ptr_compute_type = ptr->metadata.compute_device;
       if (ptr->dealloc()) {

        // std::cout << "attempting to free pointer: " << ptr << ":"<<ptr->allocation_counts<<"\n";
        for (auto key: ptr->cached_massaged_pointers){
            std::cout << "deallocing cached pointer \n";
            if (compute_mapping_deallocators.find(std::get<1>(key.first)) != compute_mapping_deallocators.end()){
                compute_mapping_deallocators[std::get<1>(key.first)](key.second, ptr);
            }else{
                std::cerr << "No compute mapping deallocator found for compute type " << std::get<1>(key.first) << "{" << int(std::get<1>(key.first)) << "} on device " << this_device_type << std::endl;
                throw std::runtime_error("No compute mapping deallocator found for compute type");
            }
        };
        
        if (compute_device_deallocators.find(ptr_compute_type) != compute_device_deallocators.end()){
            
                compute_device_deallocators[ptr_compute_type](ptr->data);
                
                // std::cout << "Freeing pointer: " << ptr << "\n";
                free(ptr);
            
            }else{
                std::cerr << "No deallocator found for default compute type " << ptr_compute_type << "{" << int(ptr_compute_type) << "} on device " << this_device_type << std::endl;
                throw std::runtime_error("No deallocator found for default compute type");
            }
        }
    }

    template <typename T>
    MassagedMemory<T> get_massaged_pointer(BaseMemoryAllocation* ptr, AllocationMetadata meta){

        if(ptr->cached_massaged_pointers.find(meta.hash()) != ptr->cached_massaged_pointers.end()){
            return MassagedMemory<T>(meta, (T*)ptr->cached_massaged_pointers[meta.hash()], ptr);
        }

        ComputeType target_type = meta.compute_device;

        if (target_type == ptr->metadata.compute_device){
            return MassagedMemory<T>(meta, (T*)ptr->data, ptr);
        }else {
            auto key = std::make_tuple(ptr->metadata.compute_device,target_type);
        
            if (compute_type_converters.find(key) != compute_type_converters.end()){
                void* result = compute_type_converters[key](ptr->data, ptr,  meta);
                ptr->cached_massaged_pointers[meta.hash()] = result;
                return MassagedMemory<T>(meta, (T*)result, ptr);
            }else{
                std::cerr << "No compute type converter found for conversion from " << ptr->metadata.compute_device << "{"<< int(ptr->metadata.compute_device) << "} to " << target_type << "{"<< int(target_type) << "}" << std::endl;
                std::cerr << "Ptr: " << ptr << " on device " << this_device_type << ": equal:" <<  (ptr->metadata.compute_device == target_type) << std::endl;
                throw std::runtime_error("No compute type converter found for requested conversion");
            }
        }
    }

    BaseMemoryAllocation* convert_memory_type(void* ptr, AllocationMetadata meta) {
        MemoryType target_type = meta.storage_device;
        if (memory_type_converters.find(target_type) != memory_type_converters.end()){
            return memory_type_converters[target_type](ptr, meta);
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
        {MemoryType::kDISK, false},
        {MemoryType::kUnknown_MEM, false}
    };
    
    MemoryType default_memory_type = MemoryType::kUnknown_MEM;
    int compute_units = 0;
    size_t shared_memory_size = 0;
    
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
AllocationMap* create_disk_mapper(int device_id);
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
    mapper->default_allocator_type = ComputeType::kCPU;
    mapper->supports_compute_device[ComputeType::kCPU] = true;
    
    mapper->compute_device_allocators[ComputeType::kCPU] = [](AllocationMetadata meta, void* existing_data){
        
        void* data = malloc(meta.byte_size);
        return new BaseMemoryAllocation(meta, data);
    };

    mapper->compute_device_deallocators[ComputeType::kCPU] = [](void* ptr) {
        free(ptr);
    };
    
    mapper->memory_type_converters[MemoryType::kDDR] = [mapper](void* ptr, AllocationMetadata meta)
    {
        std::cout << "Converting/creating alloc for " << meta << "\n";
        return mapper->compute_device_allocators[meta.compute_device](meta,ptr);
        // if(compute_type==ComputeType::kOPENGLTEXTURE){
        //     return  mapper->compute_device_allocators[ComputeType::kOPENGLTEXTURE](size,bitsize,storage_ptr, metadata); // No conversion needed
        // }
    };


    // mapper->compute_type_converters[std::tuple<ComputeType,ComputeType>({ComputeType::kCPU,ComputeType::kCPU})] = [mapper](void* data, BaseMemoryAllocation* storage, AllocationMetadata metadata){
    //     std::cout << storage->metadata.compute_device << "->" << metadata.compute_device << "\n";
    //     return data;
    // };

    

    


    mapper->this_device_type = MemoryType::kDDR;
    

    return mapper;
}


__weak ComputeDeviceBase* create_cpu_compute_device(int device_id){
    ComputeDeviceBase* device = new ComputeDeviceBase();
    device->supports_memory_location[MemoryType::kDDR] = true;
    device->default_memory_type = MemoryType::kDDR;

    device->compute_units = 16; // get thread mapper later

    return device;
}





struct DeviceManager{
    public:

    std::map<MemoryType, AllocationMap**> memory_devices;
    std::map<MemoryType, int> device_counts;
    std::map<ComputeType, ComputeDeviceBase**> compute_devices;
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
        device_counts[MemoryType::kDISK] = 1; // for now, just support one disk device
        int nym_vulkan_devices = count_vulkan_devices();
        int nym_opengl_devices = count_opengl_devices();
        
        memory_devices[MemoryType::kDDR] = new AllocationMap*[device_counts[MemoryType::kDDR]];
        memory_devices[MemoryType::kCUDA_VRAM] = new AllocationMap*[device_counts[MemoryType::kCUDA_VRAM]];
        memory_devices[MemoryType::kHIP_VRAM] = new AllocationMap*[device_counts[MemoryType::kHIP_VRAM]];
        memory_devices[MemoryType::kDISK] = new AllocationMap*[1]; // for now, just support one disk device

        compute_devices[ComputeType::kCPU] = new ComputeDeviceBase*[device_counts[MemoryType::kDDR]];
        compute_devices[ComputeType::kCUDA] = new ComputeDeviceBase*[device_counts[MemoryType::kCUDA_VRAM]];
        compute_devices[ComputeType::kHIP] = new ComputeDeviceBase*[device_counts[MemoryType::kHIP_VRAM]];
        compute_devices[ComputeType::kVULKAN] = new ComputeDeviceBase*[nym_vulkan_devices];

        compute_device_counts[ComputeType::kCPU] = device_counts[MemoryType::kDDR];
        compute_device_counts[ComputeType::kCUDA] = device_counts[MemoryType::kCUDA_VRAM];
        compute_device_counts[ComputeType::kHIP] = device_counts[MemoryType::kHIP_VRAM];
        compute_device_counts[ComputeType::kVULKAN] = nym_vulkan_devices;
        compute_device_counts[ComputeType::kOPENGL] = nym_opengl_devices;

        for (int i = 0; i < device_counts[MemoryType::kDDR]; i++){
            AllocationMap* mapper = create_cpu_mapper(i);
            memory_devices[MemoryType::kDDR][i] = mapper;
            ComputeDeviceBase* device = create_cpu_compute_device(i);
            compute_devices[ComputeType::kCPU][i] = device;
        }

        memory_devices[MemoryType::kDISK][0] = create_disk_mapper(0);

        for (int i = 0; i < device_counts[MemoryType::kCUDA_VRAM]; i++){
            AllocationMap* mapper = create_cuda_mapper(i);
            memory_devices[MemoryType::kCUDA_VRAM][i] = mapper;
        }

        for (int i = 0; i < device_counts[MemoryType::kHIP_VRAM]; i++){
            AllocationMap* mapper = create_hip_mapper(i);
            memory_devices[MemoryType::kHIP_VRAM][i] = mapper;
        }

        for (int i = 0; i < nym_opengl_devices; i++){
        }

        for (int i = 0; i < device_counts[MemoryType::kCUDA_VRAM]; i++){
            ComputeDeviceBase* device = create_cuda_compute_device(i);
            compute_devices[ComputeType::kCUDA][i] = device;
        }

        for (int i = 0; i < device_counts[MemoryType::kHIP_VRAM]; i++){
            ComputeDeviceBase* device = create_hip_compute_device(i);
            compute_devices[ComputeType::kHIP][i] = device;
        }

        for (int i = 0; i < nym_vulkan_devices; i++){
            ComputeDeviceBase* device = create_vulkan_compute_device(i);
            compute_devices[ComputeType::kVULKAN][i] = device;
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
        return *(memory_devices[device])[device_id];
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
        return *(compute_devices[device_type])[device_id];
    }
};

__weak DeviceManager global_device_manager;



__weak AllocationMap* create_disk_mapper(int device_id){
    // No special properties for disk memory, but we can implement swapping to disk later
    AllocationMap* mapper = new AllocationMap();
    mapper->default_compute_type = ComputeType::kCPU; // default to CPU compute for disk memory
    mapper->default_allocator_type = ComputeType::kFILE;
    mapper->supports_compute_device[ComputeType::kCPU] = true;
    mapper->supports_compute_device[ComputeType::kFILE] = true;
    mapper->compute_device_allocators[ComputeType::kFILE] = [mapper](AllocationMetadata meta, void* existing_data){
        std::string filename = mapper->device_name.empty() ? "tensor_swap_file.bin" : mapper->device_name;
        bool file_exists = std::ifstream(filename).good();
        FILE* file = fopen(filename.c_str(), file_exists ? "r+b" : "w+b");
        if (file == nullptr) {
            std::cerr << "Failed to create swap file on disk" << std::endl;
            std::cerr << "Error: " << strerror(errno) << std::endl;
            throw("error creating file\n");
        }

        if (existing_data != nullptr) {
            fwrite(existing_data, meta.type_size, meta.shape.total_size(), file);
        } else {
            if (!file_exists) {
                // Brand new file — allocate full size
                fseek(file, meta.byte_size - 1, SEEK_SET);
                fputc(0, file);
                fseek(file, 0, SEEK_SET);
            } else {
                // File exists — check if it's large enough, expand if not
                fseek(file, 0, SEEK_END);
                long current_size = ftell(file);
                if (current_size < static_cast<long>(meta.byte_size)) {
                    fseek(file, meta.byte_size - 1, SEEK_SET);
                    fputc(0, file);
                }else{
                    meta.byte_size = current_size; // update byte size to actual file size if file already exists and is larger than requested allocation
                    meta.shape = Shape<-1>{meta.byte_size / meta.type_size}; // update shape accordingly
                }
                fseek(file, 0, SEEK_SET);
            }
        }
        return new BaseMemoryAllocation(meta, file);
    };

    mapper->compute_device_deallocators[ComputeType::kFILE] = [](void* ptr) {

        FILE* file = (FILE*)ptr;
        if (file != nullptr) {
            fclose(file);
        }
    };

    // mapper->memory_type_converters[MemoryType::kDISK] = [](Shape<-1> size, size_t bitsize, ComputeType compute_type, void* ptr, void* storage_ptr, AllocationMetadata metadata) {
    //     return ptr; // No conversion needed for now, but we can implement swapping to disk later
    // };

    // mapper->memory_type_converters[MemoryType::kDDR] = [](Shape<-1> size, size_t bitsize, ComputeType compute_type, void* ptr, void* storage_ptr, AllocationMetadata metadata) {
    //     // read data from file back into memory
    //     auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
    //     void* host_ptr = host_device.allocate(size, bitsize, compute_type); // dont pass existing data to host allocator, it doesnt know how to handle that
    //     FILE* file = (FILE*)storage_ptr;
    //     size_t readfrom = (size_t)ptr; // this is actually the offset for this data in the file, which the CPU allocator can use to seek to the correct position in the file for reading/writing
    //     fseek(file, readfrom, SEEK_SET);
    //     if (file != nullptr) {
    //         fread(host_ptr, bitsize, size.total_size(), file);
    //     }
    //     // reset file pointer to beginning for future reads/writes
    //     fseek(file, 0, SEEK_SET);    
    //     return host_ptr;
    // };

    mapper->compute_type_converters[{ComputeType::kFILE,ComputeType::kCPU}] = [mapper](void* ptr, BaseMemoryAllocation* base, AllocationMetadata meta){
        
            // Get file descriptor
            int fd = fileno((FILE*)base->data);

            // Get file size
            struct stat st;
            fstat(fd, &st);
            size_t size = st.st_size;

            // mmap
            void* map = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (map == MAP_FAILED) {
                perror("mmap");
                return (void*)nullptr;
            }

            return map;
    };

     mapper->compute_mapping_deallocators[ComputeType::kCPU] = [](void* ptr, BaseMemoryAllocation* original) {

        munmap(ptr, original->metadata.byte_size);
    };
    

    AllocationMap& kddr = global_device_manager.get_device(MemoryType::kDDR, 0);
    // converter from kddr to disk as well
    kddr.memory_type_converters[MemoryType::kDISK] = [mapper](void* data, AllocationMetadata metadata){
        // write data from memory back to file
        return mapper->allocate(metadata, data);
    };

    mapper->this_device_type = MemoryType::kDISK;

    return mapper;
}

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

    MemoryLocation(const char* disk_path) : memory_type(MemoryType::kDISK), device_id(0) {
        allocation_map = &global_device_manager.get_device(MemoryType::kDISK, 0);
        allocation_map->device_name = std::string(disk_path);
    }

    MemoryLocation(std::string disk_path) : memory_type(MemoryType::kDISK), device_id(0) {
        allocation_map = &global_device_manager.get_device(MemoryType::kDISK, 0);
        allocation_map->device_name = disk_path;
    }

    operator AllocationMap&() {
        return *allocation_map;
    }
};






#endif // DEVICE_TYPE