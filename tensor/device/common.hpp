#ifndef DEVICE_MANAGER_COMMON_HPP
#define DEVICE_MANAGER_COMMON_HPP

#include <iostream>
#include <map>
#include <functional>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "enums/device.hpp"
#include "shape.hpp"

// ---------------------------------------------------------------------------
//  Compute pointer type helpers
// ---------------------------------------------------------------------------

template <ComputeType compute_type, typename T>
struct ComputeDevicePointerType {
    using type = T*;
};

template <typename T>
struct ComputeDevicePointerType<kOPENGL, T> {
    using type = uint;
};

template <ComputeType CT, typename T>
struct ComputePointer {
    ComputeDevicePointerType<CT, T> storage_tensor;
};

// ---------------------------------------------------------------------------
//  Allocation metadata
// ---------------------------------------------------------------------------

struct AllocationMetadata {
    MemoryType storage_device;
    AllocationFlags rwstatus;
    ComputeType compute_device;
    size_t byte_size = 0;
    size_t type_size = 0;
    int format = 0;
    Shape<-1> shape = {};
    int device_id = 0;

    template <typename T>
    static AllocationMetadata create(
        const Shape<-1>& shapein,
        MemoryType mt = kDDR,
        ComputeType ct = kCPU,
        int informat = 0,
        AllocationFlags rwstatusin = AllocationFlags::kRW,
        int deviceid = 0)
    {
        AllocationMetadata tocreate;
        tocreate.storage_device = mt;
        tocreate.compute_device = ct;
        tocreate.rwstatus = rwstatusin;
        tocreate.type_size = sizeof(T);
        tocreate.shape = shapein;
        tocreate.byte_size = shapein.total_size() * tocreate.type_size;
        tocreate.format = informat;
        tocreate.device_id = deviceid;
        return tocreate;
    }

    friend std::ostream& operator<<(std::ostream& os, AllocationMetadata& inp) {
        os << "AllocMeta(";
        os << "storage:" << inp.storage_device << ",";
        os << "compute:" << inp.compute_device << ",";
        os << "shape:" << inp.shape << ",";
        os << "format:" << inp.format << ",";
        os << "device_id:" << inp.device_id;
        os << ")";
        return os;
    }

    auto hash() const {
        return std::tuple<int, ComputeType, int>(storage_device, compute_device, rwstatus);
    }
};

// ---------------------------------------------------------------------------
//  Memory wrappers
// ---------------------------------------------------------------------------

template <typename T = void>
struct MemoryWithMetadata {
    AllocationMetadata metadata;
    T* data;
    MemoryWithMetadata(AllocationMetadata meta, T* da) : metadata(meta), data(da) {};
};

struct BaseMemoryAllocation : public MemoryWithMetadata<void> {
    using MemoryWithMetadata<void>::MemoryWithMetadata;

    std::map<std::tuple<int, ComputeType, int>, void*> cached_massaged_pointers;
    size_t allocation_counts = 1;

    void alloc() {
        allocation_counts++;
    }

    bool dealloc() {
        if (allocation_counts == 0) {
            throw std::runtime_error("trying to double deallocate");
        }
        allocation_counts--;
        return allocation_counts == 0;
    }

    BaseMemoryAllocation& operator=(const BaseMemoryAllocation&) = delete;
};

template <typename T>
struct MassagedMemory : public MemoryWithMetadata<T> {
    BaseMemoryAllocation* base_memory;

    MassagedMemory() : MemoryWithMetadata<T>(AllocationMetadata(), nullptr), base_memory(nullptr) {};

    operator T*() { return this->data; };
    operator const T*() { return this->data; };
    operator void*() { return (void*)this->data; };
    operator const void*() { return (const void*)this->data; };

    MassagedMemory(AllocationMetadata meta, T* da, BaseMemoryAllocation* base)
        : MemoryWithMetadata<T>(meta, da), base_memory(base) {};

    MassagedMemory operator+=(size_t offset) {
        this->data += offset;
        return *this;
    }

    MassagedMemory operator+(size_t offset) {
        return MassagedMemory<T>(this->metadata, this->data + offset, this->base_memory);
    }

    MassagedMemory operator-(size_t offset) {
        return MassagedMemory<T>(this->metadata, this->data - offset, this->base_memory);
    }

    bool operator==(const MassagedMemory& other) const {
        return this->data == other.data;
    }

    bool operator==(const T* other) const {
        return this->data == other;
    }
};

// ---------------------------------------------------------------------------
//  AllocationMap — per-device allocator/deallocator/converter registry
// ---------------------------------------------------------------------------

struct AllocationMap {
    std::string device_name;

    void register_allocation(BaseMemoryAllocation* ptr) {
        ptr->alloc();
    }

    AllocationMap& operator=(const AllocationMap&) = delete;

  public:
    MemoryType this_device_type = MemoryType::kUnknown_MEM;

    operator MemoryType() const {
        return this_device_type;
    }

    std::map<ComputeType, bool> supports_compute_device;

    std::map<ComputeType, std::function<BaseMemoryAllocation*(AllocationMetadata, void*)>> compute_device_allocators;
    std::map<ComputeType, std::function<void(void*)>> compute_device_deallocators;
    std::map<ComputeType, std::function<void(void*, BaseMemoryAllocation*)>> compute_mapping_deallocators;
    std::map<MemoryType, std::function<BaseMemoryAllocation*(void*, AllocationMetadata)>> memory_type_converters;
    std::map<std::tuple<ComputeType, ComputeType>, std::function<void*(void*, BaseMemoryAllocation*, AllocationMetadata)>> compute_type_converters;

    std::function<void()> synchronize_function = []() {};

    ComputeType default_compute_type = ComputeType::kUnknown;
    ComputeType default_allocator_type = ComputeType::kUnknown;

    int device_id = 0;

    BaseMemoryAllocation* allocate(AllocationMetadata meta, void* existing_data = nullptr) {
        ComputeType compute_type = meta.compute_device;
        auto allocation_compute_type = compute_type == ComputeType::kUnknown ? default_compute_type : compute_type;

        if (compute_device_allocators.find(allocation_compute_type) != compute_device_allocators.end()) {
            return compute_device_allocators[allocation_compute_type](meta, existing_data);
        } else {
            std::cerr << "No allocator found for default compute type " << allocation_compute_type
                      << "{" << int(allocation_compute_type) << "} on device " << this_device_type << std::endl;
            throw std::runtime_error("No allocator found for default compute type");
        }
    }

    void deallocate(BaseMemoryAllocation* ptr) {
        auto ptr_compute_type = ptr->metadata.compute_device;
        if (ptr->dealloc()) {
            for (auto& key : ptr->cached_massaged_pointers) {
                if (compute_mapping_deallocators.find(std::get<1>(key.first)) != compute_mapping_deallocators.end()) {
                    compute_mapping_deallocators[std::get<1>(key.first)](key.second, ptr);
                } else {
                    std::cerr << "No compute mapping deallocator found for compute type "
                              << std::get<1>(key.first) << "{" << int(std::get<1>(key.first))
                              << "} on device " << this_device_type << std::endl;
                    throw std::runtime_error("No compute mapping deallocator found for compute type");
                }
            }

            if (compute_device_deallocators.find(ptr_compute_type) != compute_device_deallocators.end()) {
                compute_device_deallocators[ptr_compute_type](ptr->data);
                free(ptr);
            } else {
                std::cerr << "No deallocator found for default compute type " << ptr_compute_type
                          << "{" << int(ptr_compute_type) << "} on device " << this_device_type << std::endl;
                throw std::runtime_error("No deallocator found for default compute type");
            }
        }
    }

    template <typename T>
    MassagedMemory<T> get_massaged_pointer(BaseMemoryAllocation* ptr, AllocationMetadata meta) {
        if (ptr->cached_massaged_pointers.find(meta.hash()) != ptr->cached_massaged_pointers.end()) {
            return MassagedMemory<T>(meta, (T*)ptr->cached_massaged_pointers[meta.hash()], ptr);
        }

        ComputeType target_type = meta.compute_device;

        if (target_type == ptr->metadata.compute_device) {
            return MassagedMemory<T>(meta, (T*)ptr->data, ptr);
        } else {
            auto key = std::make_tuple(ptr->metadata.compute_device, target_type);
            if (compute_type_converters.find(key) != compute_type_converters.end()) {
                void* result = compute_type_converters[key](ptr->data, ptr, meta);
                ptr->cached_massaged_pointers[meta.hash()] = result;
                return MassagedMemory<T>(meta, (T*)result, ptr);
            } else {
                std::cerr << "No compute type converter found for conversion from "
                          << ptr->metadata.compute_device << "{" << int(ptr->metadata.compute_device)
                          << "} to " << target_type << "{" << int(target_type) << "}" << std::endl;
                throw std::runtime_error("No compute type converter found for requested conversion");
            }
        }
    }

    BaseMemoryAllocation* convert_memory_type(void* ptr, AllocationMetadata meta) {
        MemoryType target_type = meta.storage_device;
        if (memory_type_converters.find(target_type) != memory_type_converters.end()) {
            return memory_type_converters[target_type](ptr, meta);
        } else {
            std::cerr << "No memory type converter found for target type " << target_type << std::endl;
            // list current memory type converters
            for (const auto& [mem_type, converter] : memory_type_converters) {
                std::cerr << "  Available converter for memory type " << mem_type << std::endl;
            };
            std::cerr << "Requested target type: " << target_type << std::endl;
            std::cerr << "Memory type of the provided pointer: " << this->this_device_type << std::endl;
            std::cerr << "This device pointer: " << this << std::endl;
            
            throw std::runtime_error("No memory type converter found for target type");
        }
    }
};

// ---------------------------------------------------------------------------
//  ComputeDeviceBase
// ---------------------------------------------------------------------------

struct ComputeDeviceBase {
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

    ComputeDeviceBase() = default;
};

// ---------------------------------------------------------------------------
//  Forward declaration — DeviceManager (full definition below)
// ---------------------------------------------------------------------------

struct DeviceManager;

// ---------------------------------------------------------------------------
//  Plugin C ABI
//
//  Every backend plugin (.so) must export these extern "C" functions.
//  The DeviceManager dlopen's each .so, reads plugin_priority(), sorts
//  ascending, then calls plugin_register(dm) in that order.
//
//  plugin_register() is responsible for:
//    - counting its devices
//    - creating AllocationMaps and ComputeDeviceBases
//    - registering them via dm->register_memory_device / register_compute_device
//    - registering cross-device converters on already-created AllocationMaps
// ---------------------------------------------------------------------------

extern "C" {
    typedef const char* (*plugin_name_fn)();
    typedef int (*plugin_priority_fn)();
    typedef void (*plugin_register_fn)(DeviceManager*);
    typedef void (*plugin_init_fn)(DeviceManager*);
}

// ---------------------------------------------------------------------------
//  DeviceManager — discovers, loads, and owns all backend plugins
// ---------------------------------------------------------------------------

struct DeviceManager {
  public:
    // device_id → AllocationMap*
    std::map<MemoryType, std::vector<AllocationMap*>> memory_devices;
    // device_id → ComputeDeviceBase*
    std::map<ComputeType, std::vector<ComputeDeviceBase*>> compute_devices;

    // keep dlopen handles alive (lambdas inside AllocationMaps reference plugin code)
    std::vector<void*> plugin_handles;

    // plugins that export plugin_init — can be lazily initialised on demand
    // (e.g. OpenGL, which needs a GL context before it can register devices)
    struct LazyPlugin {
        void* handle;
        std::string name;
        plugin_init_fn init_fn;
        bool initialized = false;
    };
    std::vector<LazyPlugin> lazy_plugins;

    bool initialized = false;
    bool initializing = false;

    DeviceManager() {
        initialize_all_devices();
    }

    ~DeviceManager() {
        // Intentionally do NOT dlclose — std::function lambdas registered by
        // plugins still point into the .so text and may be invoked during
        // teardown of global AllocationMaps.
    }

    // ---- plugin discovery --------------------------------------------------

    static std::vector<std::string> find_so_files(const std::string& dir) {
        std::vector<std::string> result;
        DIR* d = opendir(dir.c_str());
        if (!d) return result;
        struct dirent* entry;
        while ((entry = readdir(d)) != nullptr) {
            std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            std::string full_path = dir + "/" + name;

            // try to open as subdirectory
            DIR* subdir = opendir(full_path.c_str());
            if (subdir) {
                struct dirent* subentry;
                while ((subentry = readdir(subdir)) != nullptr) {
                    std::string subname = subentry->d_name;
                    if (subname.size() > 3 && subname.substr(subname.size() - 3) == ".so") {
                        result.push_back(full_path + "/" + subname);
                    }
                }
                closedir(subdir);
            } else if (name.size() > 3 && name.substr(name.size() - 3) == ".so") {
                result.push_back(full_path);
            }
        }
        closedir(d);
        return result;
    }

    // ---- main initialisation ------------------------------------------------

    void initialize_all_devices() {
        if (initialized || initializing) return;
        initializing = true;

        std::cout << "Initializing all devices..." << std::endl;

        const char* env_dir = std::getenv("DEVICE_PLUGIN_DIR");
        std::string plugin_dir = env_dir ? env_dir : "tensor/device/plugins";

        std::cout << "Plugin directory: " << plugin_dir << std::endl;

        auto so_files = find_so_files(plugin_dir);

        if (so_files.empty()) {
            std::cerr << "Warning: no plugins found in " << plugin_dir << std::endl;
            std::cerr << "  Set DEVICE_PLUGIN_DIR env var to change the search path." << std::endl;
        }

        // Load all plugins, read priority
        struct PluginEntry {
            void* handle;
            int priority;
            std::string path;
            std::string name;
        };
        std::vector<PluginEntry> entries;

        for (auto& path : so_files) {
            void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!handle) {
                std::cerr << "  Failed to load " << path << ": " << dlerror() << std::endl;
                continue;
            }

            auto priority_fn = (plugin_priority_fn)dlsym(handle, "plugin_priority");
            int priority = priority_fn ? priority_fn() : 100;

            auto name_fn = (plugin_name_fn)dlsym(handle, "plugin_name");
            std::string pname = name_fn ? name_fn() : path;

            std::cout << "  Found plugin: " << pname << " (priority " << priority << ")" << std::endl;
            entries.push_back({handle, priority, path, pname});
        }

        // Sort by priority (ascending — lower loads first)
        std::sort(entries.begin(), entries.end(),
                  [](const PluginEntry& a, const PluginEntry& b) {
                      return a.priority < b.priority;
                  });

        // Register in priority order
        for (auto& entry : entries) {
            std::cout << "Registering plugin: " << entry.name << std::endl;
            auto register_fn = (plugin_register_fn)dlsym(entry.handle, "plugin_register");
            if (!register_fn) {
                std::cerr << "  Plugin " << entry.name << " missing plugin_register symbol" << std::endl;
                continue;
            }
            try {
                register_fn(this);
                plugin_handles.push_back(entry.handle);
            } catch (const std::exception& e) {
                std::cerr << "  Plugin " << entry.name << " failed to register: " << e.what() << std::endl;
                std::cerr << "  Continuing without " << entry.name << " support." << std::endl;
            }

            // Check for optional plugin_init — deferred initialisation entry point
            auto init_fn = (plugin_init_fn)dlsym(entry.handle, "plugin_init");
            if (init_fn) {
                lazy_plugins.push_back({entry.handle, entry.name, init_fn, false});
                std::cout << "  Plugin " << entry.name << " has deferred init (plugin_init)" << std::endl;
            }
        }

        initialized = true;
        initializing = false;

        std::cout << "Device initialization complete." << std::endl;
    }

    // ---- deferred plugin init ---------------------------------------------
    //
    //  Some backends (e.g. OpenGL) cannot create their devices until an
    //  external resource exists — a GL context, a Vulkan instance, etc.
    //  Those plugins export plugin_init() instead of doing all their work
    //  in plugin_register().  The display layer calls this method after
    //  it has created the necessary context.
    //
    //  Returns true if the plugin was found and initialised successfully
    //  (or was already initialised).
    bool init_plugin(const std::string& name) {
        for (auto& lp : lazy_plugins) {
            if (lp.name != name) continue;
            if (lp.initialized) return true;
            std::cout << "Deferred init for plugin: " << lp.name << std::endl;
            try {
                lp.init_fn(this);
                lp.initialized = true;
                return true;
            } catch (const std::exception& e) {
                std::cerr << "  Plugin " << lp.name << " deferred init failed: " << e.what() << std::endl;
                return false;
            }
        }
        std::cerr << "No deferred plugin named '" << name << "' found" << std::endl;
        return false;
    }

    // ---- registration API (called by plugins) ------------------------------

    void register_memory_device(MemoryType type, int device_id, AllocationMap* mapper) {
        if ((int)memory_devices[type].size() <= device_id) {
            memory_devices[type].resize(device_id + 1, nullptr);
        }
        memory_devices[type][device_id] = mapper;
    }

    void register_compute_device(ComputeType type, int device_id, ComputeDeviceBase* device) {
        if ((int)compute_devices[type].size() <= device_id) {
            compute_devices[type].resize(device_id + 1, nullptr);
        }
        compute_devices[type][device_id] = device;
    }

    // ---- lookup API --------------------------------------------------------

    AllocationMap& get_device(MemoryType device, int device_id = 0) {
        if (memory_devices.find(device) == memory_devices.end() ||
            device_id >= (int)memory_devices[device].size() ||
            !memory_devices[device][device_id]) {
            if (!initializing) {
                initialize_all_devices();
            }
        }
        if (memory_devices.find(device) == memory_devices.end() ||
            device_id >= (int)memory_devices[device].size() ||
            !memory_devices[device][device_id]) {
            std::cerr << "Invalid device id " << device_id << " for device type " << device << std::endl;
            throw std::runtime_error("Invalid device id");
        }
        return *memory_devices[device][device_id];
    }

    ComputeDeviceBase& get_compute_device(ComputeType device_type, int device_id = 0) {
        if (compute_devices.find(device_type) == compute_devices.end() ||
            device_id >= (int)compute_devices[device_type].size() ||
            !compute_devices[device_type][device_id]) {
            if (!initializing) {
                initialize_all_devices();
            }
        }
        if (compute_devices.find(device_type) == compute_devices.end() ||
            device_id >= (int)compute_devices[device_type].size() ||
            !compute_devices[device_type][device_id]) {
            std::cerr << "Invalid device id " << device_id << " for compute device type " << device_type << std::endl;
            throw std::runtime_error("Invalid device id");
        }
        return *compute_devices[device_type][device_id];
    }
};

// Single global instance.  `inline` guarantees one definition across all
// translation units in the main binary.  When the main binary is linked with
// -rdynamic, plugins loaded via dlopen(RTLD_GLOBAL) resolve their reference
// to this same instance.
inline DeviceManager global_device_manager;

// ---------------------------------------------------------------------------
//  MemoryLocation — convenience wrapper around global_device_manager
// ---------------------------------------------------------------------------

struct MemoryLocation {
    int device_id;
    MemoryType memory_type;
    AllocationMap* allocation_map;

    MemoryLocation(MemoryType memory_type = MemoryType::kDDR, int device_id = 0)
        : memory_type(memory_type), device_id(device_id) {
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

#endif // DEVICE_MANAGER_COMMON_HPP
