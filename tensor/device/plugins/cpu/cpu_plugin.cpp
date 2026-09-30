// cpu_plugin.cpp — CPU + Disk backend plugin
//
// Built with:  g++-14 -std=c++20 -fPIC -shared -I.. -I../../tensor \
//                  -o plugins/cpu/libcpu_plugin.so cpu_plugin.cpp -ldl
//
// Priority 10 — loaded first so that GPU backends can register converters
// onto the CPU AllocationMap.

#include "plugin.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

// ---------------------------------------------------------------------------
//  CPU AllocationMap
// ---------------------------------------------------------------------------

static AllocationMap* create_cpu_mapper(int device_id) {
    AllocationMap* mapper = new AllocationMap();
    mapper->device_id = device_id;
    mapper->default_compute_type = ComputeType::kCPU;
    mapper->default_allocator_type = ComputeType::kCPU;
    mapper->supports_compute_device[ComputeType::kCPU] = true;

    mapper->compute_device_allocators[ComputeType::kCPU] = [](AllocationMetadata meta, void* existing_data) {
        void* data = malloc(meta.byte_size);
        if (existing_data) {
            memcpy(data, existing_data, meta.byte_size);
        }
        return new BaseMemoryAllocation(meta, data);
    };

    mapper->compute_device_deallocators[ComputeType::kCPU] = [](void* ptr) {
        free(ptr);
    };

    mapper->memory_type_converters[MemoryType::kDDR] = [mapper](void* ptr, AllocationMetadata meta) {
        return mapper->compute_device_allocators[meta.compute_device](meta, ptr);
    };

    mapper->this_device_type = MemoryType::kDDR;
    return mapper;
}

static ComputeDeviceBase* create_cpu_compute_device(int device_id) {
    ComputeDeviceBase* device = new ComputeDeviceBase();
    device->supports_memory_location[MemoryType::kDDR] = true;
    device->default_memory_type = MemoryType::kDDR;
    device->compute_units = 16;
    return device;
}

// ---------------------------------------------------------------------------
//  Disk AllocationMap  (file-backed memory via mmap)
// ---------------------------------------------------------------------------

static AllocationMap* create_disk_mapper(int device_id) {
    AllocationMap* mapper = new AllocationMap();
    mapper->device_id = device_id;
    mapper->default_compute_type = ComputeType::kCPU;
    mapper->default_allocator_type = ComputeType::kFILE;
    mapper->supports_compute_device[ComputeType::kCPU] = true;
    mapper->supports_compute_device[ComputeType::kFILE] = true;

    mapper->compute_device_allocators[ComputeType::kFILE] = [mapper](AllocationMetadata meta, void* existing_data) {
        std::string filename = mapper->device_name.empty() ? "tensor_swap_file.bin" : mapper->device_name;
        bool file_exists = std::ifstream(filename).good();

        // Read-only (kR without kW): open an existing file as it is — never
        // created, resized or written.  Its CPU view is mapped read-only.
        bool read_only = ((int)meta.rwstatus & AllocationFlags::kW) == 0 && ((int)meta.rwstatus & AllocationFlags::kR) != 0;
        const FileFormat* format = meta.file_format;

        // An existing file's header, parsed into m (the file mapped read-only
        // just for this).
        auto read_header = [&](AllocationMetadata& m) {
            int fd = ::open(filename.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("cannot open " + filename);
            struct stat st;
            fstat(fd, &st);
            size_t size = (size_t)st.st_size;
            void* map = size ? mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
            ::close(fd);
            if (map == MAP_FAILED) throw std::runtime_error("cannot map " + filename);
            try {
                format->read((const char*)map, size, m);
            } catch (...) {
                if (map) munmap(map, size);
                throw;
            }
            if (map) munmap(map, size);
        };

        if (read_only) {
            FILE* file = file_exists ? fopen(filename.c_str(), "rb") : nullptr;
            if (!file) throw std::runtime_error("cannot open " + filename + " for reading");
            if (format) {
                try { read_header(meta); } catch (...) { fclose(file); throw; }
                return new BaseMemoryAllocation(meta, file);
            }
            fseek(file, 0, SEEK_END);
            long size = ftell(file);
            fseek(file, 0, SEEK_SET);
            meta.byte_size = size;
            meta.shape = Shape<-1>{(long)(size / meta.type_size)};
            return new BaseMemoryAllocation(meta, file);
        }

        if (format) {
            // An existing file of this format is opened as it is when asked
            // to open it (no shape: byte_size 0) or when it already has the
            // requested shape; otherwise it is replaced by a new one.
            if (file_exists && !existing_data) {
                AllocationMetadata found = meta;
                bool parsed = true;
                std::string why;
                try { read_header(found); } catch (const std::exception& e) { parsed = false; why = e.what(); }
                if (meta.byte_size == 0 && !parsed) throw std::runtime_error(filename + ": " + why);
                if (parsed && (meta.byte_size == 0 || found.shape == meta.shape)) {
                    FILE* file = fopen(filename.c_str(), "r+b");
                    if (!file) throw std::runtime_error("cannot open " + filename + " for writing");
                    return new BaseMemoryAllocation(found, file);
                }
            }
            if (meta.byte_size == 0 && !existing_data) throw std::runtime_error("cannot open " + filename + ": no such file");
            // a new file: header, then the data (given, or the format's fill)
            FILE* file = fopen(filename.c_str(), "w+b");
            if (!file) throw std::runtime_error("cannot create " + filename + ": " + strerror(errno));
            std::vector<char> head = format->write(meta);
            fwrite(head.data(), 1, head.size(), file);
            if (existing_data) {
                fwrite(existing_data, 1, meta.byte_size, file);
            } else if (format->fill_byte() != 0) {
                std::vector<char> chunk(1 << 16, (char)format->fill_byte());
                for (size_t left = meta.byte_size; left > 0;) {
                    size_t n = std::min(left, chunk.size());
                    fwrite(chunk.data(), 1, n, file);
                    left -= n;
                }
            } else if (meta.byte_size > 0) {
                fseek(file, (long)(meta.data_offset + meta.byte_size - 1), SEEK_SET);
                fputc(0, file);
            }
            fflush(file);
            fseek(file, 0, SEEK_SET);
            return new BaseMemoryAllocation(meta, file);
        }

        FILE* file = fopen(filename.c_str(), file_exists ? "r+b" : "w+b");
        if (!file) {
            std::cerr << "Failed to create swap file on disk: " << strerror(errno) << std::endl;
            throw std::runtime_error("error creating file");
        }

        if (existing_data) {
            fwrite(existing_data, meta.type_size, meta.shape.total_size(), file);
        } else {
            if (!file_exists) {
                fseek(file, meta.byte_size - 1, SEEK_SET);
                fputc(0, file);
                fseek(file, 0, SEEK_SET);
            } else {
                fseek(file, 0, SEEK_END);
                long current_size = ftell(file);
                if (current_size < (long)meta.byte_size) {
                    fseek(file, meta.byte_size - 1, SEEK_SET);
                    fputc(0, file);
                } else {
                    meta.byte_size = current_size;
                    meta.shape = Shape<-1>{(long)(meta.byte_size / meta.type_size)};
                }
                fseek(file, 0, SEEK_SET);
            }
        }
        return new BaseMemoryAllocation(meta, file);
    };

    mapper->compute_device_deallocators[ComputeType::kFILE] = [](void* ptr) {
        FILE* file = (FILE*)ptr;
        if (file) fclose(file);
    };

    mapper->compute_type_converters[{ComputeType::kFILE, ComputeType::kCPU}] = [mapper](void* ptr, BaseMemoryAllocation* base, AllocationMetadata meta) {
        int fd = fileno((FILE*)base->data);
        struct stat st;
        fstat(fd, &st);
        size_t size = st.st_size;
        bool writable = ((int)base->metadata.rwstatus & AllocationFlags::kW) != 0;
        void* map = mmap(nullptr, size, writable ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            perror("mmap");
            return (void*)nullptr;
        }
        // the view starts at the data (after a file format's header)
        return (void*)((char*)map + base->metadata.data_offset);
    };

    mapper->compute_mapping_deallocators[ComputeType::kCPU] = [](void* ptr, BaseMemoryAllocation* original) {
        // the whole file was mapped; the view is data_offset into it
        struct stat st;
        size_t size = original->metadata.data_offset + original->metadata.byte_size;
        if (original->data && fstat(fileno((FILE*)original->data), &st) == 0) size = (size_t)st.st_size;
        munmap((char*)ptr - original->metadata.data_offset, size);
    };

    // Register converter on the CPU device (created earlier, priority 10)
    try {
        AllocationMap& kddr = global_device_manager.get_device(MemoryType::kDDR, 0);
        kddr.memory_type_converters[MemoryType::kDISK] = [mapper](void* data, AllocationMetadata metadata) {
            return mapper->allocate(metadata, data);
        };
    } catch (...) {
        std::cerr << "Disk plugin: CPU device not available for converter registration" << std::endl;
    }

    // Disk → host memory: the tensor data is already mmapped host memory,
    // so copying it into a fresh host allocation is a plain memcpy.
    mapper->memory_type_converters[MemoryType::kDDR] = [](void* data, AllocationMetadata metadata) {
        AllocationMap& kddr = global_device_manager.get_device(MemoryType::kDDR, 0);
        metadata.storage_device = MemoryType::kDDR;
        metadata.compute_device = ComputeType::kCPU;
        return kddr.allocate(metadata, data);
    };

    mapper->this_device_type = MemoryType::kDISK;
    return mapper;
}

// ---------------------------------------------------------------------------
//  Plugin C ABI
// ---------------------------------------------------------------------------

extern "C" const char* plugin_name() {
    return "cpu";
}

extern "C" int plugin_priority() {
    return 10;
}

extern "C" void plugin_register(DeviceManager* dm) {
    std::cout << "[cpu] registering CPU + Disk devices" << std::endl;

    // CPU
    int cpu_count = 1;
    for (int i = 0; i < cpu_count; i++) {
        AllocationMap* mapper = create_cpu_mapper(i);
        dm->register_memory_device(MemoryType::kDDR, i, mapper);
        ComputeDeviceBase* dev = create_cpu_compute_device(i);
        dm->register_compute_device(ComputeType::kCPU, i, dev);
    }

    // Disk (single device)
    AllocationMap* disk_mapper = create_disk_mapper(0);
    dm->register_memory_device(MemoryType::kDISK, 0, disk_mapper);
}
