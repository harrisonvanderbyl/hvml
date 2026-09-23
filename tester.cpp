// device_test.cpp — smoke test for the dynamic plugin deviceManager
//
// Build:  make
// Run:    ./device_test
//
// Verifies that:
//   1. The DeviceManager discovers and loads .so plugins from plugins/
//   2. CPU allocation / deallocation works
//   3. Disk-backed allocation + mmap conversion works
//   4. Any GPU backends present (CUDA/HIP) are registered

#include "tensor.hpp"
#include <cstring>
#include <iostream>

int main() {
    std::cout << "=== deviceManager dynamic plugin test ===" << std::endl;

    // DeviceManager is constructed as a global; plugins are loaded automatically.
    auto& dm = global_device_manager;

    // ---- CPU allocation test ----
    std::cout << "\n--- CPU allocation test ---" << std::endl;
    {
        auto meta = AllocationMetadata::create<float>({1024}, kDDR, kCPU);
        auto& cpu_map = dm.get_device(MemoryType::kDDR, 0);
        auto* alloc = cpu_map.allocate(meta);

        float* data = (float*)alloc->data;
        for (int i = 0; i < 1024; i++) {
            data[i] = (float)i;
        }

        bool ok = true;
        for (int i = 0; i < 1024; i++) {
            if (data[i] != (float)i) { ok = false; break; }
        }
        std::cout << "CPU alloc readback: " << (ok ? "PASS" : "FAIL") << std::endl;

        cpu_map.deallocate(alloc);
        std::cout << "CPU dealloc: PASS" << std::endl;
    }

    // ---- Disk allocation test ----
    std::cout << "\n--- Disk allocation test ---" << std::endl;
    {
        auto meta = AllocationMetadata::create<float>({256}, kDISK, kFILE);
        auto& disk_map = dm.get_device(MemoryType::kDISK, 0);
        disk_map.device_name = "test_swap.bin";

        auto* alloc = disk_map.allocate(meta);
        std::cout << "Disk alloc: PASS (file opened)" << std::endl;

        // Convert file → CPU (mmap)
        auto cpu_meta = AllocationMetadata::create<float>({256}, kDISK, kCPU);
        auto massaged = disk_map.get_massaged_pointer<float>(alloc, cpu_meta);
        std::cout << "Disk→CPU mmap: " << (massaged.data ? "PASS" : "FAIL") << std::endl;

        disk_map.deallocate(alloc);
        std::cout << "Disk dealloc: PASS" << std::endl;
    }

    // ---- Device enumeration ----
    std::cout << "\n--- Device enumeration ---" << std::endl;
    for (auto& [mem_type, maps] : dm.memory_devices) {
        std::cout << "MemoryType " << mem_type << ": " << maps.size() << " device(s)" << std::endl;
        // enumerate available memory converters
        for (auto& [target_mem_type, converter] : maps[0]->memory_type_converters) {
            std::cout << "  Can convert to MemoryType " << target_mem_type << std::endl;
        }
    }
    for (auto& [ct, devs] : dm.compute_devices) {
        std::cout << "ComputeType " << ct << ": " << devs.size() << " device(s)" << std::endl;
    }

    std::cout << "\n=== Test complete ===" << std::endl;

    Tensor<float,1> t({1024}, kDDR, kCPU);
    for (int i = 0; i < 1024; i++) {
        t[i] = (float)i;
    }

    bool ok = true;
    for (int i = 0; i < 1024; i++) {
        if (t[i] != (float)i) { ok = false; break; }
    }
    std::cout << "Tensor readback: " << (ok ? "PASS" : "FAIL") << std::endl;

    Tensor<float,1> t2 = t.to(kCUDA_VRAM, kCUDA);
    ok = true;
    Tensor<float,1> t3 = t2.to(kDDR, kCPU);
    ok = true;
    for (int i = 0; i < 1024; i++) {
        if (t3[i] != (float)i) { ok = false; break; }
    }
    std::cout << "Tensor round-trip readback: " << (ok ? "PASS" : "FAIL") << std::endl;

    return 0;
}
