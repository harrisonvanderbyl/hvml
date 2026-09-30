// device_interface_test.cpp — the same addition on every device, the way
// user code picks devices.
//
//   ./hvcc examples/device_interface_test.cpp -o device_interface_test      (CUDA + HIP + Vulkan)
//   ./vulkcc examples/device_interface_test.cpp -o device_interface_test -I./tensor -std=c++20 -O2
//   DEVICE_PLUGIN_DIR=tensor/device/plugins ./device_interface_test
//
// Devices that are not present are skipped.

#include "tensor.hpp"
#include "ops/ops.hpp"
#include <cstdio>

static int failures = 0;

static void check(const char* what, const Tensor<float, 1>& c, ComputeType expect_compute) {
    Tensor<float, 1> h = c.to(MemoryLocation(MemoryType::kDDR), ComputeType::kCPU);
    bool ok = c.data.metadata.compute_device == expect_compute;
    for (int i = 0; i < 16; i++) ok = ok && h.data.data[i] == float(2 * i + 16);
    if (!ok) failures++;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "  (memory " << c.device->this_device_type << " "
              << c.device->device_id << ", compute " << c.data.metadata.compute_device << ")  " << c << std::endl;   // printing reads back from any device
}

static bool has_device(ComputeType ct, int i) {
    auto it = global_device_manager.compute_devices.find(ct);
    return it != global_device_manager.compute_devices.end() && i < (int)it->second.size() && it->second[i];
}

__weak int main() {
    Tensor<float, 1> a({16}, MemoryType::kDDR);
    for (int i = 0; i < 16; i++) a[{i}] = i;
    Tensor<float, 1> b({16}, MemoryType::kDDR);
    for (int i = 0; i < 16; i++) b[{i}] = i + 16;
    check("cpu", a + b, ComputeType::kCPU);

    if (has_device(ComputeType::kCUDA, 0)) {
        auto acuda = a.to(MemoryType::kCUDA_VRAM, ComputeType::kCUDA);
        auto bcuda = b.to(MemoryType::kCUDA_VRAM, ComputeType::kCUDA);
        check("cuda", acuda + bcuda, ComputeType::kCUDA);
    }
    if (has_device(ComputeType::kHIP, 0)) {
        auto ahip = a.to(MemoryType::kHIP_VRAM, ComputeType::kHIP);
        auto bhip = b.to(MemoryType::kHIP_VRAM, ComputeType::kHIP);
        check("hip", ahip + bhip, ComputeType::kHIP);
    }

    // Every Vulkan device.  Its tensors live in the memory it allocates in
    // (its GPU's CUDA / HIP memory, host memory, or its own map); asking for
    // kVULKAN there allocates Vulkan buffers with that map's Vulkan allocator
    // and gives their kernel view (device addresses), so + runs vulkcc kernels.
    for (int i = 0; has_device(ComputeType::kVULKAN, i); i++) {
        auto& vk = global_device_manager.get_compute_device(ComputeType::kVULKAN, i);

        // explicit memory type + compute type, as in the CUDA / HIP cases
        MemoryLocation mem(vk.default_memory_type, vk.default_memory_device_id);
        auto avulkan = a.to(mem, ComputeType::kVULKAN);
        auto bvulkan = b.to(mem, ComputeType::kVULKAN);
        std::string name = "vulkan " + std::to_string(i) + ", a.to(mem, kVULKAN)";
        check(name.c_str(), avulkan + bvulkan, ComputeType::kVULKAN);

        // or the device's location, which carries kVULKAN
        MemoryLocation loc(vk);
        name = "vulkan " + std::to_string(i) + ", a.to(MemoryLocation(vk))";
        check(name.c_str(), a.to(loc) + b.to(loc), ComputeType::kVULKAN);

        // the same Vulkan buffer seen through the memory's own compute type
        // (CPU for host memory, CUDA / HIP for VRAM) and back
        if (vk.default_memory_type != MemoryType::kUnknown_MEM) {
            auto as_native = avulkan.to_compute(mem.allocation_map->default_compute_type);
            auto back = as_native.to_compute(ComputeType::kVULKAN);
            name = "vulkan " + std::to_string(i) + ", to_compute round trip";
            check(name.c_str(), back + bvulkan, ComputeType::kVULKAN);
        }
    }

    std::cout << (failures ? "SOME CHECKS FAILED" : "all checks passed") << std::endl;
    return failures ? 1 : 0;
}
