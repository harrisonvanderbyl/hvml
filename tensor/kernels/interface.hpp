
#include "enums/device.hpp"

template <DeviceType device_type, typename... Args>
class Kernel {
public:
    Kernel() {
        // Constructor implementation
    }

    void call(Args... args) {
        // Method implementation
        std::cout << "Kernel called with device type: " << static_cast<int>(device_type) << std::endl;
        throw std::runtime_error("Kernel not implemented for this device type");
    }
    
    
};