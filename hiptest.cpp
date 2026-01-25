#include <iostream>
#include "ops/ops.hpp"
#include "kernels/interface.hpp"
#include "tensor.hpp"
#include "module/linear/linear.hpp"

struct DualAntiReal {
    float real;    // x
    float dual;    // ε
    float anti;    // u

    __host__ __device__ DualAntiReal operator*(const DualAntiReal &other) const {
        return {
            real * other.real,                       // real part
            real * other.dual + dual * other.real,  // dual part (ε^2 = 0)
            other.anti * other.anti + anti + other.anti + real + other.real // anti-zero
        };
    }

    friend std::ostream &operator<<(std::ostream &os, const DualAntiReal &dar) {
        os << "{" << dar.real << " + " << dar.dual << "ε + " << dar.anti << "u}";
        return os;
    }
};


__weak int main() {
    Tensor<DualAntiReal,1> a({300},kDDR);
    for (int i = 0; i < 300; i++) {
        a.data[i].real = 1.0f;
        a.data[i].dual = 2.0f;
        a.data[i].anti = 3.0f;
    }
    auto acuda = a.to(kCUDA_VRAM);
    auto aHip = a.to(kHIP_VRAM);

    std::cout << aHip * aHip;
    std::cout << "\n";
    std::cout << acuda * acuda;
}