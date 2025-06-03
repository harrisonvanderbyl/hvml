#include "shape.hpp"
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "dtypes/complex32.hpp"
#include "file_loaders/safetensors.hpp"
#include <ops/ops.h>
#include <string>
#include "display/display.hpp"
#include "module/linear/linear.hpp"
#include "models/rwkv7/rwkv7.hpp"

class pixel {
    float4 offset;
    mat4* parent;
};

class eventQueue {
    
};

int main(){
    int objects = 4;
    Tensor<float, 2> myobjects({objects, 4}, DeviceType::kCPU);
    Tensor vecview = myobjects.view<float4,1>({objects});
    for (int i = 0; i < objects; i++) {
        vecview[i] = float4(i%512, i/512, 0, 1); // x, y, z, w
    }

    VectorDisplay display({512,512});

    display[{{}}] = uint84{0, 0, 0, 255};
    display.on_update = [&](){
        for (int i = 0; i < objects; i++) {
            auto mat = vecview[i];
            display[{mat.x, mat.y}] = uint84{255, 0, 0, 255}; // red
        }
    };
    display.displayLoop();
}