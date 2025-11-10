#include "shape.hpp"
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "dtypes/complex32.hpp"
#include "file_loaders/safetensors.hpp"
// #include "file_loaders/gltf.hpp"
#include <ops/ops.h>
#include <string>
#include "display/display.hpp"
#include "module/linear/linear.hpp"
#include "models/rwkv7/rwkv7.hpp"



int main(){



    VectorDisplay display({512,512}, false, false, false, false);
    Tensor<float4,1> particles({512},kCPU);
    Tensor<float4,2> particleField({512,512}, kCPU);
    

    // std::cout << display;
        
    display.add_on_update([&](CurrentScreenInputInfo& input_info) {
        particles[0].w += 0.1;
        particles[0].x = 200 + 100*sin(particles[0].w);
        particles[0].y = 200 + 100*cos(particles[0].w);
        display[{{}}] = 0;
        for(int i = 0; i < 512; i++){
            // std::cout << i << particles[i] << "\n";
            display[particles[i].x, particles[i].y] = 0xffffffff;
        }
    });

    display.displayLoop();

}