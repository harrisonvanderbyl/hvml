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


    Linear<float, NONE> linear(4, 4, MemoryType::kDDR);
    std::cout << linear << std::endl;
    linear.weight[0][0] = 1.0f;
    linear.weight[1][1] = 1.0f;
    linear.weight[2][2] = 1.0f;
    linear.weight[3][3] = 1.0f;

    Tensor<float,2> input({2,4}, MemoryType::kDDR);
    input[0][0] = 1.0f;
    input[0][1] = 2.0f;
    input[0][2] = 3.0f;
    input[0][3] = 4.0f;
    input[1][0] = 5.0f;
    input[1][1] = 6.0f;
    input[1][2] = 7.0f;
    input[1][3] = 8.0f;
    auto output = linear.forward(input);
    std::cout << "Input: " << input << std::endl;
    std::cout << "Output: " << output << std::endl;

    // VectorDisplay display({512,512}, false, false, false, false);

    // display[{{0,20,4},{0,20}}] = 0xffffffff; // set upper right square to black


    // std::cout << display;
        
    // display.add_on_update([&](CurrentScreenInputInfo& input_info) {
    //     // Update the model's position or any other properties here
    //     // if(input_info.isMouseLeftButtonPressed()){
    //     //     // set pixel at mouse position to white
    //     //     float32x2 mouse_pos = input_info.getGlobalMousePosition();
    //     //     int mx = static_cast<int>(mouse_pos.x);
    //     //     int my = static_cast<int>(mouse_pos.y);
    //     //     if(mx >=0 && mx < display.shape[1] && my >=0 && my
    //     //         < display.shape[0]){
    //     //             display[my][mx] = 0xFFFFFFFF; // white pixel
    //     //         }
    //     // }
    // });

    // display.displayLoop();

}