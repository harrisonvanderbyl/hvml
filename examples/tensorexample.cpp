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


template <typename R = float>
struct TimeShiftmod: Module<Tensor<R, 2>>
{
    public:
    Tensor<R,2> state;
    Tensor<R,1> buffer;
    TimeShiftmod(int batch, int dim): state({batch, dim}), buffer(Shape<1>{dim}), Module<Tensor<R,2>>({state, "state"}){

    };

    Tensor<R,3> forward(Tensor<R,3> x){
        
        if(x.device_type == kCPU){
            timeshift_cpu(x.data, state.data, buffer.data, x.shape[0], x.shape[1], x.shape[2]);
        }
        return x;
    }

};



class EinObj{
    Tensor<float,2> rearange(){

    }
};

int main(){


    Tensor<float, -1> myobjects({4, 4}, DeviceType::kCPU);
    Tensor<float, 2> myobjects2({4, 4}, DeviceType::kCPU);
    std::cout << myobjects << std::endl;

    auto mod = TimeShift(4, 512);

    auto output = EinObj({a,b},{ab}).rearange(myobjects2);
    std::cout << myobjects;
    mod.state = 0;
    std::cout << mod << std::endl;

    safetensors tensors;

    mod.save_to_safetensors(tensors, "timeshift.");

    tensors.save("safetensors.st");

    mod.load_from_safetensors(tensors, "timeshift.");

    std::cout << mod << std::endl;

    // VectorDisplay display({512,512}, false, false, false, false);

    // display[{{0,20,4},{0,20}}] = 0xffffffff; // set upper right square to black


    // std::cout << display;
        
    // display.add_on_update([&](CurrentScreenInputInfo& input_info) {
    //     // Update the model's position or any other properties here
    //     // if(input_info.isMouseLeftButtonPressed()){
    //     //     // set pixel at mouse position to white
    //     //     float2 mouse_pos = input_info.getGlobalMousePosition();
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