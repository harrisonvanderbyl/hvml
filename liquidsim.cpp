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


    Tensor<float32x4, 2> myobjects({512,512}, DeviceType::kCPU);
    VectorDisplay display(myobjects.shape, false, false, false, false);

    display.displayLoop();


}