#include "shape.hpp"
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "dtypes/complex32.hpp"
#include "file_loaders/safetensors.hpp"
#include <ops/ops.h>
#include <string>
// #include "display/display.hpp"
#include "module/linear/linear.hpp"
#include "models/rwkv7/rwkv7.hpp"


int main(){
    
    Tensor<float,3> teststriding = {{4,2,2}};
    TimeShift A(4,2);
    
    // std::cout <<teststriding << std::endl;
    // teststriding[0,0] = 0;
    // teststriding[SliceList{0,0}] = 0;
    // teststriding[SliceList{0,1}] = 2;

    // std::cout << *(float*)(teststriding.data) << std::endl;
    // std::cout << *((float*)(teststriding.data)+1) << std::endl;
    // std::cout << *((float*)(teststriding.data)+2) << std::endl;
    // std::cout << *((float*)(teststriding.data)+3) << std::endl;
    // std::cout <<teststriding.flatget(0) << std::endl;
    // std::cout <<teststriding.flatget(1) << std::endl;
    // std::cout <<teststriding.flatget(2) << std::endl;
    // std::cout <<teststriding.flatget(3) << std::endl;
    std::cout <<teststriding[{0,1}] << std::endl;
    teststriding[{0,0}] = 1;
    teststriding[{0,1}] = 2;
    teststriding[{1,0}] = 3;
    teststriding[{1,1}] = 4;
    teststriding[{2,0}] = 5;
    teststriding[{2,1}] = 6;
    teststriding[{3,0}] = 7;
    teststriding[{3,1}] = 8;
    teststriding[{{0,4,2},{},1}] = 9;
    // std::cout << teststriding[{{0,2},1,0}] << std::endl;
    std::cout <<teststriding << std::endl;
    std::cout <<teststriding[0] << std::endl;
    std::cout <<teststriding[1] << std::endl;
    std::cout <<teststriding[2] << std::endl;
    std::cout <<teststriding[3] << std::endl;

    auto out = A.forward(teststriding);

    std::cout <<out << std::endl;
    std::cout <<out[0] << std::endl;
    std::cout <<out[1] << std::endl;
    std::cout <<out[2] << std::endl;
    std::cout <<out[3] << std::endl;
    // teststriding[SliceList{1,0}] = 2;
    // teststriding[SliceList{1,1}] = 3;
    // teststriding[SliceList{2,0}] = 4;
    // teststriding[SliceList{2,1}] = 5;
    // teststriding[SliceList{3,0}] = 6;
    // teststriding[SliceList{3,1}] = 7;

    // SliceList a = {0,Slice{1},3,Slice{1}};
    // std::cout << a.reducedims << std::endl;
    // std::cout << teststriding << std::endl;
    // SliceList b = 5;
    // std::cout << b.reducedims << std::endl;
    // teststriding[0] = 4;
    
    // std::cout << teststriding.shape << std::endl;
    // std::cout << teststriding.shape.slice() << std::endl;
    // std::cout << teststriding.shape.slice<2>() << std::endl;
    // teststriding[1][1] = 1;
   

    // teststriding[Slice{}][1] = 2;
    // std::cout << teststriding<< std::endl;

    // std::cout << teststriding[{0,4,2}]<< std::endl;
    // std::cout << teststriding[Slice{0,1}]<< std::endl;
    // std::cout << teststriding[{}].strides<< std::endl;
    // std::cout << teststriding.shape<< std::endl;
    // std::cout << teststriding[{}].shape<< std::endl;
    // teststriding[1,0] = 2;
    // teststriding[1,1] = 3;
    // teststriding[{{},{},{}}] = 2;
    // teststriding[{{},{},{}}] = 3;
    // std::cout <<teststriding[0] << std::endl;
    // std::cout <<teststriding[{}][1] << std::endl;
    // teststriding[{{},0,{}}] += Tensor<float>{10.f};
    // teststriding[{{},1,{}}] += Tensor<float>{20.f};
    // teststriding[{{},2,{}}] += Tensor<float>{30.f};
    // teststriding[{0,{},{}}] += Tensor<float>{100.f};
    // teststriding[{1,{},{}}] += Tensor<float>{200.f};
    // teststriding[{2,{},{}}] += Tensor<float>{300.f};

//    FFN a = FFN(10,20);
//    std::cout << a << std::endl;
//    Tensor<float> b = Tensor<float>({10,20});
//    b = 1;
//    Tensor<float> c = Tensor<float>({20,10});
//    c = 2;
//    safetensors s;
//    s.add("key.weight", b);
//    s.add("value.weight", c);
//    s.save("test.bin");
//    a.load_from_safetensors("test.bin");
//    a.to_safetensors().save("test2.safetensors");

//    safetensors s2 = safetensors("test2.bin");
//    // print keys
//    std::cout << "Keys: " << std::endl;
//    for (auto key : s2.keys())
//    {
//        std::cout << key << std::endl;
//    }
//    a.load_from_safetensors("test2.bin");
   // a.to_safetensors().save("test2.bin");
   // auto test2 = safetensors("test2.bin");
   // std::cout << test2 << std::endl;

//    std::cout << a << std::endl;
}