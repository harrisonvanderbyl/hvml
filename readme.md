### HVML
Intuitive c++ tensor library based on the pytorch interface.

```c++

#include "tensor.hpp"

int main(){
        
    Tensor<float> item = {{1024,1024},kCPU};
    item = 0; // set all entries to 0;
    item[0] = 1; // set first row to 1;
    item[{1,1}] = 2; // set first row, first item to 2
    item[{2,{0,20}}] = 3; // set second row, first 20 items to 3;
    item[{3,{0,20,2}}] = 4; // set third row, every second object to 4
    std::cout << item[0] << std::endl;
    std::cout << item[1] << std::endl;
    std::cout << item[2] << std::endl;
    std::cout << item[{3,{0,4}}] << std::endl;

    /*
    (dtype=32bit float, shape=[1024], strides=[1], device_type=CPU)[1, 1, ..., 1, 1]
    (dtype=32bit float, shape=[1024], strides=[1], device_type=CPU)[0, 2, ..., 0, 0]
    (dtype=32bit float, shape=[1024], strides=[1], device_type=CPU)[3, 3, ..., 0, 0]
    (dtype=32bit float, shape=[4], strides=[1], device_type=CPU)[4, 0, 4, 0]
    */

}

```