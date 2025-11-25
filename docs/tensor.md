## HVML c++ Tensor Library

Tensors are mathematical objects, representing a collection of numbers.
They are useful in inumerable contexts.

### Creating a tensor object

A tensor has the following characteristics.

1) Shape 

>A tensor has a shape.
For example, a shape of [3,3] describes a 9 object grid, with width and height of 3.
[3,3,3] would represent a cube.

2) Data

>A pointer to the underlying data.

3) Stride

>Stride describes how each step along the data that a single step in a particular dimension takes.
for example:
[1,2,3,4,5,6] might be the underlying data,
A tensor that points to that data, with the stride of [2] and a shape of [3],
would look like this:
[1,3,5]

4) Datatype

>The data type describes what the data represents, for example, float, double, character, integer.

5) Device

> The device describes what device the data is stored on, for example, CUDA, CPU, VULKAN

Heres how you would use a tensor.

```cpp
#include "tensor.hpp"

int main(){
    Tensor<float> = {{3,3},kCPU};
}
```

The interface is:

`Tensor<datatype, |dimensions|>(shape, |pointer|, |device|)`

### Supported Actions

Indexing and slicing:

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