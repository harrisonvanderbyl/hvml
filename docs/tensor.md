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