
#ifndef DEVICE_TYPE
#define DEVICE_TYPE
#include <string.h>
#include "file_loaders/json.hpp"
#include <iostream>
#include "enums/device_support/device.hpp"


__weak std::ostream &operator<<(std::ostream &os, const ComputeType &dtype)
{
    std::string s;
    to_json(s, dtype);
    os << s;
    return os;
}

__weak  std::ostream &operator<<(std::ostream &os, const MemoryType &mtype)
{
    std::string s;
    to_json(s, mtype);
    os << s;
    return os;
}












#if defined(__CUDACC__)
#include "enums/device_support/cuda/device.cuh"
#endif


#if defined(__HIPCC__)
#include "enums/device_support/hip/device.hpp"
#endif

#include "enums/device_support/vulkan/device.hpp"
#include "enums/device_support/opengl/device.hpp"





        

#endif