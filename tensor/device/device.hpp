#ifndef TENSOR_DEVICE_DEVICE_HPP_
#define TENSOR_DEVICE_DEVICE_HPP_
#include "device/common.hpp"

#if defined(__CUDACC__)
#include "device/device_support/cuda/device.cuh"
#endif


#if defined(__HIPCC__)
#include "device/device_support/hip/device.hpp"
#endif

#include "device/device_support/vulkan/device.hpp"
#include "device/device_support/opengl/device.hpp"
#endif // TENSOR_DEVICE_DEVICE_HPP_