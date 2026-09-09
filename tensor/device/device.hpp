#ifndef DEVICE_MANAGER_DEVICE_HPP
#define DEVICE_MANAGER_DEVICE_HPP

//
//  device.hpp — top-level entry point for the deviceManager.
//
//  Include this header to get access to the global DeviceManager and all
//  core types.  Backend support (CPU, CUDA, HIP, OpenGL, Vulkan, ...) is
//  loaded at runtime from .so plugins in the `plugins/` directory (or the
//  directory pointed to by the DEVICE_PLUGIN_DIR environment variable).
//
//  The DeviceManager constructor automatically discovers and loads plugins
//  in priority order.  No compile-time #include of backend headers is needed.
//

#include "common.hpp"

// global_device_manager is defined inline in common.hpp — including this
// header is sufficient to use it.

#endif // DEVICE_MANAGER_DEVICE_HPP
