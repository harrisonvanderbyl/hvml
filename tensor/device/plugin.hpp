#ifndef DEVICE_MANAGER_PLUGIN_HPP
#define DEVICE_MANAGER_PLUGIN_HPP

//
//  plugin.hpp — shared interface between the deviceManager core and backend
//  plugins (.so files loaded at runtime via dlopen).
//
//  Every plugin MUST export these three `extern "C"` symbols:
//
//      const char* plugin_name();        // human-readable backend name
//      int         plugin_priority();    // load order (lower = earlier)
//      void        plugin_register(DeviceManager* dm);
//
//  A plugin MAY optionally export a fourth symbol for deferred init:
//
//      void        plugin_init(DeviceManager* dm);
//
//  This is used by backends that depend on an external resource which does
//  not exist at program start — e.g. OpenGL needs a live GL context before
//  it can query the vendor or register allocators.  When plugin_init is
//  present, plugin_register() should do only lightweight setup (no device
//  creation).  The display layer calls dm->init_plugin("name") after the
//  required context has been created.
//
//  Inside plugin_register() the plugin:
//    1. Counts its devices.
//    2. Creates an AllocationMap per device and registers it via
//       dm->register_memory_device(MemoryType, device_id, AllocationMap*).
//    3. Creates a ComputeDeviceBase per device and registers it via
//       dm->register_compute_device(ComputeType, device_id, ComputeDeviceBase*).
//    4. Registers any cross-device converters on AllocationMaps that were
//       created by earlier (lower-priority) plugins, obtained via
//       dm->get_device(MemoryType, device_id).
//
//  Priority convention:
//      10  CPU     (always first — other backends register converters on it)
//      20  Disk    (built into core, see common.hpp)
//      30  CUDA
//      40  HIP
//      50  Vulkan
//      60  OpenGL  (depends on a GL context existing; usually last)
//

#include "common.hpp"

// Re-export the C ABI typedefs so plugins can write the function signatures
// without re-declaring them.
using PluginNameFn     = const char* (*)();
using PluginPriorityFn = int (*)();
using PluginRegisterFn = void (*)(DeviceManager*);
using PluginInitFn     = void (*)(DeviceManager*);

#endif // DEVICE_MANAGER_PLUGIN_HPP
