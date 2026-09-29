
#ifndef DEVICE_TYPE
#define DEVICE_TYPE
#include <string.h>
#include "file_loaders/json.hpp"
#include <iostream>

#define __weak __attribute__((weak))

enum ComputeType
{
    kCPU,
    kCUDA,
    kHIP,
    kVULKAN,
    kVULKANTEXTURE,
    kOPENGL,
    kOPENGLTEXTURE,
    kFILE,
    kUnknown,
    ComputeTypeCount
};

enum MemoryType
{
    kDDR,
    kCUDA_VRAM,
    kHIP_VRAM,
    kDISK,
    kUnknown_MEM
};

// ---------------------------------------------------------------------------
//  AllocationFlags — access flags plus "what can this memory be used as".
//
//  kR / kW are host/transfer access.  The remaining bits are *view* flags:
//  they tell a graphics allocator (the Vulkan plugin) which kinds of GPU
//  objects to create for the allocation, and — when passed to
//  Tensor::to_compute(ct, flags) — which kind of in-place view to create
//  over an existing allocation.
//
//      kSURFACE      render target: colour attachment (or depth attachment
//                    for depth formats) — can be rendered into
//      kTEXTURE      sampled image — bind as `sampler2D`
//      kTEXELBUFFER  buffer read through a VkBufferView — bind as
//                    `samplerBuffer` (texelFetch)
//      kSTORAGE      storage image / storage texel buffer (`image2D`,
//                    `imageBuffer`)
//      kDEPTH        depth format (D32_SFLOAT)
//      kLINEAR       image memory is plain row-major and shared with a
//                    VkBuffer, so the same allocation can be viewed in place
//                    as a buffer, a texel buffer, a texture, a render target,
//                    or a HIP/CUDA pointer.  This is the default for colour
//                    images whenever the device supports it.
//      kOPTIMAL      opt out of kLINEAR: GPU-only tiled image (fastest to
//                    sample / render; can't be viewed as a buffer)
//
//  Images get every usage their format supports (a surface can always be
//  sampled, a texture can always be rendered into), so the flags you
//  allocate with don't limit the views you can take later.
// ---------------------------------------------------------------------------
enum AllocationFlags
{
    kR           = (1<<1),
    kW           = (1<<2),
    kRW          = (1<<1) | (1<<2),
    kSURFACE     = (1<<3),
    kTEXTURE     = (1<<4),
    kTEXELBUFFER = (1<<5),
    kSTORAGE     = (1<<6),
    kDEPTH       = (1<<7),
    kLINEAR      = (1<<8),
    kOPTIMAL     = (1<<9),
};

// Bits that select a *kind of view* (as opposed to host read/write access).
constexpr int kVIEW_FLAGS = kSURFACE | kTEXTURE | kTEXELBUFFER | kSTORAGE | kDEPTH | kLINEAR | kOPTIMAL;

inline AllocationFlags operator|(AllocationFlags a, AllocationFlags b) {
    return (AllocationFlags)((int)a | (int)b);
}

inline AllocationFlags operator&(AllocationFlags a, AllocationFlags b) {
    return (AllocationFlags)((int)a & (int)b);
}

inline bool has_flag(AllocationFlags flags, AllocationFlags bit) {
    return ((int)flags & (int)bit) != 0;
}

enum AssignmentType {
    Direct,
    InplaceAdd,
    NoAssignment
};


NLOHMANN_JSON_SERIALIZE_ENUM(ComputeType, {
                                             {kCPU, "CPU"},
                                                {kCUDA, "CUDA"},
                                                {kHIP, "HIP"},
                                                {kVULKAN, "Vulkan"},
                                                {kVULKANTEXTURE, "VulkanTexture"},
                                                {kOPENGL, "OpenGL"},
                                                {kOPENGLTEXTURE, "OpenGLTexture"},
                                                {kFILE, "File"},
                                                {kUnknown, "Unknown"}
                                         })

NLOHMANN_JSON_SERIALIZE_ENUM(MemoryType, {
                                             {kDDR, "DDR_RAM"},
                                                {kCUDA_VRAM, "CUDA_VRAM"},
                                                {kHIP_VRAM, "HIP_VRAM"},
                                                {kDISK, "DISK"},
                                                {kUnknown_MEM, "UNKNOWN"}
                                         })



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

        

#endif