
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

enum AllocationFlags
{
    kR,//(1<<1),
    kW,//(1<<2),
    kRW,//(1<<1) | (1<<2),
    kSURFACE,//(1<<3),
    kTEXTURE//(1<<4),
};

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
                                                {kOPENGL, "OpenGL"},
                                                {kOPENGLTEXTURE, "OpenGLTexture"},
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