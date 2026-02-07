#ifndef TEXTURE_LOADER_HPP
#define TEXTURE_LOADER_HPP
#include <tensor.hpp>
#include "file_loaders/image.hpp"

struct sampler2D: public Tensor<uint84, 2>
{
    using Tensor<uint84, 2>::Tensor;

    sampler2D(Tensor<uint84, 2> other):Tensor<uint84, 2>(other){}
};

static sampler2D load_texture(std::string filename) {
    int w, h, channels;
    uint8_t* data = (uint8_t*)(void*)stbi_load(filename.c_str(), &w, &h, &channels, 0);

    Tensor<uint8_t,3> imagergb (
        Shape<3>{(long)w, (long)h, (long)channels},
        data,
        MemoryType::kDDR
    );

    if(channels == 4){
        return *(sampler2D*)&imagergb;
    }

    Tensor<uint8_t,3> imagergba (
        Shape<3>{(long)w, (long)h, 4},
        MemoryType::kDDR
    );
    
    imagergba[{{},{},{0,3}}] = imagergb;
    imagergba[{{},{},{3}}] = 255;

    return *(sampler2D*)&imagergba;
};


#endif