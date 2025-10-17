
#ifndef GLTF_HPP
#define GLTF_HPP

#include "tensor.hpp"
#include "file_loaders/json.hpp"
#include "enums/dtype.hpp"
#include <unordered_map>
#include <fstream>
#include <span>
using json = nlohmann::json;
#include <iostream>
#include <cstring>
#include <utility>
// png and jpeg libs
#define STB_IMAGE_IMPLEMENTATION
#include "image/image.hpp"


enum class PrimitiveType
{
    TRIANGLES,
    LINES,
    POINTS,
    LINE_LOOP,
    LINE_STRIP,
    POLYGON,
    TRIANGLE_STRIP,
    TRIANGLE_FAN
};

struct Primitive
{

    std::map<std::string, Tensor<void, 1>> attributes;

    Tensor<int, 1> indices; // Indices of the mesh

    PrimitiveType type = PrimitiveType::TRIANGLES; // Default to TRIANGLES

    static PrimitiveType fromInt(int mode)
    {
        switch (mode)
        {
        case 0:
            return PrimitiveType::POINTS;
        case 1:
            return PrimitiveType::LINES;
        case 2:
            return PrimitiveType::LINE_LOOP;
        case 3:
            return PrimitiveType::LINE_STRIP;
        case 4:
            return PrimitiveType::TRIANGLES;
        case 5:
            return PrimitiveType::TRIANGLE_STRIP;
        case 6:
            return PrimitiveType::TRIANGLE_FAN;
        default:
            std::cerr << "Unknown primitive type: " << mode << std::endl;
            return PrimitiveType::TRIANGLES; // Default fallback
        }
    }

    int materialIndex = -1; // Index of the material in the glTF file, -1 if no material is assigned
};


struct Mesh
{
    std::string name;
    std::vector<Primitive> primitives;
    Mesh(std::string name) : name(name) {}
    Mesh() : name("") {}
    void addPrimitive(Primitive primitive)
    {
        primitives.push_back(primitive);
    }
    void clear()
    {
        primitives.clear();
    }
    void setName(std::string name)
    {
        this->name = name;
    }
};


struct Material
{
    std::string name = "default";
    bool doubleSided = false;
    int baseColorTextureIndex = -1;
    int metallicRoughnessTextureIndex = -1;
    int normalTextureIndex = -1;
};

class Texture: public  Tensor<uint8_t, 3>
{
public:
    std::string filename;
    Texture(std::string filename, DeviceType device_type = DeviceType::kCPU): Tensor<uint8_t, 3>(Shape<3>{0, 0, 0}, nullptr, device_type), filename(filename) 
         {
            
            int w, h, channels;

            data = (uint8_t*)(void*)stbi_load(filename.c_str(), &w, &h, &channels, 0);
            shape.A = w;
            shape.B = h;
            shape.C = channels;
            strides.A = shape.B * shape.C;
            strides.B = shape.C;
            strides.C = 1;
            this->device_type = device_type;
            bitsize = sizeof(uint84);
            std::cout << "Loaded texture: " << filename << " with shape: " << shape << std::endl;
            calculate_metadata();
        // stb_image


        }
};

class Skeleton: public Tensor<mat4, 1>
{
public:
    std::string name;
    std::vector<std::string> jointNames; // Names of the joints in the skeleton

    Skeleton(std::string name, Shape<1> shape = Shape<1>{}) : Tensor<mat4, 1>(shape), name(name) {
        for (int i = 0; i < shape.total_size(); i++) {
            this->operator[](i) = mat4::identity(); // Initialize with identity matrices
        }
    }

   
};

class gltf
{
public:
    std::vector<Tensor<char, 1>> data;
    std::vector<Tensor<char, 1>> bufferViews;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<Mesh> meshes;
    std::vector<Skeleton> skeletons; // Assuming Skeleton is defined elsewhere

    gltf(std::basic_istream<char> &in, std::string path = "")
    {
        json j;
        in >> j;

        if (j.is_null())
        {
            throw std::runtime_error("Failed to parse JSON");
        }
        if (!j.contains("asset") || !j["asset"].contains("version"))
        {
            throw std::runtime_error("Invalid glTF file: missing asset version");
        }


        // textures
        if (j.contains("textures"))
        {
            auto texturesJson = j["textures"];
            for (const auto &textureJson : texturesJson)
            {
                if (textureJson.contains("source"))
                {
                    int sourceIndex = textureJson["source"];
                    if (j.contains("images") && j["images"].size() > sourceIndex)
                    {
                        auto imageJson = j["images"][sourceIndex];
                        if (imageJson.contains("uri"))
                        {
                            std::string uri = imageJson["uri"];
                            Texture texture(path + "/" + uri);
                            textures.push_back(texture);
                        }
                    }
                }
            }
        }

        // materials
        if (j.contains("materials"))
        {
            auto materialsJson = j["materials"];
            for (const auto &materialJson : materialsJson)
            {
                Material material;
                if (materialJson.contains("name"))
                {
                    material.name = materialJson["name"];
                }
                if (materialJson.contains("doubleSided"))
                {
                    material.doubleSided = materialJson["doubleSided"];
                }
                if (materialJson.contains("pbrMetallicRoughness"))
                {
                    auto pbr = materialJson["pbrMetallicRoughness"];
                    if (pbr.contains("baseColorTexture") && pbr["baseColorTexture"].contains("index"))
                    {
                        material.baseColorTextureIndex = pbr["baseColorTexture"]["index"];
                    }
                    if (pbr.contains("metallicRoughnessTexture") && pbr["metallicRoughnessTexture"].contains("index"))
                    {
                        material.metallicRoughnessTextureIndex = pbr["metallicRoughnessTexture"]["index"];
                    }
                }
                if (materialJson.contains("normalTexture") && materialJson["normalTexture"].contains("index"))
                {
                    material.normalTextureIndex = materialJson["normalTexture"]["index"];
                }
                materials.push_back(material);
            }
        }

        // Read the binary data
        if (j.contains("buffers"))
        {
            auto buffers = j["buffers"];
            if (!buffers.empty() && buffers[0].contains("uri"))
            {
                std::string uri = buffers[0]["uri"];
                std::ifstream bin(path + "/" + uri, std::ios::binary);
                if (!bin)
                {
                    throw std::runtime_error("Failed to open buffer file: " + uri);
                }
                bin.seekg(0, std::ios::end);
                size_t size = bin.tellg();
                bin.seekg(0, std::ios::beg);
                auto dataa = Tensor<char, 1>(Shape<1>{size}, DeviceType::kCPU);
                bin.read(dataa.data, size);
                data.push_back(dataa);
            }
        }

        if (j.contains("bufferViews"))
        {
            auto bufferViewsJson = j["bufferViews"];

            for (size_t i = 0; i < bufferViewsJson.size(); ++i)
            {
                auto &view = bufferViewsJson[i];
                if (view.contains("buffer") && view.contains("byteLength"))
                {
                    size_t bufferIndex = view["buffer"];
                    size_t byteLength = view["byteLength"];
                    size_t byteOffset = view.contains("byteOffset") ? (size_t)view["byteOffset"] : 0;
                    size_t byteStride = view.contains("byteStride") ? (size_t)view["byteStride"] : 1;
                    bufferViews.push_back(data[bufferIndex][{{byteOffset, byteOffset + byteLength, 1}}]);
                }
            }
        }



        if (j.contains("meshes"))
        {
            int length = j["meshes"].size();
            auto meshesJson = j["meshes"];
            skeletons.push_back(Skeleton("baseskeleton", {length}));
                    
            int primcount = 0;
            for (const auto &meshJson : meshesJson)
            {
                Mesh mesh(meshJson["name"]);
                if (meshJson.contains("primitives"))
                {

                    
                    for (const auto &primitiveJson : meshJson["primitives"])
                    {
                        if (primitiveJson.contains("indices"))
                        {
                            int indicesBufferViewIndex = primitiveJson["indices"];
                            auto IndiceAccessor = j["accessors"][indicesBufferViewIndex];
                            int byteOffset = IndiceAccessor.contains("byteOffset") ? int(IndiceAccessor["byteOffset"]) : 0;
                            int indicecount = IndiceAccessor["count"];
                            int componentType = IndiceAccessor["componentType"];
                            if (componentType != 5125)
                            {
                                std::cerr << "Unsupported component type for indices: " << componentType << std::endl;
                                continue;
                            }
                            Tensor<int,1> indices = bufferViews[IndiceAccessor["bufferView"]][{{byteOffset, byteOffset + indicecount * sizeof(int), 1}}].view<int>();
                            int primitiveType = primitiveJson.contains("mode") ? int(primitiveJson["mode"]) : 4; // Default to TRIANGLES

                            Primitive primitive = {
                                .attributes = std::map<std::string, Tensor<void, 1>>(),
                                .indices = indices,
                                .type = Primitive::fromInt(primitiveType),
                                .materialIndex = primitiveJson.contains("material") ? int(primitiveJson["material"]) : -1
                            };

                            if (primitiveJson.contains("attributes"))
                            {
                                int positionlength = 0;
                                for (const auto &attr : primitiveJson["attributes"].items())
                                {
                                    std::string attrName = attr.key();
                                    int AccessorIndex = attr.value();
                                    auto Accessor = j["accessors"][AccessorIndex];
                                    int byteOffset = Accessor.contains("byteOffset") ? int(Accessor["byteOffset"]) : 0;
                                    int componentType = Accessor["componentType"];
                                    int count = Accessor["count"];
                                    positionlength = count;
                                    int bufferViewIndex = Accessor["bufferView"];
                                  

                                    auto bufferView = bufferViews[int(bufferViewIndex)];
                                    if (componentType == 5126)
                                    { // FLOAT
                                        if(Accessor["type"] == "VEC3"){
                                        primitive.attributes.emplace(std::pair(attrName,bufferView[{{byteOffset, byteOffset + count * sizeof(float3)}}].view<float3>()));
                                        }else if(Accessor["type"] == "VEC2"){
                                          primitive.attributes.emplace(std::pair(attrName,bufferView[{{byteOffset, byteOffset + count * sizeof(float2)}}].view<float2>()));
                                        }else if(Accessor["type"] == "VEC4"){
                                           primitive.attributes.emplace(std::pair(attrName,bufferView[{{byteOffset, byteOffset + count * sizeof(float4)}}].view<float4>()));
                                        }else if(Accessor["type"] == "SCALAR"){
                                           primitive.attributes.emplace(std::pair(attrName,bufferView[{{byteOffset, byteOffset + count * sizeof(float)}}].view<float>()));
                                        }
                                        
                                    }
                                    else if (componentType == 5123)
                                    { // UNSIGNED_SHORT
                                        primitive.attributes.emplace(std::pair(attrName,bufferView[{{byteOffset, byteOffset + count * sizeof(uint16_t)}}].view<uint16_t>()));
                                    }
                                    else if (componentType == 5125)
                                    { // UNSIGNED_INT
                                        primitive.attributes.emplace(std::pair(attrName,bufferView[{{byteOffset, byteOffset + count * sizeof(uint32_t)}}].view<uint32_t>()));
                                    }
                                    else
                                    {
                                        std::cerr << "Unsupported component type: " << componentType << std::endl;
                                    }
                                }
                                // emplace the bone_ids 
                                auto a = Tensor<int, 1>(Shape<1>{positionlength}, DeviceType::kCPU);
                                // a = uint32_t(primcount);
                                for (int i = 0; i < positionlength; i++)
                                {
                                    a[i] = primcount;
                                }
                                std::cout << a << std::endl;
                                primitive.attributes.emplace(std::pair("bone_ids", a));
                                std::cout << "Hasbone_ids: " << primitive.attributes.contains("bone_ids") << std::endl;

                                std::cout << "Added primitive bone_id "<<primcount<<" with bones of leng " << positionlength << " attributes." << std::endl;
                                // sey the bone_ids to = primcount
                                primcount+=1;
                            }



                            mesh.addPrimitive(primitive);
                        }
                    }
                }
                meshes.push_back(mesh);
            }
        }
    }

    gltf(std::string path, std::string filename)
    {
        std::ifstream bin(path + "/" + filename, std::ios::binary);
        *this = gltf(bin, path);
    }

    ~gltf()
    {

    }

    friend std::ostream &operator<<(std::ostream &os, const gltf &model)
    {
        os << "gltf model with " << model.meshes.size() << " meshes." << std::endl;
        for (const auto &mesh : model.meshes)
        {
            os << "Mesh: " << mesh.name << " with " << mesh.primitives.size() << " primitives." << std::endl;
            for (const auto &primitive : mesh.primitives)
            {
                os << "  Primitive with " << primitive.attributes.size() << " attributes." << std::endl;
                for (const auto &attr : primitive.attributes)
                {
                    os << "    Attribute: " << attr.first << std::endl;
                }
            }
        }
        os << "Materials: " << model.materials.size() << std::endl;
        for (const auto &material : model.materials)
        {
            os << "  Material: " << material.name << ", Double Sided: " << material.doubleSided
               << ", Base Color Texture Index: " << material.baseColorTextureIndex
               << ", Metallic Roughness Texture Index: " << material.metallicRoughnessTextureIndex
               << ", Normal Texture Index: " << material.normalTextureIndex << std::endl;
        }
        os << "Textures: " << model.textures.size() << std::endl;
        for (const auto &texture : model.textures)
        {
            os << "  Texture: " << texture.filename << ", Shape: " << texture.shape << std::endl;
        }
        os << "Skeletons: " << model.skeletons.size() << std::endl;
        for (const auto &skeleton : model.skeletons)
        {
            os << "  Skeleton: " << skeleton.name << ", Joints: " << skeleton.shape.total_size() << std::endl;
            os << skeleton << std::endl;
        }
        return os;
    }
};

#endif // GLTF_HPP