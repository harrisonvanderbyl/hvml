//
// Created by mfuntowicz on 3/28/23.
//

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <span>
#include <fstream>
#include "tensor.hpp"
#include "file_loaders/json.hpp"

using json = nlohmann::json;


    
    struct metadata_t {
        DataType dtype;
        std::vector<size_t> shape;
        std::pair<size_t, size_t> data_offsets;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(metadata_t, dtype, shape, data_offsets)

    /**
     *
     */
    class safetensors {

    public:

        std::unordered_map<std::string, const metadata_t> metas;

        
       
        const char* storage = nullptr;
        /**
         *
         * @return
         */
        inline size_t size() const { return metas.size(); }

        /**
         *
         * @param name
         * @return
         */
        
         template <typename T = void, int rank = -1>
         Tensor<T, rank> operator[](const char *name) const {
                if(!contains(name)){
                    std::cout << "Key not found:" << name << "\n";
                    exit(0);
                }

                const auto& meta = metas.at(name);
                char* data_begin = const_cast<char*>(storage) + meta.data_offsets.first;
                // char* data_end = const_cast<char*>(storage.data()) + meta.data_offsets.second;

                if (typeid(T)!=typeid(void)){
                    if (meta.dtype != get_dtype<T>()){
                        std::cerr << "Data type mismatch, tensor data type is " << meta.dtype << " but requested type is " << get_dtype<T>() << std::endl;
                        exit(0);
                    }
                }
                
                switch (meta.dtype)
                {
                    case DataType::kFLOAT_32:
                        return Tensor<float, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kFLOAT_64:
                        return Tensor<double, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kINT_32:
                        return Tensor<int32_t, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kINT_64:
                        return Tensor<int64_t, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kINT_8:
                        return Tensor<int8_t, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kUINT_8:
                        return Tensor<uint8_t, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kUINT_16:
                        return Tensor<uint16_t, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kUINT_32:
                        return Tensor<uint32_t, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kUINT_64:
                        return Tensor<uint64_t, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kFLOAT_16:
                        return Tensor<float16, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    case DataType::kBFLOAT_16:
                        return Tensor<bfloat16, rank>(meta.shape, data_begin, DeviceType::kCPU);
                    default:
                        std::cerr << "Unsupported data type" << std::endl;
                        exit(0);
                    }
                
                
            }

        template <typename T = void, int rank = -1>
         Tensor<T, rank> operator[](std::string name) const{
                return operator[](name.c_str());
         }

         /**
         *
         * @param name
         * @return
         */
        inline std::vector<const char*> keys() const {
            std::vector<const char*> keys;
            keys.reserve(metas.size());
            for (auto &item: metas) {
                keys.push_back(item.first.c_str());
            }
            return keys;
        }

        // contains key
        inline bool contains(const char* name) const {
            // auto keys = this->keys();
            // bool found = false;

            // for (auto key : keys){
            //     if (strcmp(key, name) == 0){
            //         found = true;
            //     }

            // }
            // return found;
            return metas.find(name) != metas.end();
        }
        inline bool contains(std::string name) const {
            return contains(name.c_str());
        }

        safetensors(){};

        safetensors(std::basic_istream<char> &in) {
                uint64_t header_size = 0;

                // todo: handle exception
                in.read(reinterpret_cast<char *>(&header_size), sizeof header_size);

                std::vector<char> meta_block(header_size);
                in.read(meta_block.data(), static_cast<std::streamsize>(header_size));
                const auto metadatas = json::parse(meta_block);

                // How many bytes remaining to pre-allocate the storage tensor
                in.seekg(0, std::ios::end);
                std::streamsize f_size = in.tellg();
                in.seekg(8 + header_size, std::ios::beg);
                const auto tensors_size = f_size - 8 - header_size;

                metas = std::unordered_map<std::string, const metadata_t>(metadatas.size());
                // allocate in a way that prevents it from being freed
                // storage = new char[tensors_size];
                posix_memalign((void**)&storage, 128, tensors_size);
                
               

                // Read the remaining content
                in.read((char*)storage, static_cast<std::streamsize>(tensors_size));

                // Populate the meta lookup table
                if (metadatas.is_object()) {
                    for (auto &item: metadatas.items()) {
                        if (item.key() != "__metadata__") {
                            const auto name = std::string(item.key());
                            const auto& info = item.value();

                            const metadata_t meta = {info["dtype"].get<DataType>(), info["shape"], info["data_offsets"]};
                            metas.insert(std::pair<std::string, metadata_t>(name, meta));
                        }
                    }
                }

            }


            safetensors(const char* filename) {
                std::ifstream bin(filename, std::ios::binary);
                *this = safetensors(bin);
            }

            safetensors(const std::string& filename) {
                std::ifstream bin(filename, std::ios::binary);
                *this = safetensors(bin);
            }

            template <typename T = void, int size = -1>
            inline void add(const char* name, const Tensor<T, size>& tensor) {
                const auto dtype = get_dtype<T>();
                auto shape = std::vector<size_t>();
                for (int i = 0; i < tensor.shape.ndim(); i++) {
                    shape.push_back(((unsigned long*)&(tensor.shape))[i]);
                }
                const metadata_t meta = {dtype, shape, {(unsigned long )(tensor.data), (unsigned long)(tensor.total_bytes)}};
                metas.insert(std::pair<std::string, metadata_t>(name, meta));
            }

            inline void save(std::basic_ostream<char> &out) {
                json metadatas = json::object();
                size_t offset = 0;
                for (auto &item: metas) {
                    const auto name = item.first;
                    auto meta = item.second;
                    meta.data_offsets.first = offset;
                    offset += meta.data_offsets.second;
                    metadatas[name] = meta;
                }

                const auto meta_str = metadatas.dump();
                const uint64_t header_size = meta_str.size();
                out.write(reinterpret_cast<const char *>(&header_size), sizeof header_size);
                out.write(meta_str.c_str(), meta_str.size());

                for (auto &item: metas) {
                    const auto meta = item.second;
                    const auto data = meta.data_offsets.first;
                    out.write((char*)data, meta.data_offsets.second);
                }
            }

            inline void save(const char* filename) {
                std::ofstream bin(filename, std::ios::binary);
                save(bin);
            }
    };





    

    


#endif //SAFETENSORS_H