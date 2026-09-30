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

    // A safetensors file on the disk map: 8-byte header length, the JSON
    // header, then the tensor data.  The allocation's CPU view starts at the
    // data; the parsed header is in its metadata.
    struct SafetensorsHeader : public FileHeader {
        uint64_t header_size = 0;
        json header;
    };

    struct SafetensorsFormat : public FileFormat {
        const char* name() const override { return "safetensors"; }
        void read(const char* file, size_t size, AllocationMetadata& meta) const override {
            if (size < 8) throw std::runtime_error("not a safetensors file (too short)");
            uint64_t n;
            memcpy(&n, file, 8);
            if (8 + n > size) throw std::runtime_error("not a safetensors file (header length " + std::to_string(n) + ")");
            auto h = std::make_shared<SafetensorsHeader>();
            h->header_size = n;
            h->header = json::parse(file + 8, file + 8 + n);
            meta.data_offset = 8 + n;
            meta.byte_size = size - meta.data_offset;
            meta.shape = Shape<-1>{(long)(meta.byte_size / meta.type_size)};
            meta.header = h;
        }
        std::vector<char> write(AllocationMetadata&) const override {
            throw std::runtime_error("safetensors files are written with safetensors::save");
        }
        static const SafetensorsFormat* instance() {
            static SafetensorsFormat format;
            return &format;
        }
    };

    /**
     *
     */
    class safetensors: public Tensor<char, 1>
    {

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
         Tensor<T, rank> get(const char *name) const {
                if(!contains(name)){
                    std::cout << "Key not found:" << name << "\n";
                    exit(0);
                }

                const auto& meta = metas.at(name);
                void* data_begin = const_cast<char*>(storage) + meta.data_offsets.first;
                // char* data_end = const_cast<char*>(storage.data()) + meta.data_offsets.second;

                if constexpr (!std::is_same_v<T, void>){
                    if (meta.dtype != get_dtype<T>()){
                        // float formats convert (into host memory); anything else is an error
                        auto as_float = [&](size_t i) -> float {
                            switch (meta.dtype) {
                                case DataType::kFLOAT_32: return ((const float*)data_begin)[i];
                                case DataType::kBFLOAT_16: return float(((const bfloat16*)data_begin)[i]);
                                case DataType::kFLOAT_16: return float(((const float16*)data_begin)[i]);
                                default: throw std::runtime_error("unconvertible dtype");
                            }
                        };
                        bool convertible = (meta.dtype == DataType::kFLOAT_32 || meta.dtype == DataType::kBFLOAT_16 ||
                                            meta.dtype == DataType::kFLOAT_16) &&
                                           (std::is_same_v<T, float> || std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>);
                        if (!convertible) {
                            std::cerr << "Data type mismatch for " << name << ": tensor data type is " << meta.dtype
                                      << " but requested type is " << get_dtype<T>() << std::endl;
                            throw std::runtime_error("safetensors: data type mismatch");
                        }
                        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
                            Tensor<T, rank> out(Shape<rank>(meta.shape), MemoryLocation(MemoryType::kDDR), ComputeType::kCPU);
                            size_t n = out.shape.total_size();
                            for (size_t i = 0; i < n; i++) out.data.data[i] = T(as_float(i));
                            return out;
                        }
                    }
                }
                
                switch (meta.dtype)
                {
                    case DataType::kFLOAT_32:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kFLOAT_64:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kINT_32:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kINT_64:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kINT_8:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kUINT_8:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kUINT_16:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kUINT_32:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kUINT_64:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kFLOAT_16:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    case DataType::kBFLOAT_16:
                        return Tensor<T, rank>(meta.shape, (T*)data_begin, MemoryType::kDISK, this->storage_pointer);
                    default:
                        std::cerr << "Unsupported data type" << std::endl;
                        exit(0);
                    }
                
                
            }

        // only if T is not a Tensor
        template <typename T = void, int rank = -1, typename std::enable_if_t<!std::is_base_of_v<Tensor<typename T::value_type, T::rank>, T>, int> = 0>
         Tensor<T, rank> get(std::string name) const{
                return get<T, rank>(name.c_str());
         }

         template <typename T>
         T get(std::string name) const{
                return get<typename T::value_type, T::tensor_rank>(name.c_str());
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

        void init() {
                // the disk map parsed the header (SafetensorsFormat); the data view starts after it
                const SafetensorsHeader* h = this->storage_pointer->metadata.template header_as<SafetensorsHeader>();
                if (!h) throw std::runtime_error("safetensors: not opened as a safetensors file");
                std::cout << "Header size: " << h->header_size << std::endl;
                const json& metadatas = h->header;
                metas = std::unordered_map<std::string, const metadata_t>(metadatas.size());
                storage = this->data.data;   // the start of the tensor data

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

            // The file on the disk map, opened as a safetensors file.
            static Tensor<char, 1> open_file(const std::string& filename) {
                if (!std::ifstream(filename).good()) throw std::runtime_error("safetensors: cannot open " + filename);
                MemoryLocation disk(filename);
                AllocationMetadata meta = AllocationMetadata::create<char>(Shape<1>{0}, MemoryType::kDISK, ComputeType::kFILE, 0,
                                                                         AllocationFlags::kRW, disk.device_id);
                meta.file_format = SafetensorsFormat::instance();
                return Tensor<char, 1>(meta);
            }


            safetensors(const char* filename): Tensor<char, 1>(open_file(filename)) {
                std::cout << "Loading safetensors file: " << filename << "\n" << "\n";
                init();
            }

            safetensors(const std::string& filename): Tensor<char, 1>(open_file(filename)) {
                std::cout << "Loading safetensors file: " << filename << "\n" << "\n";
                init();
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
                bin.close();
            }
    };





    

    


#endif //SAFETENSORS_H