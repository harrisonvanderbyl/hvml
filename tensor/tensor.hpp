#ifndef TENSOR
#define TENSOR
#include "stdlib.h"
#include "enums/dtype.hpp"
#include "vector"
#include <stdarg.h>
#include <string>
#include "device/device.hpp"
#include "shape.hpp"
#include <iostream>
#include <string.h>


template <typename R = float, int rank = -1>
class Tensor
{
public:
    using value_type = R;
    static constexpr int tensor_rank = rank;
    Shape<rank> shape;
    Shape<rank> strides;
    MassagedMemory<R> data;
    BaseMemoryAllocation *storage_pointer = NULL;
    unsigned long bitsize;

    size_t total_size = 0;
    size_t total_bytes = 0;
    AllocationMap* device = &global_device_manager.get_device(MemoryType::kDDR,0);
    // Row gather (see tensor_index): the leading `ndim - indexer_tail` dims
    // walk `indexer` (a device array of row numbers) and the row found is
    // multiplied by `indexer_stride`; the trailing `indexer_tail` dims use
    // `strides` as usual.  indexer_tail = 0, indexer_stride = 1 is a plain
    // per-element index map.
    unsigned long* indexer = nullptr;
    long indexer_stride = 1;
    int indexer_tail = 0;

    template <typename T, int orank>
    void copy_indexer(const Tensor<T, orank>& other)
    {
        indexer = other.indexer;
        indexer_stride = other.indexer_stride;
        indexer_tail = other.indexer_tail;
    }

    void calculate_metadata()
    {
        total_bytes = shape.total_size() * bitsize;
        
        if(shape.ndim() == 0){
            total_size = 1;
            return;
        }

        strides[shape.ndim() - 1] = 1;
        total_size = shape.total_size();
        for (int i = shape.ndim() - 2; i >= 0; i--)
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    Tensor(){}

    Tensor(Shape<rank> __a, MemoryLocation memory_device, ComputeType compute_type = ComputeType::kUnknown)
    {
        this->bitsize = sizeof(R);
        this->device = memory_device.allocation_map;
        // printf("ndim: %d\n", __a.ndim());
        this->shape = __a;
        this->strides = __a.clone();
        storage_pointer = device->allocate(AllocationMetadata::create<R>(__a, memory_device.memory_type, (compute_type==kUnknown)?device->default_allocator_type:compute_type, 0, AllocationFlags::kRW, memory_device.device_id));
        this->shape = storage_pointer->metadata.shape;
        this->strides = storage_pointer->metadata.shape.calc_strides();
        calculate_metadata();
        data = this->device->template get_massaged_pointer<R>(storage_pointer, AllocationMetadata::create<R>(__a, memory_device.memory_type, device->default_compute_type, 0, AllocationFlags::kRW, memory_device.device_id));
    }

    Tensor(AllocationMetadata metadata){
        this->bitsize = sizeof(R);
        this->device = MemoryLocation(metadata.storage_device, metadata.device_id).allocation_map;
        this->shape = metadata.shape;
        this->strides = this->shape.clone();
        calculate_metadata();
        storage_pointer = device->allocate(metadata);
        this->shape = storage_pointer->metadata.shape;
        this->strides = storage_pointer->metadata.shape.calc_strides();
        calculate_metadata();
        data = this->device->template get_massaged_pointer<R>(storage_pointer, AllocationMetadata::create<R>(metadata.shape, metadata.storage_device, device->default_compute_type, 0, AllocationFlags::kRW, metadata.device_id));
    
    }
    
    
    Tensor(Shape<rank> __a, R *datain, MemoryLocation memory_device, BaseMemoryAllocation *storage_pointer = nullptr)
    {
        this->device = memory_device.allocation_map;
        this->bitsize = sizeof(R);
        this->shape = __a;
        this->strides = __a.clone();
        calculate_metadata();
        if (storage_pointer == nullptr){
            this->storage_pointer = new BaseMemoryAllocation(
                AllocationMetadata::create<R>(
                    __a,
                    memory_device.memory_type,
                    this->device->default_compute_type,
                    0,
                    AllocationFlags::kRW,
                    memory_device.device_id
                ),
                datain
            );
        }else{
            this->storage_pointer = storage_pointer;
        }
        this->data = MassagedMemory<R>(AllocationMetadata::create<R>(__a, memory_device.memory_type, device->default_compute_type, 0, AllocationFlags::kRW, memory_device.device_id), datain, this->storage_pointer);

        this->device->register_allocation(this->storage_pointer);
    }
    

    Tensor(Shape<rank> __a, MassagedMemory<R> datain, MemoryLocation memory_device, BaseMemoryAllocation *storage_pointer)
    {
        this->device = memory_device.allocation_map;
        this->bitsize = sizeof(R);
        this->shape = __a;
        this->strides = __a.clone();
        calculate_metadata();
        
        this->storage_pointer = storage_pointer;
     
        this->data = datain;

        this->device->register_allocation(this->storage_pointer);
    }

    // Tensor(R a)
    // {
    //     this->device_type = MemoryType::kDDR;
    //     this->shape = Shape(1);
    //     this->strides = Shape(1);
    //     calculate_metadata();
    //     data = malloc(total_bytes);
    //     *((R *)data) = a;
    // }

    template <typename M>
    inline Tensor<R, rank> operator=(M a)
    {
        tensor_copy(*this, a);
        return *this;
    }


    Tensor<R, rank>& operator=(const Tensor<R, rank>& other){
        if (this->storage_pointer == nullptr){
            this->device = other.device;
            this->shape = other.shape;
            this->strides = other.strides;
            this->bitsize = other.bitsize;
            this->total_size = other.total_size;
            this->total_bytes = other.total_bytes;
            this->storage_pointer = other.storage_pointer;
            if (other.storage_pointer != nullptr){
                this->device->register_allocation(this->storage_pointer);
            }

            this->data = other.data;
            calculate_metadata();
            this->strides = other.strides;
            copy_indexer(other);
        }else{
            tensor_copy(*this, other);
        }
        return *this;
    };

    template <typename M, int V>
    Tensor<R, rank>& operator=(const Tensor<M,V>& other)
    {
        tensor_copy(*this, other);
        return *this;
    }

    template <typename X = SliceList<-1>, int newrank = X::reducedims<0?-1:std::max(rank-X::reducedims,-1)>
    // result of operator[] is a tensor of rank - T::reducedims if rank-T::reducedims > 0 else it is a scalar of type R
    inline std::conditional_t<newrank == 0, R&, Tensor<R, newrank>>
    gather(X inp) const
    {
        // return *this;
        // Slice i = inp.args.args[0];
        // static constexpr int reducedims = SliceArray<Slice>::reducedims;


        // if (i.start < -shape[0] || i.end <= -shape[0] || i.start >= shape[0] || i.end > shape[0])
        // {
        //     std::cerr << "Index out of range" << std::endl;
        //     std::cerr << "Index: " << i.start << " to " << i.end << " Shape: " << shape << std::endl;
        //     throw std::runtime_error("Index out of range");
        // }

        // i.start = i.start % shape[0];
        // if(
        //     i.end < 0
        // ){
        //     i.end = shape[0] - (i.end + (reducedims!=0));
        // }
        MassagedMemory<R> startingpointer = data;//(R*)device->get_massaged_pointer((R *)data, device->default_compute_type);
        int ndim = shape.ndim();
        // dims before `index_dims` of a tensor_index view move through the
        // index array instead of the data
        int index_dims = indexer ? ndim - indexer_tail : 0;
        unsigned long* index_start = indexer;
        int ii = 1;
        for (; ii <= ndim ; ii++)
        {
            long shapeofset = shape[-ii];
            long start = (static_cast<long>(inp[ndim-ii].start) + shapeofset) % shapeofset;
            if (ndim - ii < index_dims) index_start += start * strides[-ii];
            else startingpointer += start * strides[-ii];
        }
        
        if constexpr (newrank == 0)
        {
            // should only return cpu editable scalar if possible
            if (indexer) return *(startingpointer.data + (*index_start) * indexer_stride);
            return *(startingpointer.data);
        }
        else{
        auto newshape = Shape<newrank>();
        
        auto newstrides = Shape<newrank>();
        int i = 0;
        int j = 0;
        // int multiplier = 1;
        for(
            ; i < ndim; i+=1
        ){
            if (inp[i].is_slice)
            {
                if (inp[i].is_empty)
                {
                    newshape[j] = shape[i];
                    newstrides[j] = strides[i];
                    // multiplier *= shape[i];
                }
                else
                {
                    // negative indices count from the end; `end == shape` is a
                    // full slice (the old modulo turned it into length 0)
                    long st = static_cast<long>(inp[i].start);
                    long en = static_cast<long>(inp[i].end);
                    if (st < 0) st += shape[i];
                    if (en < 0) en += shape[i];
                    long spot = en - st;
                    if(inp[i].end.is_default){
                        spot = shape[i] - st;
                    }
                    // ceil(spot / step): {1::2} on a shape of 5 gives 2, {0,5,2} gives 3
                    newshape[j] = spot / inp[i].step + (spot % inp[i].step != 0);
                    newstrides[j] = this->strides[i] * inp[i].step;
                    // std::cout << "not implemented" << std::endl;
                }

                // std::cout << "Stride " << i << ": " << newstrides[j] << std::endl;
                // std::cout << "Shape " << i << ": " << newshape[j] << std::endl;
                j++;
            }
            else
            {
                // if(i > 0){

                // }
                
            }
        }
        
        Tensor<R, newrank> b = {newshape, startingpointer, *device, storage_pointer};
        
        b.strides = newstrides;
        if (indexer) {
            int tail = 0;
            for (int d = index_dims; d < ndim; d++) tail += inp[d].is_slice;
            b.indexer = index_start;
            b.indexer_stride = indexer_stride;
            b.indexer_tail = tail;
        }
        
        // // std::cout << (i.end - i.start) / i.step << std::endl;
        // // std::cout << "Start: " << i.start << " End: " << i.end << " Step: " << i.step << std::endl;
        
        // // std::cout << "Shape: " << b.shape << std::endl;
        // b.data = (void *)((uint8_t *)b.data + i.start * strides[0] * bitsize);
        // std::cout << b.strides[newrank-1] << std::endl;
        // std::cout << newstrides[newrank-1] << std::endl;
        // std::cout << X::reducedims << ":" << newrank << std::endl;

        return b;
        }
    }

    std::conditional_t<rank == 0, R&, Tensor<R, std::max(rank - 0,-1)>>
    operator[](const SliceList<0>& i) const
    {
        return gather(i);
    }

    std::conditional_t<rank == 1, R&, Tensor<R, std::max(rank - 1,-1)>>
    operator[](const SliceList<1>& i) const
    {
        return gather(i);
    }

    std::conditional_t<rank == 2, R&, Tensor<R, std::max(rank - 2,-1)>>
    operator[](const SliceList<2>& i) const
    {
        return gather(i);
    }
    std::conditional_t<rank == 3, R&, Tensor<R, std::max(rank - 3,-1)>>
    operator[](const SliceList<3>& i) const
    {
        return gather(i);
    }

    std::conditional_t<rank == 4, R&, Tensor<R, std::max(rank - 4,-1)>>
    operator[](const SliceList<4>& i) const
    {
        return gather(i);
    }

    std::conditional_t<rank == 1, R&, Tensor<R, std::max(rank - 1,-1)>>
    operator[](const int& i) const
    {
       return operator[](SliceList<1>({i}));
    }


    inline Tensor transpose()
    {
        Tensor a(shape, data, *device, storage_pointer);
        auto shapea = shape.clone();
        auto stridesa = strides.clone();
        a.shape[-1] = shapea[-2];
        a.shape[-2] = shapea[-1];
        a.strides[-1] = stridesa[-2];
        a.strides[-2] = stridesa[-1];
        
        return a;
    }

    // Swap any two dimensions (a view; negative dims count from the end).
    inline Tensor transpose(int a, int b) const
    {
        int ndim = shape.ndim();
        if (a < 0) a += ndim;
        if (b < 0) b += ndim;
        if (indexer && ((a < ndim - indexer_tail) != (b < ndim - indexer_tail)))
            throw std::runtime_error("transpose: cannot swap an indexed dim with a row dim of a tensor_index view");
        Tensor t = *this;
        t.shape[a] = shape[b];
        t.shape[b] = shape[a];
        t.strides[a] = strides[b];
        t.strides[b] = strides[a];
        return t;
    }

    inline Tensor contiguous() const
    {
        Tensor a = Tensor{shape, *device};
        a = *this;
        return a;
    }

    inline Tensor<R,(rank == -1 ? -1 : rank+1)> unsqueeze(const int& dim) const
    {
        Shape<(rank == -1 ? -1 : rank+1)> newshape;
        Shape<(rank == -1 ? -1 : rank+1)> newstrides;
        int ndim = shape.ndim();
        for (int i = 0; i < ndim + 1; i++)
        {
            if (i < dim)
            {
                newshape[i] = shape[i];
                newstrides[i] = strides[i];
            }
            else if (i == dim)
            {
                newshape[i] = 1;
                newstrides[i] = 0;
            }
            else
            {
                newshape[i] = shape[i - 1];
                newstrides[i] = strides[i - 1];
            }
        }
        Tensor<R,(rank == -1 ? -1 : rank+1)> b{newshape, data, *device, storage_pointer,};
        b.strides = newstrides;
        b.copy_indexer(*this);
        if (indexer && dim > ndim - indexer_tail) b.indexer_tail++;
        return b;
    }



    template <int v = rank>
    Tensor<R, v> broadcast(const Shape<v>& a) const
    {

        // if shapes are equal, return self
        
        
    


        Tensor<R, v> b{a, data, *device, storage_pointer};

        b.copy_indexer(*this);

        b._broadcast(a);
        return b;
    }

    void _broadcast(const Shape<rank>& a)
    {
        for (size_t i = 1; i < a.ndim() + 1; i++)
        {
            if (shape.ndim() < i || shape[-i] == 1)
            {
                strides[-i] = 0;
            }
            else if (a[-i] != shape[-i])
            {
                std::cerr << "Incompatible shapes for broadcast" << std::endl;
                std::cerr << "Shape: " << shape << " Broadcast shape: " << a << std::endl;
                throw std::runtime_error("Incompatible shapes for broadcast");
            }
        }
        shape = a;
    };


    // Rows of this tensor picked by an index tensor (64-bit ints), as a view:
    //   weight [V, D].tensor_index(ids [T]) → [T, D],  row t = weight[ids[t]]
    //   ids [B, T] → [B, T, D]
    // Nothing is copied: any operation reading the view gathers the rows
    // (e.g. `Tensor<float, 2> x(...); x = weight.tensor_index(ids);`).  The
    // view keeps a pointer to `index_tensor`'s data, so keep it alive (and on
    // the same device) while the view is used.
    template <typename I, int v = rank>
    Tensor<R, ((v < 0 || rank < 0) ? -1 : v + rank - 1)> tensor_index(const Tensor<I, v>& index_tensor) const
    {
        static_assert(std::is_integral_v<I> && sizeof(I) == sizeof(unsigned long),
                      "tensor_index: index tensor must hold 64-bit integers (unsigned long / long)");
        constexpr int newrank = (v < 0 || rank < 0) ? -1 : v + rank - 1;
        int ni = index_tensor.shape.ndim(), nd = shape.ndim();
        Shape<newrank> newshape;
        Shape<newrank> newstrides;
        for (int i = 0; i < ni; i++) {
            newshape[i] = index_tensor.shape[i];
            newstrides[i] = index_tensor.strides[i];
        }
        for (int i = 1; i < nd; i++) {
            newshape[ni + i - 1] = shape[i];
            newstrides[ni + i - 1] = strides[i];
        }
        Tensor<R, newrank> b{newshape, data, *device, storage_pointer};
        b.strides = newstrides;
        b.indexer = (unsigned long*)index_tensor.data.data;
        b.indexer_stride = strides[0];
        b.indexer_tail = nd - 1;
        return b;
    }

    template <typename T = R>
    inline Tensor<T, rank> view()
    {
        Shape<rank> newshape = shape;
        float scale =  float(bitsize) / sizeof(T);
        float newlastdim = shape[-1] * scale;
        if (newlastdim != (int)newlastdim){
            std::cout << "Last dimension is not divisible by " << sizeof(T) << std::endl;
            std::cout << "Last dimension: " << shape[-1] << " Bitsize: " << bitsize << " New last dimension: " << newlastdim << std::endl;
            std::cout << *this << std::endl;
            throw( std::runtime_error("Last dimension is not divisible by sizeof(T)"));
        }
        newshape[-1] = newlastdim;
        Tensor<T, rank> b = Tensor<T, rank>(newshape, data.template reinterpret<T>(), *device, storage_pointer);
        return b;   
    }

    template <typename T = R, int Z = -1>
    inline Tensor<T, Z> view(Shape<Z> newshape)
    {
        bool has_neg = false;
        size_t known_elements = 1;
        for(int i = 0; i < newshape.ndim(); i++){
            if(newshape[i] == -1){
                if (has_neg){
                    std::cerr << "Only one dimension can be -1" << std::endl;
                    throw std::runtime_error("Only one dimension can be -1");
                }
                has_neg = true;
                continue;
            }
            known_elements *= static_cast<size_t>(newshape[i]);
        }

        if (has_neg) {
            const size_t element_bytes = sizeof(T);
            const size_t new_total_elements = total_bytes / element_bytes;
            if (known_elements == 0 || (new_total_elements % known_elements) != 0) {
                std::cerr << "Cannot infer -1 dimension for view" << std::endl;
                throw std::runtime_error("Cannot infer -1 dimension for view");
            }
            const long missing = static_cast<long>(new_total_elements / known_elements);
            for (int i = 0; i < newshape.ndim(); i++) {
                if (newshape[i] == -1) {
                    newshape[i] = missing;
                    break;
                }
            }
        }


        if (total_bytes != newshape.total_size() * sizeof(T))
        {
            std::cerr << "Incompatible shapes for view" << std::endl;
            std::cerr << "Shape: " << shape << " New shape: " << newshape << std::endl;
            std::cerr << "Data size: " << total_bytes << " New total bytes: " << newshape.total_size() * sizeof(T) << std::endl;
            if (typeid(T) != typeid(R))
            {
                std::cerr << "Data size: " << sizeof(T) << " Tensor size: " << bitsize << std::endl;
                std::cout << "Total bytes: " << total_bytes << " New total bytes: " << newshape.total_size() * sizeof(T) << std::endl;
            }
            throw std::runtime_error("Incompatible shapes for view");
        }


        return Tensor<T, Z>{newshape, 
            data.template reinterpret<T>()
            , *device, storage_pointer};   
    }

    inline R& flatget(size_t i)
    {
        if(indexer == nullptr){
            R *ptr = (R *)data;
            for (int j = 1; j < shape.ndim()+1; j++)
            {
                int cstride = strides[-j];
                int cshape = shape[-j];
                int index = ((i%cshape) * cstride);

                i = i / cshape;
                ptr += index;       
            }
            return *ptr;
        }
        else{
            unsigned long *ptr = indexer;
            R *row = (R *)data;
            int ndim = shape.ndim();
            for (int j = 1; j < ndim+1; j++)
            {
                long cstride = strides[-j];
                long cshape = shape[-j];
                long index = ((i%cshape) * cstride);

                i = i / cshape;
                if (ndim - j < ndim - indexer_tail) ptr += index;
                else row += index;
            }
            return *(row + (*ptr) * indexer_stride);
        }
        
    }

    template <typename M>
    inline Tensor<M, rank> astype()
    {
        Tensor<M, rank> a = {shape, device};
        for (int i = 0; i < total_size; i++)
        {
            a.flatget(i) = (M)flatget(i);
        }
        return a;
    }

    std::ostream& print(std::ostream& os, Tensor<R, rank>* original_tensor = nullptr)
    {
        // auto tensorc = Tensor<R, rank>(tensorin.shape, tensorin.device_type);
        // tensorc = tensorin;
       
        if(!device->supports_compute_device[kCPU])
        {
            device->synchronize_function();
            auto tensor = to(MemoryType::kDDR, ComputeType::kCPU);
            device->synchronize_function();
            return tensor.print(os, this);
        }

        if (original_tensor == nullptr){
            original_tensor = this;
        }
        
        


        os << "(";
        os << "dtype="<< get_type_string<R>() << ", ";
        os << "shape=" << original_tensor->shape << ", ";
        os << "strides=" << original_tensor->strides << ", ";
        os << "device_type=" << original_tensor->device->this_device_type << "{"<<int(original_tensor->device->this_device_type)<<"}, ";
        os << "device_id=" << original_tensor->device->device_id << "";
        if(original_tensor->indexer != nullptr){
            os << ", indexed";
        }
        os << ")" << "[";
        if(total_size <= 4){
            for (int i = 0; i < total_size; i++)
            {
                os << flatget(i);
                if (i != total_size - 1)
                {
                    os << ", ";
                }
            }
        }
        else
        {
            for (int i = 0; i < 2; i++)
            {
                os << flatget(i);
                if (i != 2)
                {
                    os << ", ";
                }
            }
            
            os << "..., ";

            for (int i = total_size - 2; i < total_size; i++)
            {
                os << flatget(i);
                if (i != total_size - 1)
                {
                    os << ", ";
                }
            }
        }
        os << "]";
       
        
        
        return os;
    }

    // print tensor
    friend std::ostream &operator<<(std::ostream &os, Tensor<R, rank> tensorin)
    {
        
        return tensorin.print(os);
        
    }

    // operator = for Tensor void
    void operator=(const Tensor<void, rank>& other)
    {
        assert(other.dtype == get_dtype<R>());
        assert(other.shape == shape);
        this->device = other.device;
        this->shape = other.shape;
        this->strides = other.strides;
        this->bitsize = other.bitsize;
        this->data = other.data;
        this->indexer = other.indexer;
        this->total_size = other.total_size;
        this->total_bytes = other.total_bytes;
        this->storage_pointer = other.storage_pointer;

        this->device->register_allocation(this->storage_pointer);
    }

    Tensor<R,rank> to(MemoryLocation device_type, ComputeType compute_type = ComputeType::kUnknown) const{
        
        if(this->device->this_device_type == device_type.memory_type && this->device->device_id == device_type.device_id && (this->device->default_compute_type == compute_type || compute_type == kUnknown)){
            return *this;
        }
        

        BaseMemoryAllocation* result;

        AllocationMap& target_device = global_device_manager.get_device(device_type.memory_type, device_type.device_id);

        if(indexer != nullptr || strides != shape.calc_strides()){
            // For Vulkan compute types, we can't use to_compute() (no kernel ops).
            // Instead, make a contiguous copy on CPU first, then convert.
            if (compute_type == ComputeType::kVULKAN || compute_type == ComputeType::kVULKANTEXTURE) {
                Tensor<R,rank> contiguous = this->to(MemoryLocation(MemoryType::kDDR, 0), ComputeType::kCPU);
                return contiguous.to(device_type, compute_type);
            }
            std::cout << "Shape: " << shape << " Strides: " << strides << " Calculated strides: " << shape.calc_strides() << std::endl;
            Tensor output = {shape, device_type, compute_type == ComputeType::kUnknown ? target_device.default_allocator_type : compute_type};
            output = this->to_compute(compute_type);
            return output;
        }
        else{
            result = device->convert_memory_type((void*)this->data.data, AllocationMetadata::create<R>(shape,device_type.memory_type, compute_type == ComputeType::kUnknown ? target_device.default_allocator_type : compute_type, 0, AllocationFlags::kRW, device_type.device_id));
        
            return {
                shape,
                device_type.allocation_map->get_massaged_pointer<R>(
                    result,
                    AllocationMetadata::create<R>(
                        shape,
                        device_type.memory_type,
                        device_type.allocation_map->default_compute_type,
                        0,
                        AllocationFlags::kRW,
                        device_type.device_id
                    )
                ),
                device_type,
                result
            };
        };
    };

    // In-place view conversion: returns a tensor over the *same* allocation,
    // seen through `compute_type`.  `flags` selects the kind of view for
    // backends that have several (e.g. Vulkan: kTEXTURE, kSURFACE,
    // kTEXELBUFFER, kSTORAGE).  Views are created once and cached on the
    // allocation; they are released together with it.
    Tensor<R,rank> to_compute(ComputeType compute_type, AllocationFlags flags = AllocationFlags::kRW) const{
        
        if(!this->device->supports_compute_device[compute_type]){
            std::cerr << "Compute type " << compute_type << " not supported on device type " << this->device->this_device_type << std::endl;
            throw std::runtime_error("Compute type not supported on device type");
        }

        size_t offset = 0;
        if (
            this->storage_pointer->metadata.compute_device == this->data.metadata.compute_device 
        ){
            offset = this->data.data - (R*)this->storage_pointer->data;
        }else{
            offset = this->data.data - (R*)this->storage_pointer->cached_massaged_pointers[this->data.metadata.hash()];
        }

        int compute_device_id = this->data.metadata.device_id;
        if (compute_type == ComputeType::kCPU || compute_type == ComputeType::kFILE || compute_type == ComputeType::kUnknown) {
            compute_device_id = 0;
        } else if (this->data.metadata.compute_device != compute_type) {
            compute_device_id = this->device->device_id;
        }

        auto result = this->device->template get_massaged_pointer<R>(storage_pointer, AllocationMetadata::create<R>(shape,device->this_device_type,compute_type, 0, flags, compute_device_id));
        return Tensor<R,rank>{
            shape,
            result + offset,
            *device,
            storage_pointer
        };
    };

    // destructor
    ~Tensor()
    {
        if(data.data != NULL){
            device->deallocate(storage_pointer);
        }
    }
    // template <int output>//, typename std::enable_if<(rank == -1)>::type* = nullptr>
    // operator Tensor<R,output>(){
    //     assert(this->shape.ndim() == output);// "Output not correct ndims"
    //     return *this;
    // }

    // define copy constructor so that data pointer is copied but not the own_data flag
    Tensor(const Tensor<R, rank> &other)
    {
        this->device = other.device;
        this->shape = other.shape;
        this->strides = other.strides;
        this->bitsize = other.bitsize;
        this->data = other.data;
        copy_indexer(other);
        this->total_size = other.total_size;
        this->total_bytes = other.total_bytes;
        this->storage_pointer = other.storage_pointer;
        if(this->storage_pointer != nullptr){
            this->device->register_allocation(this->storage_pointer);
        }
    }

    // Between a dynamic-rank tensor (-1) and a fixed rank, e.g. the result of
    // an operation into `Tensor<float, 2>`.  Shares the data; the number of
    // dimensions is checked at runtime.
    template <int orank, typename = std::enable_if_t<orank != rank && (orank == -1 || rank == -1)>>
    Tensor(const Tensor<R, orank>& other)
    {
        this->device = other.device;
        this->shape = Shape<rank>(other.shape);
        this->strides = Shape<rank>(other.strides);
        this->bitsize = other.bitsize;
        this->data = other.data;
        copy_indexer(other);
        this->total_size = other.total_size;
        this->total_bytes = other.total_bytes;
        this->storage_pointer = other.storage_pointer;
        if(this->storage_pointer != nullptr){
            this->device->register_allocation(this->storage_pointer);
        }
    }
};


template <int rank>
class Tensor<void, rank> {

    public:
    Shape<rank> shape;
    Shape<rank> strides;
    void *data = NULL;
    BaseMemoryAllocation* storage_pointer = NULL;
    AllocationMap* device = &global_device_manager.get_device(MemoryType::kDDR,0);
    unsigned long bitsize;
    DataType dtype;

    // Tensor<void, rank> operator[](Slice<true> i) = delete;
    Tensor<void, rank> operator[](int i) = delete;
    Tensor<void, rank> view() = delete;
    Tensor<void, rank> view(Shape<rank> newshape) = delete;
    Tensor(Shape<rank> __a, MemoryType device_type = MemoryType::kDDR) = delete;
    Tensor(Shape<rank> __a, void *datain, MemoryType device_type = MemoryType::kDDR) = delete;
    friend std::ostream &operator<<(std::ostream &os, Tensor<void, rank> tensor) = delete;
    template <typename T, int orank = rank>
    Tensor(const Tensor<T, orank>& other){
        this->device = other.device;
        this->shape = other.shape;
        this->strides = other.strides;
        this->bitsize = other.bitsize;
        this->data = other.data.data;
        this->dtype = get_dtype<T>();
        this->storage_pointer = other.storage_pointer;
        device->register_allocation(this->storage_pointer);
    }

    Tensor(){

    };
    // copy constructor
    Tensor(const Tensor<void, rank> &other)
    {
        this->device = other.device;
        this->shape = other.shape;
        this->strides = other.strides;
        this->bitsize = other.bitsize;
        this->data = other.data;
        this->dtype = other.dtype;
        this->storage_pointer = other.storage_pointer;
        device->register_allocation(this->storage_pointer);
    }

    // copy assignment — shares the allocation and keeps the reference count
    // balanced (the implicit one copied the pointer without registering it,
    // so `map[name] = tensor` freed the allocation one reference too early)
    Tensor<void, rank>& operator=(const Tensor<void, rank>& other)
    {
        if (this == &other) return *this;
        if (other.storage_pointer != nullptr) other.device->register_allocation(other.storage_pointer);
        if (this->data != NULL && this->storage_pointer != nullptr) device->deallocate(this->storage_pointer);
        this->device = other.device;
        this->shape = other.shape;
        this->strides = other.strides;
        this->bitsize = other.bitsize;
        this->data = other.data;
        this->dtype = other.dtype;
        this->storage_pointer = other.storage_pointer;
        return *this;
    }

    template <typename T>
    operator T(){
        if(get_dtype<T>() != dtype){
            std::cerr << "Data type mismatch, tensor data type is " << dtype << " but requested type is " << get_dtype<T>() << std::endl;
            throw std::runtime_error("Data type mismatch");
        }
        return *((T *)data);
    }

    template <typename T>
    operator Tensor<T, rank>(){
        if(get_dtype<T>() != dtype){
            std::cerr << "Data type mismatch, tensor data type is " << dtype << " but requested type is " << get_dtype<T>() << std::endl;
            throw std::runtime_error("Data type mismatch");
        }
        return {shape, (T*)data, *device, (T*)storage_pointer};
    }

    template <typename T>
    operator Tensor<T, rank>() const{
        if(get_dtype<T>() != dtype){
            std::cerr << "Data type mismatch, tensor data type is " << dtype << " but requested type is " << get_dtype<T>() << std::endl;
            throw std::runtime_error("Data type mismatch");
        }
        return {shape, (T*)data, *device, storage_pointer};
    }

    // destructor
    ~Tensor()
    {
        if(data != NULL){
            device->deallocate(storage_pointer);
        }
    }
    
};

#endif