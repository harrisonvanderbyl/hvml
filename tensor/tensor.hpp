
// include malloc
#include "stdlib.h"
#include "enums/dtype.hpp"
#include "vector"
#include "enums/device.hpp"
#include <stdarg.h>
#include <string>
#include "shape.hpp"
#include <iostream>
#include <string.h>
#ifndef TENSOR
#define TENSOR


class Slice
{
    public:
    int start;
    int end;
    int step;
    bool is_slice;
    bool is_empty;

    
    // Constructor: full parameters
    Slice(int starti, int endi, int stepi = 1)
        : start(starti), end(endi), step(stepi), is_slice(true), is_empty(false) {};

    // Constructor: reduced slice
    Slice(int starti)
        : start(starti), end(-1), step(1), is_slice(false), is_empty(false) {};

    // Constructor: empty slice
    Slice()
        : start(0), end(-1), step(1) , is_slice(true), is_empty(true) {};

    // Slice(const Slice &other)
    //     : start(other.start), end(other.end), step(other.step), is_slice(other.is_slice), is_empty(other.is_empty) {};
};


template <int myreducedims = -1>
class SliceList {
public:
    static constexpr int reducedims = myreducedims;

    Slice A;
    Slice B;
    Slice C;
    Slice D;
    Slice E;
    Slice F;
    Slice G;
    // make sure amount of ints is equal to reducedims
    template <typename a = Slice, typename b = Slice, typename c = Slice, typename d = Slice, typename e = Slice, typename f = Slice, typename g = Slice, typename = std::enable_if_t<std::is_same_v<a,int> + std::is_same_v<b,int> + std::is_same_v<c,int> + std::is_same_v<d,int> + std::is_same_v<e,int> + std::is_same_v<f,int> + std::is_same_v<g,int> == reducedims>>
    SliceList(a aa=Slice(), b ba=Slice(), c ca=Slice(), d da=Slice(), e ea=Slice(), f fa=Slice(), g ga=Slice())
        : A(aa), B(ba), C(ca), D(da), E(ea), F(fa), G(ga) {};



    
    Slice& operator[](int i)
    {
        if (i == 0)
        {
            return A;
        }
        else if (i == 1)
        {
            return B;
        }
        else if (i == 2)
        {
            return C;
        }
        else if (i == 3)
        {
            return D;
        }
        else if (i == 4)
        {
            return E;
        }
        else if (i == 5)
        {
            return F;
        }
        else
        {
            return G;
        }
    }

};


template <typename R = float, int rank = -1>
class Tensor
{
public:
    Shape<rank> shape;
    Shape<rank> strides;
    R *data = NULL;
    unsigned long bitsize;

    size_t total_size = 0;
    size_t total_bytes = 0;
    DeviceType device_type = DeviceType::kCPU;

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

    Tensor(Shape<rank> __a, DeviceType device_type = DeviceType::kCPU)
    {
        this->bitsize = sizeof(R);
        this->device_type = device_type;
        // printf("ndim: %d\n", __a.ndim());
        this->shape = __a;
        this->strides = __a.clone();
        calculate_metadata();
        if(device_type == DeviceType::kCPU){
            data = (R*)DeviceAllocator<DeviceType::kCPU>::allocate(total_bytes);
        }else if(device_type == DeviceType::kCUDA){
            data = (R*)DeviceAllocator<DeviceType::kCUDA>::allocate(total_bytes);
        }
        else if(device_type == DeviceType::kHIP){
            data = (R*)DeviceAllocator<DeviceType::kHIP>::allocate(total_bytes);
        }
    }
    
    
    Tensor(Shape<rank> __a, R *datain, DeviceType device_type = DeviceType::kCPU)
    {
        this->device_type = device_type;
        this->bitsize = sizeof(R);
        this->shape = __a;
        this->strides = __a.clone();
        calculate_metadata();
        this->data = datain;
    }

    // Tensor(R a)
    // {
    //     this->device_type = DeviceType::kCPU;
    //     this->shape = Shape(1);
    //     this->strides = Shape(1);
    //     calculate_metadata();
    //     data = malloc(total_bytes);
    //     *((R *)data) = a;
    // }

    template <typename M>
    inline int operator=(M a)
    {
        for (int i = 0; i < total_size; i++)
        {
           flatget(i) = a;
        }
            
        return 0;
    }

    template <typename M>
    inline int operator=(Tensor<M>& other)
    {
        if (data == NULL)
        {
            data = other.data;
            shape = other.shape;
            strides = other.strides;
            bitsize = other.bitsize;
            device_type = other.device_type;
            total_size = other.total_size;
            total_bytes = other.total_bytes;

            return 0;
        }
        
        if (typeid(R) != typeid(M))
        {
            std::cerr << "Data types do not match" << std::endl;
            throw std::runtime_error("Data types do not match");
        }

        if(
            shape.total_size() == ((Tensor*)(&other))->shape.total_size() && strides.total_size() == ((Tensor*)(&other))->strides.total_size()
        ){
            std::cout << "Copying data from other tensor, both are same shape and strides" << std::endl;
            memcpy(data, other.data, total_bytes);
            return 0;
        }

        if(
            other.shape.ndim() == 0 || (other.shape[0] == 1 && other.shape.ndim() == 1)
        ){
            auto bytes = sizeof(R);
            std::cout << "Bytes: " << bytes * total_size << std::endl;
            std::cout << "Total bits: " << total_size << std::endl;
            std::fill_n((uint8_t *)data, total_size, *((uint8_t *)other.data));
            
                
            return 0;
        }

        if (shape.ndim() * other.shape.ndim() == 0)
        {
            std::cout << "One of the tensors has zero dimensions" << std::endl;
            auto bytes = bitsize;
            memcpy(data, other.data, bytes);
            return 0;

        }
    
        auto bcast = ((Tensor*)(&other))->broadcast(shape);

        std::cout << "Broadcasting tensor of shape: " << bcast.shape << " to shape: " << shape << std::endl;
        for (int i = 0; i < shape[0]; i++)
        {
            auto bbx = bcast[i];
            auto aax = this->gather(i);
            aax = bbx;
        }
        
        return 0;
    }

    template <typename X = SliceList<-1>, int newrank = X::reducedims<0?-1:std::max(rank-X::reducedims,-1)>
    // result of operator[] is a tensor of rank - T::reducedims if rank-T::reducedims > 0 else it is a scalar of type R
    inline std::conditional_t<newrank == 0, R&, Tensor<R, newrank>>
    gather(X inp)
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
        R* startingpointer = (R *)data;
        int ndim = shape.ndim();
        int ii = 1;
        for (; ii <= ndim ; ii++)
        {
            int shapeofset = shape[-ii];
            int start = (inp[ndim-ii].start + shapeofset) % shapeofset;
            startingpointer += start * strides[-ii];
        }
        
        if constexpr (newrank == 0)
        {
            return *((R *)startingpointer);
        }
        else{
        auto newshape = Shape<newrank>();
        
        auto newstrides = Shape<newrank>();
        int i = 0;
        int j= 0;
        int multiplier = 1;
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
                    newshape[j] = ((inp[i].end - inp[i].start + shape[i])%shape[i]) / inp[i].step;
                    newstrides[j] = this->strides[i] * inp[i].step;
                    // std::cout << "not implemented" << std::endl;
                }
                j++;
            }
            else
            {
                // if(i > 0){

                // }
                
            }
        }
        
        Tensor<R, newrank> b = {newshape, startingpointer, device_type};
        b.strides = newstrides;
        
        // // std::cout << (i.end - i.start) / i.step << std::endl;
        // // std::cout << "Start: " << i.start << " End: " << i.end << " Step: " << i.step << std::endl;
        
        // // std::cout << "Shape: " << b.shape << std::endl;
        // b.data = (void *)((uint8_t *)b.data + i.start * strides[0] * bitsize);
        
        

        return b;
        }
    }

    std::conditional_t<rank == 0, R&, Tensor<R, std::max(rank - 0,-1)>>
    operator[](SliceList<0> i)
    {
        return gather(i);
    }

    std::conditional_t<rank == 1, R&, Tensor<R, std::max(rank - 1,-1)>>
    operator[](SliceList<1> i)
    {
        return gather(i);
    }

    std::conditional_t<rank == 2, R&, Tensor<R, std::max(rank - 2,-1)>>
    operator[](SliceList<2> i)
    {
        return gather(i);
    }
    std::conditional_t<rank == 3, R&, Tensor<R, std::max(rank - 3,-1)>>
    operator[](SliceList<3> i)
    {
        return gather(i);
    }

    std::conditional_t<rank == 4, R&, Tensor<R, std::max(rank - 4,-1)>>
    operator[](SliceList<4> i)
    {
        return gather(i);
    }

    std::conditional_t<rank == 1, R&, Tensor<R, std::max(rank - 1,-1)>>
    operator[](int i)
    {
       return operator[](SliceList<1>({i}));
    }

    // auto operator[](int i)
    // {
    //     return operator[](SliceList({i}));
    // }

    // template <typename T = R>
    // typename std::conditional<(rank == 1), T&, Tensor<T, std::max(rank - 1,-1)>>::type
    // operator[](int i)
    // {

    //     int v = (i+shape[0]) % shape[0];
        
    //     void *ptr = (void *)((uint8_t *)data + i * strides[0] * bitsize);
        
    //     if constexpr (rank == 1)
    //     {
    //         return *((T *)ptr);
    //     }
    //     else
    //     {
    //         auto s = shape.slice();
    //         auto a = Tensor<T, std::max(rank - 1,-1)>(s, ptr, device_type);
    //         return a;
    //     }
        
    // }

    inline Tensor transpose()
    {
        Tensor a;
        a.shape = Shape(shape[1], shape[0]);
        a.strides = Shape(strides[1], strides[0]);
        a.bitsize = bitsize;
        a.device_type = device_type;
        a.total_size = total_size;
        a.total_bytes = total_bytes;
        a.data = data;
        
        return a;
    }

    inline Tensor contiguous()
    {
        Tensor a = {shape, device_type};
        a = *this;
        return a;
    }



    template <int v = rank>
    Tensor<R, v> broadcast(Shape<v>& a)
    {
        Tensor<R, v> b{a, data, device_type};
        for (size_t i = 1; i < a.ndim()+1; i++)
        {
            if(shape.ndim() > i && a[-i%a.ndim()] != shape[-i%shape.ndim()] && shape[-i%shape.ndim()] != 1){
                std::cerr << "Incompatible shapes for broadcast" << std::endl;
                std::cerr << i << "\n";
                std::cerr << "Shape: " << shape << " Broadcast shape: " << a << std::endl;
                std::cerr << "Shape: " << shape[-i%shape.ndim()] << " Broadcast shape: " << a[-i%a.ndim()] << std::endl;
                throw std::runtime_error("Incompatible shapes for broadcast");
            }
            if (shape.ndim() < i || shape[-i] == 1)
            {
                b.strides[-i] = 0;
                if(i < shape.ndim()){
                    for (size_t j = i+1; j < a.ndim()+1; j++)
                    {
                        b.strides[-j] = b.strides[-j]/ shape[-j];
                    }
                }
            }
        }

        return b;
    }

    template <typename T = R>
    inline Tensor<T, rank> view()
    {
        Shape<rank> newshape = shape.clone();
        float scale =  float(bitsize) / sizeof(T);
        float newlastdim = shape[-1] * scale;
        if (newlastdim != (int)newlastdim){
            std::cout << "Last dimension is not divisible by " << sizeof(T) << std::endl;
            std::cout << "Last dimension: " << shape[-1] << " Bitsize: " << bitsize << " New last dimension: " << newlastdim << std::endl;
            std::cout << *this << std::endl;
            throw( std::runtime_error("Last dimension is not divisible by sizeof(T)"));
        }
        newshape[-1] = newlastdim;
        Tensor<T, rank> b = Tensor<T, rank>(newshape, (T*)data, device_type);
        return b;   
    }

    template <typename T = R, int Z = -1>
    inline Tensor<T, Z> view(Shape<Z> newshape)
    {
        bool has_neg = false;
        for(int i = 0; i < newshape.ndim(); i++){
            if(newshape[i] == -1){
                if (has_neg){
                    std::cerr << "Only one dimension can be -1" << std::endl;
                    throw std::runtime_error("Only one dimension can be -1");
                }
                
                int total = -newshape.total_size();
                int missing = (total_size*bitsize) / (total * sizeof(T));
                newshape[i] = missing;
                has_neg = true;
            }
        }


        if (total_bytes != newshape.total_size() * sizeof(T))
        {
            std::cerr << "Incompatible shapes for view" << std::endl;
            std::cerr << "Shape: " << shape << " New shape: " << newshape << std::endl;
            if (typeid(T) != typeid(R))
            {
                std::cerr << "Data size: " << sizeof(T) << " Tensor size: " << bitsize << std::endl;
                std::cout << "Total bytes: " << total_bytes << " New total bytes: " << newshape.total_size() * sizeof(T) << std::endl;
            }
            throw std::runtime_error("Incompatible shapes for view");
        }


        return Tensor<T, Z>{newshape, (T*)data, device_type};   
    }

    inline R& flatget(size_t i)
    {
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

    // print tensor
    friend std::ostream &operator<<(std::ostream &os, Tensor<R, rank> tensor)
    {
        os << "(";
        os << "dtype="<< get_type_string<R>() << ", ";
        os << "shape=" << tensor.shape << ", ";
        os << "strides=" << tensor.strides << ", ";
        os << "device_type=" << tensor.device_type << "";
        os << ")" << "[";
        if(tensor.total_size <= 4){
            for (int i = 0; i < tensor.total_size; i++)
            {
                os << tensor.flatget(i);
                if (i != tensor.total_size - 1)
                {
                    os << ", ";
                }
            }
        }
        else
        {
            for (int i = 0; i < 2; i++)
            {
                os << tensor.flatget(i);
                if (i != 2)
                {
                    os << ", ";
                }
            }
            
            os << "..., ";

            for (int i = tensor.total_size - 2; i < tensor.total_size; i++)
            {
                os << tensor.flatget(i);
                if (i != tensor.total_size - 1)
                {
                    os << ", ";
                }
            }
        }
        os << "]";
       
        
        
        return os;
    }

    // operator = for Tensor void
    void operator=(const Tensor<void, rank>& other)
    {
        assert(other.dtype == get_dtype<R>());
        assert(other.shape == shape);
        this->device_type = other.device_type;
        this->shape = other.shape;
        this->strides = other.strides;
        this->bitsize = other.bitsize;
        this->data = other.data;
    }

    Tensor<R,rank> to(DeviceType device_type){
        if(this->device_type == device_type){
            return *this;
        }
        Tensor a = {shape, device_type};

        if(strides != a.strides){
            std::cerr << "Cannot convert non-contiguous tensor yet" << std::endl;
            throw std::runtime_error("Cannot convert non-contiguous tensor yet");
        }

        if(device_type == DeviceType::kCPU && this->device_type == DeviceType::kCUDA){
            #if defined(__CUDACC__) 
            cudaMemcpy(a.data, this->data, a.total_bytes, cudaMemcpyDeviceToHost);
            #else
            std::cerr << "CUDA not enabled" << std::endl;
            throw std::runtime_error("CUDA not enabled");
            #endif
        }else if(device_type == DeviceType::kCUDA && this->device_type == DeviceType::kCPU){
            #if defined(__CUDACC__) 
            cudaMemcpy(a.data, this->data, a.total_bytes, cudaMemcpyHostToDevice);
            #else
            std::cerr << "CUDA not enabled" << std::endl;
            throw std::runtime_error("CUDA not enabled");
            #endif
        }else if(device_type == DeviceType::kCPU && this->device_type == DeviceType::kHIP){
            #if defined(__HIPCC__)
            hipMemcpy(a.data, this->data, a.total_bytes, hipMemcpyDeviceToHost);
            #else
            std::cerr << "HIP not enabled" << std::endl;
            throw std::runtime_error("HIP not enabled");
            #endif
        }else if(device_type == DeviceType::kHIP && this->device_type == DeviceType::kCPU){
            #if defined(__HIPCC__)
            hipMemcpy(a.data, this->data, a.total_bytes, hipMemcpyHostToDevice
            );
            #else
            std::cerr << "HIP not enabled" << std::endl;
            throw std::runtime_error("HIP not enabled");
            #endif
        }else{
            std::cerr << "Unsupported device type conversion" << std::endl;
            throw std::runtime_error("Unsupported device type conversion");
        }
        return a;
    };

    
    // template <int output>//, typename std::enable_if<(rank == -1)>::type* = nullptr>
    // operator Tensor<R,output>(){
    //     assert(this->shape.ndim() == output);// "Output not correct ndims"
    //     return *this;
    // }
};


template <int rank>
class Tensor<void, rank> {

    public:
    Shape<rank> shape;
    Shape<rank> strides;
    void *data = NULL;
    unsigned long bitsize;
    DataType dtype;
    DeviceType device_type = DeviceType::kCPU;

    // Tensor<void, rank> operator[](Slice<true> i) = delete;
    Tensor<void, rank> operator[](int i) = delete;
    Tensor<void, rank> view() = delete;
    Tensor<void, rank> view(Shape<rank> newshape) = delete;
    Tensor(Shape<rank> __a, DeviceType device_type = DeviceType::kCPU) = delete;
    Tensor(Shape<rank> __a, void *datain, DeviceType device_type = DeviceType::kCPU) = delete;
    friend std::ostream &operator<<(std::ostream &os, Tensor<void, rank> tensor) = delete;
    template <typename T>
    Tensor(const Tensor<T, rank>& other){
        this->device_type = other.device_type;
        this->shape = other.shape;
        this->strides = other.strides;
        this->bitsize = other.bitsize;
        this->data = other.data;
        this->dtype = get_dtype<T>();
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
        return {shape, (T*)data, device_type};
    }

    template <typename T>
    operator Tensor<T, rank>() const{
        if(get_dtype<T>() != dtype){
            std::cerr << "Data type mismatch, tensor data type is " << dtype << " but requested type is " << get_dtype<T>() << std::endl;
            throw std::runtime_error("Data type mismatch");
        }
        return {shape, (T*)data, device_type};
    }

    
};

#endif