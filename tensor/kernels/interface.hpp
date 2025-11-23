
#include "enums/device.hpp"
#include "shape.hpp"


#ifndef KERNELS_INTERFACE_HPP
#define KERNELS_INTERFACE_HPP
// Order determines fallback priority


template <DeviceType dtype, typename... returnTypes>
class KernelBufferAllocator {
    void * pointers[sizeof...(returnTypes)] = {nullptr};
    Shape<sizeof...(returnTypes)> sizes;
    public:

    // input is the sizes of each each buffer to allocate, ie, type size * number of elements
    KernelBufferAllocator(Shape<sizeof...(returnTypes)> bufferSizes):sizes(bufferSizes){
      
    }

    // Will not allocate if buffer allready allocated

    template <int i = sizeof...(returnTypes)>
    void Allocate(){
        if constexpr (i > 0){
            if(pointers[i-1] == nullptr){
                size_t size = sizes[i-1];
                // make sure is unallocated
                if(pointers[i-1] == nullptr){
                   pointers[i-1] = DeviceAllocator<dtype>::allocate(size * sizeof(typename std::tuple_element<i-1, std::tuple<returnTypes...>>::type));
                }
            }
            Allocate<i-1>();
        }
    }

    // can also preallocate by doing getPointer<x>() = ...
    template <int i>
    typename std::tuple_element<i, std::tuple<returnTypes...>>::type*& getPointer(){
        void*& ptr = pointers[i];
        return reinterpret_cast<typename std::tuple_element<i, std::tuple<returnTypes...>>::type*&>(ptr);
    }

};


template <DeviceType device_type, typename... Args>
class Kernel {
public:

    Kernel() = default;
   
    virtual void call(Args...);
    


    public:
    void operator()(Args... args) {
        
        call(args...); // For GPUs etc.
    }

    template<typename... ResponseArgs>
    KernelBufferAllocator<device_type, ResponseArgs...> CreateBuffers(Shape<sizeof...(ResponseArgs)> sizes){
        return KernelBufferAllocator<device_type, ResponseArgs...>(sizes);
    }

    
};



#endif //KERNELS_INTERFACE_HPP