
#include "enums/device.hpp"
#include "shape.hpp"



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
    void Allocate(){
        for(int i = 0; i < sizeof...(returnTypes); i++){
            if(pointers[i] == nullptr){
                pointers[i] = DeviceAllocator<dtype>::allocate(sizes[i]);
            }
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

    
};


