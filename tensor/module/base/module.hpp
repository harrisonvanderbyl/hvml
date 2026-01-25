#include "shape.hpp"
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "file_loaders/safetensors.hpp"
#include <ops/ops.hpp>
#include <string>

#include "module/base/reflist.hpp"

#ifndef HVTLMODULE
#define HVTLMODULE

template <typename ...modclasses>
class Module: public ReferenceList<modclasses...>
{
    public:
    Module(Submodule<modclasses>... mods): ReferenceList<modclasses...>(mods...)
    {
    }
    
    


    // template <typename... params>
    // auto forward(params... args)
    // {
    //     std::cout << "Forward" << std::endl;
    //     return 1;
    // }

    
    // template <typename... params>
    // auto operator()(params... args)
    // {
    //     return forward(args...);
    // }
};


#endif //HVTLMODULE