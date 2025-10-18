#ifndef TENSOR_MODULE_REFLIST_HPP
#define TENSOR_MODULE_REFLIST_HPP

#include <file_loaders/safetensors.hpp>

template <typename T>
struct Submodule
{
    T* ptr;
    const char* name;
    Submodule (T& p, const char* name): ptr(&p)
    {
        this->ptr = &p;
        this->name = name;
    }

    operator void*() const
    {
        return ptr;
    }

    operator const char*() const
    {
        return name;
    }

    
};

#include <type_traits>

// Primary template with a static assertion
// for a meaningful error message
// if it ever gets instantiated.
// We could leave it undefined if we didn't care.

template<typename, typename T>
struct has_load_from_safetensors {
    static_assert(
        std::integral_constant<T, false>::value,
        "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_load_from_safetensors<C, Ret(Args...)> {
private:
    template<typename T>
    static constexpr auto check(T*)
    -> typename
        std::is_same<
            decltype( std::declval<T>().load_from_safetensors( std::declval<Args>()... ) ),
            Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        >::type;  // attempt to call it and see if the return type is correct

    template<typename>
    static constexpr std::false_type check(...);

    typedef decltype(check<C>(0)) type;

public:
    static constexpr bool value = type::value;
};


template <typename... U>
struct ReferenceList
{
public:
    void* mods[sizeof...(U)];
    const char* names[sizeof...(U)];
    
    ReferenceList(Submodule<U>... mods): mods{(void*)mods.ptr...}, names{mods.name...}
    {
        // Constructor to initialize the reference list with submodules
    }

    
    // Helper function to print a specific module at index I with its correct type
    template<size_t I>
    static void print_module(std::ostream& os, const ReferenceList& list, 
                            typename std::enable_if<(I < sizeof...(U))>::type* = nullptr) {
        // Get the Ith type from the parameter pack
        using IthType = typename std::tuple_element<I, std::tuple<U...>>::type;
        
        // Cast and print the module
        os << " " << list.names[I] << ": ";
        os << *(IthType*)(list.mods[I]) << "\n";
        
        // Recursive call to print next element
        print_module<I+1>(os, list);
    }
    
    // Base case to end recursion
    template<size_t I>
    static void print_module(std::ostream& os, const ReferenceList& list,
                            typename std::enable_if<(I >= sizeof...(U))>::type* = nullptr) {
        // Do nothing, end recursion
    }
    
    friend std::ostream& operator<<(std::ostream& os, const ReferenceList& list) {
        os << "(\n";
            print_module<0>(os, list);
        os << ")";
        return os;
    }

    // recursively loop through the modules and attempt to load them
    template<size_t I>
    void load_from_safetensors(safetensors tensors, std::string key = "", 
                                typename std::enable_if<(I < sizeof...(U))>::type* = nullptr) {
        // Get the Ith type from the parameter pack
        using IthType = typename std::tuple_element<I, std::tuple<U...>>::type;
        auto keyname = key + names[I];
        
        // Check if the module exists in the safetensors
        if constexpr (has_load_from_safetensors<IthType, void(safetensors, std::string)>::value) {
                // Load the tensors from the safetensors
                IthType* mod = (IthType*)mods[I];
                mod->load_from_safetensors(tensors, keyname + ".");
        }else{
            if (tensors.contains(keyname)) {
                // Load the module
                IthType* mod = (IthType*)mods[I];
                *mod = tensors[keyname];
                // std::cout << "Loaded " << keyname << " from safetensors" << std::endl;
            }
            else{
                std::cout << "Failed to load " << keyname << " from safetensors" << std::endl;
            }
                // *mod = tensors[key + names[I]];
        }
     
        // Recursive call to load next element
        load_from_safetensors<I+1>(tensors, key);
    }

    // Base case to end recursion
    template<size_t I>
    void load_from_safetensors(safetensors tensors, std::string key = "", 
                                typename std::enable_if<(I >= sizeof...(U))>::type* = nullptr) {
        // Do nothing, end recursion
    }

    // load from safetensors
    // this function will load the tensors from the safetensors file
    // and assign them to the modules

   void load_from_safetensors(safetensors tensors, std::string key = "") {
        load_from_safetensors<0>(tensors, key);
    }

    template<size_t I>
    void save_to_safetensors(safetensors& tensors, std::string key = "", 
                                typename std::enable_if<(I < sizeof...(U))>::type* = nullptr) {
        // Get the Ith type from the parameter pack
        using IthType = typename std::tuple_element<I, std::tuple<U...>>::type;
        
        std::string keyname = key + names[I];
        // Check if the module exists in the safetensors
        if constexpr (has_load_from_safetensors<IthType, void(safetensors, std::string)>::value) {
                // Load the tensors from the safetensors
                IthType* mod = (IthType*)mods[I];
                mod->save_to_safetensors(tensors, keyname + ".");
        }else{
            // if can be cast to Tensor<void,-1> then save it
            // if constexpr (std::is_constructible<Tensor<void, -1>, IthType>::value) {
            IthType& mod = *(IthType*)mods[I];
            tensors.add(keyname.c_str(), mod);
                 std::cout << "Saved " << keyname << " to safetensors" << std::endl;
            //  }
            //  else{
                // std::cout << "Failed to save " << keyname << " to safetensors" << std::endl;
            //  }
        }
       
        
        // Recursive call to load next element
        save_to_safetensors<I+1>(tensors, key);
    }

    // Base case to end recursion
    template<size_t I>
    void save_to_safetensors(safetensors& tensors, std::string key = "", 
                                typename std::enable_if<(I >= sizeof...(U))>::type* = nullptr) {
        // Do nothing, end recursion
    }

    void save_to_safetensors(safetensors& tensors, std::string key = "") {
        save_to_safetensors<0>(tensors, key);
    }

    safetensors to_safetensors(safetensors tensors = safetensors(), std::string key = "") {
        // loop through the modules and save them to the safetensors
        save_to_safetensors<0>(tensors, key);

        return tensors;
    }
    
    
};


#endif //TENSOR_MODULE_REFLIST_HPP