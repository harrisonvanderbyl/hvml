#ifndef TENSOR_MODULE_REFLIST_HPP
#define TENSOR_MODULE_REFLIST_HPP

#include <file_loaders/safetensors.hpp>
#include <memory>
#include <vector>

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


// Detects a `move_to(MemoryLocation)` member (modules and module lists).
template <typename T, typename = void>
struct has_move_to : std::false_type {};
template <typename T>
struct has_move_to<T, std::void_t<decltype(std::declval<T&>().move_to(std::declval<MemoryLocation>()))>> : std::true_type {};

// Replace a tensor's storage with a copy on `loc` (for weights loaded from
// disk).  Tensor::operator= would copy *into* the old storage, so the tensor
// is re-constructed instead.
template <typename T>
inline void move_tensor_to(T& tensor, MemoryLocation loc) {
    if (tensor.storage_pointer == nullptr) return;          // not loaded
    if (tensor.device->this_device_type == loc.memory_type && tensor.device->device_id == loc.device_id &&
        (loc.compute_type == ComputeType::kUnknown || tensor.data.metadata.compute_device == loc.compute_type)) return;
    T moved = tensor.to(loc);
    tensor.~T();
    new (&tensor) T(moved);
}

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
                *mod = tensors.get<IthType>(keyname);
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

    // Move every tensor of this module (recursively) to `loc`.
    template<size_t I = 0>
    void move_to(MemoryLocation loc) {
        if constexpr (I < sizeof...(U)) {
            using IthType = typename std::tuple_element<I, std::tuple<U...>>::type;
            IthType* mod = (IthType*)mods[I];
            if constexpr (has_move_to<IthType>::value) mod->move_to(loc);
            else move_tensor_to(*mod, loc);
            move_to<I + 1>(loc);
        }
    }

    void save_to_safetensors(const char* filename, std::string key = "") {
        safetensors tensors = to_safetensors(safetensors(), key);
        tensors.save(filename);
    }
    
    
};


// ---------------------------------------------------------------------------
//  ModuleList<T> — a numbered list of submodules ("layers.0.", "layers.1.", ...)
// ---------------------------------------------------------------------------

// Modules keep pointers to their own members, so the items are heap-allocated
// and never move.
template <typename T>
struct ModuleList
{
    std::vector<std::unique_ptr<T>> items;

    ModuleList() {}

    // make(i) returns a `new T(...)` for item i.
    template <typename F>
    ModuleList(size_t count, F make) {
        items.reserve(count);
        for (size_t i = 0; i < count; i++) items.emplace_back(make(i));
    }

    ModuleList(const ModuleList&) = delete;
    ModuleList& operator=(const ModuleList&) = delete;

    T& operator[](size_t i) { return *items[i]; }
    const T& operator[](size_t i) const { return *items[i]; }
    size_t size() const { return items.size(); }

    void load_from_safetensors(safetensors tensors, std::string key = "") {
        for (size_t i = 0; i < items.size(); i++) {
            items[i]->load_from_safetensors(tensors, key + std::to_string(i) + ".");
        }
    }

    void save_to_safetensors(safetensors& tensors, std::string key = "") {
        for (size_t i = 0; i < items.size(); i++) {
            items[i]->save_to_safetensors(tensors, key + std::to_string(i) + ".");
        }
    }

    void move_to(MemoryLocation loc) {
        for (auto& item : items) item->move_to(loc);
    }

    friend std::ostream& operator<<(std::ostream& os, const ModuleList& list) {
        os << "[" << list.items.size() << " x module]";
        return os;
    }
};

#endif //TENSOR_MODULE_REFLIST_HPP