
#include "tensor.hpp"
#include "ops/ops.hpp"
#include "display/display.hpp"
#include <assert.h>
#include <atomic>
#include "module/linear/linear.hpp"
#include "camera/camera.hpp"
#include "file_loaders/gltf.hpp"
#include "display/materials/materials.hpp"
#include "display/scene/scene.hpp"

#ifndef __shared__
#define __shared__
#endif

enum OperationLevel{
    Thread,
    Warp,
    Block
};

enum StorageType{
    Thread,
    Shared,
    Global
};

template <typename T, int rank, OperationLevel op_level, StorageType storage_type>
struct TileTensor
{
};

// block level class,
// warp level procedures,
// thread level procedures

// each warp level procedure must have dependencies on other warp level procedures.
// warp procedures can be async jobs.

// block procedures require syncronization between warps.

// global storage block needs to be split into independent objects, these can be run on different blocks, as not need for syncronization, can combine to be pipelined
// so there needs to be a load_to_block procedure, the purpose is to return an independent block of data that will be passed to a block level procedure.
// also can return an importance value, that dictates that it will be scheduled to load after all lower values are both loaded and processed. 


template <typename InputType,typename... Args>
struct DataLoader{ // perhaps dataloader is the input block
    int lane = -1;
    int importance = 0;
    int current_block_id = -1; // this is set by the scheduler when it schedules this dataloader to load its data, this is used to determine which block procedure will process the data loaded by this dataloader, as well as to determine when to load the data, as it should be loaded before the block procedure that processes it is scheduled to run.
    mytuple<Parameter<Args>...> params;

    DataLoader(const Tensor<Args>&... args): params(args...){
        set_importance(params...);
        set_lane(params...);
    };

    __device__ __host__ int32x2 get_num_lanes_and_blocks(){
        // this should return the number of lanes and blocks that this dataloader needs to load its data, this is used by the scheduler to determine how to schedule the loading of the data.
        return int32x2(1, 0); // default is 1 lane and 1 block, meaning it will be loaded by a single block, and no lanes
    };

    virtual __device__ __host__ void set_importance(Parameter<Args>... params){
        importance = 0; // default importance, can be overridden by user defined loaders, higher importance means it will be scheduled to load after lower importance ones are loaded *and* processed, -1 means it doesnt matter, objects with same importance will be scheduled to be loaded after each other in any order.
    };

    virtual __device__ __host__ void set_lane(Parameter<Args>... params){
        lane = -1;
    }

    virtual __device__ __host__ InputType load(Parameter<Args>... params){
        return InputType();
    };

    __device__ __host__ auto load_to_block(){ // runs on gpu, returns a block of data to be processed by a block procedure.
        return load(params...);
    };
    
};

// // saves data from block procedure back to global memory, run after everything in a lane is done, so can be async with other lanes, but not with other block procedures in the same lane, as they may be using the same shared memory for their input data.
// struct DataUnloader{

// };

// block procedures run at block level, can syncronize between warps, public variables are shared memory, can call warp procedures, but not other block procedures
template <typename BlockInputType>
struct BlockProcedure {
    // Any public variables of a block procedure are able to be accessed by any block procedures that run after it, and are shared memory.
    // Also is the input variables, that are set by the data loader, prior to the start of the block procedure, these are also shared memory.

    BlockInputType blockinput;

    // any child classes will have variable that will act like long storage between blocks

    __device__ __host__ virtual void main(const BlockInputType& input){

    };

    __device__ __host__ void run(
        BlockInputType* input,
        int num_inputs
    ){ params
        for (int i = 0; i < num_inputs; i++){
            this->blockinput = input[i];

            // run the procedure on this input, can access shared memory and public variables, can call warp procedures, but not other block procedures.
            main(blockinput);
        }
    };
};

// warp procedures run at warp level, can sync between threads in a warp, can call thread procedures
struct WarpProcedure {

};

template <typename T>
struct SharedMemoryObject : public T{
    using T::T; // Inherit constructors
    // this is an object stored in shared memory, it must have a method of loading data into it that can be called at the thread, warp, and block level
    // it must also have a method of processing the data that can be called at the warp and block level, but not thread level, as it may require synchronization between threads in a warp.
    // it must also have a method of saving the data back to global memory that can be called at the block level, but not warp or thread level, as it may require synchronization between warps.

};

template <typename T>
struct WarpMemoryObject : public T{
    using T::T; // Inherit constructors
    // this is an object stored in warp level memory, it must have a method of loading data into it that can be called at the thread and warp level, but not block level, as it may require synchronization between warps.
    // it must also have a method of processing the data that can be called at the warp level, but not thread or block level, as it may require synchronization between threads in a warp.
    // it must also have a method of saving the data back to global memory that can be called at the warp level, but not thread or block level, as it may require synchronization between threads in a warp.
};

template <typename SubKernel>
struct TileKernel {

    // get return tupe of SubKernel::load and use as type for member variable tile_input
    using TileInputType = WarpTile;
    TileInputType tile_inputs_swap_ins[32]; // every second warp shares a tile input, so one can load while the other is processing

    // __host__ __device__ void load_to_tile(typename SubKernel::params params){
    //     tile_input = SubKernel().load(params);
    // };

    __host__ __device__ int get_warp_id(){
        #if defined(__CUDA_ARCH__)
        return threadIdx.x / 32; // assuming warp size of 32
        #elif defined(__HIP_DEVICE_COMPILE__)
        return threadIdx.x / 64; // assuming warp size of 64 for AMD GPUs
        #else
        return 0; // fallback for CPU, just return 0 as there is only one warp
        #endif    
    };

    __host__ __device__ int get_thread_id(){
        #if defined(__CUDA_ARCH__)
        return threadIdx.x % 32; // assuming warp size of 32
        #elif defined(__HIP_DEVICE_COMPILE__)
        return threadIdx.x % 64; // assuming warp size of 64 for AMD GPUs
        #else
        return 0; // fallback for CPU, just return 0 as there is only one thread
        #endif    
    };

};



// warp level vector for warp level vector ops
template <typename T>
struct TVec
{
    T data;

    __host__ __device__ TVec()
    {
    };

    __host__ __device__ TVec(const T& val)
    {
        data = val;
    };

    __host__ __device__ const T operator[](int index) const
    {
        T result;
        // load data from adjacent threads in the block using warp shuffle instructions (e.g. __shfl_sync in CUDA)
        #if defined(__CUDA_ARCH__)
        result = __shfl_sync(0xffffffff, this->data, index);
        #elif defined(__HIP_DEVICE_COMPILE__)
        result = __shfl(this->data, index);
        #else
        // fallback for CPU, just return the data of the current thread (no shuffling)
        result = this->data;
        #endif
        return result;
    };

    __host__ __device__ T sum()
    {
        T result = data;
        // perform a reduction across the block using warp shuffle instructions
        #if defined(__CUDA_ARCH__)
        // accumulate by shuffle // not shuffle down, but swap each one with the second, then with 4th, etc, so that all threads get the final result
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            result += __shfl_xor_sync(0xffffffff, result, offset);
        }
        #elif defined(__HIP_DEVICE_COMPILE__)
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            result += __shfl(this->data, offset);
        }
        #else
        // fallback for CPU, just return the data of the current thread (no reduction)
        result = this->data;
        #endif
        return result;
    };

    __host__ __device__ TVec operator+(const TVec& other) const
    {
        TVec result;
        result.data = this->data + other.data;
        return result;
    };

    __host__ __device__ TVec operator*(const TVec& other) const
    {
        TVec result;
        result.data = this->data * other.data;
        return result;
    };

    __host__ __device__ TVec operator/(const TVec& other) const
    {
        TVec result;
        result.data = this->data / other.data;
        return result;
    };

    __host__ __device__ TVec operator-(const TVec& other) const
    {
        TVec result;
        result.data = this->data - other.data;
        return result;
    };

    __host__ __device__ T dot(const TVec& other) const
    {
        T result = this->data * other.data;
        // perform a reduction across the block using warp shuffle instructions
        #if defined(__CUDA_ARCH__)
        // accumulate by shuffle // not shuffle down, but swap each one with the second, then with 4th, etc, so that all threads get the final result
        for (int offset = 1; offset < 32; offset <<= 1)
        {            result += __shfl_xor_sync(0xffffffff, result, offset);
        }

        #elif defined(__HIP_DEVICE_COMPILE__)
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            result += __shfl_xor(this->data, offset);
        }
        #else
        // fallback for CPU, just return the product of the current thread (no reduction)
        result = this->data * other.data;
        #endif
        return result;
    };

    __host__ __device__ T max() const
    {
        T result = data;
        // perform a reduction across the block to find the maximum value using warp shuffle instructions
        #if defined(__CUDA_ARCH__)
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            result = max(result, __shfl_xor_sync(0xffffffff, result, offset));
        }
        #elif defined(__HIP_DEVICE_COMPILE__)
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            result = max(result, __shfl_xor(this->data, offset));
        }
        #else
        // fallback for CPU, just return the data of the current thread (no reduction)
        result = this->data;
        #endif
        return result;
    };

    __host__ __device__ TVec operator-() const
    {
        TVec result;
        result.data = -this->data;
        return result;
    };

    __host__ __device__ TVec operator+=(const TVec& other)
    {
        this->data += other.data;
        return *this;
    };

    __host__ __device__ TVec operator-=(const TVec& other)
    {
        this->data -= other.data;
        return *this;
    };

    __host__ __device__ TVec operator*=(const TVec& other)
    {
        this->data *= other.data;
        return *this;
    };

    __host__ __device__ TVec operator/=(const TVec& other)
    {
        this->data /= other.data;
        return *this;
    };

    __host__ __device__ T min() const
    {     T result = data;
        // perform a reduction across the block to find the minimum value using warp shuffle instructions
        #if defined(__CUDA_ARCH__)
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            result = min(result, __shfl_xor_sync(0xffffffff, result, offset));
        }
        #elif defined(__HIP_DEVICE_COMPILE__)
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            result = min(result, __shfl_xor(this->data, offset));
        }
        #else
        // fallback for CPU, just return the data of the current thread (no reduction)
        result = this->data;
        #endif
        return result;
    };

};

// Block storage of warp level vector
template <typename T>
struct SVect : public Hvec<Hvec<T, 32>, 32>
{
    using Hvec<Hvec<T, 32>, 32>::Hvec; // Inherit constructors

    __host__ __device__ operator TVec<T>() const
    {
        TVec<T> result;
        // load data from adjacent threads in the block using warp shuffle instructions (e.g. __shfl_sync in CUDA)
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return TVec<T>(this->data[threadIdx.x][threadIdx.y]);
        #else
        // fallback for CPU, just return the data of the current thread (no shuffling)
        result.data = this->data[0][0];
        #endif
        return result;
    };

    __host__ __device__ void save(const TVec<T>& vec)
    {
        // write data to adjacent threads in the block using warp shuffle instructions (e.g. __shfl_sync in CUDA)
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        this->data[threadIdx.x][threadIdx.y] = vec.data;
        #else
        // fallback for CPU, just write the data of the current thread (no shuffling)
        this->data[0][0] = vec.data;
        #endif
    };

    __host__ __device__ TVec<T> sum(int dim = 0) const
    {
        TVec<T> SideSum;

        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (dim == 0) {
            // sum across rows
            SideSum.data = this->data[threadIdx.x][threadIdx.y];
        } else {
            // sum across columns
            SideSum.data = this->data[threadIdx.y][threadIdx.x];
        }       
         // perform a reduction across the block using warp shuffle instructions
        __syncthreads();
        return SideSum.sum();
        #else
        // fallback for CPU, just return the data of the current thread (no reduction)
        SideSum.data = this->data[0][0];
        return TVec<T>(SideSum.data);
        #endif
    };
};


struct BlockTensorTile: public Hvec<Hvec<bfloat16, 16>, 16>
{
    using Hvec<Hvec<bfloat16, 16>, 16>::Hvec; // Inherit constructors

    __device__ __host__ void single_warp_load(const Parameter<bfloat16>& global_tensor, int warp_id = 0)
    {
        #if defined(__CUDA_ARCH__)
        // load data from global memory into shared memory using coalesced accesses
            int mywarp_id = threadIdx.y;
            int lane_id = threadIdx.x;
            if (mywarp_id == warp_id) {
                // each thread in the warp loads one element of the tile
                int index = lane_id;
                // each thread loads 8 values (128 bits) at a time
                Hvec<bfloat16, 8>& data_vec = *(Hvec<bfloat16, 8>*)((&this->data[0][0]) + index * 8);
                bfloat16* start = global_tensor.data + index * 8; // just assume its a 16*16 tensor for now
                data_vec = *((Hvec<bfloat16, 8>*)start);
                // 
64
            }
        #else
        // fallback for CPU, just copy the data from global memory to shared memory (no shuffling)
        
        #endif
    }

    template <typename T>
    __device__ __host__ void single_warp_save(Parameter<T>& global_tensor, int warp_id = 0)
    {
        #if defined(__CUDA_ARCH__)
        // write data from shared memory back to global memory using coalesced accesses
            int mywarp_id = threadIdx.y;
            int lane_id = threadIdx.x;
            if (mywarp_id == warp_id) {
                // each thread in the warp writes one element of the tile
                int index = lane_id;
                // each thread writes 8 values (128 bits) at a time
                Hvec<bfloat16, 8>& data_vec = *(Hvec<bfloat16, 8>*)((&this->data[0][0]) + index * 8);
                T* start = global_tensor.data + index * 8; // just assume its a 16*16 tensor for now
                *((Hvec<T, 8>*)start) = data_vec;
            }
        #else
        // fallback for CPU, just copy the data from shared memory back to global memory (no shuffling)
        
        #endif
    }
};

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

struct MatmulTileKernel : public TileKernel<MatmulTileKernel>
{

    BlockTensorTile tileA; // shared memory for this tile lane
    BlockTensorTile acc;
    // set warptyle type here:
    using WarpTile = BlockTensorTile;
    // wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag; // accumulator fragment

    int get_lane_count(
        Parameter<bfloat16> A,
        Parameter<bfloat16> B,
        Parameter<float> C
    ){
        return (A.shape[0] + 15) / 16;
    };

    __host__ __device__ int load_shared_and_get_tile_count(
        int lane_id,
        Parameter<bfloat16> A,
        Parameter<bfloat16> B,
        Parameter<float> C
    ){
        tileA.single_warp_load(A, 0); // only first warp loads tileA for now
        return (B.shape[1] + 15) / 16;
    };

    __host__ __device__ auto load(
        int tile_id_for_this_lane,
        int lane_id,
        Parameter<bfloat16> A,
        Parameter<bfloat16> B,
        Parameter<float> C,
        const BlockTensorTile& tileB // shared memory
    ){
        tileB.single_warp_load(B, get_warp_id()); // have each lane load a different tile from global memory into shared memory, for now just have one tile, but eventually will have multiple tiles per lane, and will need to calculate the correct tile id for each lane to load based on the lane id and the tile id for this lane, as well as the total number of tiles and lanes.
        return tileB;
    };
    
        

    __host__ __device__ auto process(
        const BlockTensorTile& inptile
    ){
        #if defined(__CUDA_ARCH__)
        // we want loading and processing to be interleaved, some warps loading, some warps processing;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
        wmma::fill_fragment( acc_frag, 0.0f );
        a_frag.load(tileA);
        b_frag.load(inptile);
        wmma::mma_sync( acc_frag, a_frag, b_frag, acc_frag );
        
        // save the accumulator fragment back to shared memory, so that it can be accumulated with the next tile's result

        #else
        // fallback for CPU, just return an empty accumulator fragment (no computation)
        #endif

    }

    __host__ __device__ void write_back(
        Parameter<bfloat16> A,
        Parameter<bfloat16> B,
        Parameter<float> C
    ){
        // write the results back to global memory, this should only be called after all tiles for this lane have been processed, as it may require synchronization between warps in the same lane, but can be async with other lanes.
        tileA.single_warp_save(C, 0); // only first warp saves tileA for now
    }
};





// block level matrix


// warp level matrix storage for matrix multiplication
template <typename warp_a_or_b>
struct WarpTensorTile: public wmma::fragment<warp_a_or_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>
{
    using wmma::fragment<warp_a_or_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>::fragment; // Inherit constructors

    __host__ __device__ void load(const BlockTensorTile& shared_tile)
    {
        wmma::load_matrix_sync(*this, (__nv_bfloat16*)(void*)shared_tile.data, 16);
    }
};

// 

template <typename TileKernel>
__global__ void run_tile_kernel(int lane_count, Parameter<bfloat16> a, Parameter<bfloat16> b, Parameter<float> c){
    __shared__ TileKernel shared_tile;
    #if defined(__CUDA_ARCH__)
        int tile_count_for_this_lane = shared_tile.load_shared_and_get_tile_count(lane_count, a, b, c); // for now just have one lane, so lane id is 0
        for (int tile_id = 0; tile_id < tile_count_for_this_lane; tile_id++){
            auto inptile = shared_tile.load(tile_id, lane_count, a, b, c);
            shared_tile.process(inptile);
        }
        // __syncthreads(); // synchronize to make sure the tile is fully loaded before computation
        // // __shared__ __half B[16*16];
        // WarpTensorTile<wmma::matrix_a> a_frag;
        // WarpTensorTile<wmma::matrix_b> b_frag;
        // wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
        // wmma::fill_fragment( acc_frag, 0.0f );
        // a_frag.load(shared_tile);
        // wmma::load_matrix_sync( b_frag, (__nv_bfloat16*)(void*)b.data, 16 );
        // wmma::mma_sync( acc_frag, a_frag, b_frag, acc_frag );
        // wmma::store_matrix_sync( c.data, acc_frag, 16, wmma::mem_row_major );

    // kernel.block_idx = blockIdx.x;
    // kernel.load_tile(a, b);
    // kernel.do_tile_computation();
    // __syncthreads();
    // kernel.write_tile(c);
    #endif
    // __syncthreads();
}
#endif

template <ComputeType device_type>
__weak void launch_tile_kernel(const Tensor<bfloat16>& a, const Tensor<bfloat16>& b, const Tensor<float>& c){
    std::cout << "Launching tile kernel for device type: " << device_type << std::endl;
}

template <>
void launch_tile_kernel<ComputeType::kCUDA>(const Tensor<bfloat16>& a, const Tensor<bfloat16>& b, const Tensor<float>& c);

#if defined(__CUDACC__)
template <>
void launch_tile_kernel<ComputeType::kCUDA>(const Tensor<bfloat16>& a, const Tensor<bfloat16>& b, const Tensor<float>& c){
    run_tile_kernel<TileKernelMatmul><<<1, dim3(32,1,1)>>>(a, b, c);
}
#endif

__weak int main(){
        Tensor<bfloat16> acpu({16,16}, MemoryType::kDDR);
        Tensor<bfloat16> bcpu({16,16}, MemoryType::kDDR);
        for (int i = 0; i < 16; i++){
            for (int j = 0; j < 16; j++){
                acpu[{i,j}] = bfloat16(rand() % 10);
                bcpu[{i,j}] = bfloat16(rand() % 10);
            }
        }
        
        
        Tensor<float> c({16,16}, MemoryType::kCUDA_VRAM);

        launch_tile_kernel<ComputeType::kCUDA>(acpu.to(kCUDA_VRAM), bcpu.to(kCUDA_VRAM), c);
        std::cout << "Output from tile kernel: " << std::endl;
        c.device->synchronize_function();
        std::cout << c << std::endl;
        
        // compare with CPU matmul
        Tensor<float,2> c_cpu({16,16}, MemoryType::kDDR);
        for (int i = 0; i < 16; i++){
            for (int j = 0; j < 16; j++){
                float sum = 0.0f;
                for (int k = 0; k < 16; k++){
                    sum += (*acpu[{i,k}].data) * (*bcpu[{k,j}].data);
                }
                c_cpu[{i,j}] = sum;
            }
        }
        std::cout << "Output from CPU matmul: " << std::endl;
        std::cout << c_cpu << std::endl;

        return 0;
}