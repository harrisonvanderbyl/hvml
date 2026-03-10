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
#include "particle/field.hpp"
#include "particle/particle.hpp"


struct ChunkOfSpace{
    bfloat16x4 velocity; // 64 bits
    Particle* occupant = nullptr; // 64 bits
    
    // ChunkOfSpace* neighboringChunks[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
     __device__ __host__ Particle* addFilled(Particle* increment = nullptr, bool onlyifzero = false){
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (!onlyifzero){
            return (Particle*)(void*)atomicExch((unsigned long long*)&this->occupant, (unsigned long long)increment);
        }
        else{
            Particle* expected = nullptr;
            return (Particle*)(void*)atomicCAS((unsigned long long*)&this->occupant, (unsigned long long)expected, (unsigned long long)increment);
            
        }
        // auto oldOccupant = this->occupant;
        // this->occupant = increment;
        // return oldOccupant;
        #else
        Particle* oldOccupant = std::atomic<Particle*>{this->occupant}.exchange(increment);
        return oldOccupant;
        #endif
    };

    __device__ __host__ ChunkOfSpace() : velocity(0.0f, 0.0f, 0.0f, 0.0f), occupant(nullptr) {};




    __device__ __host__ float32x4 swap0(){
        // swap with zero, using copy elision
        // 64 bit atomic swap
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // bfloat16x4 is equivalent to uint64_t for atomic operations
        // auto oldvelocity = this->velocity;
        // this->velocity = bfloat16x4(0.0f, 0.0f, 0.0f, 0.0f);
        auto oldvelocity = atomicExch((unsigned long long*)&this->velocity, 0);
        bfloat16x4 oldVelocity = *(bfloat16x4*)&oldvelocity;
        return oldVelocity;
        
        #else
        return std::atomic<bfloat16x4>{this->velocity}.exchange(bfloat16x4(0.0f, 0.0f, 0.0f, 0.0f));
        #endif
    };

    __device__ __host__ void atomic_plus_equals(const float32x4& a){
        #if defined(__CUDA_ARCH__)  
        atomicAdd((__nv_bfloat162*)&this->velocity[0], __float22bfloat162_rn(*(float2*)&a[0]));
        atomicAdd((__nv_bfloat162*)&this->velocity[2], __float22bfloat162_rn(*(float2*)&a[2]));
        // this->velocity += bfloat16x4(a[0], a[1], a[2], 0);
        #elif defined(__HIP_DEVICE_COMPILE__)
        // atomicAdd((__hip_bfloat16*)&this->velocity[0], a[0]);
        // atomicAdd((__hip_bfloat16*)&this->velocity[1], a[1]);
        // atomicAdd((__hip_bfloat16*)&this->velocity[2], a[2]);
        #else
        // std::atomic<bfloat16>{this->velocity[0]}.fetch_add(a[0]);
        // std::atomic<bfloat16>{this->velocity[1]}.fetch_add(a[1]);
        // std::atomic<bfloat16>{this->velocity[2]}.fetch_add(a[2]);
        #endif
    };

    // cout
    friend std::ostream &operator<<(std::ostream &os, const ChunkOfSpace& q)
    {
        os << "Velocity: (" << q.velocity[0] << ", " << q.velocity[1] << ", " << q.velocity[2] << ")";
        if (q.occupant != nullptr) {
            os << ", Occupant: " << *(q.occupant);
        } else {
            os << ", Occupant: None";
        }
        return os;
    }
};





template <size_t x, size_t y, size_t z>
struct ParticleChunk{
    static constexpr bool NoScalarType = true;
    Hvec<Hvec<Hvec<ChunkOfSpace, z>, y>, x> chunks;
    Hvec<Hvec<Hvec<ParticleChunk*, 3>, 3>, 3> neighboringChunks; // 27 pointers to neighboring chunks, including self at the center, indexed by -1, 0, 1 in each dimension
    bool am_being_simulated = false;
    __device__ __host__ void operator = (const int& value){
        for (size_t i = 0; i < x; i++){
            for (size_t j = 0; j < y; j++){
                for (size_t k = 0; k < z; k++){
                    chunks[i][j][k] = ChunkOfSpace();
                }
            }
        }
    };

    __device__ __host__ inline ChunkOfSpace& get_chunk_relative(const int32x3& position) {
        // Compute which chunk the position falls in and the local position within it
        auto floordiv = [](int a, int b) -> int {
            return a / b - (a % b != 0 && (a ^ b) < 0);
        };
        auto floormod = [](int a, int b) -> int {
            int r = a % b;
            return r + (r != 0 && (r ^ b) < 0) * b;
        };

        int cx = floordiv(position[0], (int)x);
        int cy = floordiv(position[1], (int)y);
        int cz = floordiv(position[2], (int)z);

        int lx = floormod(position[0], (int)x);
        int ly = floormod(position[1], (int)y);
        int lz = floormod(position[2], (int)z);

        if (cx == 0 && cy == 0 && cz == 0) {
            return chunks[lx][ly][lz];
        }

        // Clamp chunk offset to ±1 for neighbor lookup
        int nx = cx < -1 ? -1 : (cx > 1 ? 1 : cx);
        int ny = cy < -1 ? -1 : (cy > 1 ? 1 : cy);
        int nz = cz < -1 ? -1 : (cz > 1 ? 1 : cz);

        ParticleChunk* neighbor = neighboringChunks[nx + 1][ny + 1][nz + 1];
        if (neighbor == nullptr) {
            return chunks[lx][ly][lz]; // fallback: return local cell
        }

        // Recurse into neighbor with adjusted local position
        int32x3 remapped = int32x3(
            position[0] - cx * (int)x,
            position[1] - cy * (int)y,
            position[2] - cz * (int)z
        );
        return neighbor->get_chunk_relative(remapped);
    }

    friend std::ostream &operator<<(std::ostream &os, const ParticleChunk& chunk)
    {        os << "ParticleChunk with dimensions (" << x << ", " << y << ", " << z << ")\n";
            return os;
    };
};

template<size_t chunkx, size_t chunky, size_t chunkz>
struct ChunkGrid: public Tensor<ParticleChunk<chunkx, chunky, chunkz>, 3>
{
    using Tensor<ParticleChunk<chunkx, chunky, chunkz>,3>::Tensor; // Inherit constructors
};
