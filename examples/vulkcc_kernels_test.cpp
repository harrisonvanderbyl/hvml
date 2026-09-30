// vulkcc_kernels_test.cpp — CUDA-style kernels compiled for Vulkan by vulkcc.
//
//   ./vulkcc examples/vulkcc_kernels_test.cpp -o vulkcc_kernels_test -I./tensor -std=c++20 -O2
//   DEVICE_PLUGIN_DIR=tensor/device/plugins ./vulkcc_kernels_test
//
// Plain CUDA code: __global__ / __device__ functions, <<<>>> launches, structs
// with methods, pointers, __shared__ memory, atomics and warp shuffles.
// Vulkan tensors give the device pointers (Tensor::data is the buffer's
// device address, the kVULKAN kernel view).

#include "tensor.hpp"
#include "ops/nn.hpp"
#include <cstdio>
#include <vector>

// ---- device code ---------------------------------------------------------------

struct Particle {
    float x, y, vx, vy;
    int hits;
    bool alive;

    __host__ __device__ void step(float dt) {
        x += vx * dt;
        y += vy * dt;
        if (x < 0.0f || x > 1.0f) { vx = -vx; hits++; }
        if (y < 0.0f || y > 1.0f) { vy = -vy; hits++; }
    }
    __host__ __device__ float speed2() const { return vx * vx + vy * vy; }
};

struct Params {
    float dt;
    int steps;
    float* energy;   // pointer inside a by-value argument
};

__device__ void bump(int& counter, int by) { counter += by; }   // reference into memory

__global__ void simulate(Particle* ps, int n, Params p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle& q = ps[i];
    for (int s = 0; s < p.steps; s++) q.step(p.dt);
    if (q.speed2() > 0.5f) q.alive = false;
    bump(q.hits, 1);
    atomicAdd(p.energy, q.speed2());
}

// shared-memory tiled transpose (2-D blocks)
template <int TILE>
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];
    int c = blockIdx.x * TILE + threadIdx.x;
    int r = blockIdx.y * TILE + threadIdx.y;
    if (r < rows && c < cols) tile[threadIdx.y][threadIdx.x] = in[r * cols + c];
    __syncthreads();
    int oc = blockIdx.y * TILE + threadIdx.x;
    int orow = blockIdx.x * TILE + threadIdx.y;
    if (orow < cols && oc < rows) out[orow * rows + oc] = tile[threadIdx.x][threadIdx.y];
}

// warp-shuffle sum + one atomic per warp; grid-stride loop
template <typename T>
__device__ T warp_sum(T v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

template <typename T>
__global__ void sum(const T* x, long n, T* total) {
    T acc = 0;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) acc += x[i];
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) atomicAdd(total, acc);
}

// pointer arithmetic, histogram with integer atomics
__global__ void histogram(const unsigned int* values, int n, unsigned int* bins, int nbins) {
    const unsigned int* p = values + blockIdx.x * blockDim.x;
    int i = threadIdx.x;
    if (blockIdx.x * blockDim.x + i < n) atomicAdd(bins + (p[i] % nbins), 1u);
}

// ---- host ------------------------------------------------------------------------

static MemoryLocation gpu() { return MemoryLocation(MemoryType::kUnknown_MEM, 0); }
static int failures = 0;

static void check(const char* name, bool ok, const std::string& detail = "") {
    if (!ok) failures++;
    printf("%s %-34s %s\n", ok ? "[PASS]" : "[FAIL]", name, detail.c_str());
}

template <typename T>
static Tensor<T, 1> upload(const std::vector<T>& v) {
    return tensor_from_host(Shape<1>{(long)v.size()}, v.data(), gpu());
}

int main() {
    // particles
    {
        int n = 1000;
        std::vector<Particle> host(n);
        for (int i = 0; i < n; i++) {
            float a = i * 0.37f;
            host[i] = Particle{0.5f + 0.3f * sinf(a), 0.5f + 0.3f * cosf(a), 0.8f * cosf(3 * a), 0.7f * sinf(5 * a), 0, true};
        }
        auto ps = tensor_from_host(Shape<1>{(long)n * (long)sizeof(Particle)}, (const uint8_t*)host.data(), gpu());
        std::vector<float> zero = {0.0f};
        auto energy = upload(zero);
        Params p{0.01f, 50, energy.data};
        simulate<<<(n + 127) / 128, 128>>>((Particle*)ps.data.data, n, p);

        std::vector<Particle> ref = host;
        float e = 0;
        for (auto& q : ref) {
            for (int s = 0; s < 50; s++) q.step(0.01f);
            if (q.speed2() > 0.5f) q.alive = false;
            q.hits++;
            e += q.speed2();
        }
        auto got = tensor_to_host(ps);
        const Particle* g = (const Particle*)got.data();
        double worst = 0;
        int mismatched = 0;
        for (int i = 0; i < n; i++) {
            worst = std::max(worst, (double)std::fabs(g[i].x - ref[i].x));
            worst = std::max(worst, (double)std::fabs(g[i].y - ref[i].y));
            if (g[i].hits != ref[i].hits || g[i].alive != ref[i].alive) mismatched++;
        }
        float ge = tensor_to_host(energy)[0];
        check("structs, methods, references", worst < 1e-4 && mismatched == 0,
              "max|dx| " + std::to_string(worst) + ", " + std::to_string(mismatched) + " mismatched");
        check("float atomicAdd via pointer in struct", std::fabs(ge - e) < 1e-2 * std::max(1.0f, e),
              std::to_string(ge) + " vs " + std::to_string(e));
    }

    // transpose
    {
        int rows = 70, cols = 45;
        std::vector<float> m(rows * cols);
        for (int i = 0; i < rows * cols; i++) m[i] = (float)i * 0.5f;
        auto in = upload(m);
        std::vector<float> z(rows * cols, 0.0f);
        auto out = upload(z);
        transpose<16><<<dim3((cols + 15) / 16, (rows + 15) / 16), dim3(16, 16)>>>(in.data, out.data, rows, cols);
        auto t = tensor_to_host(out);
        bool ok = true;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) ok = ok && t[c * rows + r] == m[r * cols + c];
        check("__shared__ tile transpose (2-D blocks)", ok);
    }

    // shuffle reduction
    {
        long n = 100000;
        std::vector<float> x(n);
        double ref = 0;
        for (long i = 0; i < n; i++) { x[i] = (float)((i * 7919) % 1000) * 0.001f; ref += x[i]; }
        auto xs = upload(x);
        std::vector<float> zero = {0.0f};
        auto total = upload(zero);
        sum<float><<<64, 256>>>(xs.data, n, total.data);
        float got = tensor_to_host(total)[0];
        check("warp shuffles + atomics (float)", std::fabs(got - ref) < 1e-3 * ref,
              std::to_string(got) + " vs " + std::to_string(ref));

        std::vector<int> xi(n);
        long refi = 0;
        for (long i = 0; i < n; i++) { xi[i] = (int)((i * 31) % 100); refi += xi[i]; }
        auto xis = upload(xi);
        std::vector<int> zi = {0};
        auto ti = upload(zi);
        sum<int><<<64, 256>>>(xis.data, n, ti.data);
        int goti = tensor_to_host(ti)[0];
        check("warp shuffles + atomics (int)", goti == refi, std::to_string(goti) + " vs " + std::to_string(refi));
    }

    // histogram
    {
        int n = 5000, nbins = 13;
        std::vector<unsigned int> v(n);
        std::vector<unsigned int> ref(nbins, 0);
        for (int i = 0; i < n; i++) { v[i] = (unsigned)(i * 2654435761u); ref[v[i] % nbins]++; }
        auto vs = upload(v);
        std::vector<unsigned int> zb(nbins, 0);
        auto bins = upload(zb);
        histogram<<<(n + 255) / 256, 256>>>(vs.data, n, bins.data, nbins);
        auto got = tensor_to_host(bins);
        check("pointer arithmetic + uint atomics", got == ref);
    }

    printf("%s\n", failures ? "SOME CHECKS FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
