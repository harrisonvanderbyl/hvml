# Writing operations

Every computation in hvml is an *operation*: a struct with a static `apply`
that computes one element. `run(...)` dispatches it to the compute device of
its first tensor argument (CPU, CUDA or HIP), so an op is written once.

Operations never deal with indices or strides. Which elements meet is decided
by the tensors passed in, and those are usually *views*: slices, `unsqueeze`,
`transpose(a, b)`, `view(shape)`, `broadcast` and `tensor_index`. None of these
copy data.

## Element-wise operations

`apply` receives one element from each argument (broadcast to a common shape)
and returns the output element:

```cpp
struct OpSiluMul : public HardamardOperation<OpSiluMul> {
    __host__ __device__ static inline float apply(const float& gate, const float& up) {
        return gate / (1.0f + expf(-gate)) * up;
    }
};

Tensor<float, 2> y = OpSiluMul::run(gate, up);    // new tensor, same shape
```

If `apply` takes its first argument by non-const reference and returns
`void`, it writes in place (that's how `+=` and tensor copies work). The first
argument can be a view, so an op can update part of a tensor. For example,
rotary embeddings rotate the two halves of each head in place:

```cpp
struct OpRotateHalf : public HardamardOperation<OpRotateHalf> {
    __host__ __device__ static inline void apply(float& x1, float& x2, const float& c, const float& s) {
        float a = x1, b = x2;
        x1 = a * c - b * s;
        x2 = b * c + a * s;
    }
};

OpRotateHalf::run(x[{{}, {}, {0, half}}], x[{{}, {}, {half, 2 * half}}], cos.unsqueeze(1), sin.unsqueeze(1));
```

Shapes broadcast from the left, so give a smaller operand explicit size-1
dimensions: `pos.unsqueeze(0)` for `[B, T, D] + [T, D]`, and `bias.unsqueeze(0)`
for `[M, N] + [N]`.

Results of `run` have a dynamic rank (`Tensor<float, -1>`). Assigning one to a
`Tensor<float, 2>` shares the data, and the rank is checked at runtime.

## Reductions

`ReductionOperation<OP, dim>` sums `apply` along one dimension of the
broadcast shape and drops that dimension. `DotProduct<dim>` and
`ReduceSum<dim>` are the two provided. Most of linear algebra is a
`DotProduct` over views:

```cpp
// x [M, K] · w [N, K]ᵀ → [M, N]: broadcast to [M, N, K], reduce K
Tensor<float, 2> y = DotProduct<-1>::run(x.unsqueeze(1), w.unsqueeze(0));

// sum of squares per row → [M]
auto ss = DotProduct<-1>::run(x, x);
```

`DotProduct` accepts mixed types. Put the float operand first (`x · bfloat16
weights`) so the product is float.

**Attention.** Attention (`ops/nn.hpp`) is two dot products over transposed
views: `q` becomes `[Hk, G, T, 1, D]` and `k` becomes `[Hk, 1, 1, S, D]`.
Grouped-query heads broadcast over the size-1 dimension, so keys are never
repeated.

**Convolution.** `Conv2d` stacks strided slices `x[:, :, kh::s, kw::s]` into
an im2col tensor, then does one `DotProduct`.

Put the reduced dimension where both operands are contiguous (usually last).
This makes the CPU loop walk memory sequentially.

**How reductions run on each device:**
- **CPU:** each output element sums its own run along the reduced dimension,
  and outputs are split across threads when built with `-fopenmp`.
- **CUDA and HIP:** each thread adds consecutive elements that land on the same
  output locally, then issues one atomic add.

## Max and other folds

A reduction that isn't a sum can be written with slices. `row_max` halves the
row and folds the upper half onto the lower with an in-place `max` until one
column is left:

```cpp
while (n > 1) {
    long h = n / 2;
    OpMaxEq::run(m[{{}, {0, h}}], m[{{}, {n - h, n}}]);
    n -= h;
}
```

## Gathering rows: `tensor_index`

`table.tensor_index(ids)` is a view of the rows of `table` picked by an
`unsigned long` index tensor. For example, `[V, D]` indexed by `[T]` gives
`[T, D]`. Any operation reading the view gathers the rows as it goes, so an
embedding lookup is:

```cpp
Tensor<float, 2> x(Shape<2>{T, D}, loc);
x = embed_tokens.weight.tensor_index(ids);       // gathers + converts bf16 → float
```

The view holds a pointer to `ids`, so keep `ids` alive while the view is used.

## Ready-made operations (`ops/nn.hpp`)

| function / op                          | computes                                           |
|----------------------------------------|----------------------------------------------------|
| `linear(x, w)`                         | `x · wᵀ` via `DotProduct`                          |
| `rms_norm(x, w, eps)`, `layer_norm(x, w, b, eps)` | row normalisation                       |
| `rope_tables(positions, dim, θ, loc)`, `rope(x, tables)` | rotary embedding (rotate-half)   |
| `attention(q, k, v, scale, q_offset, causal)` | GQA attention over one sequence             |
| `row_max(x)`, `softmax_rows(x)`        | slice folds + `ReduceSum`                          |
| `gelu(x)`, `OpSiluMul`, `OpCos`, `OpSin` | activations / float trig                         |
| `OpPower`, `OpLog10Clamp`, `OpWhisperNorm` | Whisper log-mel front end                      |
| `tensor_from_host`, `tensor_to_host`   | host ↔ device copies                               |
| `working_location(device)`             | where to allocate activations (never the disk device) |
