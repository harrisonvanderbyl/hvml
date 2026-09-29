#ifndef MODULE_EMBEDDING_HPP
#define MODULE_EMBEDDING_HPP
#include "module/base/module.hpp"
#include "ops/nn.hpp"

//  Token embedding table [vocab, dim].
//  forward(ids [T]) → [T, dim]: a view of the table's rows (nothing copied).
//  Keep `ids` alive while the view is read.

template <typename W = float>
struct Embedding : public Module<Tensor<W, 2>>
{
    Tensor<W, 2> weight;

    Embedding() : Module<Tensor<W, 2>>({weight, "weight"}) {}

    Embedding(size_t vocab, size_t dim, MemoryLocation loc = MemoryType::kDDR)
        : Module<Tensor<W, 2>>({weight, "weight"}), weight(Shape<2>{(long)vocab, (long)dim}, loc) {}

    template <typename I, int R>
    auto forward(const Tensor<I, R>& ids) const { return weight.tensor_index(ids); }

    template <typename I, int R>
    auto operator()(const Tensor<I, R>& ids) const { return forward(ids); }
};

#endif //MODULE_EMBEDDING_HPP
