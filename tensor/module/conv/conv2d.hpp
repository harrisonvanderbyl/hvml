#ifndef MODULE_CONV2D_HPP
#define MODULE_CONV2D_HPP
#include "module/base/module.hpp"
#include "ops/nn.hpp"

//  2-D convolution with bias.  weight [Co, Ci, KH, KW], input [B, Ci, H, W]
//  (any view) → [B, Co, Ho, Wo].
//
//  im2col with slices: tap (kh, kw) of the padded input is the strided slice
//  x[:, :, kh::stride, kw::stride]; the taps are stacked into
//  col [B, Ho, Wo, Ci·KH·KW] and contracted with weight viewed as
//  [Co, Ci·KH·KW]:
//      out = DotProduct<-1>( [B, Ho, Wo, 1, K] , [1, 1, 1, Co, K] ) + bias   (then viewed as [B, Co, Ho, Wo])
//  Images are processed in groups so `col` stays under 64 MB.

template <typename W = float>
struct Conv2d : public Module<Tensor<W, 4>, Tensor<W, 1>>
{
    Tensor<W, 4> weight;
    Tensor<W, 1> bias;
    int stride = 1;
    int padding = 0;

    Conv2d(int stride = 1, int padding = 0)
        : Module<Tensor<W, 4>, Tensor<W, 1>>({weight, "weight"}, {bias, "bias"}), stride(stride), padding(padding) {}

    long out_size(long in, int dim) const { return (in + 2 * padding - weight.shape[dim]) / stride + 1; }

    // optionally followed by GELU (fused with the bias add)
    Tensor<float, 4> forward(const Tensor<float, 4>& x, bool with_gelu = false) const {
        MemoryLocation loc = working_location(weight);
        long B = x.shape[0], Ci = x.shape[1], H = x.shape[2], Wd = x.shape[3], Co = weight.shape[0];
        long Ho = out_size(H, 2), Wo = out_size(Wd, 3);
        int KH = weight.shape[2], KW = weight.shape[3];

        auto pad = [&]() -> Tensor<float, 4> {
            if (padding == 0) return x;
            Tensor<float, 4> p(Shape<4>{B, Ci, H + 2 * padding, Wd + 2 * padding}, loc);
            p = 0.0f;
            p[{{}, {}, {padding, padding + H}, {padding, padding + Wd}}] = x;
            return p;
        };
        Tensor<float, 4> xp = pad();

        long taps = (long)KH * KW, K = Ci * taps;
        Tensor<W, 4> w = weight;
        Tensor<W, 2> w2 = w.view(Shape<2>{Co, K});                                  // [Co, Ci·KH·KW]

        Tensor<float, 4> out(Shape<4>{B, Co, Ho, Wo}, loc);
        long group = std::max(1L, (1L << 24) / (K * Ho * Wo));
        for (long b0 = 0; b0 < B; b0 += group) {
            long b1 = std::min(B, b0 + group), n = b1 - b0;
            // col[b, i, j, c, tap] = xp[b, c, kh + stride·i, kw + stride·j]
            Tensor<float, 5> col(Shape<5>{n, Ho, Wo, Ci, taps}, loc);
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    Tensor<float, 4> tap = xp[{{b0, b1}, {}, {kh, kh + stride * (Ho - 1) + 1, stride},
                                               {kw, kw + stride * (Wo - 1) + 1, stride}}];     // [n, Ci, Ho, Wo]
                    col[{{}, {}, {}, {}, kh * KW + kw}] = tap.transpose(1, 2).transpose(2, 3);    // [n, Ho, Wo, Ci]
                }
            }
            // [n, Ho, Wo, 1, K] · [1, 1, 1, Co, K] → [n, Ho, Wo, Co] → [n, Co, Ho, Wo]
            Tensor<float, 4> y = DotProduct<-1>::run(col.view(Shape<4>{n, Ho, Wo, K}).unsqueeze(3),
                                                     w2.unsqueeze(0).unsqueeze(0).unsqueeze(0));
            out[{{b0, b1}}] = y.transpose(2, 3).transpose(1, 2);
        }
        auto b4 = bias.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        if (with_gelu) return ChainOperations<OperationAdd, OpGelu>::run(out, b4);
        out += b4;
        return out;
    }
    Tensor<float, 4> operator()(const Tensor<float, 4>& x) const { return forward(x); }
};

#endif //MODULE_CONV2D_HPP
