#ifndef RWKV7_TIMEMIX_HPP
#define RWKV7_TIMEMIX_HPP

#include "module/base/module.hpp"
#include "module/linear/linear.hpp"
#include "module/normalization/normalization.hpp"
#include "models/rwkv7/timeshift/kernels.hpp"
#include "models/rwkv7/rwkv7_op.hpp"
#include "ops/activations.hpp"
#include "ops/matmul.hpp"

template <typename T = float>
struct RWKV_TimeMix : public Module<
    Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>,
    Tensor<T, 3>, Tensor<T, 2>, Tensor<T, 2>,
    Tensor<T, 3>, Tensor<T, 2>, Tensor<T, 2>,
    Tensor<T, 3>, Tensor<T, 2>, Tensor<T, 2>,
    Tensor<T, 2>, Tensor<T, 2>,
    Tensor<T, 3>, Tensor<T, 3>,
    Tensor<T, 2>,
    Linear<T>, Linear<T>, Linear<T>, Linear<T>,
    GroupNorm<T>
>
{
    public:
    size_t dim_att;
    size_t n_layer;
    size_t n_embd;
    size_t layer_id;
    size_t n_head;
    size_t head_size;
    size_t head_size_divisor;
    
    // Parameters
    Tensor<T, 3> x_r, x_w, x_k, x_v, x_a, x_g;  // [1, 1, C]
    Tensor<T, 3> w0;  // [1, 1, C]
    Tensor<T, 2> w1, w2;  // decay LoRA
    Tensor<T, 3> a0;  // [1, 1, C]
    Tensor<T, 2> a1, a2;  // aaa LoRA
    Tensor<T, 3> v0;  // [1, 1, C] (optional)
    Tensor<T, 2> v1, v2;  // mv LoRA (optional)
    Tensor<T, 2> g1, g2;  // gate LoRA
    Tensor<T, 3> k_k, k_a;  // [1, 1, C]
    Tensor<T, 2> r_k;  // [H, N]
    
    // Linear layers
    Linear<T> receptance, key, value, output;
    GroupNorm<T> ln_x;
    
    // Time shift state
    Tensor<T, 2> shift_state;
    
    // State
    Tensor<T, 4> wkvstate;  // [B, H, head_size, head_size]
    
    RWKV_TimeMix(
        size_t layer_id,
        size_t n_layer,
        size_t n_embd,
        size_t n_head,
        size_t head_size,
        size_t dim_att,
        size_t decay_lora,
        size_t aaa_lora,
        size_t mv_lora,
        size_t gate_lora,
        DeviceType device_type = DeviceType::kCPU
    ) : dim_att(dim_att), n_layer(n_layer), n_embd(n_embd), layer_id(layer_id),
        n_head(n_head), head_size(head_size), head_size_divisor(8),
        x_r({1, 1, n_embd}, device_type), x_w({1, 1, n_embd}, device_type),
        x_k({1, 1, n_embd}, device_type), x_v({1, 1, n_embd}, device_type),
        x_a({1, 1, n_embd}, device_type), x_g({1, 1, n_embd}, device_type),
        w0({1, 1, n_embd}, device_type), w1({n_embd, decay_lora}, device_type), w2({decay_lora, n_embd}, device_type),
        a0({1, 1, n_embd}, device_type), a1({n_embd, aaa_lora}, device_type), a2({aaa_lora, n_embd}, device_type),
        v0({1, 1, n_embd}, device_type), v1({n_embd, mv_lora}, device_type), v2({mv_lora, n_embd}, device_type),
        g1({n_embd, gate_lora}, device_type), g2({gate_lora, n_embd}, device_type),
        k_k({1, 1, n_embd}, device_type), k_a({1, 1, n_embd}, device_type),
        r_k({n_head, head_size}, device_type),
        receptance(n_embd, n_embd, false, device_type),
        key(n_embd, n_embd, false, device_type),
        value(n_embd, n_embd, false, device_type),
        output(n_embd, n_embd, false, device_type),
        ln_x(n_head, n_embd, 64e-5, device_type),
        shift_state({1, n_embd}, device_type),
        wkvstate({1, n_head, head_size, head_size}, device_type),
        Module<
            Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>, Tensor<T, 3>,
            Tensor<T, 3>, Tensor<T, 2>, Tensor<T, 2>,
            Tensor<T, 3>, Tensor<T, 2>, Tensor<T, 2>,
            Tensor<T, 3>, Tensor<T, 2>, Tensor<T, 2>,
            Tensor<T, 2>, Tensor<T, 2>,
            Tensor<T, 3>, Tensor<T, 3>,
            Tensor<T, 2>,
            Linear<T>, Linear<T>, Linear<T>, Linear<T>,
            GroupNorm<T>
        >(
            Submodule<Tensor<T, 3>>(x_r, "x_r"), Submodule<Tensor<T, 3>>(x_w, "x_w"), Submodule<Tensor<T, 3>>(x_k, "x_k"), 
            Submodule<Tensor<T, 3>>(x_v, "x_v"), Submodule<Tensor<T, 3>>(x_a, "x_a"), Submodule<Tensor<T, 3>>(x_g, "x_g"),
            Submodule<Tensor<T, 3>>(w0, "w0"), Submodule<Tensor<T, 2>>(w1, "w1"), Submodule<Tensor<T, 2>>(w2, "w2"),
            Submodule<Tensor<T, 3>>(a0, "a0"), Submodule<Tensor<T, 2>>(a1, "a1"), Submodule<Tensor<T, 2>>(a2, "a2"),
            Submodule<Tensor<T, 3>>(v0, "v0"), Submodule<Tensor<T, 2>>(v1, "v1"), Submodule<Tensor<T, 2>>(v2, "v2"),
            Submodule<Tensor<T, 2>>(g1, "g1"), Submodule<Tensor<T, 2>>(g2, "g2"),
            Submodule<Tensor<T, 3>>(k_k, "k_k"), Submodule<Tensor<T, 3>>(k_a, "k_a"),
            Submodule<Tensor<T, 2>>(r_k, "r_k"),
            Submodule<Linear<T>>(receptance, "receptance"), Submodule<Linear<T>>(key, "key"), 
            Submodule<Linear<T>>(value, "value"), Submodule<Linear<T>>(output, "output"),
            Submodule<GroupNorm<T>>(ln_x, "ln_x")
        )
    {
        // Initialize parameters to zeros (will be loaded from checkpoint)
        for (size_t i = 0; i < x_r.total_size; i++) x_r.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < x_w.total_size; i++) x_w.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < x_k.total_size; i++) x_k.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < x_v.total_size; i++) x_v.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < x_a.total_size; i++) x_a.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < x_g.total_size; i++) x_g.flatget(i) = static_cast<T>(0.0);
        
        for (size_t i = 0; i < w0.total_size; i++) w0.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < a0.total_size; i++) a0.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < k_k.total_size; i++) k_k.flatget(i) = static_cast<T>(0.0);
        for (size_t i = 0; i < k_a.total_size; i++) k_a.flatget(i) = static_cast<T>(0.0);
        
        if (layer_id != 0) {
            for (size_t i = 0; i < v0.total_size; i++) v0.flatget(i) = static_cast<T>(0.0);
        }
        
        // Initialize state to zeros
        for (size_t i = 0; i < wkvstate.total_size; i++) {
            wkvstate.flatget(i) = static_cast<T>(0.0);
        }
    }
    
    std::pair<Tensor<T, 3>, Tensor<T, 3>> forward(const Tensor<T, 3>& x, const Tensor<T, 3>& v_first) {
        size_t B = x.shape[0];
        size_t seq_len = x.shape[1];
        size_t C = x.shape[2];
        size_t H = n_head;
        
        // Expand state for batch size if needed
        if (wkvstate.shape[0] != B) {
            Shape<4> new_state_shape(B, H, head_size, head_size);
            Tensor<T, 4> new_state(new_state_shape, x.device_type);
            
            // Broadcast the single batch state to all batches
            for (size_t b = 0; b < B; b++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t i = 0; i < head_size; i++) {
                        for (size_t j = 0; j < head_size; j++) {
                            new_state[SliceList<1>(b)][SliceList<1>(h)][SliceList<1>(i)][SliceList<1>(j)] = 
                                wkvstate[SliceList<1>(0)][SliceList<1>(h)][SliceList<1>(i)][SliceList<1>(j)];
                        }
                    }
                }
            }
            wkvstate = new_state;
        }
        
        // Perform time shift manually
        Tensor<T, 3> xx({B, seq_len, C}, x.device_type);
        
        // Expand shift state for batch size if needed
        if (shift_state.shape[0] != B) {
            Shape<2> new_shift_shape(B, C);
            Tensor<T, 2> new_shift_state(new_shift_shape, x.device_type);
            
            // Broadcast the single batch state to all batches
            for (size_t b = 0; b < B; b++) {
                for (size_t c = 0; c < C; c++) {
                    new_shift_state[SliceList<1>(b)][SliceList<1>(c)] = 
                        shift_state[SliceList<1>(0)][SliceList<1>(c)];
                }
            }
            shift_state = new_shift_state;
        }
        
        // Apply time shift
        for (size_t b = 0; b < B; b++) {
            for (size_t t = 0; t < seq_len; t++) {
                for (size_t c = 0; c < C; c++) {
                    if (t == 0) {
                        xx[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(c)] = 
                            shift_state[SliceList<1>(b)][SliceList<1>(c)];
                    } else {
                        xx[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(c)] = 
                            x[SliceList<1>(b)][SliceList<1>(t-1)][SliceList<1>(c)];
                    }
                }
            }
            // Update shift state with last token
            for (size_t c = 0; c < C; c++) {
                shift_state[SliceList<1>(b)][SliceList<1>(c)] = 
                    x[SliceList<1>(b)][SliceList<1>(seq_len-1)][SliceList<1>(c)];
            }
        }
        
        // Get shifted inputs
        auto xr = x + (xx * x_r);
        auto xw = x + (xx * x_w);
        auto xk = x + (xx * x_k);
        auto xv = x + (xx * x_v);
        auto xa = x + (xx * x_a);
        auto xg = x + (xx * x_g);
        
        // Apply linear transformations
        auto r = receptance.forward(xr);
        auto xw_2d = xw.template view<T, 2>({B * seq_len, C});
        auto w_temp = tanh_activation(matmul(xw_2d, w1));
        auto w_temp_2d = matmul(w_temp, w2);
        auto w_temp_3d = w_temp_2d.template view<T, 3>({B, seq_len, C});
        auto w = w0 + w_temp_3d;
        auto k = key.forward(xk);
        auto v = value.forward(xv);
        
        // Compute attention coefficients
        auto xa_2d = xa.template view<T, 2>({B * seq_len, C});
        auto a_temp = matmul(xa_2d, a1);
        auto a_temp2_3d = matmul(a_temp, a2).template view<T, 3>({B, seq_len, C});
        auto a = sigmoid(a0 + a_temp2_3d);
        
        auto xg_2d = xg.template view<T, 2>({B * seq_len, C});
        auto g_temp = matmul(xg_2d, g1);
        auto g_temp_3d = matmul(g_temp, g2).template view<T, 3>({B, seq_len, C});
        auto g = sigmoid(g_temp_3d);
        
        // Normalize k
        auto kk = k * k_k;
        auto kk_2d = kk.template view<T, 2>({B * seq_len, H * head_size});
        auto kk_norm = normalize(kk_2d, -1, static_cast<T>(2.0));
        kk = kk_norm.template view<T, 3>({B, seq_len, C});
        // Create a tensor of ones
        Tensor<T, 3> ones({B, seq_len, C}, x.device_type);
        for (size_t i = 0; i < ones.total_size; i++) {
            ones.flatget(i) = static_cast<T>(1.0);
        }
        k = k * (ones + ((a - static_cast<T>(1.0)) * k_a));
        
        // Handle v_first for layers > 0
        Tensor<T, 3> v_out = v;
        if (layer_id != 0) {
            auto xv_2d = xv.template view<T, 2>({B * seq_len, C});
            auto v_mix_temp = matmul(xv_2d, v1);
            auto v_mix_3d = matmul(v_mix_temp, v2).template view<T, 3>({B, seq_len, C});
            auto v_mix = sigmoid(v0 + v_mix_3d);
            v_out = v + ((v_first - v) * v_mix);
        } else {
            v_out = v;
        }
        
        // Apply decay
        w = w0 + w;
        w = sigmoid(w) * static_cast<T>(-0.606531);
        
        // Apply RWKV7 operation
        auto [att_out, new_state] = RWKV7_OP(r, w, k, v_out, -kk, kk * a, head_size, true, &wkvstate);
        wkvstate = new_state;
        
        // Apply group normalization
        auto att_out_2d = att_out.template view<T, 2>({B * seq_len, C});
        auto x_norm_2d = ln_x.forward(att_out_2d);
        auto x_norm = x_norm_2d.template view<T, 3>({B, seq_len, C});
        
        // Add residual connection with r_k
        auto residual = Tensor<T, 3>({B, seq_len, C}, x.device_type);
        for (size_t b = 0; b < B; b++) {
            for (size_t t = 0; t < seq_len; t++) {
                for (size_t h = 0; h < H; h++) {
                    T sum = static_cast<T>(0);
                    for (size_t i = 0; i < head_size; i++) {
                        size_t idx = h * head_size + i;
                        sum += r[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)] * 
                               k[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)] * 
                               r_k[SliceList<1>(h)][SliceList<1>(i)];
                    }
                    for (size_t i = 0; i < head_size; i++) {
                        size_t idx = h * head_size + i;
                        residual[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)] = 
                            sum * v_out[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)];
                    }
                }
            }
        }
        
        x_norm = x_norm + residual;
        auto final_output = output.forward(x_norm * g);
        
        return std::make_pair(final_output, v_out);
    }
};

#endif // RWKV7_TIMEMIX_HPP
