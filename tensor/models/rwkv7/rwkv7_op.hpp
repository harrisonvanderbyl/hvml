#ifndef RWKV7_OP_HPP
#define RWKV7_OP_HPP

#include "tensor.hpp"
#include <cmath>
#include <algorithm>

// RWKV7 core operation implementation
// Based on the algorithm:
// ab = a*b + w (ab are outer product, w is diag)
// kv = k*v  ( kv is outer product)
// for t in range(T):
//     state = ab[:, t, :, :] @ state + kv[:, t, :, :]
//     out[:, t, :, :] = r[:, t, :, :] @ state
template <typename T>
std::pair<Tensor<T, 3>, Tensor<T, 4>> rwkv7_op(
    const Tensor<T, 3>& r,      // receptance [B, T, C]
    const Tensor<T, 3>& w,      // decay [B, T, C]
    const Tensor<T, 3>& k,      // key [B, T, C]
    const Tensor<T, 3>& v,      // value [B, T, C]
    const Tensor<T, 3>& u,      // bonus [B, T, C]
    const Tensor<T, 3>& s,      // state bonus [B, T, C]
    size_t head_size,
    bool training = true,
    const Tensor<T, 4>* initial_state = nullptr
) {
    size_t B = r.shape[0];  // batch size
    size_t seq_len = r.shape[1];  // sequence length
    size_t C = r.shape[2];  // channels
    size_t H = C / head_size;  // number of heads
    
    assert(C % head_size == 0);
    assert(w.shape[0] == B && w.shape[1] == seq_len && w.shape[2] == C);
    assert(k.shape[0] == B && k.shape[1] == seq_len && k.shape[2] == C);
    assert(v.shape[0] == B && v.shape[1] == seq_len && v.shape[2] == C);
    assert(u.shape[0] == B && u.shape[1] == seq_len && u.shape[2] == C);
    assert(s.shape[0] == B && s.shape[1] == seq_len && s.shape[2] == C);
    
    // Output tensor
    Tensor<T, 3> output({B, seq_len, C}, r.device_type);
    
    // State tensor [B, H, head_size, head_size]
    Tensor<T, 4> state({B, H, head_size, head_size}, r.device_type);
    
    // Initialize state
    if (initial_state != nullptr) {
        assert(initial_state->shape[0] == B && initial_state->shape[1] == H && 
               initial_state->shape[2] == head_size && initial_state->shape[3] == head_size);
        state = *initial_state;
    } else {
        // Initialize to zero
        for (size_t i = 0; i < state.total_size; i++) {
            state.flatget(i) = static_cast<T>(0);
        }
    }
    
    // Process each batch
    for (size_t b = 0; b < B; b++) {
        // Process each time step
        for (size_t t = 0; t < seq_len; t++) {
            // Process each head
            for (size_t h = 0; h < H; h++) {
                size_t head_offset = h * head_size;
                
                // Extract head-specific vectors for this timestep
                std::vector<T> r_h(head_size), w_h(head_size), k_h(head_size), 
                              v_h(head_size), u_h(head_size), s_h(head_size);
                
                for (size_t i = 0; i < head_size; i++) {
                    size_t idx = head_offset + i;
                    r_h[i] = r[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)];
                    w_h[i] = w[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)];
                    k_h[i] = k[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)];
                    v_h[i] = v[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)];
                    u_h[i] = u[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)];
                    s_h[i] = s[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)];
                }
                
                // Compute ab = a*b + w (where a=s, b=u, w is diagonal)
                // ab is [head_size, head_size] matrix
                std::vector<std::vector<T>> ab(head_size, std::vector<T>(head_size, static_cast<T>(0)));
                for (size_t i = 0; i < head_size; i++) {
                    for (size_t j = 0; j < head_size; j++) {
                        ab[i][j] = s_h[i] * u_h[j];
                        if (i == j) {  // diagonal term
                            ab[i][j] += w_h[i];
                        }
                    }
                }
                
                // Compute kv = k*v (outer product)
                // kv is [head_size, head_size] matrix
                std::vector<std::vector<T>> kv(head_size, std::vector<T>(head_size, static_cast<T>(0)));
                for (size_t i = 0; i < head_size; i++) {
                    for (size_t j = 0; j < head_size; j++) {
                        kv[i][j] = k_h[i] * v_h[j];
                    }
                }
                
                // Update state: state = ab @ state + kv
                std::vector<std::vector<T>> new_state(head_size, std::vector<T>(head_size, static_cast<T>(0)));
                
                // ab @ state
                for (size_t i = 0; i < head_size; i++) {
                    for (size_t j = 0; j < head_size; j++) {
                        for (size_t k_idx = 0; k_idx < head_size; k_idx++) {
                            T current_state = state[SliceList<1>(b)][SliceList<1>(h)][SliceList<1>(i)][SliceList<1>(k_idx)];
                            new_state[i][j] += ab[i][k_idx] * current_state;
                        }
                        // Add kv
                        new_state[i][j] += kv[i][j];
                    }
                }
                
                // Store updated state
                for (size_t i = 0; i < head_size; i++) {
                    for (size_t j = 0; j < head_size; j++) {
                        state[SliceList<1>(b)][SliceList<1>(h)][SliceList<1>(i)][SliceList<1>(j)] = new_state[i][j];
                    }
                }
                
                // Compute output: out = r @ state
                std::vector<T> out_h(head_size, static_cast<T>(0));
                for (size_t i = 0; i < head_size; i++) {
                    for (size_t j = 0; j < head_size; j++) {
                        out_h[i] += r_h[j] * new_state[j][i];
                    }
                }
                
                // Store output
                for (size_t i = 0; i < head_size; i++) {
                    size_t idx = head_offset + i;
                    output[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(idx)] = out_h[i];
                }
            }
        }
    }
    
    return std::make_pair(output, state);
}

// Simplified RWKV7_OP function that matches the Python interface
template <typename T>
std::pair<Tensor<T, 3>, Tensor<T, 4>> RWKV7_OP(
    const Tensor<T, 3>& r,
    const Tensor<T, 3>& w,
    const Tensor<T, 3>& k,
    const Tensor<T, 3>& v,
    const Tensor<T, 3>& u,
    const Tensor<T, 3>& s,
    size_t head_size,
    bool training = true,
    const Tensor<T, 4>* initial = nullptr
) {
    return rwkv7_op(r, w, k, v, u, s, head_size, training, initial);
}

#endif // RWKV7_OP_HPP
