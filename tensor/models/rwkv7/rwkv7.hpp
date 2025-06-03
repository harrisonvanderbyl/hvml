
#ifndef RWKV7_MODEL_HPP
#define RWKV7_MODEL_HPP

#include "module/base/module.hpp"
#include "module/embedding/embedding.hpp"
#include "module/normalization/normalization.hpp"
#include "module/linear/linear.hpp"
#include "models/rwkv7/block.hpp"
#include "models/rwkv7/timeshift/kernels.hpp"
#include "file_loaders/safetensors.hpp"
#include <vector>
#include <memory>

// Forward declaration
std::map<std::string, Tensor<void, -1>> loadSafeTensors(const std::string& path);

template <typename R = float>
struct TimeShift: Module<Tensor<R, 2>>
{
    public:
    Tensor<R,2> state;
    Tensor<R,1> buffer;
    TimeShift(int batch, int dim): state({batch, dim}), buffer(Shape<1>{dim}), Module<Tensor<R,2>>(Submodule<Tensor<R,2>>(state, "state")){

    };

    Tensor<R,3> forward(Tensor<R,3> x){
        
        if(x.device_type == kCPU){
            timeshift_cpu(x.data, state.data, buffer.data, x.shape[0], x.shape[1], x.shape[2]);
        }
        return x;
    }

};

// Model parameter identification function
struct ModelParams {
    size_t n_layer;
    size_t n_embd;
    size_t n_head;
    size_t head_size;
    size_t dim_ffn;
    size_t vocab_size;
    size_t decay_lora;
    size_t aaa_lora;
    size_t mv_lora;
    size_t gate_lora;
};

template <typename T = float>
ModelParams identifyModelParams(const std::map<std::string, Tensor<void, -1>>& state_dict) {
    ModelParams params;
    
    // Get vocab_size and n_embd from embedding weight
    auto emb_it = state_dict.find("emb.weight");
    if (emb_it != state_dict.end()) {
        params.vocab_size = emb_it->second.shape[0];
        params.n_embd = emb_it->second.shape[1];
    }
    
    // Get dim_ffn from first block's ffn value weight
    auto ffn_it = state_dict.find("blocks.0.ffn.value.weight");
    if (ffn_it != state_dict.end()) {
        params.dim_ffn = ffn_it->second.shape[1];
    }
    
    // Get n_head from first block's r_k
    auto rk_it = state_dict.find("blocks.0.att.r_k");
    if (rk_it != state_dict.end()) {
        params.n_head = rk_it->second.shape[0];
        params.head_size = params.n_embd / params.n_head;
    }
    
    // Count number of layers
    params.n_layer = 0;
    for (const auto& [key, tensor] : state_dict) {
        if (key.find("blocks.") == 0 && key.find(".att.r_k") != std::string::npos) {
            params.n_layer++;
        }
    }
    
    // Get LoRA dimensions
    auto w1_it = state_dict.find("blocks.0.att.w1");
    if (w1_it != state_dict.end()) {
        params.decay_lora = w1_it->second.shape[1];
    }
    
    auto a1_it = state_dict.find("blocks.0.att.a1");
    if (a1_it != state_dict.end()) {
        params.aaa_lora = a1_it->second.shape[1];
    }
    
    auto v1_it = state_dict.find("blocks.1.att.v1");
    if (v1_it != state_dict.end()) {
        params.mv_lora = v1_it->second.shape[1];
    }
    
    auto g1_it = state_dict.find("blocks.0.att.g1");
    if (g1_it != state_dict.end()) {
        params.gate_lora = g1_it->second.shape[1];
    }
    
    return params;
}

template <typename T = float>
struct RWKV7_Model : public Module<
    Embedding<T>, LayerNorm<T>, LayerNorm<T>, Linear<T>
>
{
    public:
    ModelParams params;
    
    Embedding<T> emb;
    std::vector<std::unique_ptr<RWKV7_Block<T>>> blocks;
    LayerNorm<T> ln_in;
    LayerNorm<T> ln_out;
    Linear<T> head;
    
    RWKV7_Model(
        const ModelParams& model_params,
        DeviceType device_type = DeviceType::kCPU
    ) : params(model_params),
        emb(params.vocab_size, params.n_embd, device_type),
        ln_in(params.n_embd, 1e-5, device_type),
        ln_out(params.n_embd, 1e-5, device_type),
        head(params.n_embd, params.vocab_size, false, device_type),
        Module<Embedding<T>, LayerNorm<T>, LayerNorm<T>, Linear<T>>(
            Submodule<Embedding<T>>(emb, "emb"), 
            Submodule<LayerNorm<T>>(ln_in, "ln_in"), 
            Submodule<LayerNorm<T>>(ln_out, "ln_out"), 
            Submodule<Linear<T>>(head, "head")
        )
    {
        // Create blocks
        for (size_t i = 0; i < params.n_layer; i++) {
            blocks.push_back(std::make_unique<RWKV7_Block<T>>(
                i, params.n_layer, params.n_embd, params.n_head, params.head_size,
                params.n_embd, params.dim_ffn, params.decay_lora, params.aaa_lora,
                params.mv_lora, params.gate_lora, device_type
            ));
        }
    }
    
    // Constructor from model file
    RWKV7_Model(
        const std::string& model_path,
        DeviceType device_type = DeviceType::kCPU
    ) : RWKV7_Model(identifyModelParams<T>(loadSafeTensors(model_path)), device_type)
    {
        loadStateDict(loadSafeTensors(model_path));
    }
    
    void loadStateDict(const std::map<std::string, Tensor<void, -1>>& state_dict) {
        // Load embedding
        auto emb_it = state_dict.find("emb.weight");
        if (emb_it != state_dict.end()) {
            emb.weight = emb_it->second;
        }
        
        // Load layer norms
        auto ln_in_weight_it = state_dict.find("ln_in.weight");
        auto ln_in_bias_it = state_dict.find("ln_in.bias");
        if (ln_in_weight_it != state_dict.end()) ln_in.weight = ln_in_weight_it->second;
        if (ln_in_bias_it != state_dict.end()) ln_in.bias = ln_in_bias_it->second;
        
        auto ln_out_weight_it = state_dict.find("ln_out.weight");
        auto ln_out_bias_it = state_dict.find("ln_out.bias");
        if (ln_out_weight_it != state_dict.end()) ln_out.weight = ln_out_weight_it->second;
        if (ln_out_bias_it != state_dict.end()) ln_out.bias = ln_out_bias_it->second;
        
        // Load head
        auto head_it = state_dict.find("head.weight");
        if (head_it != state_dict.end()) {
            head.weight = head_it->second;
        }
        
        // Load blocks
        for (size_t i = 0; i < params.n_layer; i++) {
            std::string prefix = "blocks." + std::to_string(i) + ".";
            
            // Load layer norms
            auto ln1_weight_it = state_dict.find(prefix + "ln1.weight");
            auto ln1_bias_it = state_dict.find(prefix + "ln1.bias");
            if (ln1_weight_it != state_dict.end()) blocks[i]->ln1.weight = ln1_weight_it->second;
            if (ln1_bias_it != state_dict.end()) blocks[i]->ln1.bias = ln1_bias_it->second;
            
            auto ln2_weight_it = state_dict.find(prefix + "ln2.weight");
            auto ln2_bias_it = state_dict.find(prefix + "ln2.bias");
            if (ln2_weight_it != state_dict.end()) blocks[i]->ln2.weight = ln2_weight_it->second;
            if (ln2_bias_it != state_dict.end()) blocks[i]->ln2.bias = ln2_bias_it->second;
            
            // Load attention parameters
            std::string att_prefix = prefix + "att.";
            auto load_param = [&](const std::string& name, auto& param) {
                auto it = state_dict.find(att_prefix + name);
                if (it != state_dict.end()) {
                    param = it->second;
                }
            };
            
            load_param("x_r", blocks[i]->att.x_r);
            load_param("x_w", blocks[i]->att.x_w);
            load_param("x_k", blocks[i]->att.x_k);
            load_param("x_v", blocks[i]->att.x_v);
            load_param("x_a", blocks[i]->att.x_a);
            load_param("x_g", blocks[i]->att.x_g);
            load_param("w0", blocks[i]->att.w0);
            load_param("w1", blocks[i]->att.w1);
            load_param("w2", blocks[i]->att.w2);
            load_param("a0", blocks[i]->att.a0);
            load_param("a1", blocks[i]->att.a1);
            load_param("a2", blocks[i]->att.a2);
            load_param("k_k", blocks[i]->att.k_k);
            load_param("k_a", blocks[i]->att.k_a);
            load_param("r_k", blocks[i]->att.r_k);
            
            if (i != 0) {
                load_param("v0", blocks[i]->att.v0);
                load_param("v1", blocks[i]->att.v1);
                load_param("v2", blocks[i]->att.v2);
            }
            
            load_param("g1", blocks[i]->att.g1);
            load_param("g2", blocks[i]->att.g2);
            
            // Load linear layers
            load_param("receptance.weight", blocks[i]->att.receptance.weight);
            load_param("key.weight", blocks[i]->att.key.weight);
            load_param("value.weight", blocks[i]->att.value.weight);
            load_param("output.weight", blocks[i]->att.output.weight);
            
            // Load group norm
            load_param("ln_x.weight", blocks[i]->att.ln_x.weight);
            load_param("ln_x.bias", blocks[i]->att.ln_x.bias);
            
            // Load FFN parameters
            std::string ffn_prefix = prefix + "ffn.";
            auto load_ffn_param = [&](const std::string& name, auto& param) {
                auto it = state_dict.find(ffn_prefix + name);
                if (it != state_dict.end()) {
                    param = it->second;
                }
            };
            
            load_ffn_param("x_k", blocks[i]->ffn.x_k);
            load_ffn_param("key.weight", blocks[i]->ffn.key.weight);
            load_ffn_param("value.weight", blocks[i]->ffn.value.weight);
        }
    }
    
    Tensor<T, 3> forward(const Tensor<int, 2>& idx) {
        // Embedding
        auto x = emb.forward(idx);
        
        // Input layer norm
        x = ln_in.forward(x);
        
        // Process through blocks
        Tensor<T, 3> v_first = x;  // Initialize v_first for first layer
        for (size_t i = 0; i < params.n_layer; i++) {
            auto [x_out, v_first_out] = blocks[i]->forward(std::make_pair(x, v_first));
            x = x_out;
            v_first = v_first_out;
        }
        
        // Output layer norm
        x = ln_out.forward(x);
        
        // Head projection
        auto logits = head.forward(x);
        
        return logits;
    }
    
    // Convenience method for single token input
    Tensor<T, 2> forward(const Tensor<int, 1>& idx) {
        Shape<2> input_shape(1, idx.shape[0]);
        Tensor<int, 2> input_2d(input_shape, idx.device_type);
        
        // Copy data from 1D to 2D tensor
        for (size_t i = 0; i < idx.shape[0]; i++) {
            input_2d[SliceList<1>(0)][SliceList<1>(i)] = idx[SliceList<1>(i)];
        }
        
        auto output_3d = forward(input_2d);
        return output_3d[SliceList<1>(0)];  // Remove batch dimension
    }
};

#endif // RWKV7_MODEL_HPP

// class RWKV_ChannelMix(nn.Module):
    
//     def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
//         super().__init__()

//         self.x_k = nn.Parameter(torch.zeros(1,1,n_embd))

//         self.key = torch.nn.Linear(n_embd, dim_ffn, bias=False)
//         self.value = torch.nn.Linear(dim_ffn, n_embd, bias=False)
//         self.lastlayer = layer_id == n_layer - 1
        
//         self.shift = TimeShift()
        
//     def forward(self, x):
 
    
//         xx = self.shift(x)
        
//         k = x + xx * self.x_k
//         k = torch.relu(self.key(k)) ** 2
//         return self.value(k)
    
// class RWKV_TimeMix(nn.Module):
//     def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, decay_lora, aaa_lora, mv_lora, gate_lora):
//         super().__init__()
        
//         self.dim_att = dim_att
//         self.n_layer = n_layer
//         self.n_embd = n_embd
//         self.layer_id = layer_id

//         self.n_head = n_head
//         self.head_size = head_size
//         self.head_size_divisor = 8


//         H = self.n_head
//         N = self.head_size
//         C = self.n_embd

//         self.x_r = nn.Parameter(torch.empty(1,1,C))
//         self.x_w = nn.Parameter(torch.empty(1,1,C))
//         self.x_k = nn.Parameter(torch.empty(1,1,C))
//         self.x_v = nn.Parameter(torch.empty(1,1,C))
//         self.x_a = nn.Parameter(torch.empty(1,1,C))
//         self.x_g = nn.Parameter(torch.empty(1,1,C))

//         self.w0 = nn.Parameter(torch.empty(1,1,C)) 
//         self.w1 = nn.Parameter(torch.empty(C, decay_lora)) 
//         self.w2 = nn.Parameter(torch.empty(decay_lora, C))  

//         self.a0 = nn.Parameter(torch.empty(1,1,C))
//         self.a1 = nn.Parameter(torch.empty(C, aaa_lora))
//         self.a2 = nn.Parameter(torch.empty(aaa_lora, C))

//         self.v0 = nn.Parameter(torch.empty(1,1,C)) if layer_id != 0 else None
//         self.v1 = nn.Parameter(torch.empty(C, mv_lora)) if layer_id != 0 else None
//         self.v2 = nn.Parameter(torch.empty(mv_lora, C)) if layer_id != 0 else None

//         self.g1 = nn.Parameter(torch.empty(C, gate_lora))
//         self.g2 = nn.Parameter(torch.empty(gate_lora, C))

//         self.k_k = nn.Parameter(torch.empty(1,1,C))
//         self.k_a = nn.Parameter(torch.empty(1,1,C))
//         self.r_k = nn.Parameter(torch.empty(H,N))

//         self.receptance = nn.Linear(C, C, bias=False)
//         self.key = nn.Linear(C, C, bias=False)
//         self.value = nn.Linear(C, C, bias=False)
//         self.output = nn.Linear(C, C, bias=False)
//         self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

//         self.shift = TimeShift()
        
//         self.wkvstate = None
        
//     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        
//         self.wkvstate = state_dict.pop(prefix+"wkvstate") if prefix+"wkvstate" in state_dict else None
        
//         a = super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
//         return a
    
    

//     def forward(self, x, v_first):
        
//         # Get the x sizing
//         B, T, C = x.shape
//         H = self.r_k.shape[0]

//         if self.training:
//             last_state_wkv = self.wkvstate.repeat(B,1,1,1)
//         else:
//             last_state_wkv = self.wkvstate.to(torch.float)

//         K = last_state_wkv.shape[-2]
        
//         # Perform the tokenshift
//         xx = self.shift(x)
            

//         # Get the xk, xv, xr, xg, and rkvg
//         xr = x + xx * self.x_r
//         xw = x + xx * self.x_w
//         xk = x + xx * self.x_k
//         xv = x + xx * self.x_v
//         xa = x + xx * self.x_a
//         xg = x + xx * self.x_g

//         r = self.receptance(xr)
//         w = torch.tanh(xw @ self.w1) @ self.w2
//         k = self.key(xk)
//         v = self.value(xv) 
//         a = torch.sigmoid(self.a0 + ((xa @ self.a1) @ self.a2)) # a is "in-context learning rate"
//         g = torch.sigmoid(xg @ self.g1) @ self.g2
//         kk = k * self.k_k
//         kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
//         k = k * (1 + (a-1) * self.k_a)

        
//         if self.v1 is not None:
//             v = v + ((v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) )
//         else:
//             v_first = v
        
//         # cc = IdentityFwdAllReduceBwd.apply(cc)
//         w = self.w0 + w 
//         w = (-0.606531 * torch.sigmoid(w))
        
        
//         # aa = IdentityFwdAllReduceBwd.apply(aa)
       
//         # gg = IdentityFwdAllReduceBwd.apply(gg)

//         CC = r.size(-1)
//         N = self.r_k.size(-1)
        

//         x,stateout = RWKV7_OP(r, w, k, v, -kk, kk*a, CC//H, training=self.training, initial=last_state_wkv)
//         if not self.training:
//             self.wkvstate = stateout.detach()

//         x = self.ln_x(x.view(B * T, CC)).view(B, T, CC)
//         HS = self.head_size
        
//         x = x + ((r.view(B,T,-1,HS)*k.view(B,T,-1,HS)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,-1,HS)).view(B,T,-1)
//         x = self.output(x * g)

//         # Return the logits and the state
//         return x, v_first

// class Block(nn.Module):

//     def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, dim_ffn, decay_lora, aaa_lora, mv_lora, gate_lora):
//         super().__init__()
//         self.layer_id = layer_id

//         self.ln1 = nn.LayerNorm(n_embd)
//         self.ln2 = nn.LayerNorm(n_embd)

//         self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att, decay_lora, aaa_lora, mv_lora, gate_lora)
//         self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)
        
    
//         # Setup droupout at block level

//     def forward(self, x):
//         x, v_first = x[0], x[1]
//         xa, v_first = self.att(self.ln1(x),v_first)
        
//         x = x + xa
//         x = x + self.ffn(self.ln2(x))
                
//         return [x, v_first]

// def identifyModelParams(file):
    
//     vocab_size, n_embd = file["emb.weight"].shape
    
//     dim_ffn = file[f"blocks.0.ffn.value.weight"].shape[1]
  
//     n_head = file[f"blocks.0.att.r_k"].shape[0]
    
//     headsize = n_embd // n_head
    
//     n_layer = len([x for x in file.keys() if x.startswith("blocks.") and x.endswith(".att.r_k")])
    
//     decay_lora = file[f"blocks.0.att.w1"].shape[1]

//     aaa_lora = file[f"blocks.0.att.a1"].shape[1]

//     mv_lora = file[f"blocks.1.att.v1"].shape[1] 

//     gate_lora = file[f"blocks.0.att.g1"].shape[1]

//     return n_layer, n_embd, n_head, headsize, dim_ffn, vocab_size, decay_lora, aaa_lora, mv_lora, gate_lora
    



// class v5tune( torch.nn.Module):
//     def __init__(self, model_path, device="cuda"):        
//         super(v5tune, self).__init__()
        
//         file = torch.load(model_path, map_location=device)
        
//         self.n_layer, self.n_embd, self.n_head, self.head_size, self.dim_ffn, self.vocab_size, self.decay_lora, self.aaa_lora, self.mv_lora, self.gate_lora = identifyModelParams(file)
        
//         self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        
//         self.blocks = nn.Sequential(*[
//             Block(i, self.n_layer, self.n_embd, self.n_head, self.head_size, self.n_embd, self.dim_ffn, self.decay_lora, self.aaa_lora, self.mv_lora, self.gate_lora) for i in range(self.n_layer)
//         ])
        
//         self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
//         file["ln_in.weight"] = file.pop("blocks.0.ln0.weight")
//         file["ln_in.bias"] = file.pop("blocks.0.ln0.bias")
        
//         self.ln_in = nn.LayerNorm(self.n_embd)
//         self.ln_out = nn.LayerNorm(self.n_embd)
        
//         self.load_state_dict(file)
        
//         self.requires_grad_(False)
        
        
//     def to_device(self, tensor):
//         return torch.tensor(tensor, device=self.emb.weight.device)

//     def forward(self, idx):
//         x = self.to_device(idx)
//         x = self.emb(x)
//         x = self.ln_in(x)
//         x, _ = self.blocks((x, None))
//         x = self.ln_out(x)
//         x = self.head(x)
//         return x
