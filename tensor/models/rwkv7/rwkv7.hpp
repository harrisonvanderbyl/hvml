#include "module/linear/linear.hpp"
#include <models/rwkv7/timeshift/kernels.hpp>
#include "module/layernorm/layernorm.hpp"


template <typename R = float>
struct TimeShift: Module<Tensor<R, 2>>
{
    public:
    Tensor<R,2> state;
    Tensor<R,1> buffer;
    TimeShift(int batch, int dim): state({batch, dim}), buffer(Shape<1>{dim}), Module({state, "state"}){

    };

    Tensor<R> forward(Tensor<R> x){
        
        if(x.device_type == kCPU){
            timeshift_cpu(x.data, state.data, buffer.data, x.shape[0], x.shape[1], x.shape[2]);
        }else if(x.device_type == kCUDA){
            timeshift_cuda(x.data, state.data, buffer.data, x.shape[0], x.shape[1], x.shape[2]);
        }else if(x.device_type == kHIP){
            timeshift_hip(x.data, state.data, buffer.data, x.shape[0], x.shape[1], x.shape[2]);
        }
        return x;
    }

};


// class RWKV_ChannelMix(nn.Module):
    
//     def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
//         super().__init__()

//         self.x_k = nn.Parameter(torch.zeros(1,1,n_embd))

//         self.key = torch.nn.Linear(n_embd, dim_ffn, bias=False)
//         self.value = torch.nn.Linear(dim_ffn, n_embd, bias=False)
//         self.lastlayer = layer_id == n_layer - 1
        
//         self.shift = TimeShift()
        
//     def forward(self, x):
 


template <typename T = float>
struct FFN : public Module<TimeShift<T>,Linear<T>,Linear<T>>
{
    public:
    TimeShift<T> shift;
    Linear<T> key;
    Linear<T> value;
    
    FFN(
        size_t max_batch,
        size_t n_dim,
        size_t ffn_dim,
        DeviceType device_type = MemoryType::kDDR
    ):
    key(n_dim, ffn_dim, device_type), 
    value(ffn_dim, n_dim, device_type), 
    shift(max_batch,n_dim), 
    Module({shift, "shift"},{key, "key"}, {value, "value"})
    {
        
    }
     
    Tensor <T> forward(Tensor<T> input)
    {
        auto x = shift(input);
        auto r = ChainOperations<OperationRelu, OperationSqr>::run(key(x));
        // Apply ReLU squared activation
        return value(r);
    }
    
};
    
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

class RWKV_TimeMix : public Module<
    TimeShift<float>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,   
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,3>,
    Tensor<float,2>,
    Tensor<float,2>,
    Tensor<float,2>,
    Tensor<float,2>,
    Tensor<float,2>,
    Tensor<float,2>,
    Tensor<float,2>,
    Tensor<float,2>,
    Tensor<float,2>,
    Linear<float>,
    Linear<float>,
    Linear<float>,
    Linear<float>,
    LayerNorm<float>
    >
{
    public:
    // Define all parameters here
    Tensor<float,3> x_r;
    Tensor<float,3> x_w;
    Tensor<float,3> x_k;
    Tensor<float,3> x_v;
    Tensor<float,3> x_a;
    Tensor<float,3> x_g;
    Tensor<float,3> w0;
    Tensor<float,2> w1;
    Tensor<float,2> w2;
    Tensor<float,3> a0;
    Tensor<float,2> a1;
    Tensor<float,2> a2;
    Tensor<float,3> v0;
    Tensor<float,2> v1;
    Tensor<float,2> v2;
    Tensor<float,2> g1;
    Tensor<float,2> g2;
    Tensor<float,3> k_k;
    Tensor<float,3> k_a;
    Tensor<float,2> r_k;
    TimeShift<float> shift;
    Linear<float> receptance;
    Linear<float> key;
    Linear<float> value;
    Linear<float> output;
    LayerNorm<float> ln_x;


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
        DeviceType device_type = MemoryType::kDDR
    ):
    x_r({1, 1, n_embd}, device_type),
    x_w({1, 1, n_embd}, device_type),
    x_k({1, 1, n_embd}, device_type),
    x_v({1, 1, n_embd}, device_type),
    x_a({1, 1, n_embd}, device_type),
    x_g({1, 1, n_embd}, device_type),
    w0({1, 1, n_embd}, device_type),
    w1({n_embd, decay_lora}, device_type),
    w2({decay_lora, n_embd}, device_type),
    a0({1, 1, n_embd}, device_type),
    a1({n_embd, aaa_lora}, device_type),
    a2({aaa_lora, n_embd}, device_type),
    v0(layer_id != 0 ? Tensor<float,3>({1, 1, n_embd}, device_type) : Tensor<float,3>({0, 0, 0}, device_type)),
    v1(layer_id != 0 ? Tensor<float,2>({n_embd, mv_lora}, device_type) : Tensor<float,2>({0, 0}, device_type)),
    v2(layer_id != 0 ? Tensor<float,2>({mv_lora, n_embd}, device_type) : Tensor<float,2>({0, 0}, device_type)),
    g1({n_embd, gate_lora}, device_type),
    g2({gate_lora, n_embd}, device_type),
    k_k({1, 1, n_embd}, device_type),
    k_a({1, 1, n_embd}, device_type),
    r_k({n_head, head_size}, device_type),
    shift( /*max_batch=*/1024, n_embd), // Placeholder for max_batch
    receptance(n_embd, n_embd, device_type),
    key(n_embd, n_embd, device_type),
    value(n_embd, n_embd, device_type),
    output(n_embd, n_embd, device_type),
    ln_x(n_head, n_embd, 64e-5, device_type),
    Module(
        {shift, "shift"},
        {x_r, "x_r"},
        {x_w, "x_w"},
        {x_k, "x_k"},
        {x_v, "x_v"},
        {x_a, "x_a"},
        {x_g, "x_g"},
        {w0, "w0"},
        {a0, "a0"},
        {v0, "v0"},
        {g1, "g1"},
        {k_k, "k_k"},
        {k_a, "k_a"},
        {r_k, "r_k"},
        {w1, "w1"},
        {w2, "w2"},
        {a1, "a1"},
        {a2, "a2"},
        {v1, "v1"},
        {v2, "v2"},
        {g2, "g2"},
        {receptance, "receptance"},
        {key, "key"},
        {value, "value"},
        {output, "output"},
        {ln_x, "ln_x"}
    )
    {
        
    }
     
    Tensor<float> forward(Tensor<float> x, Tensor<float> v_first)
    {

        auto xx = shift(x);
        auto xr = x + xx * x_r;
        auto xw = x + xx * x_w;
        auto xk = x + xx * x_k;
        auto xv = x + xx * x_v;
        auto xa = x + xx * x_a;
        auto xg = x + xx * x_g;

        auto r = receptance.forward(xr);
        auto w = Linear(tanh(Linear(xw)(w1)))(w2);
        auto k = key.forward(xk);
        auto v = value.forward(xv);
        auto a = sigmoid(a0 + Linear((Linear(xa)(a1))(a2))); // a is "in-context learning rate"
        auto g = Linear(sigmoid(Linear(xg)(g1)))(g2);
        auto kk = k * k_k;
        
    }
};

class Block : public Module<LayerNorm, Layernorm, FFN<float>, RWKV_TimeMix>
{
    public:
    LayerNorm<float> ln1;
    LayerNorm<float> ln2;
    FFN<float> ffn;
    RWKV_TimeMix att;
    
    Block(
        size_t layer_id,
        size_t n_layer,
        size_t n_embd,
        size_t n_head,
        size_t head_size,
        size_t dim_att,
        size_t dim_ffn,
        size_t decay_lora,
        size_t aaa_lora,
        size_t mv_lora,
        size_t gate_lora,
        DeviceType device_type = MemoryType::kDDR
    ):
    ln1(n_embd, 64e-5, device_type),
    ln2(n_embd, 64e-5, device_type),
    ffn( /*max_batch=*/1024, n_embd, dim_ffn, device_type),
    att(
        layer_id,
        n_layer,
        n_embd,
        n_head,
        head_size,
        dim_att,
        decay_lora,
        aaa_lora,
        mv_lora,
        gate_lora,
        device_type
    ),
    // att(...),
    Module({ln1, "ln1"},{ln2, "ln2"},{ffn, "ffn"},{att, "att"})
    {
        
    }
     
    Tensor<float> forward(Tensor<float> x, Tensor<float> v_first)
    {
        
        auto xa = att({ln1.forward(x), v_first}) + x;      
        
        return ffn(ln2(xa)) + xa;
                
    }
    
};

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

struct Embedding : public Module<Tensor<float,2>>
{
    public:
    Tensor<float,2> weight;
    
    
    Embedding(
        size_t vocab_size,
        size_t n_embd,
        DeviceType device_type = MemoryType::kDDR
    ):weight({vocab_size,n_embd}, device_type), Module<Tensor<float,2>>({weight, "weight"})
    {
        
    }
     
    Tensor<float,3> forward(Tensor<long,2> input)
    {
        return weight.tensor_index(input);
    }
    
};

