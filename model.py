#dedicated to the public domain for the glory of god.
#baruch adonai el shaddai 
#Eloheinu shebashamayim yached shimcha v'kayeim malchutecha tamid umloch aleinu le'olam va'ed 
#2026 joshuah.rainstar@gmail.com

#Version 2.1 EpistemicGPT
#current notes:
#your choice on whether to key the carving directions in the subspaceunembed to n_layer or a fixed budget\
#and whether to run them all in parallel or sequential
#your choice on lelu gelu or some other nonlinearity
#figure out some kind of ste alpha decay schedule or dont use


'''
Train all MLPS of form(linear,LELU,linear) with the following LR
width LR
4 2.0667168855e-01
8 1.0333584427e-01
16 5.1667922137e-02
32 2.5833961069e-02
64 1.2916980534e-02
128 6.4584902671e-03
192 4.3056601781e-03
256 3.2292451336e-03
384 2.1528300890e-03
512 1.6146225668e-03
1024 8.0731128339e-04
2048 4.0365564170e-04
4096 2.0182782085e-04
8192 1.0091391042e-04
16384 5.0456955212e-05
Generalization error bounds for two-layer
neural networks with Lipschitz loss function
Jiang Yu Nguwi∗ Nicolas Privault
 1) activation/module-scale Lipschitz factor for the RMSNorm-guarded MLP
       L_eff ~= 1.0998393201 * sqrt(dim)

    2) expected top Hessian eigenvalue for the simplified linear+squared-loss model
       with Gaussian batch inputs centered coordinatewise at 1.09984:
       lambda_max = sigma^2 + dim * (1.09984)^2

    3) hypothetical gradient-descent learning-rate edge and a conservative mean
       lr_edge = 2 / lambda_max
       lr = lr_edge / 2

further estimations, calculations suggest that the LR on attention should be the MLP but divided by the highest width of the internal representation.
in an OST model, its lr/3.

for embedding layer, LR is allowed to be quite large.
(1 / log(sqrt(max(dim, vocab))))*0.5

for unembedding(assume we will parallelize, not accumulate sequential): 
lr_unembed = (1 / (sqrt(n_slices + 1) * log(max(dim, vocab))))*0.5

Scheduling should decay the entire model gradually to a lower bound.  that lowest possible bound that is useful to train is quite simply 1 / (1099.84 * sqrt(max(dim,vocab)).
training below this may not be effective or useful. 

any change to nonlinearities, any scaling operations, any use of softmax objectively will change and invalidate this prospective learning rate schedule. 


'''
import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_boundary_ste_hook(alpha=1.0):
    def hook(module, grad_input, grad_output):
        if not grad_input or not grad_output:
            return None

        gi = grad_input[0]
        go = grad_output[0]

        if gi is None or go is None:
            return None
        if gi.shape != go.shape:
            return None

        new_gi = alpha * go + (1.0 - alpha) * gi
        return (new_gi,) + tuple(grad_input[1:])
    return hook

def make_texture_hook(ema_decay=0.95, roughness_decay=0.95, 
                      smooth_floor=0.3, rough_focus=0.8, eps=1e-8):
    """
    External memory per linear layer, living in the closure.
    
    State:
      direction: EMA of gradient (unnormalized accumulator, normalized on use)
      variance:  EMA of squared deviation from direction (roughness)
      grad_sq:   EMA of squared gradient norm (for normalizing variance)
      scale:     the integrated scalar output
    """
    state = {
        "direction": None,   # running mean gradient, same shape as grad
        "variance": None,    # running mean squared deviation from direction
        "grad_sq_ema": None, # running mean of ||g||^2 for relative roughness
        "step": 0,
    }
    
    def hook(module, grad_input, grad_output):
        g = grad_input[0]
        if g is None:
            return None
        
        with torch.no_grad():
            flat = g.reshape(g.shape[0], -1)  # (batch, features) or just (features,)
            
            # initialize state on first call
            if state["direction"] is None:
                state["direction"] = torch.zeros_like(flat)
                state["variance"] = torch.zeros(flat.shape[0], device=flat.device)
                state["grad_sq_ema"] = torch.zeros(flat.shape[0], device=flat.device)
            
            state["step"] += 1
            a = ema_decay
            ar = roughness_decay
            
            # bias correction weight
            bc = 1.0 / (1.0 - a ** state["step"])
            
            # update direction EMA
            state["direction"].mul_(a).add_(flat, alpha=1 - a)
            dir_corrected = state["direction"] * bc
            
            # normalize direction to unit vector per sample
            dir_norm = dir_corrected.norm(dim=-1, keepdim=True).clamp(min=eps)
            d_hat = dir_corrected / dir_norm
            
            # deviation: how far is current gradient from principal direction?
            proj_scalar = (flat * d_hat).sum(dim=-1, keepdim=True)
            orthogonal = flat - proj_scalar * d_hat
            dev_sq = (orthogonal ** 2).sum(dim=-1)
            grad_sq = (flat ** 2).sum(dim=-1)
            
            # update roughness and magnitude EMAs
            state["variance"].mul_(ar).add_(dev_sq, alpha=1 - ar)
            state["grad_sq_ema"].mul_(ar).add_(grad_sq, alpha=1 - ar)
            
            # relative roughness: what fraction of gradient energy is off-direction?
            # 0 = perfectly aligned with principal direction (smooth)
            # 1 = entirely orthogonal (rough)
            roughness = state["variance"] / state["grad_sq_ema"].clamp(min=eps)
            roughness = roughness.clamp(0, 1)
            
            # --- scaling logic ---
            # smoothness = 1 - roughness
            # when smooth: scale down overall (don't overcorrect)
            # when rough: focus on direction, amplify along it
            
            smoothness = 1.0 - roughness  # per-sample
            
            # overall magnitude scale: high when rough, drops to floor when smooth
            magnitude = smooth_floor + (1.0 - smooth_floor) * roughness
            magnitude = magnitude.unsqueeze(-1)
            
            # focus: how much to concentrate onto principal direction
            # rough -> high focus, smooth -> low focus (let full gradient through at reduced scale)
            focus = rough_focus * roughness.unsqueeze(-1)
            
            # decompose gradient
            g_parallel = proj_scalar * d_hat
            g_ortho = orthogonal
            
            # blend: ortho component gets attenuated by focus
            g_new = magnitude * (g_parallel + (1.0 - focus) * g_ortho)
        
        # reshape back and return
        out = list(grad_input)
        out[0] = g_new.reshape(g.shape)
        return tuple(out)
    
    return hook, state

class LELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = math.pi / math.sqrt(3.0) #logistic CDF matched scale beats gelu and is less expensive

    def forward(self, x):
        return x * torch.sigmoid(self.scale * x)



class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        w_max = torch.pi / 2 #highest frequency- one rotation in 4 tokens
        w_min = torch.pi / (max_len) #this sets the lowest scale to that sufficient to resolve needle at depth.
        #most effectively, set your context size to more than you ever intend to handle
        B = 10
        setfreqs_hi = torch.logspace(
                    start=math.log(w_min,B),
                    end=math.log(w_max,B),
                    steps=(dim // 4),
                    base=B
                )


        setfreqs_lo = w_max - torch.logspace(
            start=math.log(w_min,B),
            end=math.log(w_max,B),
            steps=(dim // 4) + 1,
            base=B
        )[:-1] # Drop last to match size and allow a smooth continuation

                # Combine
        setfreqs = torch.cat((setfreqs_hi, setfreqs_lo))
        inv_freq, _ = torch.sort(setfreqs,descending=True) #this give us a sigmoidal distribution of frequencies.
        #many large, many small, few near the midpoint.

        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def get_embeddings(self, positions, device):
        positions = positions.clamp(0, self.max_len - 1).long()
        cos = F.embedding(positions, self.cos_cached).unsqueeze(0).unsqueeze(0) # [1, 1, T, D]
        sin = F.embedding(positions, self.sin_cached).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, x, positions=None):
        # x: (B, H, T, D) where D = self.dim (head_dim)
        if positions is None:
            T = x.shape[-2]
            positions = torch.arange(T, device=x.device)

        cos, sin = self.get_embeddings(positions, x.device)
        # cos, sin: [1, 1, T, D//2]

        D = x.shape[-1]
        mid = D // 2
        h1 = x[..., :mid]   # first half of dimensions
        h2 = x[..., mid:]   # second half of dimensions

        # Adjacent pairing within each half:
        # h1 pairs: (h1[0],h1[1]), (h1[2],h1[3]), ...
        # h2 pairs: (h2[0],h2[1]), (h2[2],h2[3]), ...
        #
        # h1 gets sign-flipped convention (sin term adds instead of subtracts)
        # h2 gets standard convention
        #
        # This is achieved by negating the "b" element of each pair in h1
        # before feeding into the standard rotation formula.

        # For h1: negate odd-indexed elements, apply rotation, un-negate
        # Equivalent to: the kernel contribution for h1 pairs becomes
        #   S * cos(w*d) + A * sin(w*d)
        # instead of
        #   S * cos(w*d) - A * sin(w*d)

        # Extract paired elements for h1
        h1_a = h1[..., 0::2]   # even indices (the "real" part)
        h1_b = h1[..., 1::2]   # odd indices (the "imaginary" part)

        # Extract paired elements for h2
        h2_a = h2[..., 0::2]
        h2_b = h2[..., 1::2]

        # Split cos/sin for the two halves' frequency ranges
        # First D//4 frequencies go to h1, last D//4 to h2
        n_pairs_per_half = mid // 2  # D//4
        cos1 = cos[..., :n_pairs_per_half]
        sin1 = sin[..., :n_pairs_per_half]
        cos2 = cos[..., n_pairs_per_half:]
        sin2 = sin[..., n_pairs_per_half:]

        # H1: flipped convention (negate b before rotation)
        # Standard rotation on (a, -b):
        #   out_a = a * cos - (-b) * sin = a * cos + b * sin
        #   out_b_neg = a * sin + (-b) * cos = a * sin - b * cos
        # Then un-negate b: out_b = -(a * sin - b * cos) = -a * sin + b * cos
        out_h1_a = h1_a * cos1 + h1_b * sin1
        out_h1_b = -h1_a * sin1 + h1_b * cos1

        # H2: standard convention
        #   out_a = a * cos - b * sin
        #   out_b = a * sin + b * cos
        out_h2_a = h2_a * cos2 - h2_b * sin2
        out_h2_b = h2_a * sin2 + h2_b * cos2

        # Reassemble: interleave a,b back into adjacent pairs
        out_h1 = torch.stack([out_h1_a, out_h1_b], dim=-1).flatten(-2)
        out_h2 = torch.stack([out_h2_a, out_h2_b], dim=-1).flatten(-2)

        return torch.cat([out_h1, out_h2], dim=-1)



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = LELU()
        self.c_fc  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MLP_wide(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = LELU()
        self.c_fc  = nn.Linear(config.n_embd, config.n_embd*2, bias=config.bias)

        self.c_proj  = nn.Linear(config.n_embd*2, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MLP_bottle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd,  config.n_embd//2, bias=config.bias)
        self.act = LELU()
        self.c_proj  = nn.Linear(config.n_embd//2, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.depth = 0
        self.n_heads = config.n_head
        self.n_embd = config.n_embd
        dim = config.n_embd
        self.head_dim = dim // self.n_heads
        self.skew_basis = nn.Parameter(
            torch.randn(self.n_heads, self.head_dim, self.head_dim) * 0.02
        )


        self.q_proj = MLP(config)
        self.k_proj = nn.Linear(dim,dim,bias=False)
        self.v_proj = nn.Linear(dim,dim,bias=False)
        self.s_mlp = MLP(config)
        self.delta_proj = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.v_sink_basis_true = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))

        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.uncertainty_vec = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.mask = None #set in GPT main at model time to ensure its on GPU
        self.rope = config.rope
        limit = config.block_size // 2

        self.sd_alpha = 0.3
        self.sd_sigma = config.block_size / 2.0
        alphas = torch.linspace(0, 1, self.n_heads).view(1, self.n_heads, 1, 1)
        self.register_buffer('k_alpha', alphas)
        self.h = nn.Parameter(torch.randn(self.n_heads, self.head_dim, 1))
        self.s_gate = nn.Linear(2, 1, bias=True)

    def erode(self, B, H, T, device):
        alpha = 0.3
        sigma_frac = 0.18
    
        pos = torch.arange(T, device=device).float()              # [T]
    
        # spread head preferences across sequence: early -> middle -> late
        head_centers = torch.linspace(0, T - 1, H, device=device) # [H]
    
        # distance of each source position from each head's preferred region
        src_dist = pos.view(1, 1, T) - head_centers.view(H, 1, 1) # [H,1,T]
        src_dist = src_dist.expand(H, T, T).abs()                 # [H,T,T]
    
        sigma = max(T * sigma_frac, 1.0)
    
        # highest drop away from preferred region, lowest drop near it
        drop_probs = alpha * (1.0 - torch.exp(-(src_dist ** 2) / (2 * sigma ** 2)))
    
        # keep causal structure
        causal = torch.tril(torch.ones(T, T, device=device))
        drop_probs = drop_probs * causal.unsqueeze(0)
        keep_probs = 1.0 - drop_probs
    
        keep_mask = torch.bernoulli(
            keep_probs.unsqueeze(0).expand(B, H, T, T)
        )
        return keep_mask
        
    def forward(self, x, delta_prev=None):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

      
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
           # === V projection with delta composition ===
        v_base = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        if delta_prev is not None:
            v_combined = v_base + self.delta_proj(delta_prev)
        else:
            v_combined = v_base
        v = v_combined
        q = F.rms_norm(q, (D,),eps=1e-6) #never ever norm k, you stupid fuck

        q = self.rope(q)
        k = self.rope(k)
        scores = (q @ k.transpose(-2, -1)) * (math.log(T) * math.log(D))
        # Soft Attention

        mask = self.mask[:, :, :T, :T]
        soft_scores = F.softplus(scores) #zero point mass is log(2)/alpha or ~0.382.
        # STE: Forward sets small/neg values to 0, Backward ignores the zeroing
        # Values < 1e-6 do not participate in mass/scaling but receive gradients
        threshold = 1e-6
        pruned_scores = torch.where(soft_scores < threshold, torch.zeros_like(soft_scores), soft_scores)
        soft_scores = soft_scores + (pruned_scores - soft_scores).detach()
        soft_scores = soft_scores.masked_fill(mask == 0, 0.0) #prevent cheating here
       

        soft_sums = soft_scores.sum(dim=-1, keepdim=True)

        scale = torch.clamp(1.0 / (soft_sums + 1e-6), max=1.0)
        attn = soft_scores * scale
        attn = torch.nan_to_num(attn, nan=0.0)

        y_context = attn @ v
        v_i = v.unsqueeze(-2)              # (B, H, T, 1, D)
        delta = (attn.unsqueeze(-1) * (v.unsqueeze(-3) - v_i)).sum(dim=-2)
                #XSA-like but purely self-term
        self_weight = attn[..., torch.arange(T), torch.arange(T)]  # (B,H,T)
        self_weight = self_weight.unsqueeze(-1)  # (B,H,T,1)
        delta = delta - self_weight * v
        
        #delta = y_context - v OR: 
        # === Mixing tensor: variance of the attended distribution ===
        v_sq = attn @ (v * v)  # E[v^2] under attention weights
        mix_signal = F.softplus(v_sq - y_context * y_context)  # Var[v] per dim, (B, H, T, D)
        delta = delta * mix_signal
        # ===  sinking ===
        pos = torch.arange(T, device=x.device).float().view(1, 1, 1, T)
        query_pos = torch.arange(T, device=x.device).float().view(1, 1, T, 1)
        rel_dist = (query_pos - pos).clamp(min=0) / query_pos.clamp(min=1)
        
        # first moment: where is attention centered? (B, H, T, 1)
        mu = (attn * rel_dist).sum(dim=-1, keepdim=True)
        
        # second moment: how spread is it? (B, H, T, 1)
        var = (attn * (rel_dist - mu).pow(2)).sum(dim=-1, keepdim=True)



        s_conf = torch.sigmoid(self.s_gate(torch.cat([mu, var], dim=-1)))
        vn = F.normalize(v, dim=-1)
        y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn

        y_clean = y_context * s_conf 
        y_true = F.rms_norm(y_clean, (D,))+ self.v_sink_basis_true
        

        S_flat = y_true.transpose(1, 2).contiguous().view(B, T, -1)
        #P_flat = p.transpose(1, 2).contiguous().view(B, T, -1)

        truth = self.s_mlp(S_flat)

        if not self.training:
            return truth
        else:
            mask = self.erode(B,H,T,x.device)
            soft_scores = soft_scores.masked_fill(mask == 0, 0.0) #prevent cheating here
            soft_sums = soft_scores.sum(dim=-1, keepdim=True)
            scale = torch.clamp(1.0 / (soft_sums + 1e-6), max=1.0)
            attn = soft_scores * scale
            attn = torch.nan_to_num(attn, nan=0.0)
    
            y_context = attn @ v
            v_i = v.unsqueeze(-2)              # (B, H, T, 1, D)
            delta = (attn.unsqueeze(-1) * (v.unsqueeze(-3) - v_i)).sum(dim=-2)
                    #XSA-like but purely self-term
            self_weight = attn[..., torch.arange(T), torch.arange(T)]  # (B,H,T)
            self_weight = self_weight.unsqueeze(-1)  # (B,H,T,1)
            delta = delta - self_weight * v
            
            #delta = y_context - v OR: 
            # === Mixing tensor: variance of the attended distribution ===
            v_sq = attn @ (v * v)  # E[v^2] under attention weights
            mix_signal = F.softplus(v_sq - y_context * y_context)  # Var[v] per dim, (B, H, T, D)
            delta = delta * mix_signal
            # ===  sinking ===
            pos = torch.arange(T, device=x.device).float().view(1, 1, 1, T)
            query_pos = torch.arange(T, device=x.device).float().view(1, 1, T, 1)
            rel_dist = (query_pos - pos).clamp(min=0) / query_pos.clamp(min=1)
            
            # first moment: where is attention centered? (B, H, T, 1)
            mu = (attn * rel_dist).sum(dim=-1, keepdim=True)
            
            # second moment: how spread is it? (B, H, T, 1)
            var = (attn * (rel_dist - mu).pow(2)).sum(dim=-1, keepdim=True)
    
    
    
            s_conf = torch.sigmoid(self.s_gate(torch.cat([mu, var], dim=-1)))
            vn = F.normalize(v, dim=-1)
            y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn
    
            y_clean = y_context * s_conf 
            y_true = F.rms_norm(y_clean, (D,))+ self.v_sink_basis_true
            
    
            S_flat = y_true.transpose(1, 2).contiguous().view(B, T, -1)
            #P_flat = p.transpose(1, 2).contiguous().view(B, T, -1)

            tangent = self.s_mlp(S_flat)
    
            return tangent + (truth - tangent).detach()



def norm(x):
    return F.rms_norm(x, (x.size(-1),),eps=1e-6)

class Block(nn.Module):
    def __init__(self, config, block_idx):
        super().__init__()
        self.attn = Attention(config)
        self.attn_dir = MLP_bottle(config)
        self.config = config

    def forward(self, x):
        a = norm(x)
        z = self.attn(a)
        vn = F.normalize(self.attn_dir(a), dim=-1)
        q = x - (x * vn).sum(dim=-1, keepdim=True) * vn
        x = q + z

        return x

    

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 66 #shakespeare
    n_layer: int = 1
    n_head: int = 6

    n_embd: int = 192 #recommend 32 min per head
    dropout: float = 0.0
    bias: bool = False
    rope: nn.Module = None
    bottle: nn.Module = None
    device: str = "cuda"

class SoftplusCELoss(nn.Module):
    def __init__(self, ignore_index=-1, label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing


    def forward(self, logits, targets):
        # logits: (B, V) or (B, T, V)
        # targets: (B,) or (B, T)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)

        mask = flat_targets != self.ignore_index
        flat_logits = flat_logits[mask]
        flat_targets = flat_targets[mask]

        if flat_targets.numel() == 0:
            return flat_logits.sum() * 0.0

        sp = F.softplus( flat_logits)

        threshold = 1e-6
        pruned = torch.where(sp < threshold, torch.zeros_like(sp), sp)
        sp = sp + (pruned - sp).detach()  # STE

        sp_sum = sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (sp_sum + 1e-6), max=1.0)
        probs = sp * scale  # sub-unity simplex

        # gather target probs
        target_probs = probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)

        # NLL on the softplus-normalized probs
        loss = -torch.log(target_probs + 1e-8)

        if self.label_smoothing > 0.0:
            # smooth term: average negative log-prob across vocab
            smooth_loss = -torch.log(probs + 1e-6).mean(dim=-1)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        return loss.mean()

        

class ParallelSubspaceUnembed(nn.Module):
    """
    Parallel carveouts against the same h.
    Diversity is enforced only through cosine-distinct subspaces.

    Each slice owns:
      - a direction net that proposes a subspace direction
      - a vocab projection for the carved component

    Output:
      logits: summed vocab logits from all slices (+ residual head if enabled)
      sep_loss: pairwise squared cosine overlap penalty between slice directions
    """
    def __init__(self, d_model, vocab_size, n_slices=4, use_residual_head=True, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_slices = n_slices
        self.use_residual_head = use_residual_head
        self.eps = eps

        self.dir_nets = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_slices)
        ])
        self.projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_slices)
        ])

        if use_residual_head:
            self.residual_proj = nn.Linear(d_model, vocab_size, bias=False)
        else:
            self.residual_proj = None

    def forward(self, h):
        # h: [B, T, D]
        B, T, D = h.shape
        assert D == self.d_model

        logits = 0.0
        dirs = []
        components = []

        for i in range(self.n_slices):
            v = self.dir_nets[i](h)                            # [B,T,D]
            v = F.normalize(v, dim=-1, eps=self.eps)          # unit direction
            c = (h * v).sum(dim=-1, keepdim=True) * v         # projection of h onto v

            dirs.append(v)
            components.append(c)
            logits = logits + self.projs[i](c)

        if self.use_residual_head:
            used = torch.stack(components, dim=0).sum(dim=0)  # [B,T,D]
            residual = h - used
            logits = logits + self.residual_proj(residual)

        # pairwise squared cosine overlap penalty
        # because v_i, v_j are normalized, dot(v_i, v_j) is cosine
        sep_loss = h.new_zeros(())

        if self.training:
            count = 0
            for i in range(self.n_slices):
                for j in range(i + 1, self.n_slices):
                    cos_ij = (dirs[i] * dirs[j]).sum(dim=-1)      # [B,T]
                    sep_loss = sep_loss + (cos_ij ** 2).mean()
                    count += 1
    
            if count > 0:
                sep_loss = sep_loss / count

        return logits, sep_loss

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def bit_width(vocab_size: int) -> int:
    return max(1, math.ceil(math.log2(vocab_size)))


def hamming_parity_bits(m: int) -> int:
    r = 0
    while (2 ** r) < (m + r + 1):
        r += 1
    return r


def int_to_bits(x: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(width, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).float()


def hamming_encode_bits(data_bits: torch.Tensor) -> torch.Tensor:
    m = data_bits.shape[-1]
    r = hamming_parity_bits(m)
    n = m + r

    out = torch.zeros(*data_bits.shape[:-1], n, dtype=data_bits.dtype, device=data_bits.device)

    data_idx = 0
    for pos in range(1, n + 1):
        if (pos & (pos - 1)) != 0:
            out[..., pos - 1] = data_bits[..., data_idx]
            data_idx += 1

    for i in range(r):
        p = 2 ** i
        parity = torch.zeros(*data_bits.shape[:-1], dtype=data_bits.dtype, device=data_bits.device)
        for pos in range(1, n + 1):
            if (pos & p) and (pos != p):
                parity = torch.remainder(parity + out[..., pos - 1], 2.0)
        out[..., p - 1] = parity

    return out


class BinaryHammingPath(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, use_hamming: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.data_bits = bit_width(vocab_size)
        self.parity_bits = hamming_parity_bits(self.data_bits) if use_hamming else 0
        self.code_bits = self.data_bits + self.parity_bits
        self.proj = nn.Linear(self.code_bits, d_model, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be int32 or int64")
        if input_ids.min() < 0 or input_ids.max() >= self.vocab_size:
            raise ValueError("input_ids out of range")

        bits = int_to_bits(input_ids, self.data_bits)
        if self.parity_bits > 0:
            bits = hamming_encode_bits(bits)
        return self.proj(bits)


class FlatRollGeometry(nn.Module):
    """
    Exact construction in vocab space (V x V), then interpolate row-wise to d_model.
    This preserves unique token positions in vocab coordinates, then adapts to model width.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        scale: str = "box",
        seed: int = 0,
        freeze: bool = True,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        V = int(vocab_size)
        Dv = V
        eps = 1e-12

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        x = self._make_base(Dv, scale=scale, generator=g, dtype=dtype)  # [V]

        shifts = torch.arange(V)
        rows = [torch.roll(x, shifts=int(s.item() % Dv), dims=0) for s in shifts]
        W = torch.stack(rows, dim=0).to(dtype)  # [V, V]

        M = int(torch.argmax(x))
        pm = x[M].item()
        N = 1.0 / (pm + eps)

        r_idx = torch.arange(V)
        c_idx = (r_idx + M) % Dv
        S = torch.zeros((V, Dv), dtype=dtype)
        S[r_idx, c_idx] = N

        exact_vocab_geom = W + S  # [V, V]

        if d_model != V:
            interp = F.interpolate(
                exact_vocab_geom.unsqueeze(1),   # [V,1,V]
                size=d_model,
                mode="linear",
                align_corners=False,
            ).squeeze(1)                         # [V,d_model]
        else:
            interp = exact_vocab_geom

        interp = interp.to(dtype=dtype)
        if device is not None:
            interp = interp.to(device)

        self.embed = nn.Embedding.from_pretrained(interp, freeze=freeze)

    @staticmethod
    def _make_base(
        D: int,
        scale: str = "box",
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            complex_dtype = torch.complex64
            work_float = torch.float32
        else:
            complex_dtype = torch.complex128
            work_float = torch.float64

        X = torch.zeros(D, dtype=complex_dtype)
        X[0] = torch.tensor(0, dtype=complex_dtype)

        if D % 2 == 0:
            for k in range(1, D // 2):
                phi = torch.rand((), generator=generator, dtype=work_float) * (2 * math.pi)
                val = (torch.cos(phi) + 1j * torch.sin(phi)).to(complex_dtype)
                X[k] = val
                X[D - k] = torch.conj(val)
            X[D // 2] = 1.0 if torch.rand((), generator=generator) < 0.5 else -1.0
        else:
            for k in range(1, (D - 1) // 2 + 1):
                phi = torch.rand((), generator=generator, dtype=work_float) * (2 * math.pi)
                val = (torch.cos(phi) + 1j * torch.sin(phi)).to(complex_dtype)
                X[k] = val
                X[D - k] = torch.conj(val)

        x = torch.fft.ifft(X).real.to(work_float)

        if scale == "unit":
            x = x / (x.norm() + 1e-12)
        elif scale == "box":
            x = x / (x.abs().max() + 1e-12)
        else:
            raise ValueError("scale must be 'unit' or 'box'")

        return x.to(dtype)



class ThreePieceEmbedding(nn.Module):
    """
    final = (binary_path + cayley_path) * (1 + alpha * geometry_path)

    binary_path: identity / fingerprint
    cayley_path: structured deformational coordinates
    geometry_path: fixed atlas scaffold
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        bottleneck_width: int = 16,
        cayley_expansions: int = 3,
        use_hamming: bool = True,
        geom_scale: str = "box",
        geom_seed: int = 0,
        geom_freeze: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.binary = BinaryHammingPath(
            vocab_size=vocab_size,
            d_model=d_model,
            use_hamming=use_hamming,
        )

        self.embd = nn.Embedding(vocab_size,d_model)

        self.geometry = FlatRollGeometry(
            vocab_size=vocab_size,
            d_model=d_model,
            scale=geom_scale,
            seed=geom_seed,
            freeze=geom_freeze,
        )


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        e_bin = self.binary(input_ids)          # [B,T,d]
        e_embd = self.embd(input_ids)          # [B,T,d]
        e_geo = self.geometry.embed(input_ids)  # [B,T,d]

        base = e_bin + e_geo
        #out = base + e_embd
        return base

'''
ThreePieceEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.n_embd,
                bottleneck_width=16,#something close to the binary width
                cayley_expansions=3,
                use_hamming=True,
                geom_scale="box",
                geom_seed=0,
                geom_freeze=True),


'''
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rope = RoPE(config.n_embd // config.n_head, max_len=config.block_size)
        self.config.rope = self.rope

        self.transformer = nn.ModuleDict(dict(
        wte=nn.Embedding(config.vocab_size,config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(self.config,i) for i in range(config.n_layer)]),
        ))
        mask_tensor = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size).to(device=self.config.device)
        self.register_buffer("mask", mask_tensor)
        i = 0
        for block in self.transformer.h:
          block.attn.mask = self.mask #set here
          block.attn.depth= i
          i = i + 1

        self._boundary_handles = []
        self._linear_states = {}
        #self.register_confined_backward() experiment with at LATER TIME
        self.criterion = SoftplusCELoss(ignore_index=-1)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


  
    def register_confined_backward(self):
        states = {}
        handles = []
        mlphook = make_boundary_ste_hook(0.5)

        i = 1
        L = len(self.transformer.h)
        for block in self.transformer.h:
            handles.append(block.attn.register_full_backward_hook(make_boundary_ste_hook(float(1-2**(-i/L))))) #gently diminish contributions
            handles.append(block.attn_dir.register_full_backward_hook(mlphook)) #mlp contribution is tiny anyway
            i = i  + 1

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                hook, st = make_texture_hook()
                h = m.register_full_backward_hook(hook)
                handles.append(h)
                states[name] = st

    
        self._boundary_handles = handles
        self._linear_states=states
        


    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    def forward(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)

        for i, block in enumerate(self.transformer.h):
            x = block(x)

        x = norm(x)
        if targets is not None:
            logits  = self.lm_head(x)
            loss = self.criterion(logits, targets) 
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
