#dedicated to the public domain for the glory of god.
#baruch adonai el shaddai 
#Eloheinu shebashamayim yached shimcha v'kayeim malchutecha tamid umloch aleinu le'olam va'ed 
#2026 joshuah.rainstar@gmail.com

#Version 2.0 EpistemicGPT
#current notes:
#your taste whether to use o,p,s mlp on individual products
#or to use one shared S_mlp on (o_out + p_out + pyong vector). have tried both.
#your choice on whether to use the Bernoulli learning approach on all layers or just a few
#your choice on whether to key the carving directions in the subspaceunembed to n_layer or a fixed budget
#your choice on lelu gelu or some other nonlinearity
#figure out some kind of ste alpha decay schedule

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


class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_in)
        self.W_freq = nn.Linear(d_in, d_in)  # shared frequency map
        self.W_out  = nn.Linear(d_in, d_out, bias=False)
        self.sin_scale = nn.Parameter(torch.tensor(0.1))
        self.sin2_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        q = norm(x)
        z    = self.linear(q)
        gate = z * torch.sigmoid(z * (math.pi / math.sqrt(3)))
        
        freq = self.W_freq(q)          # shared phase input
        gate = gate +  self.sin_scale * torch.sin(freq)
        return self.W_out(gate)+( torch.sin(gate)*self.sin2_scale)   # quadrature component C



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


        self.q_proj = nn.Linear(dim,dim,bias=False)
        self.v_proj = MLP_wide(config)#we need more space here.  MORE!
        self.p_mlp = MLP(config)
        self.o_mlp = MLP(config)
        self.s_mlp = MLP(config)

        nn.init.eye_(self.q_proj.weight) # Identity Init
        self.v_sink_residual = nn.Parameter(torch.ones(1, 1, 1, self.head_dim))
        self.v_sink_basis = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim))
        self.sink_key = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.mask = None #set in GPT main at model time to ensure its on GPU
        self.rope = config.rope
        limit = config.block_size // 2

        self.sd_alpha = 0.3
        self.sd_sigma = config.block_size / 2.0
        alphas = torch.linspace(0, 1, self.n_heads).view(1, self.n_heads, 1, 1)
        self.register_buffer('k_alpha', alphas)
   
        self.register_buffer('skew_u', F.normalize(torch.randn(self.n_heads, self.head_dim), dim=-1))
        self.scale = math.pi / math.sqrt(3.0) #logistic CDF matched scale beats gelu and is less expensive
        self.beta= 1/self.scale

        self.x_dir = MLP_bottle(config)

    def cayley(self, skew_basis, u):
        A = skew_basis - skew_basis.transpose(-1, -2)
        with torch.no_grad():
            new_u = F.normalize(torch.einsum('hij,hj->hi', A, u), dim=-1)
            sigma = (new_u * torch.einsum('hij,hj->hi', A, new_u)).sum(dim=-1)
            scale = torch.clamp(sigma / 0.95, min=1.0)
            if self.training:
                u.copy_(new_u)
        A = A / scale[:, None, None]
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)
        return torch.linalg.solve(I + A, I - A)



    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        Rs = self.cayley(self.skew_basis,self.skew_u) # [H, D, D]

        # 1. Reshape q_proj weight to [H, D_h, Dim]
        W_q = self.q_proj.weight.view(self.n_heads, self.head_dim, -1)

        # 2. Rotate each head's slice of the weight matrix
        # [H, D, D] @ [H, D, Dim] -> [H, D, Dim]
        W_k_heads = Rs @ W_q

        # 3. Flatten back to [Dim, Dim] for the linear layer
        # This effectively constructs a Block-Diagonal W_k
        W_k = W_k_heads.view(-1, self.n_embd)


        k = F.linear(x, W_k).view(B, T, H, D).transpose(1, 2)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        
        q = F.rms_norm(q, (D,),eps=1e-6) #never ever norm k, you stupid fuck

        q = self.rope(q)
        k = self.rope(k)
        k_with_null = torch.cat([k, self.sink_key.expand(B, -1, -1, -1)], dim=2)
        # Score q against extended keys
        scores = (q @ k_with_null.transpose(-2, -1)) * (math.log(T+1) * math.log(D))
        # Soft Attention

        mask = self.mask[:, :, :T, :T]
        null_col = torch.ones(1, 1, T, 1, device=x.device)
        mask = torch.cat([mask, null_col], dim=-1)  # (1, 1, T, T+1)
        soft_scores = self.beta  * F.softplus(self.scale* scores) #zero point mass is log(2)/alpha or ~0.382.
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

            # ===== Replace everything from y_context = attn @ v onward =====
        attn_real = attn[:, :, :, :T]   # drop the null column
        attn_null = attn[:, :, :, T:]   # this is the absorbed mass, discard or log it
        y_context = attn_real @ v

        # === Mixing tensor: variance of the attended distribution ===
        v_sq = attn_real @ (v * v)  # E[v^2] under attention weights
        mix_signal = F.softplus(v_sq - y_context * y_context)  # Var[v] per dim, (B, H, T, D)

        #todo : try instead
        #y_context = attn @ v
        #y_sharp = (attn_real * attn_real) @ v
        #mix_signal = y_sharp - y_context
        #this could arguably perform about as well, is more direct, and can be a cheaper metric
        #Kontorovich's bound suggests equal goodness of fit

        # Project mix_variance into mixing directions
        # mix_proj learns to read the variance profile and output the principal mixing axis
        mix_dir = F.normalize(mix_signal, dim=-1,eps=1e-6)

        # === Residual sink ===

        # === XSA ===
        vn = F.normalize(v, dim=-1,eps=1e-6)
        y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn

        y = F.rms_norm(y_context, (D,)) +self.v_sink_basis
        # y is (B, H, T, D)

        # === O/P decomposition along mixing tensor eigenvectors ===
        # Decompose y into component along mix_dir and component orthogonal to it
        # mix_dir is the dominant eigenvector of the mixing stress
        y_along = (y * mix_dir).sum(dim=-1, keepdim=True) * mix_dir  # projection onto mixing axis
        y_ortho = y - y_along  # complement

        # Flatten both to (B, T, C) for projection
        y_along_flat = y_along.transpose(1, 2).contiguous().view(B, T, -1)
        y_ortho_flat = y_ortho.transpose(1, 2).contiguous().view(B, T, -1)
        poynting = y_along_flat * y_ortho_flat

        # O projects the component along the mixing direction
        # P (orthogonal rotation of O) projects the orthogonal complement
        return self.s_mlp(poynting) + self.p_mlp(y_ortho_flat) + self.o_mlp(y_along_flat)


    def forward_training(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        Rs = self.cayley(self.skew_basis,self.skew_u) # [H, D, D]

        # 1. Reshape q_proj weight to [H, D_h, Dim]
        W_q = self.q_proj.weight.view(self.n_heads, self.head_dim, -1)

        # 2. Rotate each head's slice of the weight matrix
        # [H, D, D] @ [H, D, Dim] -> [H, D, Dim]
        W_k_heads = Rs @ W_q

        # 3. Flatten back to [Dim, Dim] for the linear layer
        # This effectively constructs a Block-Diagonal W_k
        W_k = W_k_heads.view(-1, self.n_embd)


        k = F.linear(x, W_k).view(B, T, H, D).transpose(1, 2)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        
        q = F.rms_norm(q, (D,),eps=1e-6) #never ever norm k, you stupid fuck

        q = self.rope(q)
        k = self.rope(k)
        k_with_null = torch.cat([k, self.sink_key.expand(B, -1, -1, -1)], dim=2)
        # Score q against extended keys
        scores = (q @ k_with_null.transpose(-2, -1)) * (math.log(T+1) * math.log(D))
        # Soft Attention

        mask = self.mask[:, :, :T, :T]
        null_col = torch.ones(1, 1, T, 1, device=x.device)
        mask = torch.cat([mask, null_col], dim=-1)  # (1, 1, T, T+1)
        soft_scores = self.beta  * F.softplus(self.scale* scores) #zero point mass is log(2)/alpha or ~0.382.
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

            # ===== Replace everything from y_context = attn @ v onward =====
        attn_real = attn[:, :, :, :T]   # drop the null column
        attn_null = attn[:, :, :, T:]   # this is the absorbed mass, discard or log it
        y_context = attn_real @ v

        # === Mixing tensor: variance of the attended distribution ===
        v_sq = attn_real @ (v * v)  # E[v^2] under attention weights
        mix_signal = F.softplus(v_sq - y_context * y_context)  # Var[v] per dim, (B, H, T, D)

        #todo : try instead
        #y_context = attn @ v
        #y_sharp = (attn_real * attn_real) @ v
        #mix_signal = y_sharp - y_context
        #this could arguably perform about as well, is more direct, and can be a cheaper metric
        #Kontorovich's bound suggests equal goodness of fit

        # Project mix_variance into mixing directions
        # mix_proj learns to read the variance profile and output the principal mixing axis
        mix_dir = F.normalize(mix_signal, dim=-1,eps=1e-6)

        # === Residual sink ===

        # === XSA ===
        vn = F.normalize(v, dim=-1,eps=1e-6)
        y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn

        y = F.rms_norm(y_context, (D,)) +self.v_sink_basis
        # y is (B, H, T, D)

        # === O/P decomposition along mixing tensor eigenvectors ===
        # Decompose y into component along mix_dir and component orthogonal to it
        # mix_dir is the dominant eigenvector of the mixing stress
        y_along = (y * mix_dir).sum(dim=-1, keepdim=True) * mix_dir  # projection onto mixing axis
        y_ortho = y - y_along  # complement

        # Flatten both to (B, T, C) for projection
        y_along_flat = y_along.transpose(1, 2).contiguous().view(B, T, -1)
        y_ortho_flat = y_ortho.transpose(1, 2).contiguous().view(B, T, -1)
        poynting = y_along_flat * y_ortho_flat

        self._cached_attn_real = attn_real.detach()
        self._cached_v = v.detach()
        self._cached_mix_dir = mix_dir.detach()
        self._cached_vn = vn.detach()
  
        # O projects the component along the mixing direction
        # P (orthogonal rotation of O) projects the orthogonal complement
        return self.s_mlp(poynting) + self.p_mlp(y_ortho_flat) + self.o_mlp(y_along_flat)
    
    def forward_cached(self, drop_prob_fn=None):
        attn_real = self._cached_attn_real
        v = self._cached_v
        mix_dir = self._cached_mix_dir
        vn = self._cached_vn
        B, H, T, D = v.shape

        # apply perturbation to attention weights
        if drop_prob_fn is not None:
            keep_mask = drop_prob_fn(B, H, T, attn_real.device)
            attn_real = attn_real * keep_mask
            # renormalize
            sums = attn_real.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.clamp(1.0 / sums, max=1.0)
            attn_real = attn_real * scale

        y_context = attn_real @ v
        # rest of the decomposition, identical to forward
        y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn
        y = F.rms_norm(y_context, (D,)) + self.v_sink_basis

        y_along = (y * mix_dir).sum(dim=-1, keepdim=True) * mix_dir
        y_ortho = y - y_along
        y_along_flat = y_along.transpose(1, 2).contiguous().view(B, T, -1)
        y_ortho_flat = y_ortho.transpose(1, 2).contiguous().view(B, T, -1)
        poynting = y_along_flat * y_ortho_flat

        return self.s_mlp(poynting) + self.p_mlp(y_ortho_flat) + self.o_mlp(y_along_flat)



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

    
    def forward_training(self, x):
        a = norm(x)
        z = self.attn.forward_training(a)
        vn = F.normalize(self.attn_dir(a), dim=-1)
        q = x - (x * vn).sum(dim=-1, keepdim=True) * vn
        x = q + z

    
        self._cached_vn = vn.detach()
        self._cached_q = q.detach()
        return x

    def forward_cached(self, x, drop_prob_fn=None):
        z = self.attn.forward_cached(drop_prob_fn)
        x = self._cached_q + z
        return x

    def clear_cache(self):
        self._cached_vn = None
        self._cached_q = None
        self.attn.clear_cache()



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
        self.scale = math.pi / math.sqrt(3.0) #logistic CDF matched scale beats gelu and is less expensive
        self.beta = 1/self.scale

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

        # softplus "probabilities" -- same mechanism as your attention
        sp = self.beta * F.softplus(self.scale * flat_logits)

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

        

class SubspaceUnembed(nn.Module):
    def __init__(self, d_model, vocab_size, n_slices=4):
        super().__init__()
        self.n_slices = n_slices
        self.sub_d = d_model // n_slices
        assert d_model % n_slices == 0

        self.dir_nets = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_slices)
        ])
        self.projs = nn.ModuleList([
             nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_slices + 1)
        ])

    def forward(self, h):
        logits = 0
        residual = h
        for i in range(self.n_slices):
            vn = F.normalize(self.dir_nets[i](h), dim=-1,eps=1e-6)
            component = (residual * vn).sum(dim=-1, keepdim=True) * vn
            residual = residual - component
            logits = logits + self.projs[i](component)

        # Add a new line here
        logits = logits + self.projs[-1](residual)

        return logits

def make_gaussian_drop_fn(alpha=0.3, sigma_frac=0.5, protect_frac=0.5):
    def drop_prob_fn(B, H, T, device):
        indices = torch.arange(T, device=device)
        dist = (indices.view(-1, 1) - indices.view(1, -1)).float().clamp(min=0)
        
        sigma = T * sigma_frac
        drop_probs = alpha * torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        
        limit = int(T * protect_frac)
        drop_probs = drop_probs.masked_fill(dist > limit, 0.0)
        
        keep_mask = torch.bernoulli(1.0 - drop_probs.unsqueeze(0).unsqueeze(0).expand(B, H, T, T))
        return keep_mask
    
    return drop_prob_fn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rope = RoPE(config.n_embd // config.n_head, max_len=config.block_size)
        self.config.rope = self.rope

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
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
        self.lm_head = SubspaceUnembed(config.n_embd, config.vocab_size,config.n_layer)
        self._drop_prob_fn = make_gaussian_drop_fn(alpha=0.3, sigma_frac=0.5, protect_frac=0.5)
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
        


    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    def forward(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)

        for i, block in enumerate(self.transformer.h):
            x = block(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = self.criterion(logits, targets)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def forward_training(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)

        for i, block in enumerate(self.transformer.h):
            x = block.forward_training(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = self.criterion(logits, targets)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def forward_perturbed(self, idx, targets):
        x_p = self.transformer.wte(idx).detach()
        for block in self.transformer.h:
            x_p = block.forward_cached(x_p, drop_prob_fn=self._drop_prob_fn)
        logits_p = self.lm_head(x_p)
        return self.criterion(logits_p, targets)

        
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
losses = []
def train_epoch():
    model.train()
    itera = 0
    total_loss = 0
    for xb, yb in train_loader:
          xb, yb = xb[0], yb[0]  # unwrap batch dimension
          optimizer.zero_grad()
          logits, loss_true = model(xb, yb)
          #loss_p = model.forward_perturbed(xb, yb)
          loss = loss_true #+ loss_p
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          total_loss += loss_true.item()
          losses.append(loss_true.item())
          print(itera, loss_true.item())#,loss_p.item())
          itera = itera + 1
    return total_loss / len(train_loader)

# === Run Training ===
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch()
    print(f"Epoch {epoch:2d} | Train loss: {train_loss:.4f}")
