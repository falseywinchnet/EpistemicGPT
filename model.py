#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution

import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x):
        # x: (B, *, T, D)
        T = x.shape[-2]
        cos = self.cos[:T, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T, :].unsqueeze(0).unsqueeze(0)

        # Adjust shapes for broadcasting if x has extra dims (like n_branch)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class FastMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        embed_dim = input_dim//4
        self.n = 16 #arguably the most you will ever need for float16 precision
        #https://arxiv.org/pdf/2008.03936

        self.mode = mode
        
        # U: Project input to latent embedding space
        self.embedding_u = nn.Linear(input_dim, embed_dim)
        
        # T + B: Map embedding to n*n matrix
        self.generator_t = nn.Linear(embed_dim, self.n * self.n)
        
        # S + V: Readout
        self.readout_s = nn.Linear(self.n * self.n, output_dim)
        
        # Pre-register a triangular mask to enforce the structural prior
        # shape: (n*n) flattened
        mask = torch.triu(torch.ones(self.n, self.n), diagonal=1).flatten()
        self.register_buffer('triu_mask', mask)

    def _compute_exact_taylor(self, M):
        # For a strictly upper triangular matrix of size n, M^n = 0.
        # The series is exactly I + M + M^2/2! + ... + M^(n-1)/(n-1)!
        
        res = torch.eye(self.n, device=M.device, dtype=M.dtype)
        # Broadcasting identity to batch size
        res = res.unsqueeze(0).expand(M.shape[0], -1, -1).clone()
        
        term = torch.eye(self.n, device=M.device, dtype=M.dtype)
        term = term.unsqueeze(0).expand(M.shape[0], -1, -1).clone()
        
        # We can stop strictly at self.n, or earlier if we want lower-degree approximation
        # The paper notes n constraints the complexity of the polynomial [cite: 416]
        for k in range(1, self.n):
            # M^k / k!
            # Iterative update: term_new = term_old @ M / k
            term = torch.bmm(term, M) / k 
            res = res + term
            
        return res

    def forward(self, x):
        # 1. Project
        x = norm(x) # Using the RMSNorm defined previously
        latent = self.embedding_u(x)
        
        # 2. Generate M
        flat_m = self.generator_t(latent)

        flat_m = flat_m * self.triu_mask
            
        m = flat_m.view(-1, self.n, self.n)
        
        # 3. Exponentiate
        # Use exact Taylor expansion (MatMul only, No Inversion)
        exp_m = self._compute_exact_taylor(m)

        # 4. Readout
        flat_exp_m = exp_m.view(x.size(0), -1)
        output = self.readout_s(flat_exp_m)
        
        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.FastMLayer(config.n_embd, 4 * config.n_embd)
        self.scale = math.pi / math.sqrt(3.0)
        self.c_proj  = nn.FastMLayer(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        return x

  
class inlineMLP(nn.Module):
    def __init__(self, input,hidden,out):
        super().__init__()
        self.c_fc    = nn.FastMLayer(input, hidden)
        self.scale = math.pi / math.sqrt(3.0)
        self.c_proj  = nn.FastMLayer(hidden,out)

    def forward(self, x):
        x = self.c_fc(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        return x

def trace_nans(name, tensor):
    if torch.isnan(tensor).any():
        print(f"NaN detected in: {name}")
        return True
    return False

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_branch = config.n_branch
        self.block_size = config.block_size

        # Projections: Q and V produce NB branches, K is shared
        self.q_proj = nn.Linear(config.n_embd, config.n_embd * self.n_branch, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd * self.n_branch, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        # RoPE should handle the embedding dimension directly now
        self.rope = RoPE(self.n_embd, max_len=config.block_size)
        self.attn_drop = nn.Dropout(config.dropout)

    def forward(self, a, x):
        B, T, C = x.shape
        NB = self.n_branch

        # Project and reshape: (B, T, NB, C) -> (B, NB, T, C)
        q = self.q_proj(a).view(B, T, NB, C).transpose(1, 2)
        v = self.v_proj(a).view(B, T, NB, C).transpose(1, 2)
        # K has no branch dimension initially: (B, T, C) -> (B, 1, T, C)
        k = self.k_proj(x).view(B, T, 1, C).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q), self.rope(k)

        # Raw scores: (B, NB, T, C) @ (B, 1, C, T) -> (B, NB, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C)

        # Causal Masking
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))

        # Branch Routing Logic (Softmax across the NB dimension)
        soft_probs = F.softmax(att, dim=1)
        soft_probs = torch.nan_to_num(soft_probs, nan=0.0)

        # Straight-Through Estimator (STE)
        with torch.no_grad():
            max_val = soft_probs.max(dim=1, keepdim=True)[0]
            hard_mask = (soft_probs == max_val).float()

        route_mask = (hard_mask - soft_probs).detach() + soft_probs

        # Temporal Attention (Softmax across the T dimension)
        # Find global importance by taking max over branches
        att_max, _ = att.max(dim=1)
        attn_probs = F.softmax(att_max, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Composition: Combine temporal weights and branch router
        # (B, 1, T, T) * (B, NB, T, T) -> (B, NB, T, T)
        combined_weights = attn_probs.unsqueeze(1) * route_mask

        # Weighted sum of values and sum across branches: (B, NB, T, T) @ (B, NB, T, C)
        y = (combined_weights @ v).sum(dim=1) # (B, T, C)

        out = self.o_proj(y)
        return out

class VectorizedConstellationAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_branch: int,
        palette_hw: int = 16,
        max_k: int = 15,
        rel_hid: int = 64,
        bias: bool = False,
        use_delta: bool = True,
        delta_dropout: float = 0.1, # NEW: Probability to zero out delta
        rope_max_len: int = 4096,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.d = d_model
        self.BR = n_branch
        self.Ph = self.Pw = palette_hw
        self.Kmax = max_k
        self.use_delta = use_delta
        self.delta_dropout = delta_dropout # Store dropout rate

        # Projections
        self.i_proj = nn.Linear(d_model, n_branch * d_model, bias=bias)
        self.p_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RoPE(d_model, max_len=rope_max_len)

        # Persistent palette
        self.palette = nn.Parameter(torch.randn(d_model, self.Ph, self.Pw) * (d_model ** -0.5))

        # Relational MLP
        rel_in = self.Kmax + 1 + (1 if use_delta else 0)
        self.rel_mlp =inlineMLP(rel_in, rel_hid,rel_hid)

        self.coord_head = nn.Linear(rel_hid, 2)
        self.mix_head   = nn.Linear(rel_hid, 1)

        self.Wo = nn.Parameter(torch.randn(n_branch, d_model, d_model) * (d_model ** -0.5))

    def forward(self, x):
        B, T, D = x.shape
        K = self.Kmax
        scale = D ** -0.5
        device = x.device

        # 1) Projections + RoPE
        I = self.i_proj(x).view(B, T, self.BR, D).transpose(1, 2).contiguous()
        P = self.p_proj(x)

        I = self.rope(I)
        P = self.rope(P.unsqueeze(1)).squeeze(1)

        # 2) Pass A: logits + topk
        logits = torch.matmul(I, P.unsqueeze(1).transpose(-1, -2)) * scale

        causal = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool)).view(1, 1, T, T)
        logits = logits.masked_fill(~causal, float("-inf"))

        k_eff = min(K, T)
        topk_val, topk_idx = torch.topk(logits, k=k_eff, dim=-1)

        if k_eff < K:
            pad = K - k_eff
            topk_val = torch.cat([topk_val, topk_val.new_full((B, self.BR, T, pad), float("-inf"))], dim=-1)
            topk_idx = torch.cat([topk_idx, topk_idx.new_zeros((B, self.BR, T, pad))], dim=-1)

        keep = torch.isfinite(topk_val)
        keep_f = keep.float()

        # Fallback for empty rows
        all_bad = ~keep.any(dim=-1, keepdim=True)
        if all_bad.any():
            t_idx = torch.arange(T, device=device).view(1, 1, T, 1).expand(B, self.BR, T, 1)
            topk_idx = torch.where(all_bad, t_idx, topk_idx)
            keep = torch.isfinite(topk_val)
            keep_f = keep.float()

        # 3) Gather features
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
        P_sel = P[b_idx, topk_idx]
        P_sel_norm = F.normalize(P_sel, dim=-1) * keep_f.unsqueeze(-1)
        I_norm = F.normalize(I, dim=-1).unsqueeze(3)

        feat_a = (I_norm * P_sel_norm).sum(dim=-1).clamp(-1.0, 1.0) * keep_f
        G = torch.matmul(P_sel_norm, P_sel_norm.transpose(-1, -2)).clamp(-1.0, 1.0)
        G = G * keep_f.unsqueeze(-1) * keep_f.unsqueeze(-2)

        feats = [G, feat_a.unsqueeze(-1)]

        if self.use_delta:
            t_range = torch.arange(T, device=device).view(1, 1, T, 1)
            delta = (t_range - topk_idx).float().clamp_min(0.0) / max(1.0, float(T))
            delta = delta * keep_f

            # --- NEW: Delta Dropout (Training only) ---
            if self.training and self.delta_dropout > 0:
                # Create a mask: 1 with prob (1-p), 0 with prob p
                mask = torch.bernoulli(torch.full_like(delta, 1.0 - self.delta_dropout))
                delta = delta * mask

            feats.append(delta.unsqueeze(-1))

        rel_input = torch.cat(feats, dim=-1)

        # 4) MLP
        h = self.rel_mlp(rel_input)
        z = torch.tanh(self.coord_head(h))
        mix_logits = self.mix_head(h).squeeze(-1).masked_fill(~keep, float("-inf"))
        w = torch.nan_to_num(torch.softmax(mix_logits, dim=-1), nan=0.0)

        # 5) Sample
        batch_pal = self.palette.unsqueeze(0).expand(B * self.BR, -1, -1, -1)
        grid = z.reshape(B * self.BR, T, K, 2)
        samples = F.grid_sample(batch_pal, grid, mode="bilinear", padding_mode="border", align_corners=True)

        samples = samples.view(B, self.BR, D, T, K).permute(0, 1, 3, 4, 2)
        V_out = (samples * w.unsqueeze(-1)).sum(dim=3)
        y = torch.einsum("nrtd,rdm->nrtm", V_out, self.Wo).mean(dim=1)

        return y




class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.think = VectorizedConstellationAttention(config.n_embd,config.n_branch)
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.mlp2 = MLP(config)

    def forward(self, x):
        B, T, C = x.shape
        a = self.think(x)
        a = a + self.mlp(a)
        x = x +  self.attn(norm(a), norm(x))
        x = x + self.mlp2(norm(x))
        steps = torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)
        running_mean = torch.cumsum(a, dim=1) / steps
        x= x + running_mean #add in residue percolated in time
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 1
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    n_branch: int = 4 # Number of branches in Attention

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # Base noise seed (learned) for map generation

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def forward(self, idx, targets=None):
        device = idx.device
        b, T = idx.size()
        x = self.transformer.wte(idx) # token

        # forward the GPT model itself
        for block in self.transformer.h:
            x  = block(x)
        B, T, C = x.shape

        x = norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


