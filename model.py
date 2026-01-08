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

class VectorizedConstellationAttention(nn.Module):
    def __init__(
        self,
        config,
        palette_hw: int = 16,
        max_k: int = 15,
        rel_hid: int = 64,
        bias: bool = False,
        use_delta: bool = True,
        delta_dropout: float = 0.1, # NEW: Probability to zero out delta
    ):
        super().__init__()
        assert config.n_embd % 2 == 0
        self.d = config.n_embd
        self.BR = 1
        self.Ph = self.Pw = palette_hw
        self.Kmax = max_k
        self.use_delta = use_delta
        self.delta_dropout = delta_dropout # Store dropout rate

        # Projections
        self.i_proj = nn.Linear(config.n_embd, config.n_embd, bias=bias)
        self.p_proj = nn.Linear(config.n_embd, config.n_embd, bias=bias)

        self.rope = config.rope

        # Persistent palette
        self.palette = nn.Parameter(torch.randn(config.n_embd, self.Ph, self.Pw) * (config.n_embd ** -0.5))

        # Relational MLP
        rel_in = self.Kmax + 1 + (1 if use_delta else 0)
        self.rel_mlp = nn.Sequential(
            nn.Linear(rel_in, rel_hid),
            nn.GELU(),
            nn.Linear(rel_hid, rel_hid),
            nn.GELU(),
        )
        self.coord_head = nn.Linear(rel_hid, 2)
        self.mix_head   = nn.Linear(rel_hid, 1)

        self.Wo = nn.Parameter(torch.randn(1, config.n_embd, config.n_embd) * (config.n_embd ** -0.5))

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
        batch_pal = self.palette.unsqueeze(0).expand(B, -1, -1, -1)
        grid = z.reshape(B, T, K, 2)
        samples = F.grid_sample(batch_pal, grid, mode="bilinear", padding_mode="border", align_corners=True)

        samples = samples.view(B, 1, D, T, K).permute(0, 1, 3, 4, 2)
        V_out = (samples * w.unsqueeze(-1)).sum(dim=3)
        y = torch.einsum("nrtd,rdm->nrtm", V_out, self.Wo).mean(dim=1)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear( config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.scale = math.pi / math.sqrt(3.0)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_branch = 4
        self.block_size = config.block_size
        self.n_sinks = getattr(config, 'n_sinks', 4) # Default to 4 sink tokens

        # Projections: Q and V produce NB branches, K is shared
        self.q_proj = nn.Linear(config.n_embd, config.n_embd * self.n_branch, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd * self.n_branch, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Learnable sink tokens for K and V
        # K sink is (1, 1, n_sinks, C) to broadcast across batch and branches
        self.k_sink = nn.Parameter(torch.zeros(1, 1, self.n_sinks, config.n_embd))
        # V sink is (1, n_branch, n_sinks, C) as V is branched
        self.v_sink = nn.Parameter(torch.zeros(1, self.n_branch, self.n_sinks, config.n_embd))

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + self.n_sinks, config.block_size + self.n_sinks))
                             .view(1, 1, config.block_size + self.n_sinks, config.block_size + self.n_sinks))

        self.rope = config.rope
        self.attn_drop = nn.Dropout(config.dropout)
        
    def forward(self, a, x):
        B, T, C = x.shape
        NB = self.n_branch
        NS = self.n_sinks

        # Project and reshape: (B, T, NB, C) -> (B, NB, T, C)
        q = self.q_proj(a).view(B, T, NB, C).transpose(1, 2)
        v_orig = self.v_proj(a).view(B, T, NB, C).transpose(1, 2)
        # K has no branch dimension initially: (B, T, C) -> (B, 1, T, C)
        k_orig = self.k_proj(x).view(B, T, 1, C).transpose(1, 2)

        # Apply RoPE to projected sequences before adding sinks
        q, k_orig = self.rope(q), self.rope(k_orig)

        # Prepend Sinks:
        # K sinks: (1, 1, NS, C) -> (B, 1, NS, C)
        k = torch.cat([self.k_sink.expand(B, -1, -1, -1), k_orig], dim=2)
        # V sinks: (1, NB, NS, C) -> (B, NB, NS, C)
        v = torch.cat([self.v_sink.expand(B, -1, -1, -1), v_orig], dim=2)

        # Raw scores: (B, NB, T, C) @ (B, 1, C, T + NS) -> (B, NB, T, T + NS)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C)

        # Causal Masking: The mask must account for the T + NS dimension
        # Sinks are always visible, so the first NS columns are 1s
        full_T = T + NS
        mask = self.mask[:, :, NS:full_T, :full_T]
        att = att.masked_fill(mask == 0, float("-inf"))

        # Branch Routing Logic (Softmax across the NB dimension)
        soft_probs = F.softmax(att, dim=1)
        soft_probs = torch.nan_to_num(soft_probs, nan=0.0)

        # Straight-Through Estimator (STE)
        with torch.no_grad():
            max_val = soft_probs.max(dim=1, keepdim=True)[0]
            hard_mask = (soft_probs == max_val).float()

        route_mask = (hard_mask - soft_probs).detach() + soft_probs

        # Temporal Attention (Softmax across the T + NS dimension)
        att_max, _ = att.max(dim=1)
        attn_probs = F.softmax(att_max, dim=-1)

        # Composition
        combined_weights = attn_probs.unsqueeze(1) * route_mask

        # Weighted sum: (B, NB, T, T + NS) @ (B, NB, T + NS, C) -> (B, NB, T, C)
        y = (combined_weights @ v).sum(dim=1) 

        out = self.o_proj(y)
        return out



def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.think = VectorizedConstellationAttention(config)
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.mlp2 = MLP(config)


    def forward(self, x):
        B, T, C = x.shape
        q = self.think(norm(x))
        a = q + self.mlp(norm(q))
        u = x + self.attn(norm(a), norm(x))
        u = u + self.mlp2(norm(u)) #reconstructive shift
        steps = torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)
        running_mean = torch.cumsum(norm(a), dim=1) / steps
        u = u + running_mean
        return u

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 1
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    n_branch: int = 4 # Number of branches in Attention
    rope: nn.Module = None


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # Base noise seed (learned) for map generation
        self.rope = RoPE(config.n_embd, max_len=config.block_size)
        self.config.rope = self.rope

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(config.n_layer)]),
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


