#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution

import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class LELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = math.pi / math.sqrt(3.0)

    def forward(self, x):
        return x * torch.sigmoid(self.scale * x)

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

class InertialManifold(nn.Module):
    def __init__(
        self, 
        config, 
        palette_size: int = 16,     # H, W of the manifold
        kernel_size: int = 15,      # Bandwidth of the inertial filter
        expansion_factor: int = 2,  # Expansion for the output mixing
    ):
        super().__init__()
        self.d = config.n_embd
        self.h_pal = palette_size
        self.w_pal = palette_size
        self.history_dropout = getattr(config, 'history_dropout', 0.1) # Probability to drop a time-step
        
        # 1. The Territory (Palette)
        self.palette = nn.Parameter(
            torch.randn(1, config.n_embd, palette_size, palette_size) * (config.n_embd ** -0.5)
        )
        
        # 2. The Inertial Filter (Compass)
        self.norm = nn.LayerNorm(config.n_embd)
        self.inertial_conv = nn.Conv1d(
            in_channels=config.n_embd,
            out_channels=config.n_embd,
            kernel_size=kernel_size,
            padding=kernel_size - 1, # Causal padding
            groups=config.n_embd,    # Depthwise
            bias=False
        )
        
        # 3. The Navigator (now with Dropout)
        # Projects smoothed state -> (u, v) coordinates.
        self.navigator = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2,bias=config.bias),
            LELU(),
            nn.Dropout(config.dropout), # Added Dropout
            nn.Linear(config.n_embd // 2, 2,bias=config.bias),
            nn.Tanh() 
        )

        # 4. Integration (Output Projection) (now with Dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * expansion_factor,bias=config.bias),
            LELU(),
            nn.Dropout(config.dropout), # Added Dropout
            nn.Linear(config.n_embd * expansion_factor, config.n_embd,bias=config.bias),
            nn.Dropout(config.dropout)  # Added Final Dropout
        )
        
        nn.init.dirac_(self.inertial_conv.weight)

    def forward(self, x):
        B, T, D = x.shape
        
        # --- A. Inertial Smoothing ---
        # Normalize
        # Epistemic Dropout: Drop Time-Steps (History Perforation)
        # We drop the input signal at random positions.
        # The Conv1d must use the kernel's momentum to bridge the gap.
        if self.training and self.history_dropout > 0:
            # Mask shape: (B, T, 1) -> Broadcasts across features
            mask = torch.bernoulli(torch.full((B, T, 1), 1.0 - self.history_dropout, device=x.device))
            x_ = x * (mask / (1.0 - self.history_dropout))

        # Transpose for Conv1d: (B, T, D) -> (B, D, T)
        x_in = x_.transpose(1, 2)
        
        # Apply causal filter
        x_smooth = self.inertial_conv(x_in)[:, :, :T]
        x_smooth = x_smooth.transpose(1, 2)   # (B, T, D)
        
        # --- B. Navigation ---
        coords = self.navigator(x_smooth)
        grid = coords.view(B, 1, T, 2)
        
        # --- C. Territory Lookup ---
        batch_palette = self.palette.expand(B, -1, -1, -1)
        
        retrieved = F.grid_sample(
            batch_palette, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        retrieved = retrieved.squeeze(2).transpose(1, 2)
        
        # --- D. Integration ---
        y = self.output_proj(retrieved)
        
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

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_branch = 4
        self.block_size = config.block_size
        self.history_dropout = getattr(config, 'history_dropout', 0.1) # Default 10%

        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd * self.n_branch, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd * self.n_branch, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Sinks
        self.v_sink_residual = nn.Parameter(torch.zeros(1, 1, 1, config.n_embd))
        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_branch, 1, config.n_embd))

        # Register mask as non-persistent to fix the device mismatch bug
        self.register_buffer("mask", config.mask, persistent=False)

        self.rope = config.rope
        self.attn_drop = nn.Dropout(config.dropout)
       
    def forward(self, a, x):
        B, T, C = x.shape
        NB = self.n_branch
        device = x.device

        # 1. Projections
        q = self.q_proj(a).view(B, T, NB, C).transpose(1, 2)
        q = norm(q)
        v = self.v_proj(a).view(B, T, NB, C).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 1, C).transpose(1, 2)

        # 2. RoPE
        q, k = self.rope(q), self.rope(k)

        # 3. Raw Scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C)

        # 4. Masking Logic
        # A. Standard Causal Mask
        causal_mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))

        # B. Stochastic History Dropout (Epistemic Perforation)
        # We drop random connections in the history (k < t)
        # but we MUST preserve the diagonal (k == t) so the Query isn't blinded.
        if self.training and self.history_dropout > 0:
            # 1. Generate random dropout mask (True = Drop)
            # Shape: (B, 1, T, T) - Broadcasts across branches
            drop_mask = torch.rand(B, 1, T, T, device=device) < self.history_dropout
            
            # 2. Enforce strict lower-triangularity (diagonal=-1)
            # This ensures the diagonal (current position) and upper triangle are False.
            drop_mask = drop_mask.tril(diagonal=-1)
            
            # 3. Apply drop
            att = att.masked_fill(drop_mask, float("-inf"))

        # 5. Branch Routing
        soft_probs = F.softmax(att, dim=1)
        soft_probs = torch.nan_to_num(soft_probs, nan=0.0)

        with torch.no_grad():
            max_val = soft_probs.max(dim=1, keepdim=True)[0]
            hard_mask = (soft_probs == max_val).float()

        route_mask = (hard_mask - soft_probs).detach() + soft_probs

        # 6. Temporal Normalization
        scores_max, _ = att.max(dim=1) 
        s = F.softplus(scores_max)
        S = s.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (S + 1e-6), max=1.0)
        
        w = s * scale
        w = self.attn_drop(w) # Standard dropout on weights still applies

        residual = 1.0 - w.sum(dim=-1, keepdim=True)

        # 7. Composition
        combined_weights = w.unsqueeze(1) * route_mask
        y_context = (combined_weights @ v)

        branch_activity = route_mask.max(dim=-1, keepdim=True)[0] 
        y_basis = branch_activity * self.v_sink_basis 

        y_branches = (y_context + y_basis).sum(dim=1)
        y_res = residual * self.v_sink_residual.squeeze(1) 

        y = y_branches + y_res
        out = self.o_proj(y)
        return out


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.think = InertialManifold(config)
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
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    n_branch: int = 4 # Number of branches in Attention
    rope: nn.Module = None
    mask: nn.Module = None

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # Base noise seed (learned) for map generation
        self.rope = RoPE(config.n_embd, max_len=config.block_size)
        self.config.rope = self.rope
        
        mask_tensor = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("mask", mask_tensor)
        self.config.mask = self.mask
        

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


