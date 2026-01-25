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
        self.dim = dim
        self.max_len = max_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def get_embeddings(self, positions, device):
        positions = positions.clamp(0, self.max_len - 1).long()
        cos = F.embedding(positions, self.cos_cached).unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, T, D]
        sin = F.embedding(positions, self.sin_cached).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, x, positions=None):
        # x: (B, Branch, H, T, D)
        if positions is None:
            T = x.shape[-2]
            positions = torch.arange(T, device=x.device)
        
        cos, sin = self.get_embeddings(positions, x.device)
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat((y1, y2), dim=-1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class VernierRoPE(nn.Module):
    def __init__(self, dim, max_len=4096, base_1=10000.0, base_2=9973.0, analytic_init=False):
        """
        Args:
            dim: Embedding dimension (must be even).
            max_len: Pre-computed cache length.
            base_1: Primary frequency base.
            base_2: Vernier offset base (ideally prime close to base_1).
            analytic_init: Default state of the absolute position injection.
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.analytic = analytic_init

        # 1. Establish Dual Frequency Grids
        inv_freq_1 = 1.0 / (base_1 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq_2 = 1.0 / (base_2 ** (torch.arange(0, dim, 2).float() / dim))
        
        t = torch.arange(max_len).float()
        
        # Outer Products [Max_Len, Dim/2]
        freqs1 = torch.einsum('i,j->ij', t, inv_freq_1)
        freqs2 = torch.einsum('i,j->ij', t, inv_freq_2)
        
        # 2. Rotation Component (Average Frequency)
        # We use the mean to preserve standard relative phase behavior.
        avg_freqs = (freqs1 + freqs2) / 2
        self.register_buffer('cos_cached', avg_freqs.cos())
        self.register_buffer('sin_cached', avg_freqs.sin())
        
        # 3. Analytic Component (Beat Frequency)
        # cos(w1 - w2) -- The Moiré pattern.
        beat_freqs = freqs1 - freqs2
        self.register_buffer('amp_cached', beat_freqs.cos())

    def forward(self, x, positions=None, analytic=None):
        """
        Args:
            x: Input tensor (B, H, T, D) or (B, T, H, D)
            positions: Specific positions indices. If None, infers from sequence length.
            analytic: Override the default analytic injection behavior (True/False).
        """
        # Resolve toggle state
        use_analytic = self.analytic if analytic is None else analytic

        # Infer positions if not provided
        if positions is None:
            # Handle standard NLP shapes: (B, H, T, D) or (B, T, H, D)
            # We assume T is the second to last dimension.
            T = x.shape[-2] 
            positions = torch.arange(T, device=x.device)
            # Broadcast positions to batch size if needed, 
            # but usually single sequence indexing is fine for broadcasting.

        # Clamp and fetch cache
        positions = positions.clamp(0, self.max_len - 1).long()
        
        # Retrieve cached slices [T, D/2]
        cos = F.embedding(positions, self.cos_cached)
        sin = F.embedding(positions, self.sin_cached)
        
        # Reshape for broadcast against (B, H, T, D)
        # Standard RoPE typically broadcasts over Batch and Heads.
        # We need [1, 1, T, D/2]
        cos = cos.view(1, 1, -1, x.shape[-1] // 2)
        sin = sin.view(1, 1, -1, x.shape[-1] // 2)

        # Split input into even/odd
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Apply Standard Rotation
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        if use_analytic:
            # Fetch and shape Amplitude
            amp = F.embedding(positions, self.amp_cached)
            amp = amp.view(1, 1, -1, x.shape[-1] // 2)
            
            # Inject Absolute Information additively
            y1 = y1 + (x1 * amp)
            y2 = y2 + (x2 * amp)

        return torch.cat((y1, y2), dim=-1)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
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
        self.n_heads = config.n_head
        dim = config.n_embd
        self.head_dim = dim // self.n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.q_proj.weight) # Identity Init
        self.v_sink_residual = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        
        self.mask = config.mask
        self.rope = config.rope

        
        self.skew_basis = nn.Parameter(torch.randn(dim, dim) * 0.02)

    def get_orthogonal_matrix(self):
       # Enforce skew-symmetry: 
       #A = M - M.T
       skew = self.skew_basis - self.skew_basis.T 
       return torch.matrix_exp(skew)   
        
    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        R = self.get_orthogonal_matrix()
        w_k = R @ self.q_proj.weight 

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = F.linear(x, w_k).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        q = F.rms_norm(q, (D,))

        q = self.rope(q)
        k = self.rope(k)
    
        # Soft Attention
        scores = (q @ k.transpose(-2, -1))
        
        mask = self.mask[:, :, :T, :T]

        soft_scores = F.softplus(scores)
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

        current_mass = attn.sum(dim=-1, keepdim=True)
        residual = 1.0 - F.sigmoid(current_mass)
        y_res = residual * self.v_sink_residual
        y = F.rms_norm(y_context, (D,)) + self.v_sink_basis + y_res
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(y)

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 1 #note: dont use heads with less than 64 param
    #per head. you can use branching instead-
    #evaluate multiple Q against one K, score individually,
    #process though v independently, and either :
    #mean, STE-softmax route branchwise, and feed though O.
    #but, ultimately, better to just add a wider embedding.
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    rope: nn.Module = None
    mask: torch.Tensor = None

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rope = VernierRoPE(config.n_embd // config.n_head, max_len=config.block_size)
        self.config.rope = self.rope

        mask_tensor = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("mask", mask_tensor)
        self.config.mask = self.mask

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(config.n_layer)]),
            wtu = nn.Linear(config.n_embd,config.vocab_size),
        ))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)
        q = self.transformer.wtu.weight.sum(dim=0)/self.config.vocab_size
        x = x + q #stabilize composition so we dont spend energy 
        #in construction and can route instead
        for block in self.transformer.h:
            x = block(x)

        x = norm(x)

        if targets is not None:
            logits = self.transformer.wtu(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.transformer.wtu(x[:, [-1], :])
            loss = None

        return logits, loss
