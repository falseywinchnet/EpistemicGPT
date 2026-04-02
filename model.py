#copyright 2026 joshuah.rainstar@gmail.com
#MIT- take this and use it, but please credit me.
#Version 2.0 EpistemicGPT
#current notes:
#your taste whether to use o,p,s mlp on individual products
#or to use one shared S_mlp on (o_out + p_out + pyong vector)
#your choice on whether to use the rope i have applied universally or deduce a head-specific frequency
#your choice on whether to use the Bernoulli learning  approach on all layers or  just a few
#your choice on whether to key the carving directions in the subspaceunembed to layers or a fixed budget
#your choice on lelu gelu or some other nonlinearity
#recommend >32dim per head and to remember that about 60% of avail jacobian directions are exhausted in bookeeping

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
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
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
        self.n_heads = config.n_head
        self.n_embd = config.n_embd
        dim = config.n_embd
        self.head_dim = dim // self.n_heads
        self.skew_basis = nn.Parameter(
            torch.randn(self.n_heads, self.head_dim, self.head_dim) * 0.02
        )


        self.q_proj = nn.Linear(dim,dim,bias=False)
        self.v_proj = nn.Linear(dim,dim,bias=False)
        self.o_proj = nn.Linear(dim,dim,bias=False)
        self.s_proj = nn.Linear(dim, dim, bias=False)
        self.p_mlp = MLP(config)
        self.o_mlp = MLP(config)
        self.s_mlp = MLP(config)

        nn.init.eye_(self.q_proj.weight) # Identity Init
        self.v_sink_residual = nn.Parameter(torch.ones(1, 1, 1, self.head_dim))
        self.v_sink_basis = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim))

        self.mask = None #set in GPT main at model time to ensure its on GPU
        self.rope = config.rope
        limit = config.block_size // 2

        self.sd_alpha = 0.3
        self.sd_sigma = config.block_size / 2.0
        alphas = torch.linspace(0, 1, self.n_heads).view(1, self.n_heads, 1, 1)
        self.register_buffer('k_alpha', alphas)
        self.p_skew_basis = nn.Parameter(
            torch.randn(self.n_heads, self.head_dim, self.head_dim) * 0.02
        )


    def get_p_matrix(self):
        skew = self.p_skew_basis - self.p_skew_basis.transpose(-1, -2)
        return torch.matrix_exp(skew)

    def get_orthogonal_matrix(self):
        # A = M - M.T (Skew symmetric)
        # We broadcast the transpose over the last two dims
        skew = self.skew_basis - self.skew_basis.transpose(-1, -2)

        # Matrix Exp for each head independently
        # Result: [H, D_h, D_h]
        return torch.matrix_exp(skew)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # Get per-head rotations
        Rs = self.get_orthogonal_matrix() # [H, D, D]

        # 1. Reshape q_proj weight to [H, D_h, Dim]
        W_q = self.q_proj.weight.view(self.n_heads, self.head_dim, -1)

        # 2. Rotate each head's slice of the weight matrix
        # [H, D, D] @ [H, D, Dim] -> [H, D, Dim]
        W_k_heads = Rs @ W_q

        # 3. Flatten back to [Dim, Dim] for the linear layer
        # This effectively constructs a Block-Diagonal W_k
        W_k = W_k_heads.view(-1, self.n_embd)

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = F.linear(x, W_k).view(B, T, H, D).transpose(1, 2)


        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)


        q = F.rms_norm(q, (D,)) #never ever norm k, you stupid fuck

        q = self.rope(q)
        k = self.rope(k)

        # Soft Attention
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))

        mask = self.mask[:, :, :T, :T]

        soft_scores = F.softplus(scores)
        # STE: Forward sets small/neg values to 0, Backward ignores the zeroing
        # Values < 1e-6 do not participate in mass/scaling but receive gradients
        threshold = 1e-6
        pruned_scores = torch.where(soft_scores < threshold, torch.zeros_like(soft_scores), soft_scores)
        soft_scores = soft_scores + (pruned_scores - soft_scores).detach()

        soft_scores = soft_scores.masked_fill(mask == 0, 0.0) #prevent cheating here

        if self.training:
            # Create Distance Matrix [T, T]
            # dist[i, j] = i - j
            # We only care about positive distances (j <= i), which causal mask handles
            indices = torch.arange(T, device=x.device)
            dist = indices.view(-1, 1) - indices.view(1, -1)

            # Gaussian Decay Profile
            # P(drop) is high when dist is small (Recent)
            # P(drop) is low when dist is large (Distant)
            # We clamp dist to 0 to avoid NaNs, though masking handles it


            # Broadcast probabilities to batch/heads [1, 1, T, T]
            dist = dist.float().clamp(min=0)
            drop_probs = self.sd_alpha * torch.exp(-(dist**2) / (2 * self.sd_sigma**2))

            # Align dimensions: [1, 1, T, T]
            drop_probs = drop_probs.unsqueeze(0).unsqueeze(0)

            #dont allow damage to the first half, regardless
            #limited suppot positions do not get eroded
            limit = T // 2
            absolute_cutoff_mask = (dist > limit)
            drop_probs = drop_probs.masked_fill(absolute_cutoff_mask, 0.0)
            #Expand BEFORE sampling to ensure atomic independence
            # We must explicitly expand to [B, H, T, T] so Bernoulli rolls
            # a unique die for every single head and batch item.
            drop_probs_expanded = drop_probs.expand(B, H, T, T)

            # Generate Bernoulli Mask on the full tensor
            keep_mask = torch.bernoulli(1.0 - drop_probs_expanded).bool()

            # Apply Dropout
            soft_scores = soft_scores.masked_fill(~keep_mask, 0.0)
            #what this does is force attention to learn deeper patterns.
            #it also dramatically improves needles- it finds needles with same num batches,
            #despite randomly hiding needles. so it technically sees needle far sooner.


        soft_sums = soft_scores.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (soft_sums + 1e-6), max=1.0)
        attn = soft_scores * scale
        attn = torch.nan_to_num(attn, nan=0.0)

            # ===== Replace everything from y_context = attn @ v onward =====

        y_context = attn @ v  # (B, H, T, D)

        # === Mixing tensor: variance of the attended distribution ===
        v_sq = attn @ (v * v)  # E[v^2] under attention weights
        mix_variance = F.softplus(v_sq - y_context * y_context)  # Var[v] per dim, (B, H, T, D)

        # Project mix_variance into mixing directions
        # mix_proj learns to read the variance profile and output the principal mixing axis
        mix_dir = F.normalize(mix_variance, dim=-1)

        # === Residual sink ===
        current_mass = attn.sum(dim=-1, keepdim=True)
        residual_weight = 1.0 - F.sigmoid(current_mass)
        y_res = residual_weight * self.v_sink_residual

        # === XSA ===
        vn = F.normalize(v, dim=-1)
        y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn

        y = F.rms_norm(y_context, (D,)) +self.v_sink_basis + y_res
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
        Rs_p = self.get_p_matrix()  # (H, D, D)
        W_o_heads = self.o_proj.weight.view(self.n_heads, self.head_dim, -1)
        W_p_heads = Rs_p @ W_o_heads
        W_p = W_p_heads.view(-1, self.n_embd)

        o__out = self.o_proj(y_along_flat)      # mixing-axis content through O
        p__out = F.linear(y_ortho_flat, W_p)    # orthogonal content through P
        s__out = self.s_proj(poynting)      # poynting vector

        return self.s_mlp(s__out) + self.p_mlp(p__out) + self.o_mlp(o__out)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))

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

        # softplus "probabilities" -- same mechanism as your attention
        sp = F.softplus(flat_logits)

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
            smooth_loss = -torch.log(probs + 1e-8).mean(dim=-1)
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
            for _ in range(n_slices+1)
        ])

    def forward(self, h):
        logits = 0
        residual = h
        for i in range(self.n_slices):
            vn = F.normalize(self.dir_nets[i](h), dim=-1)
            component = (residual * vn).sum(dim=-1, keepdim=True) * vn
            residual = residual - component
            logits = logits + self.projs[i](component)

        # Add a new line here
        logits = logits + self.projs[-1](residual)

        return logits



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
        for block in self.transformer.h:
          block.attn.mask = self.mask #set here
        self.criterion = SoftplusCELoss(ignore_index=-1)
        self.lm_head = SubspaceUnembed(config.n_embd, config.vocab_size,config.n_layer)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)

        for i, block in enumerate(self.transformer.h):
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)

        if targets is not None:
            logits = self.lm_head(x)
            loss = self.criterion(logits, targets)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
