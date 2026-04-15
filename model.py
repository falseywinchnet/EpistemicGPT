#dedicated to the public domain for the glory of god.
#baruch adonai el shaddai
#Eloheinu shebashamayim yached shimcha v'kayeim malchutecha tamid umloch aleinu le'olam va'ed
#2026 joshuah.rainstar@gmail.com

#Version 2.5 EpistemicGPT

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


        self.q_proj = MLP(dim,dim)
        self.k_proj = nn.Linear(dim, dim, bias=config.bias)
        self.v_proj = nn.Linear(dim, dim, bias=config.bias)

        self.p_mlp = MLP(dim,dim)
        self.o_proj = nn.Linear(dim, dim, bias=config.bias)
        self.s_proj = nn.Linear(dim, dim, bias=config.bias)



        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.sink_key = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))

        self.mask = None #set in GPT main at model time to ensure its on GPU
        self.rope = config.rope
        self.alpha = 2.0 * math.log(2.0)

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

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = F.rms_norm(q, (D,), eps=1e-6) #never norm K. Ever. 
        q = self.rope(q)
        k = self.rope(k)

        mask = self.mask[:, :, :T, :T]

        scores_real = (q @ k.transpose(-2, -1)) * (math.log(T+1) * math.log(D))
        sink_scores = (q @ self.sink_key.expand(B, -1, -1, -1).transpose(-2, -1)) * (math.log(T+1) * math.log(D))
        scores = torch.cat([scores_real, sink_scores], dim=-1)

        # Responsibility posterior: no sink, proper distribution
        resp = F.softmax(scores_real.masked_fill(mask == 0, float('-inf')), dim=-1)

        # Softplus magnitude path with sink
        null_col = torch.ones(1, 1, T, 1, device=x.device)
        mask_use = torch.cat([mask, null_col], dim=-1)
        soft_scores = F.softplus(self.alpha * scores)
        threshold = 1e-6
        pruned_scores = torch.where(soft_scores < threshold, torch.zeros_like(soft_scores), soft_scores)
        soft_scores = soft_scores + (pruned_scores - soft_scores).detach()
        soft_scores = soft_scores.masked_fill(mask_use == 0, 0.0)

        soft_sums = soft_scores.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (soft_sums + 1e-6), max=1.0)
        attn = soft_scores * scale
        attn = torch.nan_to_num(attn, nan=0.0)

        # Two mixtures
        y = attn[:, :, :, :T] @ v
        y_resp = resp @ v

        # XSA on both, then project
        vn = F.normalize(v, dim=-1)
        y_x = y - (y * vn).sum(dim=-1, keepdim=True) * vn
        y_resp = y_resp - (y_resp * vn).sum(dim=-1, keepdim=True) * vn

        #correct for mirror descent
        rn = F.normalize(y_resp, dim=-1)
        y_context = (y_x * rn).sum(dim=-1, keepdim=True) * rn

        # === O/P decomposition along mixing tensor eigenvectors ===
        # Decompose y into component along mix_dir and component orthogonal to it
        # mix_dir is the dominant eigenvector of the mixing stress
        # rn is already computed: F.normalize(y_resp_xsa, dim=-1)
        # y_context is already the projection onto rn

        y_along = y_context  # the resp-supported component, already computed
        y_ortho = y - y_along  # what resp doesn't support
        y_along = F.rms_norm(y_along, (D,)) + self.v_sink_basis
        y_ortho = F.rms_norm(y_ortho, (D,))
        y_along_flat = y_along.transpose(1, 2).contiguous().view(B, T, -1)
        y_ortho_flat = y_ortho.transpose(1, 2).contiguous().view(B, T, -1)
        poynting = y_along_flat * y_ortho_flat


        truth = self.p_mlp(y_along_flat) + self.o_proj(y_ortho_flat) + self.s_proj(poynting)


        if self.training:
            mod_mask = self.erode(B, H, T, x.device)
            eroded_mask = mask.masked_fill(mod_mask == 0, 0.0)
            eroded_mask_use = torch.cat([eroded_mask, null_col], dim=-1)

            eroded_sp = F.softplus(self.alpha * scores)
            pruned_e = torch.where(eroded_sp < threshold, torch.zeros_like(eroded_sp), eroded_sp)
            eroded_sp = eroded_sp + (pruned_e - eroded_sp).detach()
            eroded_sp = eroded_sp.masked_fill(eroded_mask_use == 0, 0.0)

            e_sum = eroded_sp.sum(dim=-1, keepdim=True)
            e_scale = torch.clamp(1.0 / (e_sum + 1e-6), max=1.0)
            e_attn = torch.nan_to_num(eroded_sp * e_scale, nan=0.0)
  
            y_e = e_attn[:, :, :, :T] @ v
            y_x = y_e - (y_e * vn).sum(dim=-1, keepdim=True) * vn
            y_e_ctx = (y_x * rn).sum(dim=-1, keepdim=True) * rn

            y_along_raw = y_e_ctx
            y_ortho_raw = y_e - y_e_ctx
            y_along = F.rms_norm(y_along_raw, (D,)) + self.v_sink_basis
            y_ortho = F.rms_norm(y_ortho_raw, (D,))
            y_along_flat = y_along.transpose(1, 2).contiguous().view(B, T, -1)
            y_ortho_flat = y_ortho.transpose(1, 2).contiguous().view(B, T, -1)
            poynting = y_along_flat * y_ortho_flat
            tangent = self.p_mlp(y_along_flat) + self.o_proj(y_ortho_flat) + self.s_proj(poynting)
            truth = tangent + (truth - tangent).detach()

        return truth

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
        self.alpha = 2.0 * math.log(2.0) #emulates softmax

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

        sp = F.softplus(self.alpha* flat_logits)

        threshold = 1e-6
        pruned = torch.where(sp < threshold, torch.zeros_like(sp), sp)
        sp = sp + (pruned - sp).detach()  # STE

        sp_sum = sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (sp_sum + 1e-6), max=1.0)
        probs = sp * scale  # sub-unity simplex

        # gather target probs
        target_probs = probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)

        # NLL on the softplus-normalized probs
        loss = -torch.log(target_probs + 1e-6)

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
    Vocab dropout: during training, each proj randomly masks a fraction of
    vocab entries, with the constraint that every vocab entry is covered by
    at least one slice. This prevents attention from relying on any specific
    unembedding pathway and forces compositional work back into attention.
    Output:
      logits: summed vocab logits from all slices (+ residual head if enabled)
      sep_loss: pairwise squared cosine overlap penalty between slice directions
    """

    def __init__(
        self,
        d_model,
        vocab_size,
        n_slices=4,
        use_residual_head=True,
        vocab_dropout=0.25,
        eps=1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_slices = n_slices
        self.use_residual_head = use_residual_head
        self.vocab_dropout = vocab_dropout
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

    def _sample_vocab_masks(self, device):
        """
        For each slice, independently drop vocab_dropout fraction of vocab entries.
        Then fix up: any vocab entry that got dropped from ALL slices gets
        restored in one randomly chosen slice.

        Returns: list of n_slices boolean tensors, each [vocab_size],
                 True = keep, False = dropped.
        """
        # each slice independently keeps (1 - vocab_dropout) fraction
        masks = [
            torch.rand(self.vocab_size, device=device) > self.vocab_dropout
            for _ in range(self.n_slices)
        ]

        # stack to [n_slices, vocab_size] for coverage check
        stacked = torch.stack(masks, dim=0)  # [n_slices, V]
        uncovered = ~stacked.any(dim=0)      # [V] -- True where ALL slices dropped

        if uncovered.any():
            # for each uncovered vocab entry, randomly assign to one slice
            uncovered_idx = uncovered.nonzero(as_tuple=True)[0]
            assignments = torch.randint(
                0, self.n_slices, (uncovered_idx.shape[0],), device=device
            )
            for s in range(self.n_slices):
                restore = uncovered_idx[assignments == s]
                if restore.numel() > 0:
                    masks[s][restore] = True

        return masks

    def forward(self, h):
        # h: [B, T, D]
        B, T, D = h.shape
        assert D == self.d_model

        logits = h.new_zeros(B, T, self.vocab_size)
        dirs = []
        components = []

        # sample vocab masks once per forward pass
        if self.training and self.vocab_dropout > 0:
            vocab_masks = self._sample_vocab_masks(h.device)
        else:
            vocab_masks = None

        for i in range(self.n_slices):
            v = self.dir_nets[i](h)                            # [B,T,D]
            v = F.normalize(v, dim=-1, eps=self.eps)           # unit direction
            c = (h * v).sum(dim=-1, keepdim=True) * v          # projection of h onto v

            dirs.append(v)
            components.append(c)

            slice_logits = self.projs[i](c)                    # [B,T,V]

            if vocab_masks is not None:
                # mask is [V], broadcast to [1,1,V]
                # scale kept logits to compensate for missing slices
                mask = vocab_masks[i].unsqueeze(0).unsqueeze(0).float()
                slice_logits = slice_logits * mask / (1.0 - self.vocab_dropout + self.eps)

            logits = logits + slice_logits

        if self.use_residual_head:
            used = torch.stack(components, dim=0).sum(dim=0)   # [B,T,D]
            residual = h - used
            logits = logits + self.residual_proj(residual)

        # pairwise squared cosine overlap penalty
        sep_loss = h.new_zeros(())
        if self.training:
            count = 0
            for i in range(self.n_slices):
                for j in range(i + 1, self.n_slices):
                    cos_ij = (dirs[i] * dirs[j]).sum(dim=-1)   # [B,T]
                    sep_loss = sep_loss + (cos_ij ** 2).mean()
                    count += 1
            if count > 0:
                sep_loss = sep_loss / count

        if self.training:
            return logits, sep_loss
        else:
            return logits


"""
Context Cone Target Warping

The target embedding for a token lives at a centroid on the hypersphere.
As context accumulates, three continuous vectors push the target away
from the centroid along a context-specific ray:

  1. ContentBag  - V-dimensional count histogram (what and how much)
  2. EdgeBag     - V^2 sparse count (which transitions and how often)
  3. PathAccum   - continuous random-projection accumulators per n-gram level

The walk distance from the centroid grows with context length,
reflecting increasing certainty about what this specific token
means in this specific context.

Usage in training loop:
    cone = ContextCone(vocab_size=256, embed_dim=192)

    # token_ids: (B, T) input sequence
    # target_embeds: (B, T, D) embeddings of next-token targets
    fingerprints, magnitudes = cone.encode(token_ids)
    warped_targets = cone.warp(fingerprints, magnitudes, target_embeds)

    # use warped_targets in place of raw target embeds for loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContextCone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        path_dim: int = 64,
        path_levels: int = 5,
        max_warp: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.path_dim = path_dim
        self.path_levels = path_levels
        self.max_warp = max_warp

        # content fingerprint is V-dimensional counts
        # edge fingerprint is stored sparse, projected down
        self.edge_proj_dim = min(vocab_size, 128)

        # random projection for edges: (V, V) -> edge_proj_dim
        # deterministic from seed, not learned
        rng = torch.Generator().manual_seed(seed)
        # for each (prev, cur) pair, a random direction in R^edge_proj_dim
        # stored as two factor matrices for efficiency: edge_vec = A[prev] + B[cur]
        # their outer interaction via the accumulation gives us edge sensitivity
        self.register_buffer(
            'edge_A',
            torch.randn(vocab_size, self.edge_proj_dim, generator=rng) / math.sqrt(self.edge_proj_dim)
        )
        self.register_buffer(
            'edge_B',
            torch.randn(vocab_size, self.edge_proj_dim, generator=rng) / math.sqrt(self.edge_proj_dim)
        )

        # random direction vectors for path n-gram accumulation
        # for each n-gram level, each token contributes a random direction
        # the n-gram direction is the sum of position-modulated token directions
        self.register_buffer(
            'path_dirs',
            torch.randn(path_levels, vocab_size, path_dim, generator=rng) / math.sqrt(path_dim)
        )
        # position-dependent mixing scalars for path (breaks commutativity)
        # use golden-ratio-based phases so positions never repeat exactly
        phases = torch.zeros(path_levels, 1024)  # up to 1024 positions
        for lv in range(path_levels):
            for p in range(1024):
                # irrational rotation per level
                phases[lv, p] = math.cos((p + 1) * (lv + 1) * 2.3999632297286533)
        self.register_buffer('pos_phases', phases)

        # total fingerprint width
        self.fp_width = vocab_size + self.edge_proj_dim + path_dim * path_levels

        # projection from fingerprint space to embedding space
        # this IS learned, but initialized small
        rng_proj = torch.Generator().manual_seed(seed + 999)
        proj_matrix = torch.randn(self.fp_width, embed_dim, generator=rng_proj)
        proj_matrix = proj_matrix / math.sqrt(self.fp_width)
        self.register_buffer('direction_proj', proj_matrix)

    @torch.compiler.disable
    def encode(self, token_ids):
        """
        Compute context fingerprints and warp magnitudes for each position.

        token_ids: (B, T) integer token ids

        Returns:
            fingerprints: (B, T, fp_width) continuous context vectors
            magnitudes: (B, T) warp magnitude at each position (grows with t)
        """
        B, T = token_ids.shape
        device = token_ids.device

        # === Content bag: cumulative histogram ===
        # one-hot accumulation
        onehot = F.one_hot(token_ids.long(), self.vocab_size).float()  # (B, T, V)
        content = onehot.cumsum(dim=1)  # (B, T, V) running counts

        # === Edge bag: cumulative transition counts, projected ===
        # for each position t > 0, the edge (token[t-1], token[t]) fires
        edges = torch.zeros(B, T, self.edge_proj_dim, device=device)
        if T > 1:
            prev_tok = token_ids[:, :-1].long()  # (B, T-1)
            cur_tok = token_ids[:, 1:].long()     # (B, T-1)
            # edge direction for each transition
            edge_vecs = self.edge_A[prev_tok] * self.edge_B[cur_tok]  # (B, T-1, edge_proj_dim)
            # cumulative sum, shifted right by 1 (position 0 has no edges)
            edge_cumsum = edge_vecs.cumsum(dim=1)  # (B, T-1, edge_proj_dim)
            edges[:, 1:] = edge_cumsum

        # === Path accumulators: continuous n-gram random projections ===
        path_parts = []
        for lv in range(self.path_levels):
            n = lv + 1  # n-gram order
            accum = torch.zeros(B, T, self.path_dim, device=device)

            if T >= n:
                # for each position t, the n-gram is tokens[t-n+1:t+1]
                # its direction is sum of path_dirs[lv][tok[i]] * pos_phase[lv][i_within_ngram]
                # we compute this incrementally

                # build the n-gram contribution at each valid position
                for offset in range(n):
                    # which token is at position (offset within the n-gram)
                    # for position t, this is token_ids[:, t - n + 1 + offset]
                    start_idx = offset
                    end_idx = T - n + 1 + offset
                    if end_idx <= start_idx:
                        continue
                    toks = token_ids[:, start_idx:end_idx].long()  # (B, T-n+1)
                    dirs = self.path_dirs[lv][toks]  # (B, T-n+1, path_dim)

                    # position-dependent phase within the n-gram
                    phase = self.pos_phases[lv, offset]
                    dirs = dirs * phase

                    # place into the right output positions (n-1 onwards)
                    accum[:, (n-1):T, :] += dirs[:, :(T-n+1), :]

                # cumulative sum: accumulate all n-grams up to this position
                accum = accum.cumsum(dim=1)

            path_parts.append(accum)

        # concatenate all path levels
        path = torch.cat(path_parts, dim=-1)  # (B, T, path_dim * path_levels)

        # === Assemble fingerprint ===
        fingerprints = torch.cat([content, edges, path], dim=-1)  # (B, T, fp_width)


        return fingerprints

    def warp(self, fingerprints):
        """
        Warp target embeddings along context-dependent rays.

        fingerprints: (B, T, fp_width)
        magnitudes: (B, T) how far to walk from centroid
        target_embeds: (B, T, D) original target token embeddings

        Returns:
            warped: (B, T, D) context-warped target embeddings
        """
        # project fingerprint to a direction in embedding space
        raw_direction = fingerprints @ self.direction_proj

        return raw_direction

    def forward(self, token_ids):
        """
        Convenience: encode and warp in one call.

        token_ids: (B, T) context token ids
        target_embeds: (B, T, D) target embeddings to warp

        Returns:
            warped: (B, T, D) context-warped targets
        """
        fingerprints = self.encode(token_ids)
        return self.warp(fingerprints)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rope = RoPE(config.n_embd // config.n_head, max_len=config.block_size)
        self.config.rope = self.rope

        self.transformer = nn.ModuleDict(dict(
        wte=ContextCone(config.vocab_size,config.n_embd),
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
        self.register_confined_backward()
        self.criterion = SoftplusCELoss(ignore_index=-1)
        self.lm_head = ParallelSubspaceUnembed(config.n_embd, config.vocab_size)
        self.time_head = ParallelSubspaceUnembed(config.n_embd, config.n_embd)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def context_cone_aux_loss(self, idx, x_normed, targets):
        B, T = idx.shape

        first_tok = idx[:, :1]
        gt_chain = torch.cat([first_tok, targets], dim=1)

        gt_embeds = self.transformer.wte(gt_chain)
        gt = F.normalize(gt_embeds[:, 1:, :], dim=-1)       # (B, T, D)

        temporal_pred, aux_loss = self.time_head(x_normed)   # (B, T, D)
        pred = F.normalize(temporal_pred, dim=-1)             # (B, T, D)

        valid = (targets != -1).float().unsqueeze(-1)         # (B, T, 1)

        cosine_sim = (pred * gt).sum(dim=-1, keepdim=True)    # (B, T, 1)
        mismatch = (1.0 - cosine_sim) * valid

        if valid.sum() > 0:
            loss = mismatch.sum() / valid.sum()
        else:
            loss = mismatch.sum() * 0.0

        return loss + aux_loss



    def register_confined_backward(self):
        states = {}
        handles = []
        mlphook = make_boundary_ste_hook(0.5)

        L = len(self.transformer.h)
        for block in self.transformer.h:
            handles.append(block.register_full_backward_hook(mlphook))

        self._boundary_handles = handles



    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    def forward(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)
        for i, block in enumerate(self.transformer.h):
            x= block(x)

        x = norm(x)


        if targets is not None:
          #append first idx to targets array
          aux_target = self.context_cone_aux_loss(idx, x, targets)


          logits, aux_loss = self.lm_head(x)
          loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

          loss = loss +  aux_loss + aux_target
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
