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
            # fast path: contiguous 0..T-1 positions
            if positions.ndim == 1 and positions.numel() > 0:
                expected = torch.arange(positions.numel(), device=positions.device)
                if torch.equal(positions, expected):
                    T = positions.numel()
                    cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)
                    sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
                    return cos, sin
        
            positions = positions.clamp(0, self.max_len - 1).long()
            cos = F.embedding(positions, self.cos_cached).unsqueeze(0).unsqueeze(0)
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


        self.q_proj = nn.Linear(dim, dim, bias=config.bias)
        self.k_proj = nn.Linear(dim, dim, bias=config.bias)
        self.v_proj = nn.Linear(dim, dim, bias=config.bias)
        self.p_proj = nn.Linear(dim, dim, bias=config.bias)
        self.o_proj = nn.Linear(dim, dim, bias=config.bias)

        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.sink_key = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.sink_value = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))

        self.mask = None #set in GPT main at model time to ensure its on GPU
        self.rope = config.rope

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = F.rms_norm(q, (D,), eps=1e-6) #never norm K. Ever. 
        q = self.rope(q)
        k = self.rope(k)

        mask = self.mask[:, :, :T, :T].expand(B, H, -1, -1)
       
        scores_real = (q @ k.transpose(-2, -1)) * (math.log(T) * math.log(D))
        sink_scores = (q @ self.sink_key.expand(B, -1, -1, -1).transpose(-2, -1)) * (1 * math.log(D))
        scores = torch.cat([scores_real, sink_scores], dim=-1)
        
        
       
        # Softplus magnitude path with sink
        null_col = torch.ones(1, 1, T, 1, device=x.device).expand(B, H, -1, -1)
        mask_use = torch.cat([mask, null_col], dim=-1)
        soft_scores = F.softplus(scores)
        threshold = 1e-6
        pruned_scores = torch.where(soft_scores < threshold, torch.zeros_like(soft_scores), soft_scores)
        soft_scores = soft_scores + (pruned_scores - soft_scores).detach()
        soft_scores = soft_scores.masked_fill(mask_use == 0, 0.0)
        soft_sums = soft_scores.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (soft_sums + 1e-6), max=1.0)
        m = mask_use.float()
        mass = (m * F.softplus(scores)).sum(dim=-1, keepdim=True) + 1e-6
    
        # stage 1
        d = (m * torch.sigmoid(scores)) / mass          # (B,H,T,T+1)
    
        # stage 2
        R = d.sum(dim=-2)                               # (B,H,T+1)
    
        # stage 3
        lam_geom_raw = F.softplus(R)
        lam_geom = lam_geom_raw / (lam_geom_raw.sum(dim=-1, keepdim=True) + 1e-6)
        responsibility = lam_geom.unsqueeze(-2) - lam_geom.mean(dim=-1, keepdim=True).unsqueeze(-2)       
        
        attn = soft_scores * scale
        attn = torch.nan_to_num(attn, nan=0.0)
        attn_real = attn[:, :, :, :T]
        attn_sinks= attn[:, :, :, T:]
        attn_resp_real =  attn[:, :, :, :T] * responsibility[:, :, :, :T]
        
        y = attn_real @ v + attn_sinks * self.sink_value
        y_resp = attn_resp_real @ v


        # XSA , then project
        vn = F.normalize(v, dim=-1)
        y_context = y - (y * vn).sum(dim=-1, keepdim=True) * vn
        y_resp_context = y_resp - (y_resp * vn).sum(dim=-1, keepdim=True) * vn

        y_context = y_context + y_resp_context

        y_context = F.rms_norm(y_context, (D,)) + self.v_sink_basis
        y_context_flat = y_context.transpose(1, 2).contiguous().view(B, T, -1)
        truth = self.o_proj(y_context_flat)
       
        
        return truth

def norm(x):
    return F.rms_norm(x, (x.size(-1),),eps=1e-6)

class Block(nn.Module):
    def __init__(self, config, block_idx):
        super().__init__()
        self.attn = Attention(config)
        self.attn_dir = MLP_bottle(config)
        self.config = config
        self.ffn = MLP(config)


    def forward(self, x):
        a = norm(x)
        z = self.attn(a)
        vn = F.normalize(self.attn_dir(a), dim=-1)
        q = x - (x * vn).sum(dim=-1, keepdim=True) * vn
        x = q + z
        x = x + self.ffn(norm(x))
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



class ContinuousContextCone(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        edge_proj_dim: int = 128,
        path_dim: int = 64,
        path_levels: int = 5,
        content_proj_dim: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_proj_dim = edge_proj_dim
        self.path_dim = path_dim
        self.path_levels = path_levels
        self.content_proj_dim = content_proj_dim

        rng = torch.Generator().manual_seed(seed)

        # content: project embeddings to a fixed-dim space before accumulating
        # this replaces the one-hot histogram with a projected-sum histogram
        self.register_buffer(
            'content_proj',
            torch.randn(embed_dim, content_proj_dim, generator=rng)
            / math.sqrt(embed_dim)
        )

        # edges: two independent random projections for prev/cur
        # their elementwise product is the transition signature
        self.register_buffer(
            'edge_proj_A',
            torch.randn(embed_dim, edge_proj_dim, generator=rng)
            / math.sqrt(embed_dim)
        )
        self.register_buffer(
            'edge_proj_B',
            torch.randn(embed_dim, edge_proj_dim, generator=rng)
            / math.sqrt(embed_dim)
        )

        # paths: one random projection per level, from embed_dim to path_dim
        # replaces path_dirs[lv][tok] lookup
        path_projs = torch.randn(
            path_levels, embed_dim, path_dim, generator=rng
        ) / math.sqrt(embed_dim)
        self.register_buffer('path_projs', path_projs)

        # same golden-ratio positional phases
        phases = torch.zeros(path_levels, 1024)
        for lv in range(path_levels):
            for p in range(1024):
                phases[lv, p] = math.cos(
                    (p + 1) * (lv + 1) * 2.3999632297286533
                )
        self.register_buffer('pos_phases', phases)

        # total fingerprint width
        self.fp_width = content_proj_dim + edge_proj_dim + path_dim * path_levels

        # final projection to embedding space
        rng_out = torch.Generator().manual_seed(seed + 999)
        proj_matrix = torch.randn(self.fp_width, embed_dim, generator=rng_out)
        proj_matrix = proj_matrix / math.sqrt(self.fp_width)
        self.register_buffer('direction_proj', proj_matrix)

    @torch.compiler.disable
    def encode(self, embeddings):
        """
        embeddings: (B, T, D) continuous embedding vectors

        Returns:
            fingerprints: (B, T, fp_width)
        """
        B, T, D = embeddings.shape
        device = embeddings.device

        # === Content: cumulative projected sum ===
        # replaces one-hot histogram with projected-embedding accumulation
        projected = embeddings @ self.content_proj  # (B, T, content_proj_dim)
        content = projected.cumsum(dim=1)

        # === Edges: elementwise product of projected consecutive pairs ===
        edges = torch.zeros(B, T, self.edge_proj_dim, device=device)
        if T > 1:
            prev_proj = embeddings[:, :-1] @ self.edge_proj_A  # (B, T-1, edge_proj_dim)
            cur_proj = embeddings[:, 1:] @ self.edge_proj_B
            edge_vecs = prev_proj * cur_proj
            edges[:, 1:] = edge_vecs.cumsum(dim=1)

        # === Paths: continuous n-gram projections ===
        path_parts = []
        for lv in range(self.path_levels):
            n = lv + 1
            accum = torch.zeros(B, T, self.path_dim, device=device)

            if T >= n:
                # project all embeddings through this level's random matrix
                # replaces path_dirs[lv][tok] lookup
                level_proj = embeddings @ self.path_projs[lv]  # (B, T, path_dim)

                for offset in range(n):
                    start_idx = offset
                    end_idx = T - n + 1 + offset
                    if end_idx <= start_idx:
                        continue
                    dirs = level_proj[:, start_idx:end_idx]  # (B, T-n+1, path_dim)
                    phase = self.pos_phases[lv, offset]
                    accum[:, (n-1):T, :] += dirs[:, :(T-n+1), :] * phase

                accum = accum.cumsum(dim=1)

            path_parts.append(accum)

        path = torch.cat(path_parts, dim=-1)
        fingerprints = torch.cat([content, edges, path], dim=-1)
        return fingerprints

    def warp(self, fingerprints):
        return fingerprints @ self.direction_proj

    def forward(self, embeddings):
        return self.warp(self.encode(embeddings))

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

        self.geometry = FlatRollGeometry(
            vocab_size=vocab_size,
            d_model=d_model,
            scale=geom_scale,
            seed=geom_seed,
            freeze=geom_freeze,
        )


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        e_bin = self.binary(input_ids)          # [B,T,d]
        e_geo = self.geometry.embed(input_ids)  # [B,T,d]
        base = e_bin + e_geo
        return base


import torch
import torch.nn.functional as F

def expected_rank_of_token(scores: torch.Tensor,
                           token_ids: torch.Tensor,
                           temperature: float = 1.0) -> torch.Tensor:
    """
    Computes the expected rank of the given token at each position, without full V x V matrix.
    """
    # scores: (..., V), token_ids: (...,)

    # Gather score of the target token
    score_i = scores.gather(-1, token_ids.unsqueeze(-1))  # (..., 1)

    # Δ_j = score_j - score_i
    diff = scores - score_i  # (..., V)

    # P(j beats i)
    p = torch.sigmoid(diff / temperature)

    # Expected rank = 1 + sum_j P(j > i)
    return 1.0 + p.sum(dim=-1)  # (...,)


def rank_future_sequence_loss_soft(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_future_steps: int = 15,
    decay: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "mean",
):
    """
    Memory-efficient smooth rank loss. For each t, matches rank of x_{t+Δ} to Δ.
    logits  … (B, T, V) – model scores
    targets … (B, T)    – token ids
    """
    B, T, V = logits.shape
    device = logits.device
    total_loss = torch.tensor(0.0, device=device)

    for Δ in range(2, max_future_steps + 1):
        if Δ >= T:
            break

        # Current time-step logits (for rank eval)
        cur_logits  = logits[:, :-Δ, :]          # (B, T−Δ, V)
        fut_targets = targets[:, Δ:]             # (B, T−Δ)

        # Efficient rank of ground-truth future token
        tgt_exp_rank = expected_rank_of_token(cur_logits, fut_targets, temperature)  # (B, T−Δ)

        # Penalize distance from desired rank Δ
        step_loss = F.l1_loss(
            tgt_exp_rank,
            torch.full_like(tgt_exp_rank, float(Δ)),
            reduction=reduction
        )

        # Apply decay for further future steps
        total_loss = total_loss + step_loss * (decay ** (Δ - 1))

    return total_loss

def ordered_future_loss(logits: torch.Tensor,
                        targets: torch.Tensor,
                        N: int = 15,
                        decay: float = 0.7,
                        tau: float = 1.0,
                        reduction: str = "mean"):
    """
    Penalise when the logits at step t do *not* respect the order of the next N tokens.

        top-1 logit should match token t+1
        top-2 logit should match token t+2
        ...
        top-N logit should match token t+N

    logits  – (B, T, V)
    targets – (B, T)
    """
    B, T, V = logits.shape
    device  = logits.device

    if N < 2:
        return torch.tensor(0., device=device)

    # windows where t+N fits in sequence
    valid_T = T - (N + 1)
    if valid_T <= 0:
        return torch.tensor(0., device=device)

    # (B, valid_T, N) → future token ids for each offset 2..N
    future_ids = torch.stack([targets[:, 2+k : 2+k+valid_T] for k in range(N)],
                         dim=-1)

    # (B, valid_T, N) → gather logits of those future tokens *now* (at step t)
    step_logits = logits[:, :valid_T, :].gather(
        -1, future_ids)                       # logit(x_{t+k})

    # pair-wise differences  Δ_{k,j} = logit_k − logit_j, shape (B, valid_T, N, N)
    diff = step_logits.unsqueeze(-1) - step_logits.unsqueeze(-2)

    # upper-triangular mask k<j (ignore diag & lower triangle)
    k_lt_j = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), 1)

    # logistic ranking loss
    pair_loss = F.softplus(-diff / tau)       # log(1+e^{-Δ/τ})
    pair_loss = pair_loss[..., k_lt_j]        # keep k<j entries, now shape (B, valid_T, M)

    # geometric weights per k (distance from current step)
    k_idx = torch.arange(N, device=device)
    weight = decay ** k_idx                   # shape (N,)
    # broadcast to pair-wise (k<j) selector
    weight_pair = weight.unsqueeze(-1).expand(N, N)[k_lt_j]  # (M,)

    pair_loss = pair_loss * weight_pair       # (B, valid_T, M)

    if reduction == "mean":
        return pair_loss.mean()
    elif reduction == "sum":
        return pair_loss.sum()
    else:                                     # 'none'
        return pair_loss                      # (B, valid_T, M)
#+ rank_future_sequence_loss_soft(logits,targets) * 1e-3 + ordered_future_loss(logits,targets) * 1e-3

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rope = RoPE(config.n_embd // config.n_head, max_len=config.block_size)
        self.config.rope = self.rope

        self.transformer = nn.ModuleDict(dict(
            wte=ThreePieceEmbedding(vocab_size=config.vocab_size,d_model=config.n_embd),
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
        self.hash=ContextCone(config.vocab_size,config.n_embd)
        self._boundary_handles = []
        #self.register_confined_backward()
        self.criterion = SoftplusCELoss(ignore_index=-1)
        self.lm_head = ParallelSubspaceUnembed(config.n_embd, config.vocab_size)
        self.time_head = ParallelSubspaceUnembed(config.n_embd, config.n_embd)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def context_cone_aux_loss(self, idx, x_normed, targets):
        B, T = idx.shape

        first_tok = idx[:, :1]
        gt_chain = torch.cat([first_tok, targets], dim=1)

        gt_embeds = self.hash(gt_chain)
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
            logits = self.lm_head(x[:, -1:, :])
            loss = None
        return logits, loss
