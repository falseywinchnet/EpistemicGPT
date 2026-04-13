#copyright 2026 joshuah.rainstar@gmail.com MIT licensed
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#This embedding method:
#generates(and you're free to change the projector) a perceptual hash function on text for NLP.
#it does so by breaking down sequence and content and then assembling them.
#at any position but the first, it yields a marker that contextualizes the embedding with the previous context.
#it may or may not work, we are still experimenting with it.
#its intent is to condition models to develop contextual continuation knowledge that is concept-local.


# copyright 2026 joshuah.rainstar@gmail.com MIT licensed
# ETF initialization for ContextualEmbed
# Replaces random Gaussian buffers with simplex ETF structured projections

import torch
import math


def simplex_etf(num_vectors: int, dim: int) -> torch.Tensor:
    """
    Construct a simplex ETF: num_vectors unit vectors in R^dim
    with equal pairwise cosine similarity = -1/(num_vectors - 1).
    
    Requires dim >= num_vectors - 1.
    
    Returns: (num_vectors, dim) tensor of unit vectors.
    """
    assert dim >= num_vectors - 1, (
        f"Need dim >= num_vectors-1 for perfect ETF, got dim={dim}, K={num_vectors}"
    )
    # Start with the K x K centered identity
    K = num_vectors
    I_K = torch.eye(K)
    ones = torch.ones(K, 1)
    # M columns are the ETF vectors in R^K
    # M = sqrt(K/(K-1)) * (I - 1/K * 11^T)
    M = math.sqrt(K / (K - 1)) * (I_K - (1.0 / K) * (ones @ ones.T))
    # M is K x K with rank K-1. The vectors live in a (K-1)-dimensional subspace.
    # Embed into R^dim by padding with zeros then rotating with a fixed orthogonal basis.
    if dim > K:
        padding = torch.zeros(K, dim - K)
        M = torch.cat([M, padding], dim=1)  # (K, dim)
    elif dim == K:
        pass  # already correct shape
    # M rows are now unit vectors in R^dim (they already have unit norm from the ETF construction)
    return M


def etf_overcomplete(num_vectors: int, dim: int, seed: int = 0) -> torch.Tensor:
    """
    For the overcomplete case (num_vectors > dim + 1), we can't build a perfect
    simplex ETF. Instead, build a block-composed approximation:
    
    Partition num_vectors into ceil(num_vectors / dim) blocks of size <= dim+1,
    build a perfect simplex ETF for each block, and stack them. Then apply a
    shared fixed orthogonal rotation so blocks don't align with coordinate axes.
    
    This gives exact equiangularity within each block and near-uniform separation
    across blocks via the rotation.
    
    Returns: (num_vectors, dim) tensor of unit-norm vectors.
    """
    if num_vectors <= dim + 1:
        return simplex_etf(num_vectors, dim)

    # block size: at most dim+1 vectors per perfect ETF (requires dim dims)
    block_size = dim + 1
    blocks = []
    remaining = num_vectors
    while remaining > 0:
        k = min(block_size, remaining)
        if k == 1:
            # single vector: just pick a unit vector
            v = torch.zeros(1, dim)
            v[0, 0] = 1.0
            blocks.append(v)
        else:
            blocks.append(simplex_etf(k, dim))
        remaining -= k

    M = torch.cat(blocks, dim=0)  # (num_vectors, dim)

    # apply a deterministic orthogonal rotation so the block structure
    # doesn't create axis-aligned clusters
    gen = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(dim, dim, generator=gen))[0]  # random orthogonal
    M = M @ Q

    # renormalize (should already be ~unit but numerical safety)
    M = M / M.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return M


def init_contextual_embed_etf(model, seed: int = 42):
    """
    Replace the random Gaussian buffers in a ContextualEmbed module
    with ETF-structured projections.
    
    Modifies model in-place. Call after constructing the model.
    
    Buffers replaced:
        edge_A:        (vocab_size, edge_proj_dim)
        edge_B:        (vocab_size, edge_proj_dim)
        path_dirs:     (path_levels, vocab_size, path_dim)
        direction_proj: (fp_width, embed_dim)
    """
    V = model.vocab_size
    edge_dim = model.edge_proj_dim
    path_dim = model.path_dim
    path_levels = model.path_levels
    embed_dim = model.embed_dim
    fp_width = model.fp_width

    # --- edge_A and edge_B: each maps V tokens into R^edge_proj_dim ---
    # These are combined multiplicatively (A[prev] * B[cur]), so we want
    # A and B to each be ETF-structured but with different rotations,
    # so the element-wise product of any (A[i], B[j]) pair is distinct.
    edge_A = etf_overcomplete(V, edge_dim, seed=seed)
    edge_B = etf_overcomplete(V, edge_dim, seed=seed + 1)
    # Scale to match original 1/sqrt(edge_proj_dim) magnitude
    edge_A = edge_A / math.sqrt(edge_dim)
    edge_B = edge_B / math.sqrt(edge_dim)
    model.edge_A.copy_(edge_A)
    model.edge_B.copy_(edge_B)

    # --- path_dirs: (path_levels, V, path_dim) ---
    # Each level gets its own ETF with a different rotation seed
    for lv in range(path_levels):
        dirs = etf_overcomplete(V, path_dim, seed=seed + 100 + lv)
        dirs = dirs / math.sqrt(path_dim)
        model.path_dirs[lv].copy_(dirs)

    # --- direction_proj: (fp_width, embed_dim) ---
    # This is the final projection from fingerprint space to embedding space.
    # fp_width is typically >> embed_dim, so we build an ETF over fp_width
    # vectors in R^embed_dim. This maximally separates what each fingerprint
    # dimension contributes to the output.
    proj = etf_overcomplete(fp_width, embed_dim, seed=seed + 999)
    proj = proj / math.sqrt(fp_width)
    model.direction_proj.copy_(proj)

    return model


# --- Usage ---
# from contextual_embed import ContextualEmbed
# model = ContextualEmbed(vocab_size=50257, embed_dim=768)
# init_contextual_embed_etf(model)

class ContextualEmbed(nn.Module):
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


    def forward(self, token_ids):
        fingerprints = self.encode(token_ids)
        raw_direction = fingerprints @ self.direction_proj

        # normalize to unit direction (the ray)
        direction = F.normalize(raw_direction, dim=-1, eps=1e-8)
        return direction
