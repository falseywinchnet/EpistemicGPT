#copyright 2026 joshuah.rainstar@gmail.com
#MIT- take this and use it, but please credit me.
#Version 1.0 EpistemicGPT
#You can contribute- I need triton code

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
        # x: (B, H, T, D)
        if positions is None:
            T = x.shape[-2]
            positions = torch.arange(T, device=x.device)
        
        cos, sin = self.get_embeddings(positions, x.device)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat((y1, y2), dim=-1)



class AnchorPenaltyInjector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, radius, eta, penalty_lambda):
        ctx.save_for_backward(delta, radius)
        ctx.eta = eta
        ctx.penalty_lambda = penalty_lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        delta, radius = ctx.saved_tensors
        violation = torch.relu(radius - ctx.eta)
        # Gradient points inwards. 
        # Added small epsilon to denominator to prevent NaN on perfect zeros
        grad_penalty = (2 * ctx.penalty_lambda * violation / (radius + 1e-6)) * delta
        return grad_output + grad_penalty, None, None, None, None

class FrobHeadNorm(nn.Module):
    def __init__(self, n_heads, head_dim):
        super().__init__()
        self.eta = head_dim 
        # Adjusted for [B, H, T, D] broadcasting
        self.anchor = nn.Parameter(torch.zeros(1, n_heads, 1, head_dim))

    def forward(self, x):
        # x is [B, H, T, D]
        delta = x - self.anchor
        radius = torch.norm(delta, p=2, dim=-1, keepdim=True) + 1e-6
        
        if self.training:
            x_conditioned = AnchorPenaltyInjector.apply(
                x, delta, radius, self.eta, 1.0
            )
            scale = torch.clamp(self.eta / radius, max=1.0)
            return self.anchor + ((x_conditioned - self.anchor) * scale)
        else:

            return x

class FrobNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.eta = dim #seems to be correct, unclear? 16 worked for head dim of 16
        # Shape: (1, 1, n_heads, head_dim) for easy broadcasting with [B, T, C]
        self.anchor = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x is [B, T, C]
        delta = x - self.anchor
        # Norm is taken across the dim (C)
        radius = torch.norm(delta, p=2, dim=-1, keepdim=True) + 1e-6
        
        if self.training:
            x_conditioned = AnchorPenaltyInjector.apply(
                x, delta, radius, self.eta, 1.0
            )
            scale = torch.clamp(self.eta / radius, max=1.0)
            return self.anchor + ((x_conditioned - self.anchor) * scale)
        else:
            return x
            
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = FrobNorm(config.n_embd)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = LELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.norm(x)
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

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.q_proj.weight) # Identity Init
        self.v_sink_residual = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        
        self.mask = config.mask
        self.rope = config.rope
        limit = config.block_size // 2
            
        # alpha: Max dropout probability (at distance 0/immediate neighbor)
        #sigma- set here to be half the block size.
        #so, 30% dropout at the event horizon,
        #tapering to 0% at halfway.
        self.sd_alpha = getattr(config, 'sd_alpha', 0.3) 
        self.sd_sigma = getattr(config, 'sd_sigma',  limit / 3.0)
        self.qnorm = FrobHeadNorm(config.n_head, self.head_dim)
        self.vnorm = FrobHeadNorm(config.n_head, self.head_dim)


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

        #now, why is k a rotation on Q?
        #ensures k cannot sacrifice to q, q pressures ansi and wants iso
        #make all a tradeoff. realistically sufficient to pair them- one is lookup into other.
        #using reflection here instead of a proper aux loss on k to promote iso is for simplification.
        #the difference has not been ablated.
        
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        q = self.qnorm(q) #never ever norm k, you stupid fuck

        q = self.rope(q)
        k = self.rope(k)
    
        # Soft Attention
        scores = (q @ k.transpose(-2, -1))
        
        mask = self.mask[:, :, :T, :T].to(device=x.device)

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
            #soft_scores = soft_scores.masked_fill(~keep_mask, 0.0)
            #what this does is force attention to learn deeper patterns. 
            #it also dramatically improves needles- it finds needles with same num batches,
            #despite randomly hiding needles. so it technically sees needle far sooner.


        soft_sums = soft_scores.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (soft_sums + 1e-6), max=1.0)
        attn = soft_scores * scale
        attn = torch.nan_to_num(attn, nan=0.0)
        y_context = attn @ v

        current_mass = attn.sum(dim=-1, keepdim=True)
        residual = 1.0 - F.sigmoid(current_mass)
        y_res = residual * self.v_sink_residual
        y = self.vnorm(y_context) + self.v_sink_basis + y_res #the purpose of a sink is to inject structure
        y = y.transpose(1,2).contiguous() #get in b,t,h,d

        y = y.view(B, T, -1)

        return self.o_proj(y)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
    def forward(self, x):  
        x = x + self.attn(x) #attention is only ever a diffusive, additive adjustment. 
        x = x + self.mlp(x) #post-norm in like highway, but only on MLP.
        #the purpose of an MLP is to restore structure. 
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 66
    n_layer: int = 4
    n_head: int = 4
  
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    rope: nn.Module = None
    mask: torch.Tensor = None

class ConstrainedSimplexLoss(nn.Module):
    """
    A CrossEntropy wrapper that conditions the gradient descent to the 
    tangent cone of the probability simplex (Sum(p)=1).
    Cramer-Rao Bound for Arbitrarily Constrained Sets

    Heedong Do, Member, IEEE, Angel Lozano, Fellow, IEEE
    
    Paper alignment:
    - Enforces the 'span of the tangent cone' constraint[cite: 7, 37].
    - Applies Projection Matrix Pi derived in Theorem 3[cite: 165].
    - Valid for singular Fisher Information Matrices common in ML[cite: 6].
    """
    def __init__(self, vocab_size=66, ignore_index=-1):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
        # Construct Projection Matrix Pi = I - (1/V) * 11^T
        # This projects gradients onto the zero-sum subspace (the tangent space of the simplex).
        I = torch.eye(vocab_size)
        ones = torch.ones(vocab_size, vocab_size)
        
        # Register as a buffer so it saves with the model but doesn't update as a parameter
        self.register_buffer('Pi', I - (ones / vocab_size))

    def _project_gradient(self, grad):
        """
        Backward hook: Projects gradients onto the geometric tangent plane.
        This ensures updates strictly respect the conservation of probability mass.
        """
        # Handle flattened or batched gradients
        if grad.dim() > 1:
            return grad @ self.Pi
        return self.Pi @ grad

    def forward(self, logits, targets):
        """
        Args:
            logits: (Batch, Seq_Len, Vocab) or (Batch, Vocab)
            targets: (Batch, Seq_Len) or (Batch)
        """
        # 1. Flatten for CrossEntropy (matches your snippet's logic)
        # view(-1, vocab_size) ensures compatibility with (B, T, V) or (B, V)
        flat_logits = logits.view(-1, self.vocab_size)
        flat_targets = targets.view(-1)

        # 2. Register the geometric constraint hook on the computation graph
        # This intercepts the backward pass before it reaches the model parameters.
        if flat_logits.requires_grad:
            flat_logits.register_hook(self._project_gradient)

        # 3. Compute standard CE loss
        loss = F.cross_entropy(
            flat_logits, 
            flat_targets, 
            ignore_index=self.ignore_index
        )
        
        return loss        # This handles the singularity by clamping the ratio.
        

    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rope = RoPE(config.n_embd // config.n_head, max_len=config.block_size)
        self.config.rope = self.rope

        mask_tensor = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("mask", mask_tensor)
        self.config.mask = self.mask

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(config.n_layer)]),
        ))
        self.criterion = ConstrainedSimplexLoss(vocab_size=config.vocab_size, ignore_index=-1)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.norm = FrobNorm(config.n_embd)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)
        x = self.norm(x)

        for block in self.transformer.h:
            x = block(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = self.criterion(logits, targets)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
