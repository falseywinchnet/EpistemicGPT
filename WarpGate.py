#dedicated to the public domain for the glory of god.
#baruch adonai el shaddai 
#Eloheinu shebashamayim yached shimcha v'kayeim malchutecha tamid umloch aleinu le'olam va'ed 
#2026 joshuah.rainstar@gmail.com

#Version 2.1 EpistemicGPT
#current notes:
#your choice on whether to key the carving directions in the subspaceunembed to n_layer or a fixed budget\
#and whether to run them all in parallel or sequential
#your choice on lelu gelu or some other nonlinearity
#figure out some kind of ste alpha decay schedule or dont use


'''
Train all MLPS of form(linear,LELU,linear) with the following LR
width LR
4 2.0667168855e-01
8 1.0333584427e-01
16 5.1667922137e-02
32 2.5833961069e-02
64 1.2916980534e-02
128 6.4584902671e-03
192 4.3056601781e-03
256 3.2292451336e-03
384 2.1528300890e-03
512 1.6146225668e-03
1024 8.0731128339e-04
2048 4.0365564170e-04
4096 2.0182782085e-04
8192 1.0091391042e-04
16384 5.0456955212e-05
Generalization error bounds for two-layer
neural networks with Lipschitz loss function
Jiang Yu Nguwi∗ Nicolas Privault
 1) activation/module-scale Lipschitz factor for the RMSNorm-guarded MLP
       L_eff ~= 1.0998393201 * sqrt(dim)

    2) expected top Hessian eigenvalue for the simplified linear+squared-loss model
       with Gaussian batch inputs centered coordinatewise at 1.09984:
       lambda_max = sigma^2 + dim * (1.09984)^2

    3) hypothetical gradient-descent learning-rate edge and a conservative mean
       lr_edge = 2 / lambda_max
       lr = lr_edge / 2

further estimations, calculations suggest that the LR on attention should be the MLP but divided by the highest width of the internal representation.
in an OST model, its lr/3.

for embedding layer, LR is allowed to be quite large.
(1 / log(sqrt(max(dim, vocab))))*0.5

for unembedding(assume we will parallelize, not accumulate sequential): 
lr_unembed = (1 / (sqrt(n_slices + 1) * log(max(dim, vocab))))*0.5

Scheduling should decay the entire model gradually to a lower bound.  that lowest possible bound that is useful to train is quite simply 1 / (1099.84 * sqrt(max(dim,vocab)).
training below this may not be effective or useful. 

any change to nonlinearities, any scaling operations, any use of softmax objectively will change and invalidate this prospective learning rate schedule. 


'''
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



class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.depth = 0
        self.n_heads = config.n_head
        self.n_embd = config.n_embd
        dim = config.n_embd
        self.head_dim = dim // self.n_heads
        

        self.q_proj = nn.Linear(dim,dim,bias=False)
        self.k_proj = nn.Linear(dim,dim,bias=False)
        self.v_proj =nn.Linear(dim,dim,bias=False)
        self.o_proj =nn.Linear(dim,dim,bias=False)
        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.sink_key = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))

        self.mask = None #set in GPT main at model time to ensure its on GPU
        self.rope = config.rope


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

        q = F.rms_norm(q, (D,),eps=1e-6) #never ever norm k, you stupid fuck

        q = self.rope(q)
        k = self.rope(k)
        k2 = torch.cat([k, self.sink_key.expand(B, -1, -1, -1)], dim=2)

        scores = (q @ k2.transpose(-2, -1)) * (math.log(T+1) * math.log(D))
        # Soft Attention

        mask = self.mask[:, :, :T, :T]
        null_col = torch.ones(1, 1, T, 1, device=x.device)
        mask_use = torch.cat([mask, null_col], dim=-1) 
        soft_scores = F.softplus(scores) #zero point mass is log(2)/alpha or ~0.382.
        # STE: Forward sets small/neg values to 0, Backward ignores the zeroing
        # Values < 1e-6 do not participate in mass/scaling but receive gradients
        threshold = 1e-6
        pruned_scores = torch.where(soft_scores < threshold, torch.zeros_like(soft_scores), soft_scores)
        soft_scores = soft_scores + (pruned_scores - soft_scores).detach()
        soft_scores = soft_scores.masked_fill(mask_use == 0, 0.0) #prevent cheating here
       
        soft_sums = soft_scores.sum(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (soft_sums + 1e-6), max=1.0)
        attn = soft_scores * scale
        attn = torch.nan_to_num(attn, nan=0.0)
        attn_real = attn[:, :, :, :T]   # drop the null column
        y_context = attn_real @ v
        
        vn = F.normalize(v, dim=-1)
        y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn
        y = F.rms_norm(y_context, (D,)) + self.v_sink_basis    
        y_flat = y.transpose(1, 2).contiguous().view(B, T, -1)

        attn_flat = attn[:, :, :, :T].mean(dim=1)  # (B, T, T)
        mixture = attn_flat @ x

        xn = F.normalize(x, dim=-1)  # (B, T, C)
        mix2 = attn_flat @ xn

        weighted_src = F.normalize(mix2, dim=-1)
        mutual = mixture - (mixture * weighted_src).sum(dim=-1, keepdim=True) * weighted_src
        
        truth = self.o_proj(y_flat)  


        if self.training:
            mod_mask = self.erode(B,H,T,x.device)
            mask =  mask.masked_fill(mod_mask == 0, 0.0)
            eroded_attn  = attn_real.masked_fill(mask == 0, 0.0)
        
            y_context =eroded_attn @ v
            y_context = y_context - (y_context * vn).sum(dim=-1, keepdim=True) * vn
            y = F.rms_norm(y_context, (D,)) + self.v_sink_basis    
            y_flat = y.transpose(1, 2).contiguous().view(B, T, -1)

            tangent= self.o_proj(y_flat)
            truth = tangent + (truth - tangent).detach()

        
        return truth + mutual
        
def norm(x):
    return F.rms_norm(x, (x.size(-1),),eps=1e-6)

class Block(nn.Module):
    def __init__(self, config, block_idx):
        super().__init__()
        self.attn = Attention(config)
        self.attn_dir = MLP_bottle(config)
        self.ffn = MLP(config)
        self.config = config

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

        sp = F.softplus( flat_logits)

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

        self.embd = nn.Embedding(vocab_size,d_model)

        self.geometry = FlatRollGeometry(
            vocab_size=vocab_size,
            d_model=d_model,
            scale=geom_scale,
            seed=geom_seed,
            freeze=geom_freeze,
        )


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        e_bin = self.binary(input_ids)          # [B,T,d]
        e_embd = self.embd(input_ids)          # [B,T,d]
        e_geo = self.geometry.embed(input_ids)  # [B,T,d]

        base = e_bin + e_geo
        #out = base + e_embd
        return base

'''
ThreePieceEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.n_embd,
                bottleneck_width=16,#something close to the binary width
                cayley_expansions=3,
                use_hamming=True,
                geom_scale="box",
                geom_seed=0,
                geom_freeze=True),


'''


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

        # === Warp magnitude ===
        # grows with position, representing increasing context confidence
        # use log(t+1) so it grows but decelerates
        positions = torch.arange(T, device=device, dtype=torch.float32)
        magnitudes = torch.log1p(positions).unsqueeze(0).expand(B, -1)  # (B, T)
        # normalize so max magnitude = max_warp
        if T > 1:
            magnitudes = magnitudes * (self.max_warp / magnitudes[:, -1:].clamp(min=1e-6))
        else:
            magnitudes = magnitudes * self.max_warp

        return fingerprints, magnitudes

    def warp(self, fingerprints, magnitudes, target_embeds):
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

        # normalize to unit direction (the ray)
        direction = F.normalize(raw_direction, dim=-1, eps=1e-8)

        # remove component parallel to target (so warp is tangent to sphere)
        target_norm = F.normalize(target_embeds, dim=-1, eps=1e-8)
        parallel = (direction * target_norm).sum(dim=-1, keepdim=True) * target_norm
        tangent_direction = F.normalize(direction - parallel, dim=-1, eps=1e-8)

        # walk along the tangent direction, magnitude scales with context length
        displacement = tangent_direction #* magnitudes.unsqueeze(-1)

        # apply displacement and renormalize to preserve original norm
        warped = target_embeds + displacement
        orig_norm = target_embeds.norm(dim=-1, keepdim=True)
        warped = F.normalize(warped, dim=-1, eps=1e-8) * orig_norm

        return warped

    def forward(self, token_ids, target_embeds):
        """
        Convenience: encode and warp in one call.
        
        token_ids: (B, T) context token ids
        target_embeds: (B, T, D) target embeddings to warp
        
        Returns:
            warped: (B, T, D) context-warped targets
        """
        fingerprints, magnitudes = self.encode(token_ids)
        return self.warp(fingerprints, magnitudes, target_embeds)


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

        self._boundary_handles = []
        self._linear_states = {}
        #self.register_confined_backward() experiment with at LATER TIME
        self.criterion = SoftplusCELoss(ignore_index=-1)
        self.lm_head = ParallelSubspaceUnembed(config.n_embd, config.vocab_size)
        self.warp = ContextCone(config.vocab_size,config.n_embd)
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
        


    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    def forward(self, idx, targets=None):
        b, T = idx.size()
        x = self.transformer.wte(idx)
        x = self.warp(idx,x)
        for i, block in enumerate(self.transformer.h):
            x= block(x)

        x = norm(x)

        fingerprints, magnitudes = self.warp.encode(idx)

        if targets is not None:
          
          # h_next is the model's own representation of the target positions
          # x is (B, T, D) after all layers and norm
          # x[:, 1:] is h at positions 1..T-1, which are the targets for positions 0..T-2
          h_current = x[:, :-1]  # (B, T-1, D)
          h_next = x[:, 1:].detach()  # (B, T-1, D) -- detach so we don't backprop through target
          
          fp = fingerprints[:, :-1]  # context at positions 0..T-2
          mag = magnitudes[:, :-1]
          
          warped_next = self.warp.warp(fp, mag, h_next)
          
          h_norm = F.normalize(h_current, dim=-1)
          cos_warped = (h_norm * F.normalize(warped_next, dim=-1)).sum(dim=-1)
          cos_unwarped = (h_norm * F.normalize(h_next, dim=-1)).sum(dim=-1)
          
          specificity_loss = F.relu(cos_unwarped - cos_warped).mean()
          
          logits, aux_loss = self.lm_head(x)
          loss = self.criterion(logits, targets) +  aux_loss + specificity_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
