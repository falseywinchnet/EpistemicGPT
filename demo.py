#MIT copyright joshuah.rainstar@gmail.com 2025
#can you improve on these results? i dont mean by tweaking numbers.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- CONSTANTS ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 64
VOCAB_SIZE = 64
EMBED_DIM = 128
STEPS = 3000
LR = 2e-3
IGNORE_INDEX = -100


# --- NEW GENERATORS ---

def generate_reverse_task(batch_size, seq_len, vocab_size, device):
    """
    Task: Output the sequence in reverse order.
    Input: [t1, t2, t3, ...]
    Target: [..., t3, t2, t1]
    Note: Highly difficult for standard causal attention without Bi-RoPE.
    """
    # Generate random sequence
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Target is the reverse of x
    y = torch.flip(x, dims=[1])

    # Mask out the first half of targets to let the model see some context before predicting
    # (Optional: for pure reverse, we might want to predict immediately,
    # but let's give it a prefix context to make it solvable)
    # Actually, for a pure reverse transformer, it usually needs the full context.
    # Let's do a 'Mirror' task: Input [A, B, C], Target [C, B, A] appended.

    half = seq_len // 2
    prefix = torch.randint(0, vocab_size, (batch_size, half), device=device)

    # Input: [Prefix, Zero_Padding] (Model must fill padding)
    # Actually, standard training is Teacher Forced.
    # Input: [A, B, C, C, B, A]

    x = torch.cat([prefix, torch.flip(prefix, dims=[1])], dim=1)
    y = x.roll(-1, dims=1)

    y[:, -1] = IGNORE_INDEX
    y[:, :half-1] = IGNORE_INDEX # Only penalize the second half (the prediction)

    return x, y

def generate_pointer_task(batch_size, seq_len, vocab_size, device):
    """
    Task: The token at index i contains a value 'p'.
    The target is the content of the token at index 'p'.
    """
    # 1. Content Tokens (Random)
    content = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # 2. Pointer Tokens (Random indices pointing to valid spots)
    # We will alternate: [Content, Pointer, Content, Pointer...]
    # Or simpler: First half is content, Second half is pointers.

    half = seq_len // 2
    bank = torch.randint(0, vocab_size, (batch_size, half), device=device)

    # Pointers point to indices 0..half-1
    pointers = torch.randint(0, half, (batch_size, half), device=device)

    x = torch.cat([bank, pointers], dim=1)

    # Targets
    # For the first half (bank), we don't care (or next token prediction).
    # For the second half (pointers), the target is bank[pointer]

    targets = torch.zeros_like(x)
    targets[:, :half] = IGNORE_INDEX # Ignore bank prediction

    # Gather targets
    # We need to vectorized gather: batch_idx, pointers
    for b in range(batch_size):
        targets[b, half:] = bank[b, pointers[b]]

    # Standard Shift for Causal LM
    # The input at t should predict t+1.
    # So if input at t is a pointer, we want target at t to be the Answer.
    # Wait, standard LM: Predict Next Token.
    # Input: [Bank... | Ptr1, Ans1, Ptr2, Ans2...] is better?
    # Let's stick to the "Copy" structure: Input is given, predict next.
    # If x is [Bank | Pointers], standard LM predicts shifted x.
    # That doesn't work for Pointers.

    # Adjusted Pointer Task:
    # Input: [Content | Query_Index]
    # Target: [ ... | Content[Query_Index]]
    # We do this token by token in the second half.
    # To make it causal:
    # X: [C1, C2, ... | P1, P2, ...]
    # Y: [ ... | Val(P1), Val(P2)...] -> This implies P1 predicts Val(P1).
    # Correct.

    y = targets # The targets we calculated are exactly what we want P_i to predict.
    y[:, -1] = IGNORE_INDEX # Last one has no next token

    return x, y

def generate_parity_task(batch_size, seq_len, vocab_size, device):
    """
    Task: Copy tokens, but only from EVEN positions.
    Input: [A, B, C, D] -> Output [A, A, C, C] (Repeat even tokens)
    Or simpler: The second half should be a copy of the *Even indices* of the first half.
    """
    half = seq_len // 2
    # Ensure half is even for easier reshaping
    if half % 2 != 0: half -= 1

    source = torch.randint(0, vocab_size, (batch_size, half), device=device)

    # Target: Sequence of even indexed items repeated twice to fill length?
    # Let's do: Input [A, B, C, D] -> Target [A, C] (Skip logic)
    # X: [A, B, C, D | A, C, 0, 0]

    evens = source[:, 0::2] # Indices 0, 2, 4...

    # Construct X: Source + Evens + Padding
    # We need to fill SEQ_LEN
    payload_len = evens.shape[1]
    padding = torch.zeros((batch_size, seq_len - half - payload_len), dtype=torch.long, device=device)

    x = torch.cat([source, evens, padding], dim=1)

    # Target (Shifted left)
    y = x.roll(-1, dims=1)
    y[:, :half-1] = IGNORE_INDEX
    y[:, -1] = IGNORE_INDEX

    return x, y

def generate_copy_batch(batch_size, seq_len, vocab_size, device):
    half = seq_len // 2
    # Random first half
    first_half = torch.randint(0, vocab_size, (batch_size, half), device=device)
    # Copy to second half
    x = torch.cat([first_half, first_half], dim=1)

    # Target is x shifted left
    y = x.roll(-1, dims=1)
    y[:, -1] = IGNORE_INDEX

    # Mask out the first half targets to focus loss ONLY on the copy operation
    # We only care about predicting the second half
    y[:, :half-1] = IGNORE_INDEX

    return x, y


def generate_denoise_task(batch_size, seq_len, vocab_size, device):
    """
    Task: Identify the 'Signal' token hidden in noise.
    Input: [N, S, N, N, S, N, S...] where S is the signal.
    Target: Always predict S.
    Reasoning: Lazy Softmax allows the model to sum up the attention
    from all S occurrences, overpowering the scattered noise.
    """
    # Pick a random signal token for each batch
    signals = torch.randint(0, vocab_size, (batch_size, 1), device=device)

    # Generate random noise
    noise = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Create a mask to inject signals (e.g., 30% signal density)
    # We want enough signal for the 'sum' to work
    mask_prob = 0.35
    mask = torch.rand((batch_size, seq_len), device=device) < mask_prob

    x = torch.where(mask, signals, noise)

    # The target is always the signal, regardless of what the current input is.
    # We want the model to learn "The dominant token in the context is X"
    # Target shape must match x
    y = signals.repeat(1, seq_len)

    # We might want to ignore the first few tokens where context is low
    y[:, :4] = IGNORE_INDEX

    return x, y

def generate_modal_task(batch_size, seq_len, vocab_size, device):
    """
    Task: Identify the most frequent token (The Mode).
    Input: [A, B, A, C, A, D, E, A...] (A is dominant)
    Target: A
    Mechanism: Tests 'Mixing' ability (averaging the context to find the center of mass).
    """
    # 1. Select the "Winner" for each batch
    winner = torch.randint(0, vocab_size - 2, (batch_size, 1), device=device)

    # 2. Generate background noise
    x = torch.randint(0, vocab_size - 2, (batch_size, seq_len), device=device)

    # 3. Inject the winner (e.g., 50% density)
    # Create a mask
    mask = torch.rand((batch_size, seq_len), device=device) < 0.5
    x = torch.where(mask, winner, x)

    # Target is always the winner
    y = winner.repeat(1, seq_len)
    y[:, :10] = IGNORE_INDEX # Ignore early tokens

    return x, y

def generate_priming_task(batch_size, seq_len, vocab_size, device):
    """
    Task: Semantic Priming / Ambiguity Resolution.
    Format: [Context_Marker, ..., Ambiguous_Token] -> Target
    Logic:
      If Context == C1 and Token == T, Target = R1
      If Context == C2 and Token == T, Target = R2
    Mechanism: 'Mixing'.
    The Context_Marker must add a bias vector to the residual stream that
    shifts the ambiguous token's representation into the correct 'meaning' region.
    """
    # Define 2 Contexts and 1 Ambiguous Token
    # C1 -> Target 1
    # C2 -> Target 2

    # Let's say Contexts are tokens 0 and 1.
    # Ambiguous Token is token 2.
    # Targets are 3 and 4.
    C1, C2 = 0, 1
    AMBIG = 2
    TGT1, TGT2 = 3, 4

    # Randomly choose context for each batch
    is_c1 = torch.rand((batch_size,), device=device) > 0.5

    contexts = torch.where(is_c1, torch.tensor(C1, device=device), torch.tensor(C2, device=device))
    targets = torch.where(is_c1, torch.tensor(TGT1, device=device), torch.tensor(TGT2, device=device))

    # Construct Sequence: [Context, Noise..., Ambig]
    x = torch.randint(5, vocab_size, (batch_size, seq_len), device=device) # Noise > 5

    x[:, 0] = contexts
    x[:, -1] = AMBIG

    # Target
    y = torch.full_like(x, IGNORE_INDEX)
    y[:, -1] = targets

    return x, y

def generate_syntactic_agreement_task(batch_size, seq_len, vocab_size, device):
    """
    Task: Subject-Verb Number Agreement.
    Input: [Subject (Sing/Plural), Noise..., Verb_Root]
    Target: Verb_Form (Sing/Plural)
    Mechanism: 'Mixing'.
    The 'Subject' adds a 'Plurality' vector to the residual stream.
    The 'Verb_Root' + 'Plurality' -> 'Verb_Form'.
    """
    # Define Vocabulary Mapping for the Task
    # 0: Singular Subject (e.g., "The Dog")
    # 1: Plural Subject (e.g., "The Dogs")
    # 2: Verb Root (e.g., "run")
    # 3: Singular Verb (e.g., "runs")
    # 4: Plural Verb (e.g., "run")

    SUBJ_SING = 0
    SUBJ_PLUR = 1
    VERB_ROOT = 2
    OUT_SING = 3
    OUT_PLUR = 4

    # Randomly choose number for each batch
    is_plural = torch.rand((batch_size,), device=device) > 0.5

    subjects = torch.where(is_plural, torch.tensor(SUBJ_PLUR, device=device), torch.tensor(SUBJ_SING, device=device))
    targets = torch.where(is_plural, torch.tensor(OUT_PLUR, device=device), torch.tensor(OUT_SING, device=device))

    # Generate Noise (Index 5+)
    x = torch.randint(5, vocab_size, (batch_size, seq_len), device=device)

    # Place Subject early (random pos in first half)
    # We scatter them to prove it's not positional
    locs = torch.randint(0, seq_len // 2, (batch_size, 1), device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    x.scatter_(1, locs, subjects.unsqueeze(1))

    # Place Verb Root at the end
    x[:, -1] = VERB_ROOT

    # Target
    y = torch.full_like(x, IGNORE_INDEX)
    y[:, -1] = targets

    return x, y

def generate_semantic_intersection_task(batch_size, seq_len, vocab_size, device):
    """
    Task: Semantic Intersection (Logical AND).
    Input: [Attribute_A, Attribute_B, ..., Query]
    Target: The specific concept matching A + B.
    Example: "Red" + "Fruit" -> "Apple", "Yellow" + "Fruit" -> "Banana"
    Mechanism: 'Palette Composition'.
    The model must sum V(A) + V(B) and decode to V(Target).
    """
    # Define Attributes
    # A1, A2 (e.g., Colors: Red, Yellow)
    # B1, B2 (e.g., Categories: Fruit, Tool)

    # Targets:
    # A1 + B1 -> T1 (Red Fruit -> Apple)
    # A2 + B1 -> T2 (Yellow Fruit -> Banana)
    # A1 + B2 -> T3 (Red Tool -> Wrench?)
    # A2 + B2 -> T4 (Yellow Tool -> Tape Measure?)

    # We map them to indices:
    A1, A2 = 0, 1
    B1, B2 = 2, 3
    T1, T2, T3, T4 = 4, 5, 6, 7
    QUERY = 8

    # Randomly select A and B
    idx_a = torch.randint(0, 2, (batch_size,), device=device) # 0 or 1
    idx_b = torch.randint(0, 2, (batch_size,), device=device) # 0 or 1

    attr_a = torch.where(idx_a == 0, torch.tensor(A1, device=device), torch.tensor(A2, device=device))
    attr_b = torch.where(idx_b == 0, torch.tensor(B1, device=device), torch.tensor(B2, device=device))

    # Determine Target based on combination
    # 0,0 -> T1; 1,0 -> T2; 0,1 -> T3; 1,1 -> T4
    # Simple logic: Target = 4 + (idx_a) + 2*(idx_b)
    # idx_a=0, idx_b=0 -> 4
    # idx_a=1, idx_b=0 -> 5
    # idx_a=0, idx_b=1 -> 6
    # idx_a=1, idx_b=1 -> 7
    targets = 4 + idx_a + 2 * idx_b

    # Construct Sequence
    x = torch.randint(10, vocab_size, (batch_size, seq_len), device=device)

    # Place A and B at random positions
    # We just overwrite two random columns?
    # Let's simple-loop or use scatter.
    # We place A in first half, B in second half to ensure separation
    loc_a = torch.randint(0, seq_len // 2, (batch_size, 1), device=device)
    loc_b = torch.randint(seq_len // 2, seq_len - 1, (batch_size, 1), device=device)

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    x.scatter_(1, loc_a, attr_a.unsqueeze(1))
    x.scatter_(1, loc_b, attr_b.unsqueeze(1))

    # Trigger at end
    x[:, -1] = QUERY

    y = torch.full_like(x, IGNORE_INDEX)
    y[:, -1] = targets

    return x, y

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# Example Usage

class VernierRoPE(torch.nn.Module):
    def __init__(self, dim, base_1=10000.0, base_2=9973.0):
        super().__init__()
        self.dim = dim
        # Keep grids distinct
        self.inv_freq_1 = 1.0 / (base_1 ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq_2 = 1.0 / (base_2 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x, position_ids):
        # 1. Compute two distinct rotations
        freqs1 = torch.einsum("i,d->id", position_ids, self.inv_freq_1.to(x.device))
        freqs2 = torch.einsum("i,d->id", position_ids, self.inv_freq_2.to(x.device))
        
        # 2. The "Beat" acts as the Absolute Embedding
        # The cosine of the difference acts as a position-dependent amplitude scaler
        # This generalizes infinitely because it is analytic.
        beat_freq = (freqs1 - freqs2)
        amplitude_mod = torch.cos(beat_freq).repeat_interleave(2, dim=-1)
        
        # 3. Apply Standard RoPE (using one grid, or the mean) for the relative phase
        # Let's use the mean frequency for the rotation
        avg_freqs = (freqs1 + freqs2) / 2
        emb = torch.cat((avg_freqs, avg_freqs), dim=-1)
        cos, sin = torch.cos(emb), torch.sin(emb)
        
        # Rotate
        x_rot = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1)
        x_rotated = (x * cos) + (x_rot * sin)
        
        # 4. INJECT ABSOLUTE INFO RELATIONALLY
        # Add the beat-modulated signal. 
        # This replaces: x + x * cos(learned_pos)
        return x_rotated + (x * amplitude_mod) 


class GapedDualCrossAttention(nn.Module):
    def __init__(self, dim, use_sinks=True, lazy_softmax=False):
        super().__init__()
        self.n_heads = 2
        self.head_dim = dim // self.n_heads

        self.use_sinks = use_sinks
        self.lazy_softmax = lazy_softmax

        self.q_proj  = nn.Linear(dim, dim, bias=False)
        self.skew_basis = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.wtp = nn.Linear(1, self.head_dim)

        self.v_sink_residual = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
        self.v_sink_basis = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_dim))
        self.rope = VernierRoPE(self.head_dim)

    def get_orthogonal_matrix(self):
       # Enforce skew-symmetry:
       #A = M - M.T
       skew = self.skew_basis - self.skew_basis.T
       # Map to Lie Group SO(n): R = exp(A) # This guarantees R @ R.T = I

       return torch.matrix_exp(skew)

    def forward(self,x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        R = self.get_orthogonal_matrix()
        w_k = R @ self.q_proj.weight

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)

        k = F.linear(x, w_k).view(B, T, H, D).transpose(1, 2)
        #ensure ansitropic constraint on K or Q will crush it into the floor
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        q = F.rms_norm(q, (D,))

        # Positions
        pos = torch.arange(T, device=x.device).float()

        q = self.rope(q, pos)
        k = self.rope(k, pos)


        # Soft Attention
        scores = (q @ k.transpose(-2, -1)) * (D ** -0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))

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
        y_context = attn@ v

        current_mass = attn.sum(dim=-1, keepdim=True)
        residual = 1.0 - F.sigmoid(current_mass)
        y_res = residual * self.v_sink_residual
        y = F.rms_norm(y_context, (D,)) + self.v_sink_basis + y_res
        y = y.transpose(1, 2).contiguous().view(B, -1, C)

        return self.o_proj(y), attn


import math
class LS(nn.Module):
   #logistical CDF sigmoid
    def __init__(self):
        super().__init__()
        self.alpha = math.pi/math.sqrt(3.0)

    def forward(self, x):
        return torch.sigmoid(self.alpha * x)


# --- WRAPPER ---
class GapedModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.attn = GapedDualCrossAttention(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            LS(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM)
        )
        self.unembed = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(self, x):
        h = self.embed(x)
        h, attn = self.attn(norm(h))

        h = h + self.mlp(norm(h))
        return self.unembed(h), attn

def run_experiment(task_name, generator_func):
    print(f"Running Task: {task_name}")
    torch.manual_seed(42)
    model = GapedModelWrapper().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    losses = []
    final_attn = None

    for i in range(STEPS):
        opt.zero_grad()
        x, y = generator_func(8, SEQ_LEN, VOCAB_SIZE, DEVICE)
        logits, attn = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if i == STEPS - 1:
            final_attn = attn.detach()

    return losses, final_attn

print("Starting Experiment...")

l_rev, a_rev = run_experiment("Mirror/Reverse", generate_reverse_task)
l_ptr, a_ptr = run_experiment("Pointer/Index", generate_pointer_task)
l_par, a_par = run_experiment("Parity/Skip", generate_parity_task)
l_half , a_half = run_experiment("Copy/Half", generate_copy_batch)
l_noise, a_noise = run_experiment("Signal Denoising", generate_denoise_task)
l_modal, a_modal = run_experiment("Modal Consensus (Mixing)", generate_modal_task)
l_prime, a_prime = run_experiment("Semantic Priming", generate_priming_task)
l_agree, a_agree = run_experiment("Syntactic Agreement", generate_syntactic_agreement_task)
l_intersect, a_intersect = run_experiment("Semantic Intersection", generate_semantic_intersection_task)


# --- PLOTTING ---
plt.figure(figsize=(15, 6))

# 1. Losses
plt.subplot(1, 2, 1)
def smooth(y): return np.convolve(y, np.ones(30)/30, mode='valid')
plt.plot(smooth(l_rev), label="Mirror/Reverse",linestyle='--')
plt.plot(smooth(l_ptr), label="Pointer/Index")
plt.plot(smooth(l_par), label="Parity/Skip")
plt.plot(smooth(l_half), label="Copy/Half")
plt.plot(smooth(l_noise), label="Signal Denoising")
plt.plot(smooth(l_modal), label="Modal Consensus (Mixing)")
plt.plot(smooth(l_prime), label="Semantic Priming")
plt.plot(smooth(l_agree), label="Syntactic Agreement")
plt.plot(smooth(l_intersect), label="Semantic Intersection")

plt.title("Convergence")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
