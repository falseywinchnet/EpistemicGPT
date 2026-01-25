'''
def run_comparison():
    dim = 64
    seq_len = 4096
    base = 10000
    
    # Query at 0, Keys at 0..4096
    pos_q = torch.tensor([0]).float()
    pos_k = torch.arange(seq_len).float()


    model_new = YourNewModuleHere(dim)

    model_standard = RoPE(dim,base=base)

    # The "Ones" Vector (Pure Geometric Test)
    x_content = torch.ones(1, 1, dim)

    # --- Run New ---
    q_h = model_new(x_content, pos_q) 
    k_h = model_new(x_content, pos_k) 
    scores_h = (q_h.view(1, dim) @ k_h.view(seq_len, dim).T).squeeze().abs().detach().numpy()

    # --- Run Standard ---
    q_s = model_standard(x_content, pos_q)
    k_s = model_standard(x_content, pos_k)
    scores_s = (q_s.view(1, dim) @ k_s.view(seq_len, dim).T).squeeze().abs().detach().numpy()

    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Standard first (Grey/Background)
    plt.plot(scores_s, color='gray', alpha=0.5, linewidth=1, label="Standard RoPE (10k)")
    
    # Plot New (Blue/Foreground)
    plt.plot(scores_h, color='blue', linewidth=2, label="decayed")
    
    plt.axvline(x=1024, color='red', linestyle=':', label="Target Horizon (1024)")
    
    plt.title("Geometry Check:  Smoothing vs Standard Ripple")
    plt.xlabel("Distance")
    plt.ylabel("Attention Magnitude (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_comparison()

'''

class ERoPE(nn.Module):
    #Erotic- erm i mean Euler Rope. Uses multiple bases that cancel out each other's noise terms.
    #get it as smooth as you like with more bases. BTW, the scale/smoothing mask- thats not perfect yet.
    #maybe someone will come along and take this to the next level?
    #the above demo shows the ultimate noise convergence(ones, lol)
    #this tells us a lot about why rope struggles. How do we solve it?
    #perodicity and smoothing.
    #could we get something like this with some kind of gabriels horn type approach to the individual channels in time?
    #maybe, im not a math expert, i got here after a lot of dicking around with multi-base rope trying to solve a few problems:
    #locality- you want rope's sensitivity to *gradually and smoothly drop off* which means that future noise cant crowd out the present.
    #useful signals in the near field that are absolute but also analytical

     
    def __init__(self, dim, base=10000.0, num_bases=8, max_len=512):
        super().__init__()
        self.dim = dim
        self.num_bases = num_bases
        self.ratio = (1 + np.e) / 2
        self.base_life = float(dim)
        
        # Double precision for initialization stability
        bases = [base * (self.ratio ** k) for k in range(num_bases)]
        freqs = [1.0 / (b ** (torch.arange(0, dim, 2).double() / dim)) for b in bases]
        scales = [1.0 / (self.base_life * (self.ratio ** (-1.5 * k))) for k in range(num_bases)]
        
        self.register_buffer("stack_inv_freqs", torch.stack(freqs))
        self.register_buffer("stack_scales", torch.tensor(scales).view(-1, 1))
        self.register_buffer("cached_cos", None, persistent=False)
        self.register_buffer("cached_sin", None, persistent=False)
        
        self.max_len = 0
        self._build_cache(max_len)

    def _build_cache(self, seq_len):
        if seq_len <= self.max_len: return
        t = torch.arange(seq_len, device=self.stack_inv_freqs.device).double().view(1, 1, -1, 1)
        freqs = self.stack_inv_freqs.view(self.num_bases, 1, 1, -1)
        scales = self.stack_scales.view(self.num_bases, 1, 1, 1)
        
        theta = t * freqs
        mask = 1.0 / (1.0 + (theta * scales).pow(2))
        
        f_cos = (torch.cos(theta) * mask).mean(dim=0).squeeze(0)
        f_sin = (torch.sin(theta) * mask).mean(dim=0).squeeze(0)
        
        # Final shape (Seq, Dim)
        self.cached_cos = torch.repeat_interleave(f_cos, 2, dim=-1).float()
        self.cached_sin = torch.repeat_interleave(f_sin, 2, dim=-1).float()
        self.max_len = seq_len

    def forward(self, x, position_ids):
        # x shape: (B_total, T, D)
        target = position_ids.max().item() + 1
        if target > self.max_len: self._build_cache(int(target))
        
        # Index and unsqueeze to (1, T, D) for broadcasting against B_total
        cos = self.cached_cos[position_ids].unsqueeze(0)
        sin = self.cached_sin[position_ids].unsqueeze(0)
        
        x_half_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        # Relativistic absolute signal injection
        return (x * cos) + (x_half_rot * sin) + (x + x_half_rot) * 0.5
