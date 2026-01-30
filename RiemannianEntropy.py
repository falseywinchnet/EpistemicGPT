#copyright joshuah.rainstar@gmail.com MIT 2026

class RiemannianManifoldLoss(nn.Module):
    """
    A unified loss function that:
    1. Geometrically constrains gradients to the Tangent Cone of the Simplex [Do & Lozano, Thm 3].
    2. Kinematically constrains gradient magnitude to the Entropic Energy of the data stream.
    """
    def __init__(self, vocab_size, ignore_index=-1):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
        # --- IDEA 1: GEOMETRIC PROJECTION (The Rails) ---
        # "The key geometric object... is the tangent cone... whose span determines accuracy" 
        # Construct Projection Matrix Pi = I - (1/V) * 11^T
        I = torch.eye(vocab_size)
        ones = torch.ones(vocab_size, vocab_size)
        self.register_buffer('Pi', I - (ones / vocab_size))
        
    def _project_gradient(self, grad):
        """
        Projects gradients onto span(T_Theta), the tangent space of the simplex.
        "It suffices to test a unique Pi, which is the projection matrix onto span T_Theta".
        """
        if grad.dim() > 1:
            return grad @ self.Pi
        return self.Pi @ grad

    def _compute_entropic_energy(self, input_ids):
        """
        --- IDEA 2: KINEMATIC BOUND (The Speed Limit) ---
        Computes the Wasserstein Energy (Work) to transform Context_t -> Context_{t+1}.
        Uses the Local Vocabulary trick for efficiency.
        """
        B, T = input_ids.shape
        
        # 1. Map to Local Vocab (Efficiency)
        flat_input = input_ids.view(-1)
        unique_tokens, inverse_indices = torch.unique(flat_input, return_inverse=True)
        local_ids = inverse_indices.view(B, T)
        V_local = len(unique_tokens)
        
        # 2. Vectorized Recency Accumulation
        # Distance matrix: dists[t, history_idx]
        indices = torch.arange(T, device=input_ids.device)
        dists = (indices.unsqueeze(1) - indices.unsqueeze(0)) + 1.0
        mask = torch.tril(torch.ones(T, T, device=input_ids.device))
        weights = (1.0 / dists) * mask
        
        # Accumulate scores [B, T, V_local]
        # We loop T for safe broadcasting of weights
        scores = torch.zeros(B, T, V_local, device=input_ids.device)
        for t in range(T):
            hist = local_ids[:, :t+1]
            w = weights[t, :t+1].unsqueeze(0).expand(B, -1)
            scores[:, t, :].scatter_add_(1, hist, w)
            
        # 3. Future Promotion & Mixture Construction
        # We process t=0..T-2 (predicting 1..T-1)
        valid_steps = T - 1
        mixtures = scores[:, :valid_steps, :].clone()
        
        # Add +1.0 to Future Token (the target for the NEXT step)
        future_tokens = local_ids[:, 1:T] # [B, T-1]
        mixtures.scatter_add_(2, future_tokens.unsqueeze(2), torch.ones(B, valid_steps, 1, device=input_ids.device))
        
        # Normalize (Softmax)
        distributions = F.softmax(mixtures, dim=-1)
        
        # 4. Compute Wasserstein Energy (Transition t -> t+1)
        # Diff between Dist[t] and Dist[t+1]
        # We have dists for 0..T-2. We compare adjacent.
        d_curr = distributions[:, :-1, :]
        d_next = distributions[:, 1:, :]
        
        # Energy = L1 Norm
        energy_vec = torch.abs(d_next - d_curr).sum(dim=-1) # [B, T-2]
        
        # Prepend initial kick
        initial = torch.ones(B, 1, device=input_ids.device)
        energy_limit = torch.cat([initial, energy_vec], dim=1) # [B, T-1]
        
        return energy_limit

    def forward(self, logits, targets, input_ids):
        # 1. Geometric Projection Hook
        # We attach the hook to logits so the backward pass flows through Pi
        if logits.requires_grad:
            logits.register_hook(self._project_gradient)
            
        # 2. Compute Kinematic Energy Limit
        # We need energy for the valid prediction window (T-1)
        # Note: input_ids includes the full context.
        energy_limit = self._compute_entropic_energy(input_ids)
        
        # 3. Align Logits/Targets
        valid_steps = energy_limit.size(1)
        valid_logits = logits[:, :valid_steps, :]
        valid_targets = targets[:, :valid_steps]
        
        # 4. Raw Cross Entropy (The "Curved" Distance^2)
        raw_nll = F.cross_entropy(
            valid_logits.reshape(-1, self.vocab_size), 
            valid_targets.reshape(-1), 
            reduction='none',
            ignore_index=self.ignore_index
        ).view(energy_limit.shape)
        
        # 5. Dimensional Scaling
        # Scale = Energy / sqrt(Loss)
        # "Valid for singular Fisher Information Matrices" [cite: 6]
        # This handles the singularity by clamping the ratio.
        
        # We use sqrt to align dimensions (Distance vs Distance^2)
        metric_loss = torch.sqrt(raw_nll + 1e-6)
        
        scale = energy_limit / (metric_loss + 1e-6)
        scale = torch.clamp(scale, max=1.0)
        
        # Apply Scale (Detached)
        # We scale the *original* NLL so gradients are scaled proportionally
        final_loss = raw_nll * scale.detach()
        
        return final_loss.mean()
