"""
CardHouse Optimizer
===================

A drop-in PyTorch optimizer that replaces Newton-Muon's Cholesky-based
right-preconditioning with a Householder direction decomposition using
MAD-based (robust) scale estimation.

For matrix-shaped parameters (2D), CardHouse:
  1. Maintains k learned Householder vectors per parameter that track
     the top eigenvectors of the input activation second moment ZZ^T
     via deflated power iteration.
  2. Estimates per-direction scale using median absolute deviation (MAD)
     converted to Gaussian-equivalent sigma, which is robust to the
     heavy-tailed distributions produced by GELU/ReLU activations.
  3. Preconditions the gradient by rescaling its components along each
     Householder direction to equalize their robust scales.
  4. Applies the matrix sign (via Newton-Schulz iteration) to the
     preconditioned gradient.
  5. Updates with momentum.

For 1D parameters (biases, norms), falls back to momentum SGD.

What register_activations does: Attaches forward hooks to every nn.Linear. Each hook captures the input tensor, reshapes it to (dim, batch*seq), and stashes it. On optimizer.step(), CardHouse reads these buffers to update its Householder directions and scale estimates. If you don't call it, the optimizer falls back to using the gradient's column structure as a proxy, which is worse but functional.
What split_params does: Walks named_parameters, puts 2D tensors into the matrix bucket (CardHouse), everything else into scalar bucket (AdamW). Embeddings get heuristically detected by aspect ratio (first dim > 10x second dim) and routed to AdamW.
The k parameter: Number of Householder vectors. Start with k=8. For models where d_model is large and anisotropy is concentrated in a few directions (which the paper's Table 4 suggests is typical), k=8-16 should capture most of it. k=3 matching your spike count is the theoretical minimum for this synthetic data.
diagnostics() returns per-parameter scale info. The mean_ratio field (var_scale / robust_scale) tells you how non-Gaussian each direction is. Ratio near 1.0 means Gaussian, above 1.0 means heavy tails where the MAD estimator is earning its keep.
Call remove_hooks() when done training, or the hooks hold references to the model and leak memory.

Usage
-----
    from cardhouse import CardHouse

    optimizer = CardHouse(model.parameters(), lr=0.01)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

With activation hooks (recommended for best results):
    optimizer = CardHouse(model.parameters(), lr=0.01)
    optimizer.register_activations(model)  # auto-hooks linear layers

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

Without hooks, CardHouse still works but cannot track activation
statistics. It will precondition using only the gradient's own
column structure as a proxy for input anisotropy.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from typing import Optional, List, Dict, Any


def _msgn_ns(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Matrix sign via Newton-Schulz iteration.
    X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k

    Input:  G of shape (m, n)
    Output: U V^T where G = U S V^T (compact SVD), shape (m, n)
    """
    norm = torch.norm(G, p="fro")
    if norm < 1e-12:
        return G
    X = G / norm
    for _ in range(steps):
        XXT = X @ X.T
        X = 1.5 * X - 0.5 * (XXT @ X)
    return X


def _robust_scale(projections: torch.Tensor) -> torch.Tensor:
    """
    MAD-based robust scale estimation.
    Returns Gaussian-equivalent variance: (MAD / 0.6745)^2

    projections: (batch,) tensor of scalar projections
    """
    median = projections.median()
    abs_dev = (projections - median).abs()
    mad = abs_dev.median()
    sigma = mad / 0.6745
    return sigma * sigma


class _HouseholderState:
    """Per-parameter Householder direction state."""

    def __init__(self, n: int, k: int, device: torch.device, dtype: torch.dtype):
        self.k = k
        self.n = n
        # initialize with random orthogonal-ish directions
        vs = torch.randn(k, n, device=device, dtype=dtype)
        # QR to get orthonormal initial directions
        if k <= n:
            q, _ = torch.linalg.qr(vs.T)
            self.vs = q.T[:k].contiguous()  # (k, n)
        else:
            self.vs = torch.nn.functional.normalize(vs, dim=1)
        self.robust_scales = torch.ones(k, device=device, dtype=dtype)
        self.var_scales = torch.ones(k, device=device, dtype=dtype)
        self.initialized = False

    def update_from_activations(
        self, Z: torch.Tensor, beta: float, blend: float = 0.3
    ):
        """
        Update directions and scales from activation matrix.
        Z: (n, batch) or (batch, n) - will be handled.
        """
        if Z.shape[0] != self.n:
            if Z.shape[1] == self.n:
                Z = Z.T
            else:
                return  # shape mismatch, skip

        batch = Z.shape[1]
        if batch < 4:
            return

        with torch.no_grad():
            for i in range(self.k):
                v = self.vs[i]  # (n,)

                # project activations onto direction
                projs = Z.T @ v  # (batch,)

                # variance scale
                var = projs.var().clamp(min=1e-12)
                if self.initialized:
                    self.var_scales[i] = beta * self.var_scales[i] + (1 - beta) * var
                else:
                    self.var_scales[i] = var

                # robust scale (MAD)
                rs = _robust_scale(projs).clamp(min=1e-12)
                if self.initialized:
                    self.robust_scales[i] = (
                        beta * self.robust_scales[i] + (1 - beta) * rs
                    )
                else:
                    self.robust_scales[i] = rs

                # power iteration step: rotate toward top eigenvector of ZZ^T
                # ZZ^T v = Z (Z^T v) = Z projs
                ZZTv = (Z * projs.unsqueeze(0)).sum(dim=1) / batch  # (n,)

                # deflate against previous directions
                for p in range(i):
                    dot = ZZTv @ self.vs[p]
                    ZZTv = ZZTv - dot * self.vs[p]

                norm = ZZTv.norm()
                if norm > 1e-10:
                    new_v = (1 - blend) * v + blend * (ZZTv / norm)
                    self.vs[i] = torch.nn.functional.normalize(new_v, dim=0)

            self.initialized = True

    def update_from_gradient(self, G: torch.Tensor, beta: float, blend: float = 0.3):
        """
        Fallback: use gradient column structure as proxy for activation anisotropy.
        G: (m, n) gradient matrix. We treat columns of G^T as samples.
        """
        # G^T G / m approximates input second moment structure
        Z = G.T  # (n, m) - treat the m rows of G as "samples"
        self.update_from_activations(Z, beta, blend)

    def precondition(self, G: torch.Tensor) -> torch.Tensor:
        """
        Right-precondition gradient G of shape (m, n) using Householder directions.
        Rescales gradient components along each direction by ratio of
        mean robust scale to per-direction robust scale.
        """
        if not self.initialized:
            return G

        mean_scale = self.robust_scales.mean()
        if mean_scale < 1e-12:
            return G

        Gout = G.clone()
        for i in range(self.k):
            v = self.vs[i]  # (n,)
            rescale = mean_scale / (self.robust_scales[i] + 1e-8)

            # Gout[row, :] += (Gout[row, :] . v) * (rescale - 1) * v
            dots = Gout @ v  # (m,)
            Gout = Gout + (rescale - 1.0) * dots.unsqueeze(1) * v.unsqueeze(0)

        return Gout


class CardHouse(Optimizer):
    """
    CardHouse optimizer.

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (default: 0.01)
        momentum: momentum coefficient (default: 0.85)
        k: number of Householder vectors per matrix parameter (default: 8)
        beta: EWMA coefficient for scale tracking (default: 0.95)
        ns_steps: Newton-Schulz iterations for matrix sign (default: 10)
        refresh_every: update Householder directions every N steps (default: 4)
        blend: power iteration blending factor (default: 0.3)
        weight_decay: weight decay coefficient (default: 0.0)
        scalar_lr_scale: learning rate multiplier for 1D params (default: 0.3)
        matrix_threshold: minimum dimension to treat as matrix param (default: 2)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.85,
        k: int = 8,
        beta: float = 0.95,
        ns_steps: int = 10,
        refresh_every: int = 4,
        blend: float = 0.3,
        weight_decay: float = 0.0,
        scalar_lr_scale: float = 0.3,
        matrix_threshold: int = 2,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            k=k,
            beta=beta,
            ns_steps=ns_steps,
            refresh_every=refresh_every,
            blend=blend,
            weight_decay=weight_decay,
            scalar_lr_scale=scalar_lr_scale,
            matrix_threshold=matrix_threshold,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._activation_buffers: Dict[nn.Module, torch.Tensor] = {}
        self._param_to_module: Dict[nn.Parameter, nn.Module] = {}
        self._hooks: List[Any] = []

    def register_activations(self, model: nn.Module):
        """
        Register forward hooks on all Linear layers to capture input activations.
        Call once before training.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # map parameter to module
                self._param_to_module[module.weight] = module
                if module.bias is not None:
                    self._param_to_module[module.bias] = module

                handle = module.register_forward_hook(self._activation_hook)
                self._hooks.append(handle)

    def _activation_hook(self, module: nn.Module, input, output):
        """Capture input activations for a linear layer."""
        if not module.training:
          return
        x = input[0]
        if x.dim() == 3:
            # (batch, seq, dim) -> flatten to (batch*seq, dim)
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        # store as (dim, samples) for consistency with ZZ^T convention
        self._activation_buffers[module] = x.T.detach()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _get_house_state(
        self, p: torch.Tensor, group: dict
    ) -> Optional[_HouseholderState]:
        """Get or create Householder state for a parameter."""
        state = self.state[p]
        if "house" not in state:
            if p.dim() != 2:
                return None
            m, n = p.shape
            if min(m, n) < group["matrix_threshold"]:
                return None
            effective_k = min(group["k"], n)
            state["house"] = _HouseholderState(n, effective_k, p.device, p.dtype)
        return state["house"]

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            beta = group["beta"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            refresh = group["refresh_every"]
            blend = group["blend"]
            scalar_lr = lr * group["scalar_lr_scale"]
            do_refresh = (self._step_count % refresh == 0) or (self._step_count == 1)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # weight decay
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                # initialize momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]

                if p.dim() == 2 and min(p.shape) >= group["matrix_threshold"]:
                    # matrix parameter: CardHouse update
                    house = self._get_house_state(p, group)

                    if house is not None and do_refresh:
                        # try to get activations from hook
                        module = self._param_to_module.get(p)
                        if module is not None and module in self._activation_buffers:
                            Z = self._activation_buffers[module]
                            house.update_from_activations(Z, beta, blend)
                        else:
                            # fallback: use gradient structure
                            house.update_from_gradient(grad, beta, blend)

                    if house is not None:
                        G_precond = house.precondition(grad)
                    else:
                        G_precond = grad

                    Q = _msgn_ns(G_precond, steps=ns_steps)

                    buf.mul_(mom).add_(Q)
                    p.data.add_(buf, alpha=-lr)

                else:
                    # scalar/1D parameter: momentum SGD
                    buf.mul_(mom).add_(grad)
                    p.data.add_(buf, alpha=-scalar_lr)

        # clear activation buffers after step
        self._activation_buffers.clear()

        return loss

    def diagnostics(self) -> dict:
        """
        Return diagnostic information about Householder states.
        Useful for logging during training.

        Returns dict with per-parameter info:
            - robust_scales: MAD-based scales per direction
            - var_scales: variance-based scales per direction
            - scale_ratio: var/robust per direction (>1 means heavy tails)
            - directions: the Householder vectors
        """
        info = {}
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                state = self.state.get(p, {})
                house = state.get("house")
                if house is not None and house.initialized:
                    ratio = house.var_scales / (house.robust_scales + 1e-12)
                    info[f"param_{i}"] = {
                        "shape": tuple(p.shape),
                        "robust_scales": house.robust_scales.clone(),
                        "var_scales": house.var_scales.clone(),
                        "scale_ratio": ratio,
                        "mean_ratio": ratio.mean().item(),
                        "k": house.k,
                    }
        return info


class CardHouseAdamW(Optimizer):
    """
    Hybrid optimizer: CardHouse for 2D (matrix) parameters,
    AdamW for everything else.

    This is the recommended configuration for transformer training,
    matching the pattern used by Muon/Newton-Muon papers where
    matrix params get the spectral optimizer and embed/norm/bias
    params get Adam.

    Args:
        matrix_params: iterable of 2D parameters (Linear weights)
        scalar_params: iterable of other parameters (biases, norms, embeddings)
        matrix_lr: learning rate for matrix params (default: 0.01)
        scalar_lr: learning rate for scalar params (default: 3e-4)
        k: Householder vectors (default: 8)
        momentum: for CardHouse (default: 0.85)
        betas: for AdamW (default: (0.9, 0.999))
        weight_decay: for AdamW (default: 0.01)
        **cardhouse_kwargs: additional kwargs passed to CardHouse
    """

    def __init__(
        self,
        matrix_params,
        scalar_params,
        matrix_lr: float = 0.01,
        scalar_lr: float = 3e-4,
        k: int = 8,
        momentum: float = 0.85,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        **cardhouse_kwargs,
    ):
        matrix_params = list(matrix_params)
        scalar_params = list(scalar_params)

        self.cardhouse = CardHouse(
            matrix_params, lr=matrix_lr, momentum=momentum, k=k, **cardhouse_kwargs
        )
        self.adamw = torch.optim.AdamW(
            scalar_params, lr=scalar_lr, betas=betas, weight_decay=weight_decay
        )

        # expose param_groups for schedulers
        self.param_groups = self.cardhouse.param_groups + self.adamw.param_groups
        self.state = {}

    def register_activations(self, model: nn.Module):
        self.cardhouse.register_activations(model)

    def remove_hooks(self):
        self.cardhouse.remove_hooks()

    def zero_grad(self, set_to_none=True):
        self.cardhouse.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.cardhouse.step(closure)
        self.adamw.step()
        return loss

    def diagnostics(self):
        return self.cardhouse.diagnostics()

    def state_dict(self):
        return {
            "cardhouse": self.cardhouse.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.cardhouse.load_state_dict(state_dict["cardhouse"])
        self.adamw.load_state_dict(state_dict["adamw"])


def split_params(model: nn.Module):
    """
    Split model parameters into matrix params (for CardHouse)
    and scalar params (for AdamW).

    Returns (matrix_params, scalar_params) as lists.

    Convention: any 2D parameter with both dims >= 2 is a matrix param.
    Everything else (embeddings, biases, layernorm weights) is scalar.
    """
    matrix_params = []
    scalar_params = []
    seen = set()

    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))

        if p.dim() == 2 and min(p.shape) >= 2:
            # skip embeddings: typically (vocab, dim) with very large first dim
            # heuristic: if first dim > 10x second dim, treat as embedding
            if p.shape[0] > 10 * p.shape[1]:
                scalar_params.append(p)
            else:
                matrix_params.append(p)
        else:
            scalar_params.append(p)

    return matrix_params, scalar_params
