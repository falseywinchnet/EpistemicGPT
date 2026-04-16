"""
Gaussian Surrogate Probe with Householder Dual Projection

Taps the residual stream at one or more layers, projects through two
related linear maps (related by a learned Householder reflection),
and penalizes both projections for deviating from a Gaussian shape.

Drop-in aux loss for any x + fn(x) residual network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianProbe(nn.Module):
    """
    Projects d_model -> d_vocab (small, e.g. 128) via two Householder-related
    linear maps. Computes a surrogate loss penalizing non-Gaussianity of the
    resulting logit distributions.
    """

    def __init__(self, d_model, d_vocab=128):
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab

        # Primary projection
        self.W = nn.Linear(d_model, d_vocab, bias=False)

        # Householder reflection vector (learned)
        self.v = nn.Parameter(torch.randn(d_vocab, 1))

        # Precompute the target: standard normal quantiles for d_vocab bins
        # These are fixed, never trained
        quantiles = torch.linspace(0.5 / d_vocab, 1.0 - 0.5 / d_vocab, d_vocab)
        target = torch.erfinv(2 * quantiles - 1) * math.sqrt(2)
        self.register_buffer("target_quantiles", target)

    def _reflect(self, W):
        """Apply Householder reflection: W' = W - 2v(v^T W)"""
        v = F.normalize(self.v, dim=0)  # [d_vocab, 1]
        # W is [d_vocab, d_model]
        vT_W = v.T @ W  # [1, d_model]
        return W - 2 * v @ vT_W  # [d_vocab, d_model]

    def _gaussian_loss(self, logits):
        """
        Penalize deviation from Gaussian via sorted-quantile matching.

        logits: [B, T, d_vocab]
        Sort along the vocab dim, compare against standard normal quantiles.
        """
        # Sort along vocab dimension
        sorted_logits, _ = torch.sort(logits, dim=-1)

        # Standardize per-sample: zero mean, unit variance
        mu = sorted_logits.mean(dim=-1, keepdim=True)
        std = sorted_logits.std(dim=-1, keepdim=True).clamp(min=1e-6)
        normalized = (sorted_logits - mu) / std

        # MSE against expected Gaussian quantiles
        target = self.target_quantiles.unsqueeze(0).unsqueeze(0)  # [1, 1, d_vocab]
        loss = F.mse_loss(normalized, target.expand_as(normalized))
        return loss

    def forward(self, x):
        """
        x: [B, T, d_model] residual stream activation

        Returns:
            loss: scalar, the Gaussian surrogate penalty
            info: dict with diagnostics
        """
        # Primary projection
        logits_a = self.W(x)  # [B, T, d_vocab]

        # Reflected projection
        W_reflected = self._reflect(self.W.weight)  # [d_vocab, d_model]
        logits_b = F.linear(x, W_reflected)  # [B, T, d_vocab]

        # Gaussian penalty on both
        loss_a = self._gaussian_loss(logits_a)
        loss_b = self._gaussian_loss(logits_b)
        loss = (loss_a + loss_b) / 2.0

        # Diagnostics: how Gaussian are we actually?
        with torch.no_grad():
            def _moments(logits):
                mu = logits.mean(dim=-1)
                centered = logits - mu.unsqueeze(-1)
                var = (centered ** 2).mean(dim=-1)
                std = var.sqrt().clamp(min=1e-6)
                skew = (centered ** 3).mean(dim=-1) / (std ** 3)
                kurt = (centered ** 4).mean(dim=-1) / (var ** 2)
                return skew.mean().item(), kurt.mean().item()

            skew_a, kurt_a = _moments(logits_a)
            skew_b, kurt_b = _moments(logits_b)

        info = {
            "loss_a": loss_a.item(),
            "loss_b": loss_b.item(),
            "skew_a": skew_a,
            "kurt_a": kurt_a,  # Gaussian = 3.0
            "skew_b": skew_b,
            "kurt_b": kurt_b,
        }

        return loss, info


class GaussianProbeTap(nn.Module):
    """
    Wraps a residual block so the probe taps x after the block runs.

    Usage:
        block = TransformerBlock(...)
        probe = GaussianProbe(d_model=768, d_vocab=128)
        tapped = GaussianProbeTap(block, probe)

        # In forward pass:
        x, aux_loss, info = tapped(x)
    """

    def __init__(self, block, probe):
        super().__init__()
        self.block = block
        self.probe = probe

    def forward(self, x, **kwargs):
        x = self.block(x, **kwargs)
        loss, info = self.probe(x)
        return x, loss, info


# --- Integration helper ---

def attach_probes(model, layer_indices, d_model, d_vocab=128):
    """
    Attach Gaussian probes to specific layers of a model.

    Expects model.blocks or model.transformer.h or similar list of layers.
    Returns the list of probes (for collecting aux losses during training).

    This is a sketch -- adapt the attribute path to your model.
    """
    # Find the layer list
    if hasattr(model, 'blocks'):
        layers = model.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Can't find layer list. Adapt this function.")

    probes = []
    for idx in layer_indices:
        probe = GaussianProbe(d_model, d_vocab)
        tapped = GaussianProbeTap(layers[idx], probe)
        layers[idx] = tapped
        probes.append(probe)

    return probes


# --- Standalone test ---

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, D = 4, 64, 256
    d_vocab = 128

    probe = GaussianProbe(d_model=D, d_vocab=d_vocab)

    # Random residual stream activations
    x = torch.randn(B, T, D)

    loss, info = probe(x)
    print(f"Initial loss: {loss.item():.4f}")
    print(f"  logits_a  skew={info['skew_a']:.3f}  kurt={info['kurt_a']:.3f}")
    print(f"  logits_b  skew={info['skew_b']:.3f}  kurt={info['kurt_b']:.3f}")

    # Quick optimization to show the loss actually goes down
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for step in range(200):
        # Simulate varying residual activations
        x = torch.randn(B, T, D)
        loss, info = probe(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}: loss={loss.item():.4f}  "
                  f"skew_a={info['skew_a']:.3f} kurt_a={info['kurt_a']:.3f}  "
                  f"skew_b={info['skew_b']:.3f} kurt_b={info['kurt_b']:.3f}")

    print("\nDone. If kurt -> 3.0 and skew -> 0.0, the probe is learning to Gaussianize.")
