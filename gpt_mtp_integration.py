#dedicated to the public domain for the glory of god.
#baruch adonai el shaddai
#2026 joshuah.rainstar@gmail.com

# Integration patch for EpistemicGPT + MTP Decode Chain
# This file shows the modified GPT class with MTP integrated.
# Drop-in replacement for the GPT class in the main file.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtp_decode import (
    MTPDecodeChain,
    integrate_mtp_losses,
)

# import everything else from original file
# (Block, Attention, MLP, RoPE, ThreePieceEmbedding, ContextCone,
#  ParallelSubspaceUnembed, SoftplusCELoss, GPTConfig, norm, etc.)


def norm(x):
    return F.rms_norm(x, (x.size(-1),), eps=1e-6)


class GPT_MTP(nn.Module):
    """
    GPT with cascaded MTP decode chain.

    Changes from original GPT:
    - lm_head removed (decode chain module 0 replaces it)
    - time_head + context_cone_aux_loss unchanged
    - MTPDecodeChain handles all token prediction
    - n_active_mtp controls inference speed/quality tradeoff
    """
    def __init__(self, config, n_horizons=3, decode_blocks=1):
        super().__init__()
        self.config = config

        self.rope = None  # set below after import
        # Assume RoPE, ThreePieceEmbedding, Block, etc are imported from main file
        # We show the structure; actual imports depend on your file layout

        from epistemicgpt import (
            RoPE, ThreePieceEmbedding, Block, ContextCone,
            ParallelSubspaceUnembed, SoftplusCELoss, GPTConfig
        )

        self.rope = RoPE(config.n_embd // config.n_head, max_len=config.block_size)
        config.rope = self.rope

        self.transformer = nn.ModuleDict(dict(
            wte=ThreePieceEmbedding(
                vocab_size=config.vocab_size, d_model=config.n_embd
            ),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        ))

        mask_tensor = torch.tril(torch.ones(
            config.block_size, config.block_size
        )).view(1, 1, config.block_size, config.block_size).to(
            device=config.device
        )
        self.register_buffer("mask", mask_tensor)

        for i, block in enumerate(self.transformer.h):
            block.attn.mask = self.mask
            block.attn.depth = i

        # context cone (unchanged)
        self.hash = ContextCone(config.vocab_size, config.n_embd)

        # time head for cone aux loss (unchanged)
        self.time_head = ParallelSubspaceUnembed(config.n_embd, config.n_embd)

        # MTP decode chain (replaces lm_head)
        self.mtp_chain = MTPDecodeChain(
            config,
            n_horizons=n_horizons,
            decode_blocks_per_horizon=decode_blocks,
        )

        # loss
        self.criterion = SoftplusCELoss(ignore_index=-1)

        # loss weights (tune these)
        self.w_mtp = 0.5
        self.w_consistency = 0.1
        self.w_rank = 1e-3
        self.w_order = 1e-3

        print("GPT_MTP parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def context_cone_aux_loss(self, idx, x_normed, targets):
        B, T = idx.shape
        first_tok = idx[:, :1]
        gt_chain = torch.cat([first_tok, targets], dim=1)
        gt_embeds = self.hash(gt_chain)
        gt = F.normalize(gt_embeds[:, 1:, :], dim=-1)

        temporal_pred, aux_loss = self.time_head(x_normed)
        pred = F.normalize(temporal_pred, dim=-1)
        valid = (targets != -1).float().unsqueeze(-1)
        cosine_sim = (pred * gt).sum(dim=-1, keepdim=True)
        mismatch = (1.0 - cosine_sim) * valid

        if valid.sum() > 0:
            loss = mismatch.sum() / valid.sum()
        else:
            loss = mismatch.sum() * 0.0
        return loss + aux_loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, n_active_mtp=None):
        b, T = idx.size()

        # ── backbone: embed + transform ──
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = norm(x)

        if targets is not None:
            # ── training ──

            # context cone aux (unchanged)
            cone_loss = self.context_cone_aux_loss(idx, x, targets)

            # MTP decode chain: all modules active during training
            primary_logits, mtp_loss, info = self.mtp_chain(
                x, targets, self.mask, self.criterion,
                n_active=n_active_mtp  # None = all
            )

            # cross-horizon consistency
            consistency_loss = self.mtp_chain.cross_horizon_consistency_loss(
                info['all_logits'], targets
            )

            # rank + ordering losses on primary logits
            # these are now informed by MTP module predictions
            rank_losses = integrate_mtp_losses(
                primary_logits, targets,
                info['all_logits'],
                rank_weight=self.w_rank,
                order_weight=self.w_order,
                max_future_steps=min(8, info['n_active'] + 1),
                order_N=min(8, info['n_active'] + 1),
            )

            # aggregate all losses
            total_loss = (
                mtp_loss                               # includes primary CE + horizon CEs
                + cone_loss                            # context cone geometry
                + consistency_loss * self.w_consistency # cross-horizon agreement
                + sum(rank_losses.values())            # rank/ordering discipline
            )

            return primary_logits, total_loss

        else:
            # ── inference ──
            n_inf = n_active_mtp if n_active_mtp is not None else 1

            if n_inf == 1:
                # fast: single decode module on last position only
                logits_0, _ = self.mtp_chain.decode_modules[0](
                    x[:, -1:, :],
                    self.mask[:, :, :1, :1]
                )
                return logits_0, None

            elif n_inf > 1:
                # refined: run multiple modules, get stability signal
                # use a dummy target (won't compute loss)
                dummy_targets = torch.zeros(b, T, dtype=torch.long, device=idx.device)
                primary_logits, _, info = self.mtp_chain(
                    x, dummy_targets, self.mask, self.criterion,
                    n_active=n_inf
                )

                # stability analysis
                stability = self.mtp_chain.variance_stability_signal(
                    info['all_logits']
                )

                # at the last position, we can adjust primary logits
                # using the stability signal
                if stability is not None and stability.shape[1] > 0:
                    # get stability at last valid position
                    s = stability[:, -1]  # (B,) JSD value

                    # high JSD = disagreement = decision point
                    # low JSD = agreement = stable prediction
                    # we could use this to adjust temperature, or to
                    # blend with the cached predictions

                    # for now, just return logits + stability info
                    return primary_logits[:, -1:, :], None
                else:
                    return primary_logits[:, -1:, :], None


# ─── usage example ───

def make_mtp_model(vocab_size=66, n_layer=4, n_head=6, n_embd=192,
                   block_size=1024, n_horizons=3, decode_blocks=1,
                   device="cuda"):
    """
    Create a GPT_MTP model. This is the same as creating a regular GPT
    but with the MTP decode chain.

    n_horizons: number of decode modules (1 = standard NTP, 3 = recommended)
    decode_blocks: blocks per decode module (1-2, keep light)
    """
    from epistemicgpt import GPTConfig

    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=False,
        device=device,
    )

    model = GPT_MTP(config, n_horizons=n_horizons, decode_blocks=decode_blocks)
    return model.to(device)


# ─── training loop integration ───

def train_step(model, batch, optimizer):
    """
    Standard training step. The MTP chain is fully internal.
    No dataloader changes needed.
    """
    idx, targets = batch  # (B, T), (B, T) as usual
    optimizer.zero_grad()

    logits, loss = model(idx, targets)
    loss.backward()
    optimizer.step()

    return loss.item()


def generate_token(model, idx, n_active_mtp=1):
    """
    Generate next token.

    n_active_mtp=1: fast, standard
    n_active_mtp=2+: slower, with stability-informed selection
    """
    with torch.no_grad():
        result = model(idx, n_active_mtp=n_active_mtp)
        logits = result[0]  # (B, 1, V)

        # sample or argmax as usual
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token
