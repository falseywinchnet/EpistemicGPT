#dedicated to the public domain for the glory of god.
#baruch adonai el shaddai
#2026 joshuah.rainstar@gmail.com

# MTP Decode Chain for EpistemicGPT
# Cascaded decode modules with cross-horizon consistency

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── rank / ordering losses (from percy liang's repo, user's versions) ───

def expected_rank_of_token(scores, token_ids, temperature=1.0):
    score_i = scores.gather(-1, token_ids.unsqueeze(-1))
    diff = scores - score_i
    p = torch.sigmoid(diff / temperature)
    return 1.0 + p.sum(dim=-1)


def rank_future_sequence_loss_soft(logits, targets, max_future_steps=15,
                                   decay=0.5, temperature=1.0, reduction="mean"):
    B, T, V = logits.shape
    device = logits.device
    total_loss = torch.tensor(0.0, device=device)

    for delta in range(2, max_future_steps + 1):
        if delta >= T:
            break
        cur_logits = logits[:, :-delta, :]
        fut_targets = targets[:, delta:]
        tgt_exp_rank = expected_rank_of_token(cur_logits, fut_targets, temperature)
        step_loss = F.l1_loss(tgt_exp_rank,
                              torch.full_like(tgt_exp_rank, float(delta)),
                              reduction=reduction)
        total_loss = total_loss + step_loss * (decay ** (delta - 1))

    return total_loss


def ordered_future_loss(logits, targets, N=15, decay=0.7, tau=1.0, reduction="mean"):
    B, T, V = logits.shape
    device = logits.device

    if N < 2:
        return torch.tensor(0.0, device=device)

    valid_T = T - (N + 1)
    if valid_T <= 0:
        return torch.tensor(0.0, device=device)

    future_ids = torch.stack(
        [targets[:, 2 + k : 2 + k + valid_T] for k in range(N)], dim=-1
    )
    step_logits = logits[:, :valid_T, :].gather(-1, future_ids)
    diff = step_logits.unsqueeze(-1) - step_logits.unsqueeze(-2)
    k_lt_j = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), 1)
    pair_loss = F.softplus(-diff / tau)
    pair_loss = pair_loss[..., k_lt_j]
    k_idx = torch.arange(N, device=device)
    weight = decay ** k_idx
    weight_pair = weight.unsqueeze(-1).expand(N, N)[k_lt_j]
    pair_loss = pair_loss * weight_pair

    if reduction == "mean":
        return pair_loss.mean()
    elif reduction == "sum":
        return pair_loss.sum()
    return pair_loss


# ─── decode module: a small 1-2 block decoder for a single horizon ───

class DecodeBlock(nn.Module):
    """Lightweight decode block. Just attention + ffn, no frills."""
    def __init__(self, n_embd, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.fc = nn.Linear(n_embd, n_embd, bias=bias)
        self.fc_out = nn.Linear(n_embd, n_embd, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(y)

        # ffn
        x = x + self.dropout(self.fc_out(self.act(self.fc(F.rms_norm(x, (C,))))))
        return x


class DecodeModule(nn.Module):
    """
    Single-horizon decode module.
    Takes backbone output, runs 1-2 small blocks, produces vocab logits.
    Each module decodes at a specific horizon offset.
    """
    def __init__(self, n_embd, n_head, vocab_size, n_blocks=1,
                 dropout=0.0, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecodeBlock(n_embd, n_head, dropout, bias)
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        x_normed = F.rms_norm(x, (x.size(-1),))
        logits = self.head(x_normed)
        return logits, x  # return both logits and evolved latent


class MTPDecodeChain(nn.Module):
    """
    Cascaded multi-token prediction decode chain.

    Architecture:
        backbone -> x (rich latent, no decode pressure)
        x -> DecodeModule_1 -> logits_1 (t+1), commitment written into x
        x -> DecodeModule_2 -> logits_2 (t+2), commitment written into x
        ...
        x -> DecodeModule_N -> logits_N (t+N)

    Each module decodes from positions [0..T-offset-1] and targets
    are sliced accordingly so the dataloader doesn't change.

    At eval time, set n_active_modules to control speed/quality tradeoff.
    """
    def __init__(self, config, n_horizons=3, decode_blocks_per_horizon=1):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size

        # commitment projection: projects logits back into latent space
        # so module k+1 knows what module k decided
        self.commitment_proj = nn.Linear(config.vocab_size, config.n_embd, bias=False)

        # one decode module per horizon
        self.decode_modules = nn.ModuleList([
            DecodeModule(
                n_embd=config.n_embd,
                n_head=config.n_head,
                vocab_size=config.vocab_size,
                n_blocks=decode_blocks_per_horizon,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(n_horizons)
        ])

        # learnable mixing weight for commitment injection
        self.commitment_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, targets, mask, criterion, n_active=None):
        """
        x:       (B, T, D) backbone output (already normed or not, caller decides)
        targets: (B, T) target token ids, standard next-token targets
        mask:    (1, 1, T, T) causal mask
        criterion: loss function (e.g. SoftplusCELoss)
        n_active: how many decode modules to run (None = all)

        Returns:
            primary_logits: (B, T, V) from module 0 (the t+1 predictor)
            total_loss: scalar, aggregated across all active modules
            info: dict with per-module logits for external use
        """
        if n_active is None:
            n_active = self.n_horizons

        B, T, D = x.shape
        device = x.device

        all_logits = []
        all_losses = []
        working_x = x  # this gets commitment injections

        for i in range(min(n_active, self.n_horizons)):
            offset = i + 1  # module 0 predicts t+1, module 1 predicts t+2, etc.

            if T - offset < 1:
                break

            # slice: module i operates on positions [0 .. T-offset-1]
            # and its targets are targets[offset:]
            T_eff = T - offset
            x_slice = working_x[:, :T_eff, :]
            tgt_slice = targets[:, offset:offset + T_eff]

            # also slice mask
            mask_slice = mask[:, :, :T_eff, :T_eff]

            # run decode module
            logits_i, evolved_x = self.decode_modules[i](x_slice, mask_slice)

            # compute CE loss for this horizon
            # only on valid positions (where tgt != -1)
            loss_i = criterion(
                logits_i.reshape(-1, self.vocab_size),
                tgt_slice.reshape(-1)
            )

            # weight: closer horizons matter more
            horizon_weight = 1.0 / (offset)
            all_losses.append(loss_i * horizon_weight)
            all_logits.append(logits_i)

            # inject commitment into working_x for next module
            # the commitment is a soft embedding of what this module predicted
            if i < n_active - 1:
                # detach logits so commitment doesn't backprop through previous module
                commitment = self.commitment_proj(logits_i.detach().softmax(dim=-1))
                gate = torch.sigmoid(self.commitment_gate)

                # write commitment into positions [0..T_eff-1] of working_x
                # but working_x has T positions; we update the overlapping range
                working_x = working_x.clone()
                working_x[:, :T_eff, :] = (
                    working_x[:, :T_eff, :] * (1.0 - gate) + commitment * gate
                )

        # primary logits come from module 0
        primary_logits = all_logits[0] if all_logits else None

        # aggregate losses
        total_mtp_loss = sum(all_losses) if all_losses else torch.tensor(0.0, device=device)

        info = {
            'all_logits': all_logits,
            'n_active': len(all_logits),
        }

        return primary_logits, total_mtp_loss, info

    def cross_horizon_consistency_loss(self, all_logits, targets):
        """
        Compute consistency between overlapping predictions.

        Module i at position k predicts token at k+i+1.
        Module i-1 at position k+1 predicts token at k+1+i = k+i+1.

        So all_logits[i][:, k, :] and all_logits[i-1][:, k+1, :]
        are two predictions of the same token.

        We penalize their disagreement via symmetric KL.
        """
        if len(all_logits) < 2:
            return torch.tensor(0.0, device=all_logits[0].device)

        total = torch.tensor(0.0, device=all_logits[0].device)
        count = 0

        for i in range(1, len(all_logits)):
            # module i has T - (i+1) positions
            # module i-1 has T - i positions
            # overlap: module i at pos k  <->  module i-1 at pos k+1
            T_i = all_logits[i].shape[1]

            if T_i < 1:
                continue

            # module i-1 at positions [1..T_i] (shifted by 1)
            pred_prev = all_logits[i - 1][:, 1:1 + T_i, :]
            pred_curr = all_logits[i][:, :T_i, :]

            # symmetric KL on softmax distributions
            p = F.softmax(pred_prev, dim=-1)
            q = F.softmax(pred_curr, dim=-1)

            # avoid log(0)
            eps = 1e-8
            kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)
            kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(-1)
            sym_kl = 0.5 * (kl_pq + kl_qp)

            total = total + sym_kl.mean()
            count += 1

        if count > 0:
            total = total / count

        return total

    def variance_stability_signal(self, all_logits):
        """
        At inference time: compute per-position variance across horizon predictions.
        High variance = decision point, low variance = stable basin.

        Returns (B, T_min) tensor of stability scores (lower = more stable).
        """
        if len(all_logits) < 2:
            return None

        # find minimum overlapping length
        min_T = min(l.shape[1] for l in all_logits)

        # stack softmax distributions, aligned to same target token
        # module i at position k predicts token at k+i+1
        # to align: module 0 pos k predicts k+1
        #           module 1 pos k-1 predicts k+1
        #           module 2 pos k-2 predicts k+1
        # so we shift each module's logits accordingly
        aligned = []
        for i, logits_i in enumerate(all_logits):
            T_i = logits_i.shape[1]
            # module i at pos j predicts token j+i+1
            # to predict token at position p, module i uses pos p-i-1
            # valid range: p-i-1 >= 0 and p-i-1 < T_i
            # ie p >= i+1 and p < T_i + i + 1
            # common range across all modules: p in [n_horizons, min_T + 1)
            start = i  # start index in this module's positions
            end = start + (min_T - len(all_logits) + 1)
            if end <= start:
                return None
            aligned.append(F.softmax(logits_i[:, start:end, :], dim=-1))

        if not aligned or aligned[0].shape[1] == 0:
            return None

        # (n_horizons, B, T_aligned, V)
        stacked = torch.stack(aligned, dim=0)

        # per-position entropy of the mean distribution minus mean of entropies
        # = mutual information, a measure of disagreement
        mean_dist = stacked.mean(dim=0)  # (B, T_aligned, V)
        H_mean = -(mean_dist * torch.log(mean_dist + 1e-8)).sum(-1)
        H_each = -(stacked * torch.log(stacked + 1e-8)).sum(-1)
        mean_H = H_each.mean(dim=0)

        # Jensen-Shannon divergence (bounded, symmetric)
        jsd = H_mean - mean_H  # (B, T_aligned), >= 0

        return jsd


def integrate_mtp_losses(primary_logits, targets, all_logits,
                         rank_weight=1e-3, order_weight=1e-3,
                         max_future_steps=8, order_N=8):
    """
    Given the primary logits (from module 0) and targets,
    compute the rank and ordering losses.

    The MTP module logits provide real structure for the rank targets,
    so these losses are no longer just noise.
    """
    losses = {}

    if primary_logits is None or primary_logits.shape[1] < 4:
        return losses

    B, T, V = primary_logits.shape

    if rank_weight > 0:
        rl = rank_future_sequence_loss_soft(
            primary_logits, targets[:, 1:1+T],
            max_future_steps=min(max_future_steps, len(all_logits) + 1),
            decay=0.5, temperature=1.0
        )
        losses['rank_loss'] = rl * rank_weight

    if order_weight > 0:
        ol = ordered_future_loss(
            primary_logits, targets[:, 1:1+T],
            N=min(order_N, len(all_logits) + 1),
            decay=0.7, tau=1.0
        )
        losses['order_loss'] = ol * order_weight

    return losses


# ─── example integration into GPT.forward ───

def example_forward_with_mtp(model, idx, targets=None, n_active_mtp=None):
    """
    Shows how to integrate MTPDecodeChain into the existing GPT.forward.

    Assumes model has:
        model.mtp_chain: MTPDecodeChain instance
        model.transformer.h: main backbone blocks
        model.transformer.wte: embedding
        model.criterion: loss function
        model.mask: causal mask buffer

    This is pseudocode showing the integration pattern.
    The actual integration goes into GPT.forward.
    """
    b, T = idx.size()

    # 1. embed
    x = model.transformer.wte(idx)

    # 2. run backbone - no decode pressure here
    for block in model.transformer.h:
        x = block(x)

    x = F.rms_norm(x, (x.size(-1),))

    if targets is not None:
        # 3. context cone aux (unchanged)
        aux_target = model.context_cone_aux_loss(idx, x, targets)

        # 4. MTP decode chain
        primary_logits, mtp_loss, info = model.mtp_chain(
            x, targets, model.mask, model.criterion,
            n_active=n_active_mtp
        )

        # 5. main CE loss from primary logits (module 0 = t+1 prediction)
        main_loss = model.criterion(
            primary_logits.reshape(-1, primary_logits.size(-1)),
            targets[:, 1:1 + primary_logits.size(1)].reshape(-1)
        )

        # 6. cross-horizon consistency
        consistency_loss = model.mtp_chain.cross_horizon_consistency_loss(
            info['all_logits'], targets
        )

        # 7. rank/ordering losses on primary logits
        #    now meaningful because MTP modules provide real structure
        rank_losses = integrate_mtp_losses(
            primary_logits, targets,
            info['all_logits'],
            rank_weight=1e-3,
            order_weight=1e-3,
        )

        # 8. aggregate
        total_loss = (
            main_loss
            + mtp_loss * 0.5
            + aux_target
            + consistency_loss * 0.1
            + sum(rank_losses.values())
        )

        return primary_logits, total_loss

    else:
        # inference: only run module 0 (or more if n_active_mtp > 1)
        n_inf = n_active_mtp if n_active_mtp is not None else 1

        if n_inf == 1:
            # fast path: just decode t+1
            logits_0, _ = model.mtp_chain.decode_modules[0](
                x[:, -1:, :],
                model.mask[:, :, :1, :1]
            )
            return logits_0, None
        else:
            # multi-module inference for stability analysis
            # run on full sequence (or last N tokens for efficiency)
            primary_logits, _, info = model.mtp_chain(
                x, torch.zeros(b, T, dtype=torch.long, device=idx.device),
                model.mask, model.criterion,
                n_active=n_inf
            )

            # compute stability signal
            stability = model.mtp_chain.variance_stability_signal(
                info['all_logits']
            )

            # return last-position logits + stability info
            return primary_logits[:, -1:, :], None, stability
