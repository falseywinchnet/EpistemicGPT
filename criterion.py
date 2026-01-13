import torch
import torch.nn as nn
import torch.nn.functional as F



class ManifoldLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        mask = (targets != self.ignore_index)
        if not mask.any(): return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits = logits[mask]; targets = targets[mask]
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # 1. Sigmoid Probabilities
        probs = logits
        
        # Masking
        target_mask = torch.zeros_like(probs, dtype=torch.bool)
        target_mask.scatter_(1, targets.unsqueeze(1), True)
        
        # P_Target & P_Background
        p_target = probs[target_mask]
        p_background = probs[~target_mask].view(probs.size(0), -1)
        
        # Force 1: Targeted Brier (Drive Confidence)
        loss_target = (1.0 - p_target).pow(2)

        
        loss_margin = F.softplus(max_bg_logits - target_logits)

        # Summation
        return (loss_target  +loss_ce/2.0).mean()
