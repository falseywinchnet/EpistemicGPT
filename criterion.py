import torch
import torch.nn as nn
import torch.nn.functional as F


#
#
# Note : this wont 1:1 against CEloss. ie 1.5 does NOT mean 1.5 CE.
# this is also softer on the model than CEloss. far softer. 
#

class ManifoldLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: [Batch, Time, Vocab]
        targets: [Batch, Time]
        """
        # Flatten
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Mask
        mask = (targets != self.ignore_index)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        logits = logits[mask]
        targets = targets[mask]
        vocab_size = logits.size(-1)

        # 1. Sigmoid Probabilities (Independent axes)
        probs = torch.sigmoid(logits)

        # --- Force 1: The Simplex Meld ---
        # "Slowly meld towards a simplex."
        # Pulls the sum of probabilities towards 1.0.
        # Normalized by V to keep gradients O(1).
        p_sum = probs.sum(dim=-1)
        loss_simplex = (p_sum - 1.0).pow(2) / vocab_size

        # --- Force 2: The Adversarial Margin ---
        # "Penalize getting the target wrong, keeping the max difference."
        # This operates in Logit space for linear separation.
        # It ensures Target Logit > Max(Other Logits).
        
        # Mask target out to find the runner-up
        logits_for_max = logits.clone()
        logits_for_max.scatter_(1, targets.unsqueeze(1), float('-inf'))
        max_other_logits, _ = logits_for_max.max(dim=1)
        
        target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Softplus creates a smooth hinge. 
        # If Target >> MaxOther, this loss is 0.
        # If Target < MaxOther, this loss is linear.
        loss_margin = F.softplus(max_other_logits - target_logits)

        # --- Force 3: The Brier Relaxation ---
        # "Scaled precisely from the point where the answer is correct."
        # If Target is winning (e.g. 40% vs 20%), Margin is low, but this 
        # term persists to push 40% -> 100% via quadratic decay.
        
        p_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # (1 - p)^2
        # High gradients at low confidence, vanishing gradients at high confidence.
        loss_target_brier = (1.0 - p_target).pow(2)

        # Summation
        loss = (loss_simplex + loss_margin + loss_target_brier).mean()
        
        return loss
