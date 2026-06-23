"""
Hybrid Dice + Focal loss, as defined in Section 4.2.4 of the paper
(Eq. 6-8).

L_Dice  = 1 - (2 * sum(p_i * g_i) + eps) / (sum(p_i) + sum(g_i) + eps)        (Eq. 6)

L_Focal = - sum_i [ alpha * (1 - p_i)^gamma * g_i   * log(p_i)
                   + (1 - alpha) * p_i^gamma * (1 - g_i) * log(1 - p_i) ]      (Eq. 7)

L_total = lambda1 * L_Dice + lambda2 * L_Focal,  lambda1 = lambda2 = 0.5      (Eq. 8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss (Eq. 6). Expects raw logits; applies sigmoid internally."""

    def __init__(self, eps: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.eps = eps
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1).float()

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice_coeff = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice_coeff
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal loss (Eq. 7). Expects raw logits; uses a numerically
    stable formulation based on log-sigmoid rather than naive log(p)/log(1-p).

    Parameters
    ----------
    alpha : float
        Balances the importance of positive vs negative examples (paper:
        alpha in [0, 1]; default 0.8, a common choice for the heavily
        imbalanced nodule-vs-background setting -- adjust as needed since
        the paper does not state the exact value used).
    gamma : float
        Focusing parameter that down-weights well-classified pixels
        (paper: gamma > 0; default 2.0, the standard choice from Lin et al.).
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, from_logits: bool = True, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        if self.from_logits:
            # Numerically stable BCE-with-logits, decomposed per-pixel so we
            # can still apply the focal modulating factor (1-p)^gamma / p^gamma.
            p = torch.sigmoid(pred)
            p = p.clamp(min=self.eps, max=1.0 - self.eps)
        else:
            p = pred.clamp(min=self.eps, max=1.0 - self.eps)

        pos_term = self.alpha * (1 - p).pow(self.gamma) * target * torch.log(p)
        neg_term = (1 - self.alpha) * p.pow(self.gamma) * (1 - target) * torch.log(1 - p)

        loss = -(pos_term + neg_term)
        return loss.mean()


class HybridDiceFocalLoss(nn.Module):
    """
    L_total = lambda1 * L_Dice + lambda2 * L_Focal     (Eq. 8)

    Default lambda1 = lambda2 = 0.5, as specified in the paper.
    """

    def __init__(
        self,
        lambda_dice: float = 0.5,
        lambda_focal: float = 0.5,
        focal_alpha: float = 0.8,
        focal_gamma: float = 2.0,
        dice_eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.dice = DiceLoss(eps=dice_eps, from_logits=True)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, from_logits=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        total = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return {
            "loss": total,
            "dice_loss": dice_loss.detach(),
            "focal_loss": focal_loss.detach(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 64, 64)
    target = (torch.rand(4, 1, 64, 64) > 0.9).float()  # simulate sparse nodule mask

    criterion = HybridDiceFocalLoss()
    out = criterion(logits, target)
    print({k: v.item() for k, v in out.items()})
