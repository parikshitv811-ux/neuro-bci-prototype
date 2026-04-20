"""
Subject Adversarial Loss
=========================
Trains a small classifier to PREDICT subject identity from the EEG direction
vector, then uses the NEGATIVE gradient to REMOVE subject identity from it.

Implements gradient reversal (GRL) so the main encoder is penalized for
maintaining subject-specific information.

Architecture-safe: plugged in after the trajectory head output, never
modifies patcher/transformer/PLTA/trajectory head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gradient Reversal Layer ───────────────────────────────────────────────────

class _GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradRevFn.apply(x, alpha)


# ── Subject Classifier ────────────────────────────────────────────────────────

class SubjectClassifier(nn.Module):
    """
    Small MLP that tries to predict which subject produced a direction vector.

    Args:
        d_model:    Input direction dimension
        n_subjects: Number of subjects
        hidden:     Hidden layer size
    """

    def __init__(self, d_model: int, n_subjects: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_subjects),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Adversarial Subject Loss ──────────────────────────────────────────────────

class AdversarialSubjectLoss(nn.Module):
    """
    Adversarial loss to disentangle subject identity from EEG directions.

    During forward pass:
      1. Apply gradient reversal to direction vectors
      2. Classify subject from reversed gradient
      3. Loss = cross-entropy (reversed gradient → encoder minimizes this)

    Net effect: encoder learns to produce directions where subject is
                NOT predictable (subject-invariant representation).

    Args:
        d_model:      Direction dimension
        n_subjects:   Number of subjects in dataset
        adv_weight:   Loss weight
        grl_alpha:    Gradient reversal strength
    """

    def __init__(self,
                 d_model:    int,
                 n_subjects: int,
                 adv_weight: float = 0.02,
                 grl_alpha:  float = 1.0):
        super().__init__()
        self.adv_weight = adv_weight
        self.grl_alpha  = grl_alpha
        self.classifier = SubjectClassifier(d_model, n_subjects)

    def forward(self,
                dirs:     torch.Tensor,
                subj_ids: torch.Tensor) -> tuple:
        """
        Args:
            dirs:     (B, D) direction vectors
            subj_ids: (B,)   subject labels (0-indexed)

        Returns:
            loss, metrics dict
        """
        dirs_rev  = grad_reverse(F.normalize(dirs, dim=-1), self.grl_alpha)
        logits    = self.classifier(dirs_rev)
        l_adv     = F.cross_entropy(logits, subj_ids)

        # Accuracy (for logging only, no gradient here)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == subj_ids).float().mean().item()

        return self.adv_weight * l_adv, {
            "adv_ce":  l_adv.item(),
            "subj_acc": acc,   # low = good (subject info removed)
        }
