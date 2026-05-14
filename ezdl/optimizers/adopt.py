"""
ADOPT optimizer — Taniguchi et al., NeurIPS 2024 (arXiv:2411.02853 v3).

"ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate"

Drop-in replacement for Adam. Δύο βασικές αλλαγές:
  1. Το second moment estimate ΔΕΝ περιλαμβάνει το current gradient — το g_t
     normalizeαρεται από το v_{t-1} (previous step's second moment).
  2. Η σειρά αντιστρέφεται: πρώτα normalization, μετά momentum update.
  3. (v3) Το normalized gradient clip-αρεται σε [-c_t, c_t] με c_t = t^0.25 για
     stability στην πρώιμη εκπαίδευση — ιδιαίτερα χρήσιμο σε imbalanced data
     όπου το β2 tuning του Adam είναι ευαίσθητο.

Registered στο super-gradients OPTIMIZERS registry ως "ADOPT" ώστε να μπορεί να
χρησιμοποιηθεί μέσω YAML: `optimizer: [ADOPT]`.
"""
import torch
from torch.optim.optimizer import Optimizer

try:
    from super_gradients.common.registry.registry import register_optimizer
    _SG_AVAILABLE = True
except Exception:  # pragma: no cover — fallback αν αλλάξει το SG API
    _SG_AVAILABLE = False

    def register_optimizer(name):  # no-op decorator
        def _wrap(cls):
            return cls
        return _wrap


@register_optimizer("ADOPT")
class ADOPT(Optimizer):
    """ADOPT optimizer (NeurIPS 2024).

    Args:
        params: iterable of parameters ή param groups.
        lr: learning rate (default 1e-3).
        betas: (β1, β2) EMA coefficients (default (0.9, 0.9999)).
               Σημείωση: β2=0.9999 (όχι 0.999 του Adam) — paper default.
        eps: numerical stability term (default 1e-6).
        weight_decay: weight decay coefficient (default 0.0).
        decoupled: αν True → decoupled (AdamW-style) weight decay (default False).
        clip: αν True → εφαρμογή του v3 stability clipping c_t = t^0.25 (default True).
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.9999),
                 eps: float = 1e-6, weight_decay: float = 0.0,
                 decoupled: bool = False, clip: bool = True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, decoupled=decoupled, clip=clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            decoupled = group["decoupled"]
            clip = group["clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # --- weight decay ---
                if wd != 0:
                    if decoupled:
                        p.mul_(1.0 - lr * wd)
                    else:
                        grad = grad.add(p, alpha=wd)

                state["step"] += 1
                step = state["step"]

                # --- step 1: μόνο initialize το second moment, όχι param update ---
                if step == 1:
                    exp_avg_sq.addcmul_(grad, grad, value=1.0)
                    continue

                # --- normalize current grad με το PREVIOUS second moment ---
                denom = exp_avg_sq.sqrt().clamp_(min=eps)
                normed_grad = grad.div(denom)

                # --- (v3) stability clipping: c_t = t^0.25 ---
                if clip:
                    clip_val = float(step) ** 0.25
                    normed_grad.clamp_(-clip_val, clip_val)

                # --- momentum update με το normalized gradient ---
                exp_avg.mul_(beta1).add_(normed_grad, alpha=1.0 - beta1)

                # --- parameter update ---
                p.add_(exp_avg, alpha=-lr)

                # --- update second moment ΜΕΤΑ (για το επόμενο step's normalization) ---
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        return loss
