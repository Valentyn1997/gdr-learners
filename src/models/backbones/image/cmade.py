# maf_conditional_colored_mnist.py
# PyTorch implementation of Conditional Masked Autoregressive Flow (MAF) using MADE
# Suitable for colored MNIST (e.g., 10x10x3) or any small image vector.
# Author: you + ChatGPT

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities: preprocessing
# ---------------------------

class AtanhSquashTransform(nn.Module):
    """
    Maps x in [-1, 1] to R via:
      y = (1 - eps) * x               (keeps margin from ±1)
      z = atanh(y) = 0.5 * log((1+y)/(1-y))
    log|det J| = sum[ log(1 - eps) - log(1 - y^2) ]
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x expected in [-1, 1]
        y = x * (1.0 - self.eps)
        z = torch.atanh(y)  # available in recent PyTorch; else use 0.5*torch.log((1+y)/(1-y))
        logdet_elem = torch.log1p(-torch.tensor(self.eps).float()) - torch.log(1.0 - y * y)
        logdet = logdet_elem.sum(-1)
        return z, logdet

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.tanh(z)
        x = y / (1.0 - self.eps)
        # inverse logdet = - forward logdet evaluated at x
        logdet_elem = -torch.log1p(-torch.tensor(self.eps).float()) + torch.log(1.0 - y * y)
        logdet = logdet_elem.sum(-1)
        return x, logdet


def uniform_dequantize(x: torch.Tensor) -> torch.Tensor:
    """x in {0,...,255} or in [0,1]. Returns float in [0,1] with U(0,1/256) noise if 8-bit."""
    if x.dtype.is_floating_point:
        x = torch.clamp(x, -1.0, 1.0)
        return x
    else:
        raise TypeError("x must be floating point")


# ---------------------------
# MADE building blocks
# ---------------------------

def create_degrees(n_in: int, n_hidden: List[int], randomize: bool = True) -> List[torch.Tensor]:
    """
    Create connectivity degrees for MADE (cf. Germain et al. 2015; Papamakarios et al. 2017).
    degrees[k][i] in {1,...,D} for input/hidden layers, and {1,...,D} for output degrees (replicated).
    """
    D = n_in
    degrees = []

    # Input layer degrees: fixed 1..D (possibly permuted)
    if randomize:
        degrees_input = torch.randperm(D) + 1
    else:
        degrees_input = torch.arange(1, D + 1)
    degrees.append(degrees_input)

    # Hidden layers degrees
    prev = degrees_input
    for H in n_hidden:
        # Each hidden unit chooses an integer in {min(prev),...,D-1}
        min_prev = torch.min(prev).item()
        d = torch.randint(low=min_prev, high=D, size=(H,))  # in [min_prev, D-1]
        degrees.append(d)
        prev = d

    return degrees  # length = 1 + L (input + hidden)


class MaskedLinear(nn.Linear):
    """Linear layer with a fixed mask."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor):
        self.mask.data.copy_(mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class ConditionalMADE(nn.Module):
    """
    Conditional MADE that outputs shift (m) and log-scale (s) for all D features.
    Conditioning enters via additive projections (FiLM-like bias) to every hidden layer and the output layer.
    Masks ensure autoregressive property.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: List[int],
        n_cond: int,
        activation: nn.Module = nn.ELU(),
        random_masks: bool = True,
        output_multiplier: int = 2,  # m and s
        log_scale_clip: float = 5.0,
    ):
        super().__init__()
        self.n_in = n_in
        self.hidden_sizes = n_hidden
        self.n_cond = n_cond
        self.activation = activation
        self.random_masks = random_masks
        self.output_multiplier = output_multiplier
        self.log_scale_clip = log_scale_clip

        # Layers
        layer_sizes = [n_in] + n_hidden + [output_multiplier * n_in]
        self.net = nn.ModuleList()
        self.cond_proj = nn.ModuleList()  # conditioning to each layer (except input)

        for i in range(len(layer_sizes) - 1):
            in_f = layer_sizes[i]
            out_f = layer_sizes[i + 1]
            ml = MaskedLinear(in_f, out_f)
            self.net.append(ml)
            # Conditional projection into this layer as an additive bias
            cp = nn.Linear(n_cond, out_f, bias=False) if n_cond > 0 else None
            self.cond_proj.append(cp)

        self.degrees: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.register_masks()

        # Zero-init last layer for stable start (near identity flow)
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def register_masks(self):
        """Create and set new masks based on random degrees (supports resampling masks per block)."""
        D = self.n_in
        degrees = create_degrees(D, self.hidden_sizes, randomize=self.random_masks)
        self.degrees = degrees

        # Build masks between consecutive layers
        masks = []
        # input -> hidden1
        m = (degrees[1].unsqueeze(-1) >= degrees[0].unsqueeze(0)).float()
        masks.append(m)
        # hidden -> hidden
        for l in range(1, len(self.hidden_sizes)):
            m = (degrees[l + 1].unsqueeze(-1) >= degrees[l].unsqueeze(0)).float()
            masks.append(m)
        # last hidden -> output: output has two params per dim; degree for output dim k equals degree input k
        # We build a (2D x H_L) mask by repeating the output degrees twice (for m and s).
        last_hidden_deg = degrees[-1]
        out_deg = degrees[0]  # degrees per input dimension
        out_deg = torch.cat([out_deg, out_deg], dim=0)  # for [m, s]
        m = (out_deg.unsqueeze(-1) > last_hidden_deg.unsqueeze(0)).float()
        masks.append(m)

        self.masks = masks
        # Apply to layers
        for layer, mask in zip(self.net, masks):
            layer.set_mask(mask)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (m, s) where s is log-scale (clipped).
        x: [B, D], cond: [B, C] or None
        """
        h = x
        L = len(self.net) - 1
        for i, layer in enumerate(self.net):
            h = layer(h)
            if self.n_cond > 0 and self.cond_proj[i] is not None:
                h = h + self.cond_proj[i](cond)
            if i < L:
                h = self.activation(h)
        out = h
        m, log_s = out.chunk(2, dim=-1)
        log_s = torch.clamp(log_s, -self.log_scale_clip, self.log_scale_clip)
        return m, log_s


# ---------------------------
# MAF (stack of conditional MADEs + permutations)
# ---------------------------

class RandomPermutation(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        perm = torch.randperm(D)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(D)
        self.register_buffer("perm", perm)
        self.register_buffer("inv", inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, self.inv]


class MAFBlock(nn.Module):
    """
    One MAF block: z = (x - m(x_<i, cond)) * exp(-s(x_<i, cond))  (forward: data->base)
    logdet = -sum s
    """
    def __init__(self, made: ConditionalMADE):
        super().__init__()
        self.made = made

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        m, log_s = self.made(x, cond)
        z = (x - m) * torch.exp(-log_s)
        logdet = -log_s.sum(dim=-1)
        return z, logdet

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, cond: Optional[torch.Tensor], tol: int = 1) -> torch.Tensor:
        """
        Inverse through MADE is sequential due to autoregressive dependency.
        Compute x_i in order using masks. We exploit the degrees order in MADE.
        """
        B, D = z.shape
        x = torch.zeros_like(z)
        # Determine evaluation order from input degrees (1..D); get indices sorted by degree
        degrees_input = self.made.degrees[0]  # shape [D] with values 1..D permuted
        order = torch.argsort(degrees_input)  # indices of increasing degree
        # Fill features following order
        for idx in order:
            idx = idx.item()
            m, log_s = self.made(x, cond)  # only x_<i are valid; masked MADE ensures dependency compliance
            x[:, idx] = z[:, idx] * torch.exp(log_s[:, idx]) + m[:, idx]
        return x


class ConditionalMAF(nn.Module):
    """
    Full conditional MAF with K blocks and interleaved permutations.
    """
    def __init__(
        self,
        D: int,
        n_cond: int,
        hidden_sizes: List[int],
        K: int = 5,
        activation: nn.Module = nn.ELU(),
        log_scale_clip: float = 5.0,
    ):
        super().__init__()
        self.D = D
        self.K = K
        blocks = []
        perms = []
        for k in range(K):
            made = ConditionalMADE(
                n_in=D,
                n_hidden=hidden_sizes,
                n_cond=n_cond,
                activation=activation,
                random_masks=True,
                output_multiplier=2,
                log_scale_clip=log_scale_clip,
            )
            blocks.append(MAFBlock(made))
            if k < K - 1:
                perms.append(RandomPermutation(D))
        self.blocks = nn.ModuleList(blocks)
        self.perms = nn.ModuleList(perms)
        self.register_buffer("base_mean", torch.zeros(D))
        self.register_buffer("base_logstd", torch.zeros(D))

    def base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # Standard Normal
        log_norm_const = -0.5 * self.D * math.log(2 * math.pi)
        return log_norm_const - 0.5 * (z ** 2).sum(dim=-1)

    def log_prob(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute log p(x | cond).
        """
        h = x if len(x.shape) == 2 else x.reshape(-1, self.D)
        logdet_sum = torch.zeros(h.shape[0], device=h.device)
        cond = cond if len(cond.shape) == 2 else cond.repeat(x.shape[0], 1, 1).reshape(-1, cond.shape[-1])
        for i, block in enumerate(self.blocks):
            h, logdet = block(h, cond)
            logdet_sum = logdet_sum + logdet
            if i < len(self.perms):
                h = self.perms[i](h)
        log_pz = self.base_log_prob(h)
        return (log_pz + logdet_sum).reshape(x.shape[:-1])

    @torch.no_grad()
    def sample(self, n: tuple, cond: Optional[torch.Tensor], device=None) -> torch.Tensor:
        """
        Sample x ~ p(x | cond).
        cond: [n, C] (one per sample)
        """
        device = device or next(self.parameters()).device
        z = torch.randn(n + (self.D, ), device=device).float()

        if len(z.shape) == 2:
            h = z
        else:
            h = z.reshape(-1, self.D)
            cond = cond.unsqueeze(1).repeat(z.shape[0], 1, 1).reshape(-1, cond.shape[-1])
        # Invert flow
        for i in reversed(range(self.K)):
            if i < len(self.perms):
                h = self.perms[i].inverse(h)
            h = self.blocks[i].inverse(h, cond)
        return h.reshape(z.shape)


# ---------------------------
# Image wrapper model
# ---------------------------

class ConditionalImageFlow(nn.Module):
    """
    Wraps preprocessing (dequantization + logit) and the ConditionalMAF core.
    Exposes log_prob on raw images in [0,1] or uint8, and sampling in image space.
    """
    def __init__(self, shape: Tuple[int, int, int], cond_dim: int, hidden_sizes: List[int], K: int = 5, alpha: float = 1e-6):
        super().__init__()
        C, H, W = shape  # channel-first shape
        self.C, self.H, self.W = C, H, W
        self.D = C * H * W
        self.n_classes = cond_dim
        self.alpha = alpha

        self.squash = AtanhSquashTransform(eps=1e-2)

        self.flow = ConditionalMAF(
            D=self.D,
            n_cond=cond_dim,
            hidden_sizes=hidden_sizes,
            K=K,
            activation=nn.ELU(),
            log_scale_clip=5.0,
        )

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

    def _unflatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.C, self.H, self.W)

    def _one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, num_classes=self.n_classes).float()

    def _prep_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = uniform_dequantize(x)
        z, logdet = self.squash(x)
        return z, logdet

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] uint8 or float in [0,1]
        y: [B] long class labels
        Returns log p(x|y).
        """
        z0, logdet0 = self._prep_forward(x)
        # z = x.view(x.size(0), -1)
        # cond = self._one_hot(y).to(x.device)
        log_pz = self.flow.log_prob(z0, y)
        return log_pz + logdet0

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor, device=None) -> torch.Tensor:
        """
        Samples in image space [0,1].
        """
        device = device or next(self.parameters()).device
        # cond = self._one_hot(y.to(device))

        z = self.flow.sample(n, y, device=device)
        x_tanh, _ = self.squash.inverse(z)
        return torch.clamp(x_tanh, -1.0, 1.0) #.view(n, self.C, self.H, self.W)


# ---------------------------
# Minimal training loop example
# ---------------------------

def example_train_loop():
    """
    Minimal example for colored MNIST 10x10x3 (or similar).
    - Resize MNIST to 10x10 and colorize by duplicating channels (or use pre-colored dataset).
    - Replace the dataset block with your colored MNIST loader.
    """
    import torchvision as tv
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example dataset: grayscale MNIST -> resize -> fake "color" (repeat to 3 channels).
    # Replace with your colored MNIST dataset; the model is agnostic if shapes match.
    tfm = tv.transforms.Compose([
        tv.transforms.Resize((20, 20)),
        tv.transforms.ToTensor(),           # [0,1], shape [1,10,10]
        tv.transforms.Lambda(lambda t: t.repeat(3, 1, 1)),  # [3,10,10]
    ])
    train = tv.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    loader = DataLoader(train, batch_size=128, shuffle=True)

    # Model
    C, H, W = 3, 20, 20
    model = ConditionalImageFlow(shape=(C, H, W), n_classes=10, hidden_sizes=[512], K=6).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training
    model.train()
    for step in range(500):
        x, y  = next(iter(loader))
        x, y = x.to(device), y.to(device)
        # maximize log-likelihood == minimize negative log-likelihood
        nll = -model.log_prob(x, y).mean()
        opt.zero_grad(set_to_none=True)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 100 == 0:
            print(f"step {step} | nll: {nll.item():.3f}")
        if step == 5000:  # demo budget; extend as needed
            break

    # Sampling
    model.eval()
    with torch.no_grad():
        y_sample = torch.arange(10, device=device).repeat_interleave(8)  # 80 samples across classes
        x_gen = model.sample(y_sample.size(0), y_sample)
        # Save a grid
        grid = tv.utils.make_grid(x_gen, nrow=8)
        tv.utils.save_image(grid, "samples_maf_colormnist.png")
        print("Saved samples to samples_maf_colormnist.png")


if __name__ == "__main__":
    # Run a quick smoke test if needed.
    example_train_loop()
    # pass
