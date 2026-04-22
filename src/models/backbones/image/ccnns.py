import math
from typing import Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# FiLM-conditioned blocks
# -----------------------
class FiLM(nn.Module):
    """Per-channel scale (gamma) and shift (beta) from a conditioning vector."""
    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, num_channels, bias=True)
        self.to_beta  = nn.Linear(cond_dim, num_channels, bias=True)
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor):
        # h: [B,C,H,W], cond: [B,cond_dim]
        gamma = self.to_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta  = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + gamma) + beta


class ConvBlock(nn.Module):
    """Conv → GroupNorm → ELU (+ FiLM)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act  = nn.ELU()
        self.film = FiLM(cond_dim, out_ch) if cond_dim > 0 else None

    def forward(self, x: torch.Tensor, cond_vec: Optional[torch.Tensor]):
        h = self.conv(x)
        h = self.norm(h)
        if self.film is not None and cond_vec is not None:
            h = self.film(h, cond_vec)
        h = self.act(h)
        return h



class ConditionalCNNEncoder(nn.Module):
    """
    10x10 -> stride2 -> 5x5 bottleneck (channels increase).
    Outputs mean and logvar of latent z.
    """
    def __init__(self, mode: str, shape: Tuple[int, int, int], latent_dim: int, cond_dim: int,
                 hidden_channels: Sequence[int] = (64, 128)):
        super().__init__()
        C, H, W = shape
        assert H == 10 and W == 10, "This encoder assumes 10x10 inputs (adjust strides otherwise)."
        assert mode in ['cvae', 'cgan', 'cdiffusion']

        self.D = C * H * W
        self.latent_dim = latent_dim
        self.mode = mode

        ch1, ch2 = hidden_channels
        self.b1 = ConvBlock(C,   ch1, stride=1, cond_dim=cond_dim)   # 10x10
        self.b2 = ConvBlock(ch1, ch2, stride=2, cond_dim=cond_dim)   # 5x5
        self.b3 = ConvBlock(ch2, ch2, stride=1, cond_dim=cond_dim)   # 5x5

        feat_dim = ch2 * 5 * 5
        if self.mode == 'cvae':
            self.to_mu = nn.Linear(feat_dim, latent_dim)
            self.to_logstd = nn.Linear(feat_dim, latent_dim)
        elif self.mode == 'cgan':
            self.to_logit = nn.Linear(feat_dim, 1)
        else:
            self.to_lat = nn.Linear(feat_dim, latent_dim)

    def forward(self, cond: torch.Tensor, x: torch.Tensor):
        # cond = self.embed(y)  # [B,emb_dim]
        h = x if len(x.shape) == 2 else x.reshape(-1, self.D)
        if len(cond.shape) == 3:
            if cond.shape[:2] != x.shape[:2]:
                cond = cond.repeat(x.shape[0], 1, 1).reshape(-1, cond.shape[-1])
            else:
                cond = cond.reshape(-1, cond.shape[-1])

        h = h.reshape(-1, 3, 10, 10)
        h = self.b1(h, cond)
        h = self.b2(h, cond)
        h = self.b3(h, cond)
        h = h.flatten(1)

        if self.mode == 'cvae':
            mu, logvar = self.to_mu(h), self.to_logstd(h)
            mu_logvar = torch.cat((mu, logvar), -1)
            mu_logvar = mu_logvar.reshape(x.shape[:-1] + (2 * self.latent_dim, ))
            return mu_logvar
        elif self.mode == 'cgan':
            logit = self.to_logit(h)
            logit = logit.reshape(x.shape[:-1] + (1, ))
            return logit
        else:
            lat = self.to_lat(h)
            lat = lat.reshape(x.shape[:-1] + (self.latent_dim, ))
            return lat



class ConditionalCNNDecoder(nn.Module):
    """
    Takes z and condition; reconstructs to [-1,1] via tanh.
    """
    def __init__(self, mode: str, shape: Tuple[int, int, int], latent_dim: int, cond_dim: int,
                 hidden_channels: Sequence[int] = (64, 128)):
        super().__init__()
        C, H, W = shape
        assert H == 10 and W == 10
        assert mode in ['cvae', 'cgan', 'cdiffusion']

        self.D = C * H * W
        self.latent_dim = latent_dim
        self.mode = mode

        ch1, ch2 = hidden_channels  # mirror encoder; bottleneck uses ch2
        self.fc = nn.Linear(latent_dim + cond_dim, ch2 * 5 * 5)

        # decode 5x5 → 10x10
        self.db1 = ConvBlock(ch2, ch2, stride=1, cond_dim=cond_dim)              # 5x5
        self.up  = nn.ConvTranspose2d(ch2, ch1, kernel_size=4, stride=2, padding=1)  # 10x10
        self.norm_up = nn.GroupNorm(num_groups=min(8, ch1), num_channels=ch1)
        self.film_up = FiLM(cond_dim, ch1)
        self.act = nn.ELU()

        self.out = nn.Conv2d(ch1, C, kernel_size=3, padding=1)

    def forward(self, cond: torch.Tensor, z: torch.Tensor):
        # emb = self.embed(y)
        h = z if len(z.shape) == 2 else z.reshape(-1, self.latent_dim)

        if len(cond.shape) == 3:
            if cond.shape[:2] != z.shape[:2]:
                cond = cond.repeat(z.shape[0], 1, 1).reshape(-1, cond.shape[-1])
            else:
                cond = cond.reshape(-1, cond.shape[-1])

        h = torch.cat([h, cond], dim=-1)
        h = self.fc(h).view(h.size(0), -1, 5, 5)
        h = self.db1(h, cond)
        h = self.up(h)
        h = self.norm_up(h)
        h = self.film_up(h, cond)
        h = self.act(h)
        if self.mode in ['cvae', 'cgan']:
            x_hat = torch.tanh(self.out(h))  # [-1,1]
        else:
            x_hat = self.out(h)
        x_hat = x_hat.reshape(z.shape[:-1] + (self.D, ))
        return x_hat


class ConditionalUNet(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], latent_dim: int, cond_dim: int, hidden_channels: Sequence[int] = (64, 64)):
        super().__init__()
        C, H, W = shape
        assert H == 10 and W == 10

        self.D = C * H * W
        self.cond_enc = ConditionalCNNEncoder('cdiffusion', shape, latent_dim, cond_dim, hidden_channels)
        self.cond_dec = ConditionalCNNDecoder('cdiffusion', shape, latent_dim, cond_dim, hidden_channels)

    def forward(self, cond: torch.Tensor, x: torch.Tensor):
        # Moving t_emd from x into cond
        x, t_emb = x[..., :self.D], x[..., self.D:]
        if len(cond.shape) == 2:
            cond = torch.cat([cond, t_emb], -1)
        else:
            cond = torch.cat([cond.repeat(t_emb.shape[0], 1, 1), t_emb], -1)

        z = self.cond_enc(cond, x)
        x_rec = self.cond_dec(cond, z)
        return x_rec
