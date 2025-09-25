import torch
import torch.nn as nn

from omegaconf import DictConfig
from pyro.nn import DenseNN
from typing import Tuple, List
from torch_ema import ExponentialMovingAverage

from src.models.utils import HyperDense, SinusoidalEmbedding
from src.models.backbones.neural_cond_estimator import NeuralConditionalDistEstimator


def linear_beta_schedule(T: int, b0: float, b1: float) -> torch.Tensor:
    return torch.linspace(b0, b1, T)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Gather 1-D per-t values and reshape to x_shape."""
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

# -----------------------
# Diffusion core (DDPM)
# -----------------------
class Diffusion:
    def __init__(self, T, beta_start, beta_end, dim_out):
        betas = linear_beta_schedule(T, beta_start, beta_end).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).float(), alphas_cumprod[:-1]], dim=0)


        self.T = T
        self.dim_out = dim_out
        # Register for easy gather
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # Posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)


    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t.reshape(-1), x0.shape).reshape(x0.shape[:-1] + (1, )) * x0 + \
            extract(self.sqrt_one_minus_alphas_cumprod, t.reshape(-1), x0.shape).reshape(x0.shape[:-1] + (1, )) * noise


    def p_mean_variance(self, model: nn.Module, t_embed: nn.Module, x_t: torch.Tensor, t: torch.Tensor):
        t_emb = t_embed(t)
        eps = model(torch.cat([x_t, t_emb.unsqueeze(1).repeat(1, x_t.shape[1], 1)], dim=-1))
        # Equation 11 in DDPM paper
        mean = extract(self.sqrt_recip_alphas, t, x_t.shape) * (x_t - eps * extract((self.betas / self.sqrt_one_minus_alphas_cumprod), t, x_t.shape))
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var


    @torch.no_grad()
    def p_sample(self, model: nn.Module, t_embed: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, var, log_var = self.p_mean_variance(model, t_embed, x_t, t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        noise = torch.randn_like(x_t)
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, t_embed: nn.Module, n_sample: int) -> torch.Tensor:
        # model.eval()
        x_t = torch.randn(*n_sample, self.dim_out).float()
        for step in reversed(range(self.T)):
            t = torch.full((n_sample[0],), step, dtype=torch.long)
            x_t = self.p_sample(model, t_embed, x_t, t)
        return x_t


class CDiffusion(NeuralConditionalDistEstimator):
    """
    Conditional normalizing flow
    """

    def __init__(self, args: DictConfig = None, kind: str = None, **kwargs):
        super(CDiffusion, self).__init__(args, kind)

        # Model hyparams & Train params
        self.T = args.model[self.kind].T
        self.t_dim = args.model[self.kind].t_dim
        self.diffusion_hid_dim = args.model[self.kind].diffusion_hid_dim
        self.noise_std_X = args.model[self.kind].noise_std_X

        # Model init = Conditional VAE
        self.t_embed = SinusoidalEmbedding(self.t_dim).float()
        self.diffusion = Diffusion(self.T, 1e-4, 2e-2, self.dim_out)

        self.cond_eps_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                       param_dims=[self.diffusion_hid_dim * (self.dim_out + self.t_dim), self.diffusion_hid_dim,
                                                   self.dim_out * self.diffusion_hid_dim, self.dim_out],
                                       nonlinearity=torch.nn.ELU()).float()

        self.cond_eps = HyperDense(in_features=self.dim_out + self.t_dim, hid_features=self.diffusion_hid_dim,
                                   out_features=self.dim_out, activation=torch.nn.ELU(), cond_nn=self.cond_eps_nn)

        self.ema_optimizer = None

    def get_optimizer(self) -> torch.optim.Optimizer | List[torch.optim.Optimizer]:
        """
        Init optimizer for the nuisance flow
        """
        if self.kind == 'nuisance':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_eps_nn])
            # return torch.optim.SGD(list(modules.parameters()), lr=self.lr, momentum=0.9)
            return torch.optim.AdamW(list(modules.parameters()), lr=self.lr)
        elif self.kind == 'target':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_eps_nn])
            # optimizer = torch.optim.SGD(list(modules.parameters()), lr=self.lr, momentum=0.9)
            optimizer = torch.optim.AdamW(list(modules.parameters()), lr=self.lr)
            ema_target = ExponentialMovingAverage(list(modules.parameters()), decay=self.gamma)
            return optimizer, ema_target
        else:
            raise NotImplementedError()

    def _cond_sample(self, context, n_sample) -> torch.Tensor:
        sample = self.diffusion.sample(lambda inp: self.cond_eps(context.unsqueeze(0), inp), self.t_embed, n_sample=n_sample)
        return sample


    def _cond_training_step(self, repr_f, treat_f, out_f_scaled, eval=False, optimizer=None):
        # Representation -> Conditional distribution
        if eval:
            noised_repr_f = repr_f
        else:
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)

        context = torch.cat([noised_repr_f, treat_f], dim=-1)

        t = torch.randint(0, self.T, (out_f_scaled.shape[:-1]))
        t_emb = self.t_embed(t.reshape(-1)).reshape(out_f_scaled.shape[:-1] + (self.t_dim, ))
        noise = torch.randn_like(out_f_scaled)

        out_f_scaled_t = self.diffusion.q_sample(out_f_scaled, t, noise)
        pred = self.cond_eps(context, torch.cat([out_f_scaled_t, t_emb], -1))
        loss = (pred - noise) ** 2
        return loss

    def _cond_eval_step(self, repr_f, treat_f, out_f_scaled):
        # return self._cond_training_step(repr_f, treat_f, out_f_scaled, eval=True)
        context = torch.cat([repr_f, treat_f], dim=-1)
        out_f_sample = self.diffusion.sample(lambda inp: self.cond_eps(context.unsqueeze(0), inp), self.t_embed, n_sample=(1, out_f_scaled.shape[0]))
        out_f_sample = out_f_sample.squeeze(0)
        return (out_f_scaled - out_f_sample) ** 2
