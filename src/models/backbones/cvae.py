import numpy as np
import torch
import torch.nn as nn

from omegaconf import DictConfig
from pyro.nn import DenseNN
from typing import Tuple, List
from torch_ema import ExponentialMovingAverage

from src.models.utils import HyperDense
from src.models.backbones.neural_cond_estimator import NeuralConditionalDistEstimator
from src.models.backbones.image import ConditionalCNNEncoder, ConditionalCNNDecoder


class CVAE(NeuralConditionalDistEstimator):
    """
    Conditional normalizing flow
    """

    def __init__(self, args: DictConfig = None, kind: str = None, **kwargs):
        super(CVAE, self).__init__(args, kind)

        # Model hyparams & Train params
        self.vae_hid_dim = args.model[self.kind].vae_hid_dim
        self.vae_lat_dim = args.model[self.kind].vae_lat_dim
        self.beta = args.model[self.kind].beta
        self.noise_std_X = args.model[self.kind].noise_std_X

        # Model init = Conditional VAE
        if args.exp.mode == 'tab':
            self.cond_encoder_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                           param_dims=[self.vae_hid_dim * self.dim_out, self.vae_hid_dim,
                                                       2 * self.vae_lat_dim * self.vae_hid_dim, 2 * self.vae_lat_dim],
                                           nonlinearity=torch.nn.ELU()).float()

            self.cond_encoder = HyperDense(in_features=self.dim_out, hid_features=self.vae_hid_dim, out_features=2 * self.vae_lat_dim,
                                           activation=torch.nn.ELU(), cond_nn=self.cond_encoder_nn)

            self.cond_decoder_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                           param_dims=[self.vae_hid_dim * self.vae_lat_dim, self.vae_hid_dim,
                                                       self.dim_out * self.vae_hid_dim, self.dim_out],
                                           nonlinearity=torch.nn.ELU()).float()
            self.cond_decoder = HyperDense(in_features=self.vae_lat_dim, hid_features=self.vae_hid_dim, out_features=self.dim_out,
                                           activation=torch.nn.ELU(), cond_nn=self.cond_decoder_nn)

        elif args.exp.mode == 'img':
            self.cond_encoder = ConditionalCNNEncoder(mode='cvae', shape=(3, args.dataset.img_size, args.dataset.img_size),
                                                      latent_dim=self.vae_lat_dim,
                                                      cond_dim=self.dim_hid + self.dim_treat,
                                                      hidden_channels=(self.vae_hid_dim, self.vae_hid_dim)).float()
            self.cond_decoder = ConditionalCNNDecoder(mode='cvae', shape=(3, args.dataset.img_size, args.dataset.img_size),
                                                      latent_dim=self.vae_lat_dim,
                                                      cond_dim=self.dim_hid + self.dim_treat,
                                                      hidden_channels=(self.vae_hid_dim, self.vae_hid_dim)).float()


        else:
            raise NotImplementedError()

        self.ema_optimizer = None
        self.current_beta = 0.0

    def get_optimizer(self) -> torch.optim.Optimizer | List[torch.optim.Optimizer]:
        """
        Init optimizer for the nuisance flow
        """
        if self.hparams.exp.mode == 'tab':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_encoder_nn, self.cond_decoder_nn])
        elif self.hparams.exp.mode == 'img':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_encoder, self.cond_decoder])
        else:
            raise NotImplementedError()

        if self.kind == 'nuisance':
            return torch.optim.AdamW(list(modules.parameters()), lr=self.lr)
        elif self.kind == 'target':
            optimizer = torch.optim.AdamW(list(modules.parameters()), lr=self.lr)
            ema_target = ExponentialMovingAverage(list(modules.parameters()), decay=self.gamma)
            return optimizer, ema_target
        else:
            raise NotImplementedError()

    def _cond_sample(self, context, n_sample) -> torch.Tensor:
        if torch.tensor(n_sample).prod() > 10000 and len(n_sample) == 2 and self.hparams.exp.mode == 'img':
            sample = []
            batch_size = 500
            for i in range((n_sample[1] - 1) // batch_size + 1):
                z = torch.randn((n_sample[0], batch_size, self.vae_lat_dim)).float()
                sample.append(self.cond_decoder(context[i * batch_size: (i + 1) * batch_size, :].unsqueeze(0), z))
            sample = torch.cat(sample, dim=1)
        else:
            z = torch.randn((*n_sample, self.vae_lat_dim)).float()
            sample = self.cond_decoder(context.unsqueeze(0), z)
        return sample

    # def _post_nuisance_optimizer_step(self):
    #     # Linear warm-up to target beta over first p% of steps
    #     if not hasattr(self, "_beta_warmup"):
    #         self._beta_warmup = 0
    #     self._beta_warmup += 1
    #     p = 1.0
    #     total = max(1, int(p * self.hparams.dataset.n_samples_train / self.batch_size * self.num_epochs))
    #     frac = min(1.0, self._beta_warmup / total)
    #     self.current_beta = float(self.beta) * frac


    def _cond_training_step(self, repr_f, treat_f, out_f_scaled, eval=False, optimizer=None):
        # Representation -> Conditional distribution
        if eval:
            noised_repr_f = repr_f
        else:
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)

        context = torch.cat([noised_repr_f, treat_f], dim=-1)
        z_mu_logstd = self.cond_encoder(context, out_f_scaled)
        z_mu, z_logstd = torch.chunk(z_mu_logstd, 2, dim=-1)
        z_std = torch.exp(z_logstd.clamp(-6.0, 6.0))
        z = z_mu + torch.randn_like(z_std) * z_std

        out_f_recon = self.cond_decoder(context, z)

        mse = ((out_f_recon - out_f_scaled) ** 2).mean(-1, keepdim=True)
        kld = - 0.5 * (1 + 2 * z_std.log() - z_mu.pow(2) - z_std.pow(2)).sum(-1, keepdim=True)

        elbo = mse + self.beta * kld

        return elbo

    def _cond_eval_step(self, repr_f, treat_f, out_f_scaled):
        # context = torch.cat([repr_f, treat_f], dim=-1)
        return self._cond_training_step(repr_f, treat_f, out_f_scaled, eval=True)

