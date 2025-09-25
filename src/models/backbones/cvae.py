import torch
import torch.nn as nn

from omegaconf import DictConfig
from pyro.nn import DenseNN
from typing import Tuple, List
from torch_ema import ExponentialMovingAverage

from src.models.utils import HyperDense
from src.models.backbones.neural_cond_estimator import NeuralConditionalDistEstimator


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

        # self.cond_encoder = DenseNN(self.dim_out + self.dim_hid + self.dim_treat, [self.vae_hid_dim],
        #                             param_dims=[self.vae_lat_dim, self.vae_lat_dim], nonlinearity=torch.nn.ELU()).float()
        # self.cond_decoder = DenseNN(self.vae_lat_dim + self.dim_hid + self.dim_treat, [self.vae_hid_dim],
        #                             param_dims=[self.dim_out], nonlinearity=torch.nn.ELU()).float()


        self.ema_optimizer = None

    def get_optimizer(self) -> torch.optim.Optimizer | List[torch.optim.Optimizer]:
        """
        Init optimizer for the nuisance flow
        """
        if self.kind == 'nuisance':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_encoder_nn, self.cond_decoder_nn])
            # return torch.optim.SGD(list(modules.parameters()), lr=self.lr, momentum=0.9)
            return torch.optim.AdamW(list(modules.parameters()), lr=self.lr)
        elif self.kind == 'target':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_encoder_nn, self.cond_decoder_nn])
            # optimizer = torch.optim.SGD(list(modules.parameters()), lr=self.lr, momentum=0.9)
            optimizer = torch.optim.AdamW(list(modules.parameters()), lr=self.lr)
            ema_target = ExponentialMovingAverage(list(modules.parameters()), decay=self.gamma)
            return optimizer, ema_target
        else:
            raise NotImplementedError()

    def _cond_sample(self, context, n_sample) -> torch.Tensor:
        z = torch.randn((*n_sample, self.vae_lat_dim)).float()
        sample = self.cond_decoder(context.unsqueeze(0), z)
        # context_z = torch.cat([context.unsqueeze(0).repeat(n_sample[0], 1, 1), z], dim=-1)
        # sample = self.cond_decoder(context_z)
        return sample


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

        mse = (out_f_recon - out_f_scaled) ** 2
        kld = -0.5 * (1 + z_std.log() - z_mu.pow(2) - z_std).sum(-1, keepdim=True)

        elbo = mse + self.beta * kld

        # if eval:
        #     noised_repr_f = repr_f
        # else:
        #     noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
        # context = torch.cat([noised_repr_f, treat_f], dim=-1)
        # context_out = torch.cat([context, out_f_scaled], dim=-1)
        #
        # z_mu, z_logstd = self.cond_encoder(context_out)
        # # z_mu, z_logstd = torch.chunk(z_mu_logstd, 2, dim=-1)
        # z_std = torch.exp(z_logstd.clamp(-6.0, 6.0))
        # z = z_mu + torch.randn_like(z_std) * z_std
        #
        # context_z = torch.cat([context, z], dim=-1)
        # out_f_recon = self.cond_decoder(context_z)
        #
        # mse = (out_f_recon - out_f_scaled) ** 2
        # kld = -0.5 * (1 + z_std.log() - z_mu.pow(2) - z_std).sum(-1, keepdim=True)
        #
        # elbo = mse + self.beta * kld
        return elbo

    def _cond_eval_step(self, repr_f, treat_f, out_f_scaled):
        # context = torch.cat([repr_f, treat_f], dim=-1)
        return self._cond_training_step(repr_f, treat_f, out_f_scaled, eval=True)

