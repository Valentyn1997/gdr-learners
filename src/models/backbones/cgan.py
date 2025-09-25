import torch
import torch.nn as nn

from omegaconf import DictConfig
from pyro.nn import DenseNN
from typing import Tuple, List
from torch_ema import ExponentialMovingAverage

from src.models.utils import HyperDense
from src.models.backbones.neural_cond_estimator import NeuralConditionalDistEstimator


class CGAN(NeuralConditionalDistEstimator):
    """
    Conditional normalizing flow
    """

    def __init__(self, args: DictConfig = None, kind: str = None, **kwargs):
        super(CGAN, self).__init__(args, kind)

        # Model hyparams & Train params
        self.gan_hid_dim = args.model[self.kind].gan_hid_dim
        self.noise_std_X = args.model[self.kind].noise_std_X

        # Model init = Conditional GAN

        self.cond_generator_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                       param_dims=[self.gan_hid_dim * self.dim_out, self.gan_hid_dim,
                                                   self.dim_out * self.gan_hid_dim, self.dim_out],
                                       nonlinearity=torch.nn.ELU()).float()

        self.cond_generator = HyperDense(in_features=self.dim_out, hid_features=self.gan_hid_dim, out_features=self.dim_out,
                                         activation=torch.nn.ReLU(), cond_nn=self.cond_generator_nn)

        self.cond_discriminator_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                       param_dims=[self.gan_hid_dim * self.dim_out, self.gan_hid_dim, self.gan_hid_dim, 1],
                                       nonlinearity=torch.nn.ELU()).float()
        self.cond_discriminator = HyperDense(in_features=self.dim_out, hid_features=self.gan_hid_dim, out_features=1,
                                       activation=torch.nn.ReLU(), cond_nn=self.cond_discriminator_nn)

        # self.cond_generator = DenseNN(self.dim_out + self.dim_hid + self.dim_treat, 2 * [self.gan_hid_dim], [self.dim_out],
        #                               nonlinearity=torch.nn.ELU()).float()
        #
        # self.cond_discriminator = DenseNN(self.dim_out + self.dim_hid + self.dim_treat, 2 * [self.gan_hid_dim], [1],
        #                                   nonlinearity=torch.nn.ELU()).float()

        self.ema_optimizer = None

    def get_optimizer(self) -> torch.optim.Optimizer | List[torch.optim.Optimizer]:
        """
        Init optimizer for the nuisance flow
        """
        if self.kind == 'nuisance':
            modules_generator = torch.nn.ModuleList([self.repr_nn, self.cond_generator])
            modules_discriminator = torch.nn.ModuleList([self.repr_nn, self.cond_discriminator])
            return [(torch.optim.SGD(list(modules_generator.parameters()), lr=self.lr, momentum=0.9), 'generator'),
                    (torch.optim.SGD(list(modules_discriminator.parameters()), lr=self.lr, momentum=0.9), 'discriminator')]
        elif self.kind == 'target':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_generator_nn, self.cond_discriminator_nn])
            modules_generator = torch.nn.ModuleList([self.repr_nn, self.cond_generator_nn])
            modules_discriminator = torch.nn.ModuleList([self.repr_nn, self.cond_discriminator_nn])
            optimizer_generator = torch.optim.SGD(list(modules_generator.parameters()), lr=self.lr, momentum=0.9)
            optimizer_discriminator = torch.optim.SGD(list(modules_discriminator.parameters()), lr=self.lr, momentum=0.9)
            ema_target = ExponentialMovingAverage(list(modules.parameters()), decay=self.gamma)
            return [(optimizer_generator, 'generator'), (optimizer_discriminator, 'discriminator')], ema_target
        else:
            raise NotImplementedError()

    def _cond_sample(self, context, n_sample) -> torch.Tensor:
        z = torch.randn((*n_sample, self.dim_out)).float()
        # sample = self.cond_generator(torch.cat([context.unsqueeze(0).repeat(n_sample[0], 1, 1), z], dim=-1))
        sample = self.cond_generator(context.unsqueeze(0), z)
        return sample

    def _cond_training_step(self, repr_f, treat_f, out_f_scaled, eval=False, optimizer=None):

        _, optimizer_name = optimizer

        if eval:
            noised_repr_f = repr_f
        else:
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
        context = torch.cat([noised_repr_f, treat_f], dim=-1)

        if optimizer_name == 'generator':
            real_labels = torch.ones_like(treat_f) * 0.95
            z = torch.randn_like(out_f_scaled)
            out_f_sample = self.cond_generator(context, z).detach()
            # out_f_sample = self.cond_generator(torch.cat([context, z], dim=-1)).detach()
            logits = self.cond_discriminator(context, out_f_sample)
            # logits = self.cond_discriminator(torch.cat([context, out_f_sample], dim=-1))

            bce_loss = torch.binary_cross_entropy_with_logits(logits, real_labels)

            return bce_loss

        elif optimizer_name == 'discriminator':
            real_labels, fake_labels = torch.ones_like(treat_f) * 0.95, torch.zeros_like(treat_f) + 0.05
            z = torch.randn_like(out_f_scaled)
            out_f_sample = self.cond_generator(context, z).detach()
            # out_f_sample = self.cond_generator(torch.cat([context, z], dim=-1)).detach()

            logits_real = self.cond_discriminator(context, out_f_scaled)
            # logits_real = self.cond_discriminator(torch.cat([context, out_f_scaled], dim=-1))
            logits_fake = self.cond_discriminator(context, out_f_sample)
            # logits_fake = self.cond_discriminator(torch.cat([context, out_f_sample], dim=-1))

            bce_loss_real = torch.binary_cross_entropy_with_logits(logits_real, real_labels)
            bce_loss_fake = torch.binary_cross_entropy_with_logits(logits_fake, fake_labels)

            return bce_loss_real + bce_loss_fake

        else:
            raise NotImplementedError()

    def _cond_eval_step(self, repr_f, treat_f, out_f_scaled):
        context = torch.cat([repr_f, treat_f], dim=-1)
        z = torch.randn_like(out_f_scaled)
        out_f_sample = self.cond_generator(context, z)
        # out_f_sample = self.cond_generator(torch.cat([context, z], dim=-1))
        return (out_f_scaled - out_f_sample) ** 2

