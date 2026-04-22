from typing import Callable
import numpy as np
from torch import nn
from tqdm import tqdm
import torch
import ot
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from ray import tune
from copy import deepcopy
from scipy.stats import norm, expon
from pyro.nn import DenseNN
import math
import torch
import torch.nn as nn
from pyro.nn import PyroModule


class HyperDense(PyroModule):
    """
    A linear layer whose weight and bias are *generated per-sample*.
    """
    def __init__(
        self,
        in_features: int,
        hid_features: int,
        out_features: int,
        activation: Callable,
        cond_nn: DenseNN
    ):
        super().__init__()
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.cond_nn = cond_nn

        self.activation = activation

        # self.hyper = nn.Sequential(
        #     nn.Linear(cond_dim, hyper_hidden),
        #     nn.ReLU() if activation == "relu" else nn.Tanh(),
        #     nn.Linear(hyper_hidden, out_features * in_features + out_features),
        # )

    def forward(self, context: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        B = context.shape[:-1]
        W1, b1, W2, b2 = self.cond_nn(context)

        # W1, b1, W2, b2 = torch.split(params, [self.hid_features * self.in_features, self.hid_features,
        #                                       self.out_features * self.hid_features, self.out_features], dim=-1)
        W1 = W1.view(*B, self.hid_features, self.in_features)  # [B, O, I]
        W2 = W2.view(*B, self.out_features, self.hid_features)  # [B, O, I]

        # Out: [B, O] via einsum
        if len(W1.shape) == 3:
            out = torch.einsum("boi,bi->bo", W1, inp) + b1
            out = self.activation(out)
            out = torch.einsum("boi,bi->bo", W2, out) + b2
        else:
            out = torch.einsum("aboi,abi->abo", W1, inp) + b1
            out = self.activation(out)
            out = torch.einsum("aboi,abi->abo", W2, out) + b2
        return out


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Fixed frequencies per Transformers
        half = dim // 2
        freq = torch.exp(-math.log(10_000) * torch.arange(half) / (half - 1))
        self.register_buffer("freq", freq)


    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t assumed in [0, T-1]; map to [0,1]
        t = t.float().unsqueeze(1)  # (B,1)
        angles = t * self.freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < self.dim:
        # pad if odd dim
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
        return emb

# class HyperDense2(PyroModule):
#     """
#     A linear layer whose weight and bias are *generated per-sample*.
#     """
#     def __init__(
#         self,
#         in_features: int,
#         hid_features0: int,
#         hid_features1: int,
#         out_features: int,
#         activation: Callable,
#         cond_nn: DenseNN
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.hid_features0 = hid_features
#         self.hid_features1 = hid_features
#         self.out_features = out_features
#         self.cond_nn = cond_nn
#
#         self.activation = activation
#
#         # self.hyper = nn.Sequential(
#         #     nn.Linear(cond_dim, hyper_hidden),
#         #     nn.ReLU() if activation == "relu" else nn.Tanh(),
#         #     nn.Linear(hyper_hidden, out_features * in_features + out_features),
#         # )
#
#     def forward(self, context: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
#         B = context.shape[:-1]
#         W1, b1, W2, b2 = self.cond_nn(context)
#
#         # W1, b1, W2, b2 = torch.split(params, [self.hid_features * self.in_features, self.hid_features,
#         #                                       self.out_features * self.hid_features, self.out_features], dim=-1)
#         W1 = W1.view(*B, self.hid_features, self.in_features)  # [B, O, I]
#         W2 = W2.view(*B, self.out_features, self.hid_features)  # [B, O, I]
#
#         # Out: [B, O] via einsum
#         if len(W1.shape) == 3:
#             out = torch.einsum("boi,bi->bo", W1, inp) + b1
#             out = self.activation(out)
#             out = torch.einsum("boi,bi->bo", W2, out) + b2
#         else:
#             out = torch.einsum("aboi,abi->abo", W1, inp) + b1
#             out = self.activation(out)
#             out = torch.einsum("aboi,abi->abo", W2, out) + b2
#         return out


def get_iptw(treat_f, prop, clip_prop, normalize=False):
    if treat_f.shape[-1] == 1:  # Binary treatment
        ipwt = ((treat_f == 1.0) & (prop >= clip_prop)).float() / (prop + 1e-9) + \
               ((treat_f == 0.0) & ((1 - prop) >= clip_prop)).float() / (1 - prop + 1e-9)
    else:  # Multiple treatments
        ipwt = (prop[treat_f == 1.0] >= clip_prop).float() / (prop[treat_f == 1.0] + 1e-9)
        ipwt = ipwt.unsqueeze(-1)
    if normalize:
        ipwt_normalized = ipwt / ipwt.mean()
        return ipwt_normalized
    else:
        return ipwt


def wass_dist(sample0, sample1):
    wass_dist = []
    for i in tqdm(range(sample0.shape[1])):
        with torch.no_grad():
            s0, s1 = sample0[:, i:(i+1)], sample1[:, i:(i+1)]
            s0 = s0.squeeze(1) if len(s0.shape) == 3 else s0
            s1 = s1.squeeze(1) if len(s1.shape) == 3 else s1
            M = ot.dist(s0, s1)
            w0, w1 = torch.ones(s0.shape[0]), torch.ones(s1.shape[0])
            wass_dist.append(ot.emd2(w0 / (w0.sum()), w1 / (w1.sum()), M))
    return torch.tensor(wass_dist)


def subset_by_indices(data_dict: dict, indices: list):
    subset_data_dict = {}
    for (k, v) in data_dict.items():
        if not isinstance(data_dict[k], float) and not isinstance(data_dict[k], str):
            subset_data_dict[k] = np.copy(data_dict[k][indices])
        else:
            subset_data_dict[k] = deepcopy(data_dict[k])
    return subset_data_dict


def fit_eval_kfold(args: dict, orig_hparams: DictConfig, model_cls, train_data_dict: dict, val_data_dict: dict,
                   kind: str = None, **kwargs):
    """
    Globally defined method, used for ray tuning
    :param args: Hyperparameter configuration
    :param orig_hparams: DictConfig of original hyperparameters
    :param model_cls: class of model
    :param kwargs: Other args
    """
    new_params = deepcopy(orig_hparams)
    model_cls.set_nuisances_hparams(new_params['model']['nuisance'], args)
    # model_cls.set_subnet_hparams(new_params[subnet_name], args)
    # new_params.exp.device = 'cpu'
    torch.set_default_device('cuda')

    if val_data_dict is None:  # KFold hparam tuning
        kf = KFold(n_splits=5, random_state=orig_hparams.exp.seed, shuffle=True)
        val_metrics = []
        for train_index, val_index in kf.split(train_data_dict['cov_f']):
            ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, train_index), \
                                              subset_by_indices(train_data_dict, val_index)

            model = model_cls(new_params, kind=kind, **kwargs)
            model.fit(train_data_dict=ttrain_data_dict, log=False)
            log_dict = model. evaluate_nuisance(data_dict=val_data_dict, log=False, prefix='val')
            val_metrics.append(log_dict[model.val_metric])
        tune.report(val_metric=np.mean(val_metrics))

    else:  # predefined hold-out hparam tuning
        model = model_cls(new_params, kind=kind, **kwargs)
        model.fit(train_data_dict=train_data_dict, log=False)
        log_dict = model.evaluate_nuisance(data_dict=val_data_dict, log=False, prefix='val')
        tune.report(val_metric=log_dict[model.val_metric])


# class NormalizedRBF:
#     """
#     Normalized radial basis function, used for KDE and DKME
#     """
#     def __init__(self, sd):
#         self.sd = sd
#         self.rbf = RBF(np.sqrt(self.sd / 2))
#
#     def __call__(self, x1, x2):
#         return 1 / np.sqrt(np.pi * self.sd) * self.rbf(x1, x2)
