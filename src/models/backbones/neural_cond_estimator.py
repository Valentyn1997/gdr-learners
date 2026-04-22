import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.transforms.spline import ConditionedSpline
from omegaconf import DictConfig
import logging
import numpy as np
from pyro.nn import DenseNN
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rv_continuous
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial
from scipy.special import ndtr
from hydra.utils import instantiate
import ray
from ray import tune
from copy import deepcopy



logger = logging.getLogger(__name__)


class NeuralConditionalDistEstimator(torch.nn.Module):

    def __init__(self, args: DictConfig = None, kind: str = None, **kwargs):
        super(NeuralConditionalDistEstimator, self).__init__()

        self.device = args.exp.device
        self.kind = kind  # nuisance | target
        assert self.kind is not None
        self.has_prop_score = args.model[self.kind].has_prop_score

        # Dataset params
        self.dim_out, self.dim_cov, self.dim_treat = args.model.dim_out, args.model.dim_cov, args.model.dim_treat

        # Model hyparams & Train params
        self.hparams = args

        self.dim_hid = args.model[self.kind].hid_dim
        self.hid_layers = args.model[self.kind].hid_layers
        self.cond_dist_nn = None
        self.num_epochs = args.model[self.kind].num_epochs
        self.num_burn_in_epochs = None if not self.has_prop_score else args.model[self.kind].num_burn_in_epochs
        self.lr = args.model[self.kind].lr
        self.num_train_iter, self.num_burn_in_train_iter = None, None  # Will be calculated later
        self.batch_size = args.model[self.kind].batch_size

        self.gamma = args.model[self.kind].gamma if 'gamma' in args.model[self.kind] else None
        self.num_mc = args.model[self.kind].num_mc if 'num_mc' in args.model[self.kind] else 0

        # Model init
        if self.dim_hid > 0:
            if not self.has_prop_score:
                self.repr_nn = DenseNN(self.dim_cov, self.hid_layers * [self.dim_hid], param_dims=[self.dim_hid]).float()
            else:
                self.repr_nn = DenseNN(self.dim_cov, self.hid_layers * [self.dim_hid], param_dims=[self.dim_hid, self.dim_treat]).float()
        else:
            self.repr_nn = DenseNN(self.dim_cov, self.hid_layers * [15], param_dims=[self.dim_treat]).float()


    def _cond_training_step(self, repr_f, treat_f, out_f_scaled, optimizer=None):
        raise NotImplementedError()

    def _cond_eval_step(self, repr_f, treat_f, out_f_scaled):
        raise NotImplementedError()

    def _post_nuisance_optimizer_step(self):
        pass

    def _cond_log_prob(self, context, out) -> torch.Tensor:
        """
        Internal method for the conditional log-probability
        @param context: Tensor of the context
        @param out: Outcome tensor
        @return: Tensor with conditional log-probabilities
        """
        raise NotImplementedError()

    def _cond_sample(self, context, n_sample) -> torch.Tensor:
        """
        Internal method for the conditional log-probability
        @param context: Tensor of the context
        @param out: Outcome tensor
        @return: Tensor with conditional log-probabilities
        """
        raise NotImplementedError()

    def _cond_dist(self, context) -> torch.distributions.Distribution:
        """
        Internal method for the conditional distribution
        @param context: Tensor of the context
        @return: torch.distributions.Distribution
        """
        raise NotImplementedError()


    def training_step(self, cov_f, treat_f, out_f_scaled, prefix='train', optimizer=None):
        result = {}
        if self.dim_hid > 0:
            if not self.has_prop_score:
                repr_f = self.repr_nn(cov_f)
            else:
                repr_f, prop_preds = self.repr_nn(cov_f)
                result[f'{prefix}_prop_logits'] = prop_preds

            result[f'{prefix}_cond_dist_loss'] = self._cond_training_step(repr_f, treat_f, out_f_scaled, optimizer=optimizer)
        else:
            if self.has_prop_score:
                prop_preds = self.repr_nn(cov_f)
                result[f'{prefix}_prop_logits'] = prop_preds
            result[f'{prefix}_cond_dist_loss'] = self._cond_training_step(cov_f, treat_f, out_f_scaled, optimizer=optimizer)
        return result

    def eval_step(self, cov_f, treat_f, out_f_scaled, prefix):
        result = {}
        if self.dim_hid > 0:
            if not self.has_prop_score:
                repr_f = self.repr_nn(cov_f)
            else:
                repr_f, prop_preds = self.repr_nn(cov_f)
                result[f'{prefix}_prop_logits'] = prop_preds

            result[f'{prefix}_cond_dist_loss'] = self._cond_eval_step(repr_f, treat_f, out_f_scaled)
        else:
            if self.has_prop_score:
                prop_preds = self.repr_nn(cov_f)
                result[f'{prefix}_prop_logits'] = prop_preds
            result[f'{prefix}_cond_dist_loss'] = self._cond_eval_step(cov_f, treat_f, out_f_scaled)
        return result

    def cond_log_prob(self, treat_f, out_f_scaled, cov_f) -> torch.Tensor:
        """
        Conditional log-probability
        @param treat_f: Tensor with factual treatments
        @param out_f_scaled: Tensor with factual outcomes
        @param cov_f: Tensor with factual covariates
        @return: Tensor with log-probabilities
        """
        if self.dim_hid > 0:
            if not self.has_prop_score:
                repr_f = self.repr_nn(cov_f)
            else:
                repr_f, _ = self.repr_nn(cov_f)
            context = torch.cat([repr_f, treat_f], dim=-1)
        else:  # linear hypernetwork setup
            context = torch.cat([cov_f, treat_f], dim=-1)
        log_prob = self._cond_log_prob(context, out_f_scaled)
        return log_prob

    def cond_dist(self, treat_f, cov_f) -> torch.distributions.Distribution:
        if self.dim_hid > 0:
            if not self.has_prop_score:
                repr_f = self.repr_nn(cov_f)
            else:
                repr_f, _ = self.repr_nn(cov_f)
            context = torch.cat([repr_f, treat_f], dim=-1)
        else:  # linear hypernetwork setup
            context = torch.cat([cov_f, treat_f], dim=-1)
        cond_dist = self._cond_dist(context)
        return cond_dist

    def cond_sample(self, treat_f, cov_f, n_sample) -> torch.Tensor:
        if self.dim_hid > 0:
            if not self.has_prop_score:
                repr_f = self.repr_nn(cov_f)
            else:
                repr_f, _ = self.repr_nn(cov_f)
            context = torch.cat([repr_f, treat_f], dim=-1)
        else:  # linear hypernetwork setup
            context = torch.cat([cov_f, treat_f], dim=-1)
        cond_sample = self._cond_sample(context, n_sample)
        return cond_sample

    def get_propensity(self, cov_f) -> [torch.Tensor, torch.Tensor]:
        if self.has_prop_score:
            _, prop_preds = self.repr_nn(cov_f)
            if self.dim_treat == 1:
                prop1 = torch.sigmoid(prop_preds)
                return prop1
            else:
                prop = torch.softmax(prop_preds, dim=-1)
                return prop
        else:
            raise NotImplementedError()
