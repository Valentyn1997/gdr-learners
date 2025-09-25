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
import math
from hydra.utils import instantiate

logger = logging.getLogger(__name__)

from src.models.utils import get_iptw
from src.models import PluginNeuralConditionalDensityEstimator


class RANeuralConditionalDensityEstimator(PluginNeuralConditionalDensityEstimator):

    def __init__(self, args: DictConfig = None, **kwargs):
        super(RANeuralConditionalDensityEstimator, self).__init__(args)

        # Plug-in = nuisance model
        self.target_model = instantiate(args.model.target, args, 'target', _recursive_=False)
        self.pot_out_model = self.target_model

    def _mlflow_log_target(self, result, prefix, step):
        for k in result.keys():
            self.mlflow_logger.log_metrics({k: result[k].item()}, step=step)

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        cov_f, treat_f, out_f_scaled = super(RANeuralConditionalDensityEstimator, self).prepare_train_data(train_data_dict)
        self.target_model.num_train_iter = int(self.hparams.dataset.n_samples_train / self.target_model.batch_size * self.target_model.num_epochs)
        return cov_f, treat_f, out_f_scaled


    def fit(self, train_data_dict: dict, log: bool):
        """
        Fitting the two-stage estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        super(RANeuralConditionalDensityEstimator, self).fit(train_data_dict, log)

        # Preparing data
        cov_f, treat_f, out_f_scaled = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f_scaled, self.target_model.batch_size)

        # Preparing optimizers
        target_optimizers, self.ema_optimizer = self.target_model.get_optimizer()
        target_optimizers = [(target_optimizers, 'main')] if not isinstance(target_optimizers, list) else target_optimizers
        self.to(self.device)


        # Conditional generative model fitting
        logger.info('Fitting target models')
        for step in tqdm(range(self.target_model.num_train_iter)) if log else range(self.target_model.num_train_iter):
            cov_f, treat_f, out_f_scaled = next(iter(train_dataloader))

            for target_optimizer, optimizer_name in target_optimizers:

                target_optimizer.zero_grad()
                result_f = self.target_model.training_step(cov_f, treat_f, out_f_scaled,
                                                           optimizer=(target_optimizer, optimizer_name))

                loss, result = 0.0, {}
                for treat in self.treat_options:
                    treat_pot = torch.ones_like(treat_f) * treat
                    out_pot_sample_scaled = self.nuisance_model.cond_sample(treat_pot, cov_f, n_sample=(self.target_model.num_mc, cov_f.shape[0])).detach()
                    result_pot = self.target_model.training_step(cov_f.unsqueeze(0), treat_pot.unsqueeze(0), out_pot_sample_scaled,
                                                                 optimizer=(target_optimizer, optimizer_name))

                    loss_treat = (((treat == treat_f) * result_f['train_cond_dist_loss'] +
                                  (1 - (treat == treat_f).float()) * result_pot['train_cond_dist_loss'].mean(0))).mean()

                    assert not math.isnan(loss_treat)

                    result[f'train_loss_target_{treat}'] = loss_treat
                    loss += loss_treat

                loss.backward()

                target_optimizer.step()
                self.target_model._post_nuisance_optimizer_step()
                self.ema_optimizer.update()

                if step % 100 == 0 and log:
                    self._mlflow_log_target(result, 'train', step)

    def evaluate_cond_pot_out_dist(self, data_dict: dict, dataset, log: bool, prefix: str, kind: str):
        with self.ema_optimizer.average_parameters():
            return super(RANeuralConditionalDensityEstimator, self).evaluate_cond_pot_out_dist(data_dict, dataset, log, prefix, kind)


class DRNeuralConditionalDensityEstimator(PluginNeuralConditionalDensityEstimator):

    def __init__(self, args: DictConfig = None, **kwargs):
        super(DRNeuralConditionalDensityEstimator, self).__init__(args)

        # Plug-in = nuisance model
        self.target_model = instantiate(args.model.target, args, 'target', _recursive_=False)
        self.pot_out_model = self.target_model

    def _mlflow_log_target(self, result, prefix, step):
        for k in result.keys():
            self.mlflow_logger.log_metrics({k: result[k].item()}, step=step)

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        cov_f, treat_f, out_f_scaled = super(DRNeuralConditionalDensityEstimator, self).prepare_train_data(train_data_dict)
        self.target_model.num_train_iter = int(self.hparams.dataset.n_samples_train / self.target_model.batch_size * self.target_model.num_epochs)
        return cov_f, treat_f, out_f_scaled


    def fit(self, train_data_dict: dict, log: bool):
        """
        Fitting the two-stage estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        super(DRNeuralConditionalDensityEstimator, self).fit(train_data_dict, log)

        # Preparing data
        cov_f, treat_f, out_f_scaled = self.prepare_train_data(train_data_dict)
        prop_pred = self.nuisance_model.get_propensity(cov_f)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f_scaled, self.target_model.batch_size, prop_pred)

        # Preparing optimizers
        target_optimizers, self.ema_optimizer = self.target_model.get_optimizer()
        target_optimizers = [(target_optimizers, 'main')] if not isinstance(target_optimizers, list) else target_optimizers
        self.to(self.device)


        # Conditional generative model fitting
        logger.info('Fitting target models')
        for step in tqdm(range(self.target_model.num_train_iter)) if log else range(self.target_model.num_train_iter):
            cov_f, treat_f, out_f_scaled, _, prop_pred1 = next(iter(train_dataloader))

            for target_optimizer, optimizer_name in target_optimizers:

                target_optimizer.zero_grad()
                result_f = self.target_model.training_step(cov_f, treat_f, out_f_scaled,
                                                           optimizer=(target_optimizer, optimizer_name))
                ipwt = get_iptw(treat_f, prop_pred1, self.clip_prop, normalize=False).detach()

                loss, result = 0.0, {}
                for treat in self.treat_options:
                    treat_pot = torch.ones_like(treat_f) * treat
                    out_pot_sample_scaled = self.nuisance_model.cond_sample(treat_pot, cov_f, n_sample=(self.target_model.num_mc, cov_f.shape[0])).detach()
                    result_pot = self.target_model.training_step(cov_f.unsqueeze(0), treat_pot.unsqueeze(0), out_pot_sample_scaled,
                                                                 optimizer=(target_optimizer, optimizer_name))

                    loss_treat = ((ipwt * (treat == treat_f) * result_f['train_cond_dist_loss'] +
                                  (1 - ipwt * (treat == treat_f)) * result_pot['train_cond_dist_loss'].mean(0))).mean()

                    assert not math.isnan(loss_treat)

                    result[f'train_loss_target_{treat}'] = loss_treat
                    loss += loss_treat

                loss.backward()

                target_optimizer.step()
                self.target_model._post_nuisance_optimizer_step()
                self.ema_optimizer.update()

                if step % 100 == 0 and log:
                    self._mlflow_log_target(result, 'train', step)

    def evaluate_cond_pot_out_dist(self, data_dict: dict, dataset, log: bool, prefix: str, kind: str):
        with self.ema_optimizer.average_parameters():
            return super(DRNeuralConditionalDensityEstimator, self).evaluate_cond_pot_out_dist(data_dict, dataset, log, prefix, kind)