import torch
import math
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
import matplotlib.pyplot as plt
from hydra.utils import instantiate

logger = logging.getLogger(__name__)

from src.models.utils import get_iptw, wass_dist
from src.models.po_dist_estimator import PODistributionEstimator


# Fixing the issue with not implemented sign property
@property
def sign(self):
    return torch.ones(1).float()


ConditionedSpline.sign = sign


class PluginNeuralConditionalDensityEstimator(PODistributionEstimator):
    """
    Class for neural plugin methods / neural IPTW methods
    """

    val_metric = 'val_loss'

    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginNeuralConditionalDensityEstimator, self).__init__(args)

        self.has_prop_score = args.model.nuisance.has_prop_score
        if not self.has_prop_score:
            self.prop_alpha, self.clip_prop = None, None
        else:
            self.prop_alpha = args.model.nuisance.prop_alpha
            self.clip_prop = args.model.clip_prop

        self.cov_scaler = StandardScaler()
        self.device = args.exp.device

        # Plug-in = nuisance model
        self.nuisance_model = instantiate(args.model.nuisance, args, 'nuisance', _recursive_=False)
        self.pot_out_model = self.nuisance_model

    def get_train_dataloader(self, cov_f, treat_f, out_f, batch_size, prop_pred=None) -> DataLoader:
        if prop_pred is None:
            training_data = TensorDataset(cov_f, treat_f, out_f)
        else:
            training_data = TensorDataset(cov_f, treat_f, out_f, *prop_pred)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(train_data_dict['cov_f'].reshape(-1, self.dim_cov))

        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, train_data_dict['treat_f'], train_data_dict['out_f_scaled'], kind='torch')
        self.hparams.dataset.n_samples_train = cov_f.shape[0]
        self.nuisance_model.num_train_iter = int(self.hparams.dataset.n_samples_train / self.nuisance_model.batch_size * self.nuisance_model.num_epochs)
        if self.has_prop_score:
            self.nuisance_model.num_burn_in_train_iter = int(self.hparams.dataset.n_samples_train / self.nuisance_model.batch_size * self.nuisance_model.num_burn_in_epochs)

        logger.info(f'Effective number of training iterations: {self.nuisance_model.num_train_iter}.')
        return cov_f, treat_f, out_f_scaled

    def prepare_eval_data(self, data_dict: dict) -> Tuple[torch.Tensor]:
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, data_dict['treat_f'], data_dict['out_f_scaled'], kind='torch')
        return cov_f, treat_f, out_f_scaled

    def prepare_pot_out_data(self, data_dict: dict) -> Tuple[torch.Tensor]:
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        cov_f, _, out_pot0_scaled = self.prepare_tensors(cov_f, None, data_dict['out_pot0_scaled'], kind='torch')
        _, _, out_pot1_scaled = self.prepare_tensors(cov_f, None, data_dict['out_pot1_scaled'], kind='torch')
        return cov_f, out_pot0_scaled, out_pot1_scaled

    def _mlflow_log_nuisance(self, result, prefix, step):
        self.mlflow_logger.log_metrics({f'{prefix}_cond_dist_loss': result[f'{prefix}_cond_dist_loss'].mean().item()}, step=step)
        self.mlflow_logger.log_metrics({f'{prefix}_loss': result[f'{prefix}_loss'].item()}, step=step)
        if self.has_prop_score:
            self.mlflow_logger.log_metrics({f'{prefix}_bce_loss': result[f'{prefix}_bce_loss'].mean().item()}, step=step)

    def fit(self, train_data_dict: dict, log: bool):
        """
        Fitting the plug-in = nuisance estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f, treat_f, out_f_scaled = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f_scaled, batch_size=self.nuisance_model.batch_size)

        # Preparing optimizers
        nuisance_optimizers = self.nuisance_model.get_optimizer()
        nuisance_optimizers = [(nuisance_optimizers, 'main')] if not isinstance(nuisance_optimizers, list) else nuisance_optimizers

        self.to(self.device)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # # Saving train train_data_dict
        # self.save_train_data_to_buffer(cov_f, treat_f, out_f_scaled)

        # Conditional generative model fitting
        logger.info('Fitting plug-in / nuisance models')
        for step in tqdm(range(self.nuisance_model.num_train_iter)) if log else range(self.nuisance_model.num_train_iter):
            cov_f, treat_f, out_f_scaled = next(iter(train_dataloader))

            for nuisance_optimizer, optimizer_name in nuisance_optimizers:

                nuisance_optimizer.zero_grad()
                result = self.nuisance_model.training_step(cov_f, treat_f, out_f_scaled,
                                                           optimizer=(nuisance_optimizer, optimizer_name))

                if not self.has_prop_score:
                    loss = result['train_cond_dist_loss'].mean()

                else:
                    bce_loss = torch.binary_cross_entropy_with_logits(result['train_prop_logits'], treat_f)
                    if step > self.nuisance_model.num_burn_in_train_iter:
                        prop = torch.sigmoid(result['train_prop_logits'].detach())
                        ipwt_normalized = get_iptw(treat_f, prop, self.clip_prop, normalize=True)
                        loss = (ipwt_normalized * result['train_cond_dist_loss']).mean() + self.prop_alpha * bce_loss.mean()
                    else:
                        loss = result['train_cond_dist_loss'].mean() + self.prop_alpha * bce_loss.mean()

                    result['train_bce_loss'] = bce_loss.mean()

                assert not math.isnan(loss)

                result['train_loss'] = loss
                loss.backward()

                nuisance_optimizer.step()
                self.nuisance_model._post_nuisance_optimizer_step()

                if step % 100 == 0 and log:
                    self._mlflow_log_nuisance(result, 'train', step)


    def evaluate_nuisance(self, data_dict: dict, log: bool, prefix: str) -> dict:
        cov_f, treat_f, out_f_scaled = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            result = self.nuisance_model.eval_step(cov_f, treat_f, out_f_scaled, prefix)
            result[f'{prefix}_loss'] = result[f'{prefix}_cond_dist_loss'].mean()
            if self.has_prop_score:
                bce_loss = torch.binary_cross_entropy_with_logits(result[f'{prefix}_prop_logits'], treat_f)
                result[f'{prefix}_loss'] = result[f'{prefix}_cond_dist_loss'].mean() + self.prop_alpha * bce_loss.mean()
                result[f'{prefix}_bce_loss'] = bce_loss.mean()

        if log:
            self._mlflow_log_nuisance(result, prefix, self.nuisance_model.num_train_iter)

        # Averaging & Filtering
        avg_result = {}
        for k in result.keys():
            if k in [f'{prefix}_bce_loss', f'{prefix}_loss', f'{prefix}_cond_dist_loss']:
                avg_result[k] = result[k].mean().item()

        return avg_result

    def evaluate_cond_pot_out_dist(self, data_dict: dict, dataset, log: bool, prefix: str, kind: str):
        avg_result = {}

        self.eval()
        with torch.no_grad():
            cov_f, out_pot0_scaled, out_pot1_scaled = self.prepare_pot_out_data(data_dict)
            if kind == 'log_prob':
                for i, (treat, out_pot) in enumerate(zip(self.treat_options, [out_pot0_scaled, out_pot1_scaled])):
                    result = self.pot_out_model.eval_step(cov_f, torch.ones_like(cov_f[:, :1]) * treat, out_pot, prefix)
                    avg_result[f'{prefix}_{kind}_{i}'] = - result[f'{prefix}_cond_dist_loss'].mean().item()

            elif kind == 'wass':
                out_pot0_sacled_dist, out_pot1_scaled_dist = dataset.get_pot_cond_dist(data_dict)
                for i, (treat, out_pot_dist) in enumerate(zip(self.treat_options, [out_pot0_sacled_dist, out_pot1_scaled_dist])):
                    gt_sample = out_pot_dist.sample((self.hparams.exp.eval_num_mc, ))
                    est_sample = self.pot_out_model.cond_sample(torch.ones_like(cov_f[:, :1]) * treat, cov_f,
                                                                n_sample=(self.hparams.exp.eval_num_mc, cov_f.shape[0])).squeeze()
                    avg_result[f'{prefix}_{kind}_{i}'] = wass_dist(gt_sample, est_sample).mean().item()

            else:
                raise NotImplementedError()

        if log:
            self.mlflow_logger.log_metrics(avg_result, step=self.nuisance_model.num_train_iter)

        return avg_result






# class PluginDKME(BoundsEstimator):
#     """
#     Distributional kernel mean embeddings
#     """
#
#     val_metric = 'val_kernel_ridge_reg_neg_mse'
#
#     def __init__(self, args: DictConfig = None, **kwargs):
#         super(PluginDKME, self).__init__(args)
#
#         self.cov_scaler = StandardScaler()
#         self.scaled_out_f_bound = args.model.scaled_out_f_bound  # Support bounds for the scaled outcome
#
#         # Model hyparams
#         self.sd_x = args.model.sd_x
#         self.eps = args.model.eps
#         self.num_train_iter = 0
#
#         # Model parameters
#         self.normalized_rbf_y = []  # Will be initialized during the fit
#         self.rbf_x = RBF(np.sqrt(self.sd_x / 2))
#         self.K_inv = []
#
#     def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
#         """
#         Data pre-processing
#         :param train_data_dict: Dictionary with the training data
#         """
#         # Scaling train data
#         cov_f = self.cov_scaler.fit_transform(train_data_dict['cov_f'].reshape(-1, self.dim_cov))
#
#         cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, train_data_dict['treat_f'], train_data_dict['out_f_scaled'], kind='numpy')
#         self.hparams.dataset.n_samples_train = cov_f.shape[0]
#         return cov_f, treat_f, out_f_scaled
#
#     def prepare_eval_data(self, data_dict: dict) -> Tuple[torch.Tensor]:
#         # Scaling eval data
#         cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
#         cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, data_dict['treat_f'], data_dict['out_f_scaled'], kind='numpy')
#         return cov_f, treat_f, out_f_scaled
#
#     def set_sd_y_median_heuristic(self) -> None:
#         """
#         Calculate median heuristics (for DKME)
#         """
#         for treat_option in [0.0, 1.0]:
#             distances = np.tril(squareform(pdist(self.out_f_scaled[self.treat_f.reshape(-1) == treat_option].reshape(-1, self.dim_out), 'sqeuclidean')), -1)
#             sd_y = np.median(distances[distances > 0.0])
#             self.normalized_rbf_y.append(NormalizedRBF(sd_y / 3))
#             logger.info(f'New sd_y[{treat_option}]: {sd_y / 3}')
#
#     def fit(self, train_data_dict: dict, log: bool):
#         """
#         Fitting the estimator
#         @param train_data_dict: Training data dictionary
#         @param log: Logging to the MlFlow
#         """
#         # Preparing data
#         cov_f, treat_f, out_f_scaled = self.prepare_train_data(train_data_dict)
#
#         # Logging
#         self.mlflow_logger.log_hyperparams(self.hparams) if log else None
#
#         # Saving train train_data_dict
#         self.save_train_data_to_buffer(cov_f, treat_f, out_f_scaled)
#
#         # Conditional NFs fitting
#         logger.info('Fitting nuisance models')
#
#         # Median heuristic for sd_y
#         self.set_sd_y_median_heuristic()
#
#         # Skipping fitting while hparam tuning
#         if not log:
#             return
#
#         for treat_option in [0.0, 1.0]:
#             K = self.rbf_x(cov_f[treat_f == treat_option], cov_f[treat_f == treat_option])
#             n_cond = cov_f[treat_f == treat_option].shape[0]
#             K_inv = np.linalg.inv(K + n_cond * self.eps * np.eye(n_cond))
#             self.K_inv.append(K_inv)
#
#     def kernel_ridge_reg_neg_mse(self, treat_f, out_f_scaled, cov_f) -> np.array:
#         """
#         Negative MSE of the kernel ridge regression with the same hyperparameters as DKME
#         @param treat_f: Tensor with factual treatments
#         @param out_f_scaled: Tensor with factual outcomes
#         @param cov_f: Tensor with factual covariates
#         @return: Negative MSE
#         """
#         mses = np.zeros_like(out_f_scaled)
#         for treat_option in [0.0, 1.0]:
#             ker_ridge_reg = GaussianProcessRegressor(alpha=self.eps, kernel=self.rbf_x, optimizer=None)
#             ker_ridge_reg.fit(self.cov_f[self.treat_f == treat_option], self.out_f_scaled[self.treat_f == treat_option])
#             out_pred = ker_ridge_reg.predict(cov_f[treat_f == treat_option]).reshape(-1, self.dim_out)
#             mses[treat_f == treat_option] = ((out_pred - out_f_scaled[treat_f == treat_option]) ** 2)
#         return - mses
#
#     def evaluate(self, data_dict: dict, log: bool, prefix: str) -> dict:
#         cov_f, treat_f, out_f_scaled = self.prepare_eval_data(data_dict)
#
#         results = {
#             f'{prefix}_log_prob_f': self.cond_log_prob(treat_f, out_f_scaled, cov_f).mean(),
#             f'{prefix}_kernel_ridge_reg_neg_mse': self.kernel_ridge_reg_neg_mse(treat_f, out_f_scaled, cov_f).mean()
#         }
#
#         if log:
#             self.mlflow_logger.log_metrics(results, step=self.num_train_iter)
#         return results
#
#     def cond_log_prob(self, treat_f, out_f_scaled, cov_f) -> np.array:
#         """
#         Conditional log-probability
#         @param treat_f: Tensor with factual treatments
#         @param out_f_scaled: Tensor with factual outcomes
#         @param cov_f: Tensor with factual covariates
#         @return: Tensor with log-probabilities
#         """
#         log_prob = np.zeros_like(out_f_scaled)
#         for treat_option, normalized_rbf_y, K_inv in zip([0.0, 1.0], self.normalized_rbf_y, self.K_inv):
#             if (treat_f == treat_option).sum() > 0:
#                 L = normalized_rbf_y(out_f_scaled[treat_f == treat_option], self.out_f_scaled[self.treat_f == treat_option])
#                 K_x = self.rbf_x(self.cov_f[self.treat_f == treat_option], cov_f[treat_f == treat_option])
#                 w = np.dot(K_inv, K_x)
#                 w = w / w.sum(0, keepdims=True)
#                 log_prob[treat_f == treat_option, :] = np.log((L * w.T).sum(1, keepdims=True))
#
#         log_prob[np.isnan(log_prob)] = -1e10
#         return log_prob
#
#     def cond_cdf(self, treat_f, out_f_scaled, cov_f):
#         cdfs = np.zeros_like(out_f_scaled)
#         for treat_option, normalized_rbf_y, K_inv in zip([0.0, 1.0], self.normalized_rbf_y, self.K_inv):
#             if (treat_f == treat_option).sum() > 0:
#                 diff_mat = np.subtract.outer(out_f_scaled[treat_f == treat_option], self.out_f_scaled[self.treat_f == treat_option])
#                 K_x = self.rbf_x(self.cov_f[self.treat_f == treat_option], cov_f[treat_f == treat_option])
#                 w = np.dot(K_inv, K_x)
#                 w = w / w.sum(0, keepdims=True)
#                 cdfs[treat_f == treat_option, :] = (ndtr(np.squeeze(diff_mat) / normalized_rbf_y.sd) * w.T).sum(1, keepdims=True)
#         cdfs = np.clip(cdfs, 0.0, 1.0)
#         return cdfs
#
#     def get_bounds(self, cov_f, delta_scaled_or_alpha, mode='cdf', n_grid=200) -> Tuple[torch.Tensor]:
#         # This ideally should be vectorized in the future
#         if mode == 'cdf':
#             out_grid = np.linspace(-self.scaled_out_f_bound / 2, self.scaled_out_f_bound / 2, n_grid)
#         elif mode == 'icdf':
#             logger.warning('ICDF bounds are not implemented for DKME plugin')
#             return np.array(np.nan), np.array(np.nan)
#         else:
#             raise NotImplementedError()
#
#         cond_cdfs1_out = np.zeros((delta_scaled_or_alpha.shape[0], n_grid, cov_f.shape[0], 1))
#         cond_cdfs0_out_min_delta = np.zeros((delta_scaled_or_alpha.shape[0], n_grid, cov_f.shape[0], 1))
#
#         for g in tqdm(range(n_grid)):
#             cond_cdfs1_out_temp = self.cond_cdf(np.ones((cov_f.shape[0],)).astype(float), out_grid[g].repeat(cov_f.shape[0]).reshape(-1, 1), cov_f)
#             for d in range(delta_scaled_or_alpha.shape[0]):
#                 cond_cdfs1_out[d, g] = cond_cdfs1_out_temp
#                 cond_cdfs0_out_min_delta[d, g] = self.cond_cdf(np.zeros((cov_f.shape[0],)).astype(float), (out_grid[g] - delta_scaled_or_alpha[d]).repeat(cov_f.shape[0]).reshape(-1, 1), cov_f)
#
#         lb = np.squeeze(np.maximum((cond_cdfs1_out - cond_cdfs0_out_min_delta).max(1), np.zeros(1)))
#         ub = np.squeeze(1.0 + np.minimum((cond_cdfs1_out - cond_cdfs0_out_min_delta).min(1), np.zeros(1)))
#         return torch.tensor(lb), torch.tensor(ub)
#