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

        self.normalize_cov = args.dataset.normalize_cov
        self.cov_scaler = StandardScaler() if self.normalize_cov else None
        self.device = args.exp.device

        # Plug-in = nuisance model
        self.nuisance_model = instantiate(args.model.nuisance, args, 'nuisance', _recursive_=False)
        self.pot_out_model = self.nuisance_model

    def get_train_dataloader(self, cov_f, treat_f, out_f, batch_size, prop_pred=None, out_pot_pred=None) -> DataLoader:
        data = (cov_f, treat_f, out_f)

        if prop_pred is not None:
            data += (prop_pred,)

        if out_pot_pred is not None:
            data += out_pot_pred

        training_data = TensorDataset(*data)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=self.device))
        return train_dataloader

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        # Scaling train data
        if self.normalize_cov:
            cov_f = self.cov_scaler.fit_transform(train_data_dict['cov_f'].reshape(-1, self.dim_cov))
        else:
            cov_f = train_data_dict['cov_f'].reshape(-1, self.dim_cov)

        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, train_data_dict['treat_f'], train_data_dict['out_f_scaled'], kind='torch')
        self.hparams.dataset.n_samples_train = cov_f.shape[0]
        self.nuisance_model.num_train_iter = int(self.hparams.dataset.n_samples_train / self.nuisance_model.batch_size * self.nuisance_model.num_epochs)
        if self.has_prop_score:
            self.nuisance_model.num_burn_in_train_iter = int(self.hparams.dataset.n_samples_train / self.nuisance_model.batch_size * self.nuisance_model.num_burn_in_epochs)

        logger.info(f'Effective number of training iterations: {self.nuisance_model.num_train_iter}.')
        return cov_f, treat_f, out_f_scaled

    def prepare_eval_data(self, data_dict: dict) -> Tuple[torch.Tensor]:
        # Scaling eval data
        if self.normalize_cov:
            cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        else:
            cov_f = data_dict['cov_f'].reshape(-1, self.dim_cov)
        cov_f, treat_f, out_f_scaled = self.prepare_tensors(cov_f, data_dict['treat_f'], data_dict['out_f_scaled'], kind='torch')
        return cov_f, treat_f, out_f_scaled

    def prepare_pot_out_data(self, data_dict: dict) -> Tuple[torch.Tensor]:
        # Scaling eval data
        if self.normalize_cov:
            cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        else:
            cov_f = data_dict['cov_f'].reshape(-1, self.dim_cov)

        cov_f, _, _ = self.prepare_tensors(cov_f, None, None, kind='torch')
        out_pot_scaled_list = []
        for treat in self.treat_options:
            _, _, out_pot_scaled = self.prepare_tensors(cov_f, None, data_dict[f'out_pot{treat}_scaled'], kind='torch')
            out_pot_scaled_list.append(out_pot_scaled)
        return cov_f, out_pot_scaled_list

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
        train_dataloader_iter = iter(train_dataloader)
        for step in tqdm(range(self.nuisance_model.num_train_iter)) if log else range(self.nuisance_model.num_train_iter):
            try:
                cov_f, treat_f, out_f_scaled = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                cov_f, treat_f, out_f_scaled = next(train_dataloader_iter)

            for nuisance_optimizer, optimizer_name in nuisance_optimizers:

                nuisance_optimizer.zero_grad()
                result = self.nuisance_model.training_step(cov_f, treat_f, out_f_scaled,
                                                           optimizer=(nuisance_optimizer, optimizer_name))

                if not self.has_prop_score:
                    loss = result['train_cond_dist_loss'].mean()

                else:
                    if self.dim_treat == 1:
                        bce_loss = torch.binary_cross_entropy_with_logits(result['train_prop_logits'], treat_f)
                    else:
                        bce_loss = torch.nn.functional.cross_entropy(result['train_prop_logits'], treat_f)
                    if step > self.nuisance_model.num_burn_in_train_iter:
                        if self.dim_treat == 1:
                            prop = torch.sigmoid(result['train_prop_logits'].detach())
                        else:
                            prop = torch.softmax(result['train_prop_logits'], dim=-1)
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

                if step % 500 == 0 and log and self.hparams.exp.mode == 'img':
                    self.plot_img(digit=0, name='first_' + str(step), pot_out_model=self.nuisance_model)

                    # train_dataloader = iter(train_dataloader)


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

        logger.info('Evaluating generative potential outcome generative models')
        self.eval()
        with torch.no_grad():
            cov_f, out_pot_scaled_list = self.prepare_pot_out_data(data_dict)

            if kind == 'log_prob':
                for i, (treat, out_pot) in enumerate(zip(self.treat_options, out_pot_scaled_list)):
                    logger.info(f'Evaluating {kind} for treatment {treat}')
                    treat_pot = self._get_treat_pot(cov_f[:, :1], cov_f, treat)
                    result = self.pot_out_model.eval_step(cov_f, treat_pot, out_pot, prefix)
                    avg_result[f'{prefix}_{kind}_{i}'] = - result[f'{prefix}_cond_dist_loss'].mean().item()

            elif kind == 'wass':
                out_pot_scaled_dist_list = dataset.get_pot_cond_dist(data_dict)
                for i, (treat, out_pot_dist) in enumerate(zip(self.treat_options, out_pot_scaled_dist_list)):
                    logger.info(f'Evaluating {kind} for treatment {treat}')
                    gt_sample = out_pot_dist.sample((self.hparams.exp.eval_num_mc, ))

                    if self.hparams.exp.mode == 'img' and self.hparams.exp.plot_img:  # Plotting ground-truth
                        self.plot_img(digit=-1, sample=gt_sample, name=f'gt_sample{i}')

                    treat_pot = self._get_treat_pot(cov_f[:, :1], cov_f, treat)
                    est_sample = self.pot_out_model.cond_sample(treat_pot, cov_f, n_sample=(self.hparams.exp.eval_num_mc, cov_f.shape[0])).squeeze()
                    avg_result[f'{prefix}_{kind}_{i}'] = wass_dist(gt_sample, est_sample).mean().item()

            else:
                raise NotImplementedError()

        if log:
            self.mlflow_logger.log_metrics(avg_result, step=self.nuisance_model.num_train_iter)

        return avg_result

    def _sample_nuisance_model(self, treat_f, cov_f):
        out_pot_sample_scaled_list = []
        with torch.no_grad():
            for treat in tqdm(self.treat_options):
                treat_pot = self._get_treat_pot(treat_f, cov_f, treat)

                out_pot_sample_scaled = self.nuisance_model.cond_sample(treat_pot, cov_f, n_sample=(self.target_model.num_mc * (self.target_model.num_epochs + 1), cov_f.shape[0])).detach()
                out_pot_sample_scaled = out_pot_sample_scaled.swapaxes(0, 1)
                out_pot_sample_scaled_list.append(out_pot_sample_scaled.reshape(cov_f.shape[0], self.target_model.num_epochs + 1, self.target_model.num_mc, -1))
        return out_pot_sample_scaled_list

    def _get_treat_pot(self, treat_f, cov_f, treat):
        if self.dim_treat == 1:  # Binary treatment
            treat_pot = torch.ones_like(treat_f) * treat
        else:  # One-hot encoding
            treat_pot = torch.zeros((cov_f.shape[0], self.dim_treat)).float()
            treat_pot[0, treat] = 1.0
        return treat_pot