import torch
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
import logging
from ray import tune
import ray
from copy import deepcopy
from typing import Tuple, Union


from src.models.utils import fit_eval_kfold


logger = logging.getLogger(__name__)


class PODistributionEstimator(torch.nn.Module):
    """
    Abstract class for a PO distribution estimator
    """

    val_metric = None

    def __init__(self, args: DictConfig = None, **kwargs):
        super(PODistributionEstimator, self).__init__()

        # Dataset params
        self.dim_out, self.dim_cov, self.dim_treat = args.model.dim_out, args.model.dim_cov, 1
        assert self.dim_treat == 1
        self.treat_options = [0.0, 1.0]

        # Model hyparams
        self.hparams = args

        # MlFlow Logger
        if args.exp.logging:
            experiment_name = f'{args.model.name}/{args.model.nuisance.name}'
            if 'target' in args.model and args.model.target is not None:
                experiment_name += f'/{args.model.target.name}'
            experiment_name += f'/{args.dataset.name}'
            if args.exp.eval_num_mc != 200:
                experiment_name += '_new'
            self.mlflow_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)

    def prepare_train_data(self, train_data_dict: dict):
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        raise NotImplementedError()

    def prepare_eval_data(self, data_dict: dict):
        """
        Data pre-processing
        :param data_dict: Dictionary with the evaluation data
        """
        raise NotImplementedError()

    def prepare_tensors(self, cov=None, treat=None, out=None, kind='torch') -> Tuple[dict]:
        """
        Conversion of tensors
        @param cov: Tensor with covariates
        @param treat: Tensor with treatments
        @param out: Tensor with outcomes
        @param kind: torch / numpy
        @return: cov, treat, out
        """
        if kind == 'torch':
            cov = torch.tensor(cov).reshape(-1, self.dim_cov).float() if cov is not None else None
            treat = torch.tensor(treat).reshape(-1, self.dim_treat).float() if treat is not None else None
            out = torch.tensor(out).reshape(-1, self.dim_out).float() if out is not None else None
        elif kind == 'numpy':
            cov = cov.reshape(-1, self.dim_cov) if cov is not None else None
            treat = treat.reshape(-1).astype(float) if treat is not None else None
            out = out.reshape(-1, self.dim_out) if out is not None else None
        else:
            raise NotImplementedError()
        return cov, treat, out

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        raise NotImplementedError()

    def evaluate(self, data_dict: dict, log: bool, prefix: str) -> dict:
        raise NotImplementedError()

    def evaluate_cond_pot_out_dist(self, data_dict: dict, dataset, log: bool, prefix: str, kind: str) -> dict:
        raise NotImplementedError()

    def save_train_data_to_buffer(self, cov_f, treat_f, out_f_scaled) -> None:
        """
        Save train data for non-parametric inference of two-stage training
        @param cov_f: Tensor with factual covariates
        @param treat_f: Tensor with factual treatments
        @param out_f: Tensor with factual outcomes
        """
        self.cov_f = cov_f
        self.treat_f = treat_f
        self.out_f_scaled = out_f_scaled

    @staticmethod
    def set_nuisances_hparams(model_args: DictConfig, new_model_args: dict):
        for k in new_model_args.keys():
            assert k in model_args.keys()
            model_args[k] = new_model_args[k]

    def finetune_nuisances(self, train_data_dict: dict, resources_per_trial: dict, val_data_dict: dict = None):
        """
        Hyperparameter tuning with ray[tune]
        @param train_data_dict: Training data dictionary
        @param resources_per_trial: CPU / GPU resources dictionary
        @return: self
        """

        logger.info(f"Running hyperparameters selection with {self.hparams.model.nuisance['tune_range']} trials")
        logger.info(f'Using {self.val_metric} for hyperparameters selection')
        ray.init(num_gpus=1, num_cpus=5)

        hparams_grid = {k: getattr(tune, self.hparams.model.nuisance['tune_type'])(list(v))
                        for k, v in self.hparams.model.nuisance['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(fit_eval_kfold,
                                                 model_cls=self.__class__,
                                                 train_data_dict=deepcopy(train_data_dict),
                                                 val_data_dict=deepcopy(val_data_dict),
                                                 orig_hparams=self.hparams),
                            resources_per_trial=resources_per_trial,
                            raise_on_failed_trial=False,
                            metric="val_metric",
                            mode="min",
                            config=hparams_grid,
                            num_samples=self.hparams.model.nuisance['tune_range'],
                            name=f"{self.__class__.__name__}",
                            max_failures=1,
                            )
        ray.shutdown()

        logger.info(f"Best hyperparameters found: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        self.set_nuisances_hparams(self.hparams.model.nuisance, analysis.best_config)

        self.__init__(self.hparams)
        return self


