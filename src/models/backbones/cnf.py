import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
from omegaconf import DictConfig
from pyro.nn import DenseNN, ConditionalAutoRegressiveNN, ConditionalDenseNN
from typing import Tuple, List
from torch_ema import ExponentialMovingAverage


from src.models.backbones.neural_cond_estimator import NeuralConditionalDistEstimator


class LinerazableDenseNN(DenseNN):
    # Over-writing the default method
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            param_dims: List[int] = [1, 1],
            nonlinearity: torch.nn.Module = torch.nn.ReLU(),
    ) -> None:
        torch.nn.Module.__init__(self)

        self.input_dim = input_dim
        self.context_dim = 0
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Create masked layers
        if len(hidden_dims) > 0:
            layers = [torch.nn.Linear(input_dim, hidden_dims[0])]
            for i in range(1, len(hidden_dims)):
                layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(torch.nn.Linear(hidden_dims[-1], self.output_multiplier))
        else:
            layers = [torch.nn.Linear(input_dim, self.output_multiplier)]
        self.layers = torch.nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity


class CNFs(NeuralConditionalDistEstimator):
    """
    Conditional normalizing flow
    """

    def __init__(self, args: DictConfig = None, kind: str = None, **kwargs):
        super(CNFs, self).__init__(args, kind)

        # Model hyparams & Train params
        self.count_bins = args.model[self.kind].count_bins
        self.noise_std_X, self.noise_std_Y = args.model[self.kind].noise_std_X, args.model[self.kind].noise_std_Y
        self.scaled_out_f_bound = args.model.scaled_out_f_bound

        # Model init = Conditional NFs
        self.cond_loc = torch.nn.Parameter(torch.zeros((self.dim_out, )).float())
        self.cond_scale = torch.nn.Parameter(torch.ones((self.dim_out, )).float())
        self.cond_affine_transform = T.AffineTransform(self.cond_loc, self.cond_scale)

        if self.dim_out == 1:

            self.cond_base_dist = dist.Normal(torch.zeros(self.dim_out).float(), torch.ones(self.dim_out).float())

            if self.dim_hid > 0:
                self.cond_dist_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                            param_dims=[self.count_bins, self.count_bins, (self.count_bins - 1)],
                                            nonlinearity=torch.nn.ELU()).float()
            else:  # linear hypernetwork setup
                self.cond_dist_nn = LinerazableDenseNN(self.dim_cov + self.dim_treat, [],
                                            param_dims=[self.count_bins, self.count_bins, (self.count_bins - 1)],
                                            nonlinearity=torch.nn.ELU()).float()
            self.cond_spline_transform = T.ConditionalSpline(self.cond_dist_nn, self.dim_out,
                                                             order='quadratic',
                                                             count_bins=self.count_bins,
                                                             bound=self.scaled_out_f_bound).to(self.device)
        else:
            self.cond_base_dist = dist.MultivariateNormal(torch.zeros(self.dim_out).float(),
                                                          torch.diag(torch.ones(self.dim_out)).float())

            self.cond_dist_nn = ConditionalAutoRegressiveNN(self.dim_out, self.dim_hid + self.dim_treat,
                                                            [self.dim_hid],
                                                            param_dims=[self.count_bins, self.count_bins, (self.count_bins - 1)]).float()
            self.cond_spline_transform = T.ConditionalSplineAutoregressive(self.dim_out,
                                                                           self.cond_dist_nn,
                                                                           order='quadratic',
                                                                           count_bins=self.count_bins,
                                                                           bound=self.scaled_out_f_bound).to(self.device)

        self.cond_flow_dist = dist.ConditionalTransformedDistribution(self.cond_base_dist,
                                                                      [self.cond_affine_transform, self.cond_spline_transform])

        self.ema_optimizer = None

    def get_optimizer(self) -> torch.optim.Optimizer | List[torch.optim.Optimizer]:
        """
        Init optimizer for the nuisance flow
        """
        if self.kind == 'nuisance':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_dist_nn])
            return torch.optim.SGD(list(modules.parameters()) + [self.cond_loc, self.cond_scale], lr=self.lr, momentum=0.9)
        elif self.kind == 'target':
            modules = torch.nn.ModuleList([self.repr_nn, self.cond_dist_nn])
            parameters = list(modules.parameters()) + [self.cond_loc, self.cond_scale]
            optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=0.9)
            ema_target = ExponentialMovingAverage(parameters, decay=self.gamma)
            return optimizer, ema_target
        else:
            raise NotImplementedError()

    def _post_nuisance_optimizer_step(self):
        self.cond_flow_dist.clear_cache()

    def _cond_log_prob(self, context, out) -> torch.Tensor:
        return self.cond_flow_dist.condition(context).log_prob(out)

    def _cond_dist(self, context) -> torch.distributions.Distribution:
        return self.cond_flow_dist.condition(context)

    def _cond_sample(self, context, n_sample) -> torch.Tensor:
        return self._cond_dist(context).sample(n_sample)

    def _cond_training_step(self, repr_f, treat_f, out_f_scaled, optimizer=None):
        # Representation -> Adding noise + concat of factual treatment -> Conditional distribution
        noised_out_f_scaled = out_f_scaled + self.noise_std_Y * torch.randn_like(out_f_scaled)
        noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
        context = torch.cat([noised_repr_f, treat_f], dim=-1)
        log_prob = self._cond_log_prob(context, noised_out_f_scaled)
        return - log_prob

    def _cond_eval_step(self, repr_f, treat_f, out_f_scaled):
        context = torch.cat([repr_f, treat_f], dim=-1)
        log_prob = self._cond_log_prob(context, out_f_scaled)
        return - log_prob

