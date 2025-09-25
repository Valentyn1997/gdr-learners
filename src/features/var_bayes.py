import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
# from pyro import constraints
from pyro.nn import pyro_method
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models import GPRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from tqdm import tqdm

import pyro
import torch
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models import GPRegression
from pyro.infer import Predictive
from src.data import Synthetic



kernel0 = RBF(input_dim=1, lengthscale=torch.tensor(1.0), variance=torch.tensor(0.1))
kernel1 = RBF(input_dim=1, lengthscale=torch.tensor(1.0), variance=torch.tensor(0.1))


def _sample_gp_path(name, X, kernel):
    K = kernel(X) + 1e-3 * torch.eye(X.size(0))
    return pyro.sample(f"{name}", dist.MultivariateNormal(torch.zeros(X.size(0)), covariance_matrix=K))


def model(x, t, y):
    # Draw covariate weights once per ELBO sample
    w = pyro.sample("w", dist.Dirichlet(torch.ones(x.shape[0])))

    # GP priors over latent surfaces
    f0 = _sample_gp_path("f0", x, kernel0)
    f1 = _sample_gp_path("f1", x, kernel1)
    # Deterministic ATE
    # tau  = pyro.deterministic("tau", (w @ (f1 - f0)).sum())
    tau = pyro.sample("tau", dist.Normal((w * (f1 - f0).unsqueeze(0)).sum(-1), 1.0))

    # Choose the right surface for each unit
    f = torch.where(t.bool(), f1, f0)
    # assert f1.shape == (x.size(0),)
    with pyro.plate("data", x.size(0)):
        pyro.sample("obs", dist.Normal(f, 1.0), obs=y)

    return tau


def guide(x, t, y):
    loc = pyro.param("theta_loc", torch.tensor(0.35))
    scale = pyro.param("theta_scale", torch.tensor(0.5), constraint=torch.distributions.constraints.positive)
    pyro.sample("tau", dist.Normal(loc, scale))


def optim_config(name, param):
    # any match rule you like
    if "lengthscale" in name or 'variance' in name:
        return {"lr": 0.0}        # freeze
    return {"lr": 0.5}


dg = Synthetic(1000, 1000, cov_shift=0.5, C0=(3.0, -2, 0.5), C1=(2.0, -2, 0.5))
loss_fn = pyro.infer.Trace_ELBO(
    num_particles=256,        # 1 → 64 samples
    vectorize_particles=True  # draws in one big plate ⇒ same speed on GPU
)
svi = pyro.infer.SVI(model, guide, optim=pyro.optim.Adam(optim_config), loss=loss_fn)
train_data, test_data = dg.get_data()

cov_f, treat_f, out_f, mu0, mu1 = train_data['cov_f'], train_data['treat_f'], train_data['out_f'], test_data['mu0'], test_data['mu1']
out_scaler = StandardScaler()
out_f = out_scaler.fit_transform(out_f.reshape(-1, 1)).reshape(-1)
mu0, mu1 = out_scaler.transform(mu0.reshape(-1, 1)), out_scaler.transform(mu1.reshape(-1, 1))

cov_f, treat_f, out_f = torch.tensor(cov_f).float(), torch.tensor(treat_f).float(), torch.tensor(out_f).float()
gt_ate = (mu1 - mu0).mean()
print(f"gt ate={gt_ate}")

# pyro.clear_param_store()

prior_predictive = Predictive(model, num_samples=10, return_sites=("tau",))
samples = prior_predictive(x=cov_f, t=treat_f, y=None)
sns.distplot(np.array(samples['tau'].detach()))
plt.show()

losses = []
for i in tqdm(range(1_000)):

    # if i % 10 == 0:
    #     losses = []
    #     for j in range(100):
    #         losses.append(svi.evaluate_loss(x=cov_f, t=treat_f, y=out_f))
    #     sns.distplot(np.array(losses))
    #     plt.show()

    loss = svi.step(x=cov_f, t=treat_f, y=out_f)

    if i % 10 == 0:
        losses.append(loss)

    if i % 10 == 0:
        loc = pyro.param("theta_loc").item()
        scale = pyro.param("theta_scale").item()
        print(f"  q(τ) mean={loc: .4f},  sd={scale: .4f}")
        # print(f"  q(τ) sd={scale: .4f}")
        print(f"  loss={loss: .4f}")
        # print(f"  kernel0={kernel0.variance: .4f}, {kernel0.lengthscale: .4f}")
        # print(f"  kernel0={kernel1.variance: .4f}, {kernel1.lengthscale: .4f}")

    # if i % 100 == 0:
    #     samples = prior_predictive(x=cov_f, t=treat_f, y=None)
    #     sns.distplot(np.array(samples['tau'].detach()))
    #     plt.show()

plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")
plt.show()
