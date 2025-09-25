import numpy as np
from sklearn.datasets import make_moons
from sklearn import preprocessing
from typing import Tuple
import pyro.distributions as dist
import torch


class ConditionedMoonsDistribution:

    def __init__(self, theta, X, out_scaler, **kwargs):
        self.theta = theta
        self.X = X
        self.out_scaler = out_scaler

    def sample(self, n_samples):
        noise_theta = 0.1 * np.random.randn(*n_samples, self.X.shape[0])
        noise_Y = 0.05 * np.random.randn(*n_samples, *self.X.shape)

        theta = self.theta + noise_theta
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Y_pot = np.empty(shape=(*n_samples, *self.X.shape))
        Y_pot_scaled = np.empty_like(Y_pot)
        for j in range(n_samples[0]):
            for i in range(self.X.shape[0]):
                Y_pot[j, i, :] = self.X[i, :] @ rot[:, :, j, i] + noise_Y[j, i, :]
            Y_pot_scaled[j, :, :] = self.out_scaler.transform(Y_pot[j, :, :])
        return torch.tensor(Y_pot_scaled).float()


class SyntheticMoons:
    def __init__(self, n_samples=5000, noise=0.5, theta_0=np.pi / 4, theta_1=-np.pi / 4, **kwargs):

        self.n_samples = n_samples
        self.noise = noise

        self.out_scaler = preprocessing.StandardScaler()

        self.theta_0 = theta_0
        self.theta_1 = theta_1

    def get_data(self) -> dict:

        # Sampling random rotations
        noise_theta = 0.1 * np.random.randn(self.n_samples)
        noise_Y = 0.05 * np.random.randn(self.n_samples, 2)

        theta_0, theta_1 = self.theta_0 + noise_theta, self.theta_1 + noise_theta
        rot_0 = np.array([[np.cos(theta_0), -np.sin(theta_0)], [np.sin(theta_0), np.cos(theta_0)]])
        rot_1 = np.array([[np.cos(theta_1), -np.sin(theta_1)], [np.sin(theta_1), np.cos(theta_1)]])

        X, T = make_moons(self.n_samples, noise=self.noise)
        Y = np.empty_like(X)
        Y0_pot = np.empty_like(X)
        Y1_pot = np.empty_like(X)
        for i, t in enumerate(T):
            Y0_pot[i, :] = X[i, :] @ rot_0[:, :, i] + noise_Y[i]
            Y1_pot[i, :] = X[i, :] @ rot_1[:, :, i] + noise_Y[i]
            if t == 0:
                Y[i, :] = X[i, :] @ rot_0[:, :, i] + noise_Y[i]
            else:
                Y[i, :] = X[i, :] @ rot_1[:, :, i] + noise_Y[i]

        self.out_scaler.fit(np.concatenate([Y0_pot, Y1_pot], 0).reshape(-1, 2))
        Y_scaled = self.out_scaler.transform(Y)
        Y0_pot_scaled = self.out_scaler.transform(Y0_pot)
        Y1_pot_scaled = self.out_scaler.transform(Y1_pot)

        return {
            'cov_f': X,
            'treat_f': T.astype(float).reshape(-1, 1),
            'out_f': Y,
            'out_f_scaled': Y_scaled,
            'out_pot_0': Y0_pot,
            'out_pot_1': Y1_pot,
            'out_pot0_scaled': Y0_pot_scaled,
            'out_pot1_scaled': Y1_pot_scaled
        }

    def get_pot_cond_dist(self, data_dict) -> Tuple[ConditionedMoonsDistribution]:
        X = data_dict['cov_f']
        return (ConditionedMoonsDistribution(self.theta_0, X, self.out_scaler), ConditionedMoonsDistribution(self.theta_1, X, self.out_scaler))


