# GDR-learners: Orthogonal Learning of Generative Models for Potential Outcomes

[![Conference](https://img.shields.io/badge/ICLR2026-Paper-blue])](https://openreview.net/forum?id=bbmcIaEmJG)
[![arXiv](https://img.shields.io/badge/arXiv-2306.01424-b31b1b.svg)](https://arxiv.org/abs/2509.22953)

<img width="1658" height="617" alt="image" src="https://github.com/user-attachments/assets/d4ebf88b-9468-4874-b836-013ce4c25a48" />

A Python package for generative doubly robust (GDR) learners based on 
- (a) conditional normalizing flows (GDR-CNFs), 
- (b) conditional generative adversarial networks (GDR-CGANs), 
- (c) conditional variational autoencoders (GDR-CVAEs),
- (d) conditional diffusion models (GDR-CDMs).

The project is built with the following Python libraries:
1. [Pyro](https://pyro.ai/) - deep learning and probabilistic modelling (normalizing flows, variational inference)
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking

## Setup

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000`

To access the MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to the local browser http://localhost:5000.

### Semi-synthetic datasets setup

Before running semi-synthetic experiments, place datasets in the corresponding folders:
- [IHDP100 dataset](https://www.fredjo.com/): ihdp_npci_1-100.test.npz and ihdp_npci_1-100.train.npz to `data/ihdp100/`
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition): to `data/acic2016/`
```
 ── data/acic_2016
    ├── synth_outcomes
    |   ├── zymu_<id0>.csv   
    |   ├── ... 
    │   └── zymu_<id14>.csv 
    ├── ids.csv
    └── x.csv 
```

## Experiments

The main training script is universal for different methods and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `config/` folder.

Generic script with logging and fixed random seed is the following:
```console
PYTHONPATH=.  python3 runnables/train.py +dataset=<dataset> +model=<model> exp.seed=10
```

### Models

One needs to choose a meta-learner type, generative model backbone, and then fill in the specific hyperparameters (they are left blank in the configs):

- Plug-in learners: `+model=plugin_neural` with `model.backbone_first_stage=<backbone>`
- IPTW-learners: `+model=plugin_iptw_neural` with `model.backbone_first_stage=<backbone>`
- RA-learners: `+model=ra_neural` with `model.backbone_first_stage=<backbone>` and `model.backbone_second_stage=<backbone>`
- GDR-learners: `+model=dr_neural` with `model.backbone_first_stage=<backbone>` and `model.backbone_second_stage=<backbone>`

where <backbone> is 
- conditional normalizing flows (CNFs): `cnf`
- conditional generative adversarial networks (CGANs): `cgan`
- conditional variational autoencoders (CVAEs): `cvae`
- conditional diffusion models (CDMs): `cdiffusion`

Models already have the best hyperparameters saved (for each generative model backbone and dataset), one can access them via: `+model/backbone_first_stage/<dataset>_hparams=plugin_<backbone>` or `+model/backbone_first_stage/<dataset>_hparams/plugin_<backbone>=<dataset_param>`. Hyperparameters for all variants of meta-learners (plug-in/IPTW/RA/GDR) are the same.

To perform a manual hyperparameter tuning, use the flags `model.nuisance.tune_hparams=True`, and then, see `model.hparams_grid`. 

### Datasets
One needs to specify a dataset/dataset generator (and some additional parameters, e.g. train size for the synthetic data `dataset.n_samples_train=1000`):
- Synthetic data (adapted from https://arxiv.org/abs/2209.06203): `+dataset=synthetic`  
- [IHDP](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162) dataset: `+dataset=ihdp100` 
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition) datasets: `+dataset=acic2016`
- [HC-MNIST](https://github.com/anndvision/quince/blob/main/quince/library/datasets/hcmnist.py) dataset: `+dataset=hcmnist`
- [Colored MNIST](https://arxiv.org/abs/2401.02602) dataset: `+dataset=colored_mnist`

### Examples

Example of running an experiment with our plug-in CNFs on synthetic data with $n_{\text{train}} = 1000$ with 3 random seeds:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=synthetic +model=plugin_neural +backbone_first_stage=cnf +backbone_first_stage/synthetic_hparams/n_1250=plugin_cnf exp.logging=True model.nuisance.tune_hparams=False model.nuisance.num_epochs=100 exp.seed=10
```

Example of tuning hyperparameters of the GDR-CDMs based on the HC-MNIST dataset:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=hcmnist +model=dr_neural +backbone_first_stage=cdiffusion +backbone_second_stage=cdiffusion +backbone_first_stage/hcmnist_hparams=plugin_cdiffusion exp.logging=True model.nuisance.tune_hparams=False model.nuisance.num_epochs=20 exp.seed=10
```
