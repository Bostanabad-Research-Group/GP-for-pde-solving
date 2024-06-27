# Neural Networks with Kernel-Weighted Corrective Residuals for Solving PDEs
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Conda](https://img.shields.io/conda/v/gpytorch/gpytorch.svg)](https://anaconda.org/gpytorch/gpytorch)

Code for the paper [Neural Networks with Kernel-Weighted Corrective Residuals for Solving Partial Differential Equations](https://arxiv.org/abs/2401.03492), where we introduce kernel-weighted Corrective Residuals (CoRes) to integrate the strengths of kernel methods and deep NNs for solving nonlinear PDE systems.

The framework consists of two modules (see figure below):
- Module 1: we endow the solution with a GP prior containing a deep NN as the mean function and the Gaussian kernel. The NN parameters are fixed and the kernel parameters are estimated via heuristics (or MLE).
- Module 2: the NN parameters are estimated by minimizing a loss function which only depends on the PDE since BC/IC are automatically satisfied through the kernels.

![Flowchart](https://github.com/Bostanabad-Research-Group/GP-for-pde-solving/assets/102708675/f951e586-730d-401e-9658-582b457bd51c)

The resulting model can solve PDE systems without any labeled data inside the domain and is particularly attractive because it $(1)$ naturally satisfies the boundary and initial conditions of a PDE system in arbitrary domains, and $(2)$ can leverage any differentiable function approximator, \eg deep NN architectures, in its mean function.


## Requirements
We recommend installing the following packages via Anaconda:
- Python == 3.9.13
- PyTorch == 1.12.0
- CUDA >= 11.3
- GPyTorch == 1.7.0
- BoTorch == 0.6.4
- Dill == 0.3.5.1
- Matplotlib == 3.5.3

## Citation
If you use this code, please cite the following paper:
```bibtex
@article{mora2024neural,
  title={Neural Networks with Kernel-Weighted Corrective Residuals for Solving Partial Differential Equations},
  author={Mora, Carlos and Yousefpour, Amin and Hosseinmardi, Shirin and Bostanabad, Ramin},
  journal={arXiv preprint arXiv:2401.03492},
  year={2024}
}

