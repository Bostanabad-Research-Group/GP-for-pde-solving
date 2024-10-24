# Combining Neural Networks with Kernel Methods for Solving Nonlinear PDEs
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for the paper [A Gaussian process framework for solving forward and inverse problems involving nonlinear partial differential equations](https://link.springer.com/article/10.1007/s00466-024-02559-0), where we introduce kernel-weighted Corrective Residuals (CoRes) to integrate the strengths of kernel methods and deep NNs for solving nonlinear PDE systems.

The framework consists of two sequential modules (see figure below):
- Module 1: we endow the solution with a GP prior containing a deep NN as the mean function and the Gaussian kernel. The NN parameters are fixed and the kernel parameters are estimated via heuristics (or MLE) to faithfully reproduce the data sampled from boundary and/or initial conditions.
- Module 2: the NN parameters are estimated by minimizing a loss function which only depends on the PDE since boundary and/or initial conditions are automatically satisfied through the kernels.

![Flowchart](https://github.com/Bostanabad-Research-Group/GP-for-pde-solving/assets/102708675/f951e586-730d-401e-9658-582b457bd51c)

The resulting model can solve PDE systems without any labeled data inside the domain and is particularly attractive because it (1) naturally satisfies the boundary and initial conditions of a PDE system in arbitrary domains, and (2) can leverage any differentiable function approximator such as deep NN architectures in its mean function.

![Burgers_Elliptical_Results_Low](https://github.com/user-attachments/assets/09a9daf8-aafd-43ac-a6c4-72152fce70b6)

## Requirements
Please ensure the following packages are installed with the specified versions. If you prefer to use Anaconda, the commands for creating an environment and installing these packages through its prompt are also provided:
- Python == 3.9.13: `conda create --name NN_CoRes python=3.9.13` and then activate the environment via `conda activate NN_CoRes`
- [PyTorch](https://github.com/pytorch/pytorch) == 1.12.0 & CUDA >= 11.3: `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch`
- [GPyTorch](https://github.com/cornellius-gp/gpytorch) == 1.7.0: `conda install -c gpytorch gpytorch=1.7.0`
- [JAX](https://github.com/google/jax) == 0.4.25: `pip install jax==0.4.25 jaxlib==0.4.25 jaxtyping==0.2.25`
- Dill == 0.3.5.1: `pip install dill==0.3.5.1`
- Matplotlib == 3.5.3: `conda install -c conda-forge matplotlib=3.5.3`
- Tqdm >= 4.66.4: `pip install tqdm`

## Usage
After installing the above packages, you are all set to use our code. We provide two main files that demonstrate the application of NN-CoRes for solving the PDEs discussed in the paper.
You can test them by downloading the repo and running the following commands in your terminal:
- Burgers' equation: `python main_singleoutput.py --problem Burgers --parameter 0.003`
- Elliptic PDE: `python main_singleoutput.py --problem Elliptic --parameter 30`
- Eikonal equation: `python main_singleoutput.py --problem Eikonal --parameter 0.01`
- Lid-Driven Cavity: `python main_multioutput.py --problem LDC --parameter 5`

Alternatively, you can also simply run the files `main_singleoutput.py` or `main_multioutput.py` in your compiler.

You can use additional arguments to modify settings such as the architecture used in the mean function, optimizer, number of epochs, and more. Please refer to each main file for details.

## Contributions and Assistance
All contributions are welcome. If you notice any bugs, mistakes or have any question about the documentation, please report them by opening an issue on our GitHub page. Please make sure to label the issue according to the specific module or feature related to the problem.

## Citation
If you use this code or find our work interesting, please cite the following paper:
```bibtex
@article{mora2024neural,
  title={Neural Networks with Kernel-Weighted Corrective Residuals for Solving Partial Differential Equations},
  author={Mora, Carlos and Yousefpour, Amin and Hosseinmardi, Shirin and Bostanabad, Ramin},
  journal={arXiv preprint arXiv:2401.03492},
  year={2024}
}
