import torch
import math
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.priors import NormalPrior
from .gpregression import GPR
from .. import kernels
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F 
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from gpytorch.means import Mean
from torch.nn.parameter import Parameter
from torch.nn import init

class NN_CoRes_MultiOutput(GPR):
    """ NN_CoRes model combines the strengths of kernel methods with neural networks for PDE solving.
    Arguments:
            - X_bdy: the inputs of the sampled data at the boundaries.
            - U_bdy: the outputs of the sampled data at the boundaries.
            - X_col: the inputs of the sampled data at the boundaries.
            - quant_correlation_class: kernel used in the kernel-weighted CoRes. Default: 'RBF'
            - omega: roughness parameter used in the kernel-weighted CoRes when RBF kernel is employed. Default: 2.0
            - basis: mean function used. Default: 'neural_network'. Other choices: zero-mean ('zero') or M3 architecture ('M3') from reference (31) in the paper.
            - NN_layers: architecture of the neural network when using 'neural_network' or 'M3' as basis.
    """
    def __init__(
        self,
        X_bdy:torch.Tensor,
        U_bdy:torch.Tensor,
        X_col:torch.Tensor,
        quant_correlation_class:str = 'RBF',
        omega:float = 2.0,
        basis = 'neural_network',
        NN_layers = [20,20,20,20],
    ) -> None:
               
        quant_index = set(range(X_bdy.shape[1]-1))
        quant_correlation_class_name = quant_correlation_class

        if quant_correlation_class_name == 'RBF':
            quant_correlation_class = 'RBFKernel'

        try:
            quant_correlation_class = getattr(kernels,quant_correlation_class)
        except:
            raise RuntimeError(
                "%s not an allowed kernel" % quant_correlation_class
            )

        if quant_correlation_class_name == 'RBF':
            quant_kernel = quant_correlation_class(
                ard_num_dims = len(quant_index),
                active_dims = torch.arange(len(quant_index)),
                lengthscale_constraint = Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
            )
        
        super(NN_CoRes_MultiOutput, self).__init__(
            train_x = X_bdy, train_y = U_bdy, noise_indices = [],
            correlation_kernel = quant_kernel,
            noise = 5e-8, fix_noise = True, lb_noise = 1e-8
        )

        self.covar_module_u = quant_correlation_class(
                ard_num_dims = len(quant_index),
                active_dims = torch.arange(len(quant_index)),
                lengthscale_constraint = Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
            )
        
        self.covar_module_v = quant_correlation_class(
                ard_num_dims = len(quant_index),
                active_dims = torch.arange(len(quant_index)),
                lengthscale_constraint = Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
            )
        
        self.covar_module_p = quant_correlation_class(
                ard_num_dims = len(quant_index),
                active_dims = torch.arange(len(quant_index)),
                lengthscale_constraint = Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
            )
        
        u_idx = torch.where(X_bdy[:,2] == 0.0)[0]
        v_idx = torch.where(X_bdy[:,2] == 1.0)[0]
        p_idx = torch.where(X_bdy[:,2] == 2.0)[0]

        self.X_train_u = X_bdy[u_idx,:-1].to('cuda')
        self.X_train_v = X_bdy[v_idx,:-1].to('cuda')
        self.X_train_p = X_bdy[p_idx,:-1].to('cuda')
        self.u_train = U_bdy[u_idx].to('cuda')
        self.v_train = U_bdy[v_idx].to('cuda')
        self.p_train = U_bdy[p_idx].to('cuda')
        self.X_col = X_col
        self.omega = torch.full((1, X_bdy.shape[1]-1), omega)
        self.basis = basis

        if self.basis=='zero':
            self.mean_module = gpytorch.means.ConstantMean(prior = NormalPrior(0.,1.))
            self.mean_module.constant.data = torch.tensor([0.0])  # Set the desired value
            self.mean_module.constant.requires_grad = False  # Fix the hyperparameter

        elif self.basis=='neural_network':
            setattr(self,'mean_module_NN_All', FFNN(self, input_size = self.X_train_u.shape[1], num_classes = 3, layers = NN_layers, name = str('mean_module_0'))) 
        
        elif self.basis=='M3':
            setattr(self,'mean_module_NN_All', NetworkM3(input_dim = self.X_train_u.shape[1], output_dim = 1, layers = NN_layers)) 

    def module1(self):
        # Fix Kernel Hyperparameters
        self.covar_module_u.raw_lengthscale.data.copy_(self.omega)
        self.covar_module_u.raw_lengthscale.requires_grad = False
        self.covar_module_v.raw_lengthscale.data.copy_(self.omega)
        self.covar_module_v.raw_lengthscale.requires_grad = False
        self.covar_module_p.raw_lengthscale.data.copy_(self.omega)
        self.covar_module_p.raw_lengthscale.requires_grad = False

        # Store Cholesky decomposition of the Covariance Matrix
        self.chol_decomp_u = self.covar_module_u(self.X_train_u).cholesky()
        self.chol_decomp_v = self.covar_module_v(self.X_train_v).cholesky()
        self.chol_decomp_p = self.covar_module_p(self.X_train_p).cholesky()

    def predict(self, X_test):
        '''Predict based on the formula of the expected value of a conditional GP. Note that the mean function is shared among the
           different outputs but not the covariance matrices.
        '''
        ### Evaluate mean (NN)
        m = self.mean_module_NN_All(X_test)

        ### Corrective residuals (CoRes)
        c_u = (self.covar_module_u(self.X_train_u, X_test)).evaluate()
        c_v = (self.covar_module_v(self.X_train_v, X_test)).evaluate()
        c_p = (self.covar_module_p(self.X_train_p, X_test)).evaluate()
            
        r_u = self.u_train.unsqueeze(-1) - self.mean_module_NN_All(self.X_train_u)[:,0].unsqueeze(-1)
        r_v = self.v_train.unsqueeze(-1) - self.mean_module_NN_All(self.X_train_v)[:,1].unsqueeze(-1)
        r_p = self.p_train.unsqueeze(-1) - self.mean_module_NN_All(self.X_train_p)[:,2].unsqueeze(-1)

        C_inv_r_u = self.chol_decomp_u._cholesky_solve(r_u)
        C_inv_r_v = self.chol_decomp_v._cholesky_solve(r_v)
        C_inv_r_p = self.chol_decomp_p._cholesky_solve(r_p)

        ### Final prediction (NN + CoRes)
        eta_u = (m[:,0].unsqueeze(-1) + c_u.t() @ C_inv_r_u).squeeze(-1)
        eta_v = (m[:,1].unsqueeze(-1) + c_v.t() @ C_inv_r_v).squeeze(-1)
        eta_p = (m[:,2].unsqueeze(-1) + c_p.t() @ C_inv_r_p).squeeze(-1)

        return eta_u, eta_v, eta_p

    def calculate_loss(self):
        '''Calculate the loss function by leveraging automatic differentiation. The loss consists of a 
          a single term based on the MSE on the PDE residual. 
        '''
        ################## Forward pass (get \eta(x)) ##################
        X_col = self.X_col.clone()
        eta_u, eta_v, eta_p = self.predict(X_col)

        ################## Calculate Physics-informed Loss based on PDEs ##################
        # Get Derivatives via Automatic Differentiation
        eta_u_x_and_eta_u_y = torch.autograd.grad(eta_u, X_col, torch.ones_like(eta_u), True, True)[0]
        eta_u_x = eta_u_x_and_eta_u_y[:,0]
        eta_u_y = eta_u_x_and_eta_u_y[:,1]
        eta_u_xx_and_eta_u_xy = torch.autograd.grad(eta_u_x, X_col, torch.ones_like(eta_u_x), True, True)[0]
        eta_u_yx_and_eta_u_yy = torch.autograd.grad(eta_u_y, X_col, torch.ones_like(eta_u_y), True, True)[0]
        eta_u_xx = eta_u_xx_and_eta_u_xy[:,0]
        eta_u_yy = eta_u_yx_and_eta_u_yy[:,1]

        eta_v_x_and_eta_v_y = torch.autograd.grad(eta_v, X_col, torch.ones_like(eta_v), True, True)[0]
        eta_v_x = eta_v_x_and_eta_v_y[:,0]
        eta_v_y = eta_v_x_and_eta_v_y[:,1]
        eta_v_xx_and_eta_v_xy = torch.autograd.grad(eta_v_x, X_col, torch.ones_like(eta_v_x), True, True)[0]
        eta_v_yx_and_eta_v_yy = torch.autograd.grad(eta_v_y, X_col, torch.ones_like(eta_v_y), True, True)[0]
        eta_v_xx = eta_v_xx_and_eta_v_xy[:,0]
        eta_v_yy = eta_v_yx_and_eta_v_yy[:,1]

        eta_p_x_and_eta_p_y = torch.autograd.grad(eta_p, X_col, torch.ones_like(eta_p), True, True)[0]
        eta_p_x = eta_p_x_and_eta_p_y[:,0]
        eta_p_y = eta_p_x_and_eta_p_y[:,1]

        all_idx = torch.arange(X_col.size(0)).to(X_col.device)  # Generate indices for all collocation points
        BC_noslip_idx = torch.where((X_col[:,0] == 0.0) | (X_col[:,1] == 0.0) | (X_col[:,0] == 1.0))[0]
        BC_lid_idx = torch.where(X_col[:,1] == 1.0)[0]
        PDE_idx = torch.masked_select(all_idx, torch.logical_not(torch.isin(all_idx, torch.concat([BC_noslip_idx, BC_lid_idx]))))  # Obtain remaining indices

        # Calculate The Loss (MSE on PDE residual)
        residual_pde1 = eta_u_x[PDE_idx] + eta_v_y[PDE_idx]
        residual_pde2 = eta_u[PDE_idx]*eta_u_x[PDE_idx] + eta_v[PDE_idx]*eta_u_y[PDE_idx] + eta_p_x[PDE_idx] - 0.01*(eta_u_xx[PDE_idx]+eta_u_yy[PDE_idx])
        residual_pde3 = eta_u[PDE_idx]*eta_v_x[PDE_idx] + eta_v[PDE_idx]*eta_v_y[PDE_idx] + eta_p_y[PDE_idx] - 0.01*(eta_v_xx[PDE_idx]+eta_v_yy[PDE_idx])
        
        loss_pde = torch.mean(residual_pde1**2) + torch.mean(residual_pde2**2) + torch.mean(residual_pde3**2)

        return loss_pde

    def fit(self, optimizer: str = 'L-BFGS', lr:float = 0.01, num_iter:int = 1000, plot_hist = True, **tkwargs) -> float:
        '''Performs module 1 and 2 of the proposed framework:
            - Module 1: We set the kernel hyperparameters to a value such that the GP can faitfully reproduce BC/IC data.
            - Module 2: Optimize the NN parameters of NN-CoRes by minimizing the loss function (PDE residual). 

        Arguments of the function:
            - optimizer: optimizer used in module 2 (L-BFGS or Adam). Default: L-BFGS. 
            - lr: learning rate. The recommended value is 1e-2 for L-BFGS and 1e-3 for Adam. Default: 1e-2.
            - num_iter: number of iterations (epochs) to perform during optimization. This is the only termination criterion for the optimizer.
            - plot_hist: if True, the loss history is plotted at the end of optimization. If False, it is not plotted.
        '''
        self.to(**tkwargs)
        self.module1()
        loss_hist = []
        epochs_iter = tqdm(range(num_iter), desc='Epoch', position=0, leave=True)

        if optimizer == 'L-BFGS':
            optim = torch.optim.LBFGS(self.parameters(), lr = lr)

            for j in epochs_iter:
                def closure():
                    # Zero gradients from previous iteration
                    optim.zero_grad()
                    # Calculate loss
                    loss = self.calculate_loss()
                    # Backprop
                    loss.backward(retain_graph=True)
                    return loss
                
                optim.step(closure)
                loss = closure()
                desc = f'Epoch {j+1} - loss {loss.item():.6f}'
                epochs_iter.set_description(desc)
                loss_hist.append(loss.item())

        elif optimizer == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=lr)

            for j in epochs_iter:
                optim.zero_grad()
                loss = self.calculate_loss()
                loss.backward(retain_graph=True)
                optimizer.step()
                desc = f'Epoch {j} - loss {loss.item():.6f}'
                epochs_iter.set_description(desc)
                epochs_iter.update(1)
                loss_hist.append(loss.item())
        
        if plot_hist == True:
            plt.figure(figsize=(8,6))
            plt.plot(loss_hist)
            plt.title('Loss History')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.show()
    
    def save(self, fld):
        with open(fld, 'wb') as f:
            dill.dump(self, f)

    def evaluate(self, X_test, U_test):
        U_pred, V_pred, P_pred = self.predict(X_test)

        X_test, U_test, U_pred, V_pred, P_pred = X_test.detach().cpu(), U_test.detach().cpu(), U_pred.detach().cpu(), V_pred.detach().cpu(), P_pred.detach().cpu()

        def do_plot(x, u, ax, position, xlabel, ylabel, titleTop = ''):    
            ax = ax[position[0], position[1]]

            h = ax.tricontourf(x[:,0], x[:,1], u, levels=200, cmap='jet')
            for c in h.collections:
                c.set_edgecolor("face")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)

            cbar.ax.tick_params(labelsize=10)  
            cbar.ax.yaxis.offsetText.set(size=10)  
            cbar.ax.yaxis.offsetText.set_x(2)  
            tick_locator = ticker.MaxNLocator(nbins=8)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            cbar.formatter.set_powerlimits((0, 0))
            ax.set_aspect('equal', 'box')
            ax.set_title(titleTop, fontsize = 15, pad=5)

        pgf_with_latex = {                      
        "pgf.texsystem": "pdflatex",        
        "text.usetex": True,                
        "font.family": "serif",
        "font.serif": [],                  
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,              
        "font.size": 10,
        "legend.fontsize": 10,               
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    
            r"\usepackage[T1]{fontenc}",             
            ]
        }
        mpl.rcParams.update(pgf_with_latex)

        fig, ax = plt.subplots(3, 3, figsize=(15, 10))
        do_plot(X_test, U_test[:,0], ax, position=[0,0], titleTop='Reference ' + r'$u(\mathbf{x})$', xlabel='$x$', ylabel='$y$')
        do_plot(X_test, U_pred, ax, position=[0,1], titleTop='Predicted ' + r'$u(\mathbf{x})$', xlabel='$x$', ylabel='$y$')
        do_plot(X_test, abs(U_pred-U_test[:,0]), ax, position=[0,2], titleTop='Absolute error on ' + r'$u(\mathbf{x})$', xlabel='$x$', ylabel='$y$')
        
        do_plot(X_test, U_test[:,1], ax, position=[1,0], titleTop='Reference ' + r'$v(\mathbf{x})$', xlabel='$x$', ylabel='$y$')
        do_plot(X_test, V_pred, ax, position=[1,1], titleTop='Predicted ' + r'$v(\mathbf{x})$', xlabel='$x$', ylabel='$y$')
        do_plot(X_test, abs(V_pred-U_test[:,1]), ax, position=[1,2], titleTop='Absolute error on ' + r'$v(\mathbf{x})$', xlabel='$x$', ylabel='$y$')

        do_plot(X_test, U_test[:,2], ax, position=[2,0], titleTop='Reference ' + r'$p(\mathbf{x})$', xlabel='$x$', ylabel='$y$')
        do_plot(X_test, P_pred, ax, position=[2,1], titleTop='Predicted ' + r'$p(\mathbf{x})$', xlabel='$x$', ylabel='$y$')
        do_plot(X_test, abs(P_pred-U_test[:,2]), ax, position=[2,2], titleTop='Absolute error on ' + r'$p(\mathbf{x})$', xlabel='$x$', ylabel='$y$')

        plt.tight_layout()
        plt.show()

        abs_error_U = torch.tensor(abs(U_pred.cpu().detach().numpy()-U_test[:,0].squeeze().detach().numpy()))
        abs_error_V = torch.tensor(abs(V_pred.cpu().detach().numpy()-U_test[:,1].squeeze().detach().numpy()))
        abs_error_P = torch.tensor(abs(P_pred.cpu().detach().numpy()-U_test[:,2].squeeze().detach().numpy()))

        test_L2_err_U = torch.sqrt(torch.mean((abs_error_U)**2)) 
        test_L2_err_V = torch.sqrt(torch.mean((abs_error_V)**2)) 
        test_L2_err_P = torch.sqrt(torch.mean((abs_error_P)**2)) 

        print(f'Test L2 error (U) = {test_L2_err_U}')
        print(f'Test L2 error (V) = {test_L2_err_V}')
        print(f'Test L2 error (P) = {test_L2_err_P}')

class FFNN(gpytorch.Module):
    def __init__(self, lmgp, input_size, num_classes, layers, name):
        super(FFNN, self).__init__()
        self.dropout = nn.Dropout(0.0)
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = Linear(input_size, layers[0], bias=True, name='fci') 
            for i in range(1,self.hidden_num):
                setattr(self, 'h' + str(i), Linear(layers[i-1], layers[i], bias=True,name='h' + str(i)))
            
            self.fce = Linear(layers[-1], num_classes, bias=True,name='fce')
        else:
            self.fci = Linear(input_size, num_classes, bias=True,name='fci')

    def forward(self, x):
        if self.hidden_num > 0:       
            x = torch.tanh(self.fci(x))
            for i in range(1,self.hidden_num):
                x = torch.tanh( getattr(self, 'h' + str(i))(x) )

            x = self.fce(x)
        else:
            x = self.fci(x)

        return x
    
class NetworkM3(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1, layers = [40, 40, 40, 40], activation = 'tanh', x_col = []) -> None:
        super(NetworkM3, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'Silu':nn.SiLU(), 'Sigmoid':nn.Sigmoid()}
        activation = activation_list[activation]
        self.dim = layers[0]
  
        self.U = nn.Linear(input_dim, self.dim).to('cuda')
        self.V = nn.Linear(input_dim, self.dim).to('cuda')
        self.H1 = nn.Linear(input_dim, self.dim).to('cuda')
        self.last= nn.Linear(self.dim, output_dim).to('cuda')

        self.x_col = x_col
        self.alpha = 1.0
        self.beta = 1.0
        
        l = nn.ModuleList()
        for _ in range(len(layers)):
            l.append(nn.Linear(self.dim, self.dim))
            l.append(activation)
        self.layers = nn.Sequential(*l).to('cuda')

    def forward(self, input):        
        U = nn.Tanh()(self.U(input))
        V = nn.Tanh()(self.V(input))
        H = nn.Tanh()(self.H1(input))

        for layer in self.layers:
            Z = layer(H)
            H = (1-Z)*U + Z*V
        
        out = self.last(H)
        return out.squeeze()

class Linear(Mean):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, name=None,
                 device=None, dtype=None) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.name=str(name)
        self.register_parameter(name=str(self.name)+'weight',  parameter = Parameter(torch.empty((out_features, in_features))))
        self.register_prior(name =str(self.name)+ 'prior_m_weight_fci', prior = gpytorch.priors.NormalPrior(0.,1.), param_or_closure=str(self.name)+'weight')
        
        if bias:
            self.register_parameter(name=str(self.name)+'bias',  parameter = Parameter(torch.empty(out_features)))
            self.register_prior(name= str(self.name)+'prior_m_bias_fci', prior = gpytorch.priors.NormalPrior(0.,1.), param_or_closure=str(self.name)+'bias')
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:                                         
        init.kaiming_uniform_( getattr(self,str(self.name)+'weight'), a=math.sqrt(5))
        if getattr(self,str(self.name)+'bias') is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self,str(self.name)+'weight'))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(getattr(self,str(self.name)+'bias'), -bound, bound)

    def forward(self, input) -> Tensor:
        return F.linear(input, getattr(self,str(self.name)+'weight'), getattr(self,str(self.name)+'bias'))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


