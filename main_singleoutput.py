import torch
from NN_CoRes.models.NN_CoRes import NN_CoRes
from utils import get_data, set_seed, load
import argparse

############################### Solving single-output PDEs via NN-CoRes ##############################
############################### 0. Define Parameters in Parser #######################################
def get_parser():
    parser = argparse.ArgumentParser(description='NN-CoRes solver')
    
    # Equation parameters: \nu, \alpha or \epsilon depending on whether it is Burgers, Elliptic or Eikonal problem 
    parser.add_argument("--problem", type = str, default = 'Burgers') # Burgers Elliptic Eikonal
    parser.add_argument("--parameter", type = float, default = 0.003) 
    
    # Sampling points
    parser.add_argument("--n_pde", type = int, default = 10000)
    parser.add_argument("--n_bdy", type = int, default = 40)

    # NN layers
    parser.add_argument("--layers", type = list, default = [20,20,20,20]) 

    # RBF kernel lengthscale 
    parser.add_argument("--omega", type = int, default = 2.0)

    # Optimization settings
    parser.add_argument("--optimizer", type = str, default = 'L-BFGS') # or 'Adam'
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--epochs", type = int, default = 1000)
    parser.add_argument("--plot_loss", type = bool, default = True)

    # Random seed
    parser.add_argument("--randomseed", type = int, default = 1234)

    # Dtype, device
    parser.add_argument("--tkwargs", type = dict, default = {"dtype": torch.float, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")})

    args = parser.parse_args()    
    
    return args

options = get_parser()
set_seed(options.randomseed)

############################### 1. Generate Data ############################################
X_bdy, X_col, X_test, U_bdy, U_test = get_data(problem = options.problem,
                                        parameter = options.parameter, 
                                        n_bdy = options.n_bdy, 
                                        n_PDE = options.n_pde, 
                                        **options.tkwargs)

############################### 2. Build Model ##############################################
model = NN_CoRes(problem = options.problem,
                 parameter = options.parameter,
                 X_bdy = X_bdy, 
                 X_col = X_col, 
                 U_bdy = U_bdy, 
                 NN_layers = options.layers, 
                 omega = options.omega)

############################### 3. Train and Save Model ####################################
model.fit(optimizer = options.optimizer,
          num_iter = options.epochs, 
          lr = options.lr,
          plot_hist = options.plot_loss, 
          **options.tkwargs)

# If you want to save a model, use model.save(fld = 'Models/NN-CoRes_' + options.problem) 
# If you want to load a saved model, use model = load(fld = '...')

############################### 4. Evaluate Model ###########################################
model.evaluate(X_test, U_test)