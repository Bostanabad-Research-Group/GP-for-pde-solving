import torch
from NN_CoRes.models.NN_CoRes_MultiOutput import NN_CoRes_MultiOutput
from utils import get_data, set_seed, load
import argparse

############################ Solving multi-output PDE systems via NN-CoRes ###########################

############################### 0. Define Parameters in Parser #######################################
def get_parser():
    parser = argparse.ArgumentParser(description='NN-CoRes solver')
    
    # Equation parameters: A for LDC 
    parser.add_argument("--problem", type = str, default = 'LDC')
    parser.add_argument("--parameter", type = float, default = 5) 
    
    # Sampling points
    parser.add_argument("--n_pde", type = int, default = 10000)
    parser.add_argument("--n_bdy", type = int, default = 40)

    # NN layers
    parser.add_argument("--layers", type = list, default = [50,50,50,50,50,50]) 

    # RBF kernel lengthscale 
    parser.add_argument("--omega", type = int, default = 2.0)

    # Optimization settings
    parser.add_argument("--optimizer", type = str, default = 'L-BFGS') # or 'Adam'
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--epochs", type = int, default = 2000)
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
# Note: when dealing with multioutput data sets, we append a new feature as a label to indicate
# the corresponding output. For LDC, we use '1' for u, '2' for v and '3' for p. This is later used
# for building the corresponding covariance matrices in the corrective residuals (CoRes) 
X_bdy, X_col, X_test, U_bdy, U_test = get_data(problem = options.problem,
                                               parameter = options.parameter, 
                                               n_bdy = options.n_bdy, 
                                               n_PDE = options.n_pde, 
                                               **options.tkwargs)

############################### 2. Build Model ##############################################
model = NN_CoRes_MultiOutput(problem = options.problem,
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

model.save(fld = 'Models/NN-CoRes_' + options.problem) # If you want to load a saved model, use model = load(fld = '...')

############################### 4. Evaluate Model ###########################################
model.evaluate(X_test, U_test)