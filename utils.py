import torch
import numpy as np
import random
import dill
from jax import vmap
import jax.numpy as jnp
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse import diags
from scipy.sparse import identity
import scipy

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def load(fld):
    with open(fld, 'rb') as f:
        model = dill.load(f)
    return model

def get_data(problem = 'Burgers', parameter = 10, N_bdy = 40, N_col_boundary = 100, N_col_domain = 10000, N_test_boundary = 1000, 
                    N_test_domain = 10000, **tkwargs):
    
    if problem == 'Elliptic':
        domain = {'x':[0.0, 1.0], 'y':[0.0, 1.0]}

        ### Training data (sampled from boundaries)
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)
        # Generate the coordinates for the points on each side
        points_x = torch.linspace(domain['x'][0], domain['x'][1], N_bdy+2)[1:-1]
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_bdy)], dim=1)

        points_x = torch.linspace(domain['x'][0], domain['x'][1], N_bdy+2)[1:-1]
        top = torch.stack([points_x.squeeze(), torch.ones(N_bdy)], dim=1)

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_bdy+2)[1:-1]
        right = torch.stack([domain['x'][1]*torch.ones(N_bdy), points_y.squeeze()], dim=1)

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_bdy+2)[1:-1]
        left = torch.stack([torch.zeros(N_bdy), points_y.squeeze()], dim=1)

        corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
        
        # Concatenate the points from all sides to form the boundary tensor
        X_train = torch.cat([top, right, bottom, left, corners], dim=0)
        U_train = torch.zeros_like(X_train[:,0])
   
        ### Generate collocation points
        # At boundary
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)

        points_x = soboleng.draw(N_col_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_col_boundary)], dim=1)

        points_x = soboleng.draw(N_col_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        top = torch.stack([points_x.squeeze(), torch.ones(N_col_boundary)], dim=1)

        points_y = soboleng.draw(N_col_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        right = torch.stack([domain['x'][1]*torch.ones(N_col_boundary), points_y.squeeze()], dim=1)

        points_y = soboleng.draw(N_col_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        left = torch.stack([torch.zeros(N_col_boundary), points_y.squeeze()], dim=1)

        # Concatenate the points from all sides to form the boundary tensor
        X_col_boundary = torch.cat([top, right, bottom, left, corners], dim=0)
        
        # Inside domain
        soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble = True)
        X_col_domain = soboleng.draw(N_col_domain).to(torch.float32)
        X_col_domain[:,0] = X_col_domain[:,0]*domain['x'][1]

        X_col = torch.concat((X_col_boundary, X_col_domain), axis=0)
        #U_col = torch.sin(torch.pi*X_col[:,0])*torch.sin(torch.pi*X_col[:,1]) + 2*torch.sin(4*torch.pi*X_col[:,0])*torch.sin(4*torch.pi*X_col[:,1])

        ### Generate test data (may be also used as collocation points)        
        # At boundary
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)

        points_x = soboleng.draw(N_test_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_test_boundary)], dim=1)

        points_x = soboleng.draw(N_test_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        top = torch.stack([points_x.squeeze(), torch.ones(N_test_boundary)], dim=1)

        points_y = soboleng.draw(N_test_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        right = torch.stack([domain['x'][1]*torch.ones(N_test_boundary), points_y.squeeze()], dim=1)

        points_y = soboleng.draw(N_test_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        left = torch.stack([torch.zeros(N_test_boundary), points_y.squeeze()], dim=1)

        # Concatenate the points from all sides to form the boundary tensor
        X_test_boundary = torch.cat([top, right, bottom, left, corners], dim=0)
        
        # Inside domain
        soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble = True)
        X_test_domain = soboleng.draw(N_test_domain).to(torch.float32)
        X_test_domain[:,0] = X_test_domain[:,0]*domain['x'][1]

        X_test = torch.concat((X_test_boundary, X_test_domain), axis=0)
        U_test = torch.sin(torch.pi*X_test[:,0])*torch.sin(torch.pi*X_test[:,1]) + 2*torch.sin(4*torch.pi*X_test[:,0])*torch.sin(4*torch.pi*X_test[:,1])

    elif problem == 'Burgers':
        domain = {'x':[0.0, 1.0], 't':[0.0, 1.0]}
        ### Training data (sampled from boundaries)
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)
        # Generate the coordinates for the points on each side
        points_x = torch.linspace(-domain['x'][1], domain['x'][1], N_bdy+2)[1:-1]
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_bdy)], dim=1)

        #points_t = soboleng.draw(N_train).to(torch.float32)*t_lim # from 0 to 1
        points_t = torch.linspace(0.0, domain['t'][1], N_bdy+2)[1:-1]
        right = torch.stack([domain['x'][1]*torch.ones(N_bdy), points_t.squeeze()], dim=1)

        #points_t = soboleng.draw(N_train).to(torch.float32)
        points_t = torch.linspace(0.0, domain['t'][1], N_bdy+2)[1:-1]
        left = torch.stack([-domain['x'][1]*torch.ones(N_bdy), points_t.squeeze()], dim=1)

        corners =  torch.tensor([[-1.0, 0.0],[1.0, 0.0],[1.0, 1.0],[-1.0, 1.0]])
        U_train_corners = torch.zeros_like(corners[:,0])
        # Concatenate the points from all sides to form the boundary tensor
        X_train = torch.cat([right, bottom, left, corners], dim=0)

        U_train_bottom = -torch.sin(torch.pi*bottom[:,0])
        U_train_left = torch.zeros_like(left[:,1])
        U_train_right = torch.zeros_like(right[:,1])
        U_train = torch.cat([U_train_right, U_train_bottom, U_train_left, U_train_corners], dim=0)

        ### Generate collocation points        
        # At boundary
        points_x = (soboleng.draw(N_col_boundary).to(torch.float32))*2-1 # from -1 to 1
        points_x = points_x * torch.tensor([domain['x'][1]])
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_col_boundary)], dim=1)

        points_x = (soboleng.draw(N_col_boundary).to(torch.float32))*2-1 # from -1 to 1
        points_x = points_x * torch.tensor([domain['x'][1]])
        top = torch.stack([points_x.squeeze(), torch.ones(N_col_boundary)], dim=1)

        points_t = soboleng.draw(N_col_boundary).to(torch.float32)*domain['t'][1]
        right = torch.stack([domain['x'][1]*torch.ones(N_col_boundary), points_t.squeeze()], dim=1)

        points_t = soboleng.draw(N_col_boundary).to(torch.float32)*domain['t'][1]
        left = torch.stack([-domain['x'][1]*torch.ones(N_col_boundary), points_t.squeeze()], dim=1)

        X_col_boundary = torch.cat([right, bottom, left, top], dim=0)
        
        # Inside domain
        soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble = True)
        X_col_domain = soboleng.draw(N_col_domain).to(torch.float32)
        X_col_domain[:,0] = (X_col_domain[:,0]*2-1)*domain['x'][1]

        X_col = torch.concat((X_col_boundary, X_col_domain), axis=0)
        #U_col = torch.exp(-X_col[:,1])*torch.sin(X_col[:,0])

        ### Generate test data
        def exact_solution_burgers():
            N_pts = 120
            xx= np.linspace(0, 1, N_pts)
            yy = np.linspace(-1, 1, N_pts)
            XX, YY = np.meshgrid(xx, yy)
            X_test = np.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1)

            def u_truth_burgers(x1, x2):
                [Gauss_pts, weights] = np.polynomial.hermite.hermgauss(80)
                temp = x2-jnp.sqrt(4*parameter*x1)*Gauss_pts
                val1 = weights * jnp.sin(jnp.pi*temp) * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*parameter))
                val2 = weights * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*parameter))
                return -jnp.sum(val1)/jnp.sum(val2)
        
            test_truth = vmap(u_truth_burgers)(X_test[:,0],X_test[:,1])
            U_test = torch.tensor(np.array(test_truth))
            X_test = torch.tensor(X_test[:, [1, 0]])

            return X_test, U_test
        
        X_test, U_test = exact_solution_burgers()

    elif problem == 'Eikonal':
        domain = {'x':[0.0, 1.0], 'y':[0.0, 1.0]}
    
        ### Training data (sampled from boundaries)
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)
        # Generate the coordinates for the points on each side
        #points_x = soboleng.draw(N_train).to(torch.float32)**domain['x'][1] # from 0 to 1
        points_x = torch.linspace(domain['x'][0], domain['x'][1], N_bdy+2)[1:-1]
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_bdy)], dim=1)

        #points_x = soboleng.draw(N_train).to(torch.float32)**domain['x'][1] # from 0 to 1
        points_x = torch.linspace(domain['x'][0], domain['x'][1], N_bdy+2)[1:-1]
        top = torch.stack([points_x.squeeze(), torch.ones(N_bdy)], dim=1)

        #points_y = soboleng.draw(N_train).to(torch.float32)**domain['y'][1] # from 0 to 1
        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_bdy+2)[1:-1]
        right = torch.stack([domain['x'][1]*torch.ones(N_bdy), points_y.squeeze()], dim=1)

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_bdy+2)[1:-1]
        left = torch.stack([torch.zeros(N_bdy), points_y.squeeze()], dim=1)

        corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
        
        # Concatenate the points from all sides to form the boundary tensor
        X_train = torch.cat([top, right, bottom, left, corners], dim=0)
        U_train = torch.zeros_like(X_train[:,0])

        # At boundary
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)

        points_x = soboleng.draw(N_col_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_col_boundary)], dim=1)

        points_x = soboleng.draw(N_col_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        top = torch.stack([points_x.squeeze(), torch.ones(N_col_boundary)], dim=1)

        points_y = soboleng.draw(N_col_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        right = torch.stack([domain['x'][1]*torch.ones(N_col_boundary), points_y.squeeze()], dim=1)

        points_y = soboleng.draw(N_col_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        left = torch.stack([torch.zeros(N_col_boundary), points_y.squeeze()], dim=1)

        # Concatenate the points from all sides to form the boundary tensor
        X_col_boundary = torch.cat([top, right, bottom, left, corners], dim=0)
        
        # Inside domain
        soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble = True)
        X_col_domain = soboleng.draw(N_col_domain).to(torch.float32)
        X_col_domain[:,0] = X_col_domain[:,0]*domain['x'][1]

        X_col = torch.concat((X_col_boundary, X_col_domain), axis=0)
        #U_col = torch.sin(torch.pi*X_col[:,0])*torch.sin(torch.pi*X_col[:,1]) + 2*torch.sin(4*torch.pi*X_col[:,0])*torch.sin(4*torch.pi*X_col[:,1])

        ### Generate test data (may be also used as collocation points)        
        # At boundary
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)

        points_x = soboleng.draw(N_test_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        bottom = torch.stack([points_x.squeeze(), torch.zeros(N_test_boundary)], dim=1)

        points_x = soboleng.draw(N_test_boundary).to(torch.float32)*domain['x'][1] # from 0 to 1
        top = torch.stack([points_x.squeeze(), torch.ones(N_test_boundary)], dim=1)

        points_y = soboleng.draw(N_test_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        right = torch.stack([domain['x'][1]*torch.ones(N_test_boundary), points_y.squeeze()], dim=1)

        points_y = soboleng.draw(N_test_boundary).to(torch.float32)*domain['y'][1] # from 0 to 1
        left = torch.stack([torch.zeros(N_test_boundary), points_y.squeeze()], dim=1)

        # Concatenate the points from all sides to form the boundary tensor
        X_test_boundary = torch.cat([top, right, bottom, left, corners], dim=0)
        U_test_boundary = torch.zeros_like(X_test_boundary[:,0])
        
        # Inside domain
        N_pts = 120
        X_test_domain, Y_test_domain, U_test_domain = solve_Eikonal(N_pts-2, epsilon=parameter) # true solution
        X_test_domain = torch.tensor(np.concatenate((X_test_domain.ravel()[:, np.newaxis], Y_test_domain.ravel()[:, np.newaxis]), axis=1))
        U_test_domain = torch.tensor(U_test_domain.ravel())

        X_test = torch.concat((X_test_boundary, X_test_domain), axis=0)
        U_test = torch.concat((U_test_boundary, U_test_domain), axis=0)

    elif problem == 'LDC':
        domain = {'x':[0.0, 1.0], 'y':[0.0, 1.0]}

        ### Training data (sampled from boundaries)
        soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble = True)
        # Generate the coordinates for the points on each side
        #points_x = soboleng.draw(N_train).to(torch.float32)*domain['x'][1] # from 0 to 1
        points_x = torch.linspace(domain['x'][0], domain['x'][1], N_bdy+2)[1:-1]
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(N_bdy)], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:,0])
        v_bottom = torch.zeros_like(x_bottom[:,0])

        points_x = torch.linspace(domain['x'][0], domain['x'][1], N_bdy+2)[1:-1]
        x_top = torch.stack([points_x.squeeze(), torch.ones(N_bdy)], dim=1)
        u_top = parameter*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:,0])

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_bdy+2)[1:-1]
        x_right = torch.stack([domain['x'][1]*torch.ones(N_bdy), points_y.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:,0])
        v_right = torch.zeros_like(x_right[:,0])

        points_y = torch.linspace(domain['y'][0], domain['y'][1], N_bdy+2)[1:-1]
        x_left = torch.stack([torch.zeros(N_bdy), points_y.squeeze()], dim=1)
        u_left = torch.zeros_like(x_left[:,0])
        v_left = torch.zeros_like(x_left[:,0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]])
        u_corners = torch.zeros_like(x_corners[:,0])
        v_corners = torch.zeros_like(x_corners[:,0])

        X_train = torch.cat([x_top, x_right, x_bottom, x_left, x_corners], dim=0)
        U_train = torch.cat([u_top, u_right, u_bottom, u_left, u_corners], dim=0)
        V_train = torch.cat([v_top, v_right, v_bottom, v_left, v_corners], dim=0)
        
        T_train = torch.cat([torch.zeros_like(X_train[:,0].clone()), 1.0*torch.ones_like(X_train[:,0].clone())], dim=0)
        X_train = torch.cat([X_train.clone(), X_train.clone()], dim=0)
        X_train = torch.cat([X_train.clone(), T_train.clone().unsqueeze(-1)], dim=1)
        U_train = torch.cat([U_train.clone(), V_train.clone()], dim=0)

        ### Generate collocation points 
        def exact_solution_lidcavity():
            DATA_np = np.loadtxt(r"..\NN-CoRes\Datasets\LDC\A" + str(int(parameter)) + r".txt") 
            DATA_ = DATA_np[np.random.permutation(DATA_np.shape[0]), :]    #It has solutions for 3K points
            DATA = torch.from_numpy(DATA_[:,...]) #x , y , u , u_x , u_y , u_xx , u_yy 

            xc, yc = DATA[:,0:1] , DATA[:,1:2]
            uc, vc, pc = DATA[:,2], DATA[:,3], DATA[:,4]
            yc.requires_grad = True
            xc.requires_grad = True

            return xc, yc, uc, vc, pc
               
        x_col, y_col, _, _, _ = exact_solution_lidcavity()
        X_col = torch.stack([x_col, y_col], dim=1).squeeze().type(torch.float32)

        # Select 10,000 points for CP
        BC_idx = torch.where((X_col[:,0] == 0.0) | (X_col[:,1] == 0.0) | (X_col[:,0] == 1.0) | (X_col[:,1] == 1.0))[0]
        all_idx = torch.arange(x_col.size(0)).to(x_col.device)
        domain_idx = torch.masked_select(all_idx, torch.logical_not(torch.isin(all_idx, BC_idx)))
        
        X_col_BC = X_col[BC_idx,:]
        X_col_domain = X_col[domain_idx,:]
        
        random_indices = np.random.choice(X_col_domain.shape[0], N_col_domain, replace=False)
        X_col_domain = X_col_domain.clone()[random_indices,:]
        X_col = torch.vstack([X_col_domain, X_col_BC])

        ### Generate test data (may be also used as collocation points)        
        x_test, y_test, U_test, V_test, P_test = exact_solution_lidcavity()
        X_test = torch.stack([x_test, y_test], dim=1).squeeze().type(torch.float32)
        U_test = torch.cat([U_test.unsqueeze(-1).type(torch.float32), V_test.unsqueeze(-1).type(torch.float32), P_test.unsqueeze(-1).type(torch.float32)], dim=1)

        # Select 10,000 points for test
        BC_idx = torch.where((X_test[:,0] == 0.0) | (X_test[:,1] == 0.0) | (X_test[:,0] == 1.0) | (X_test[:,1] == 1.0))[0]
        all_idx = torch.arange(x_test.size(0)).to(x_test.device)
        domain_idx = torch.masked_select(all_idx, torch.logical_not(torch.isin(all_idx, BC_idx)))
        
        X_test_BC = X_test[BC_idx,:]
        U_test_BC = U_test[BC_idx,:]
        X_test_domain = X_test[domain_idx,:]
        U_test_domain = U_test[domain_idx,:]
        
        random_indices = np.random.choice(X_test_domain.shape[0], N_test_domain, replace=False)
        X_test_domain = X_test_domain.clone()[random_indices,:]
        U_test_domain = U_test_domain.clone()[random_indices,:]
        X_test = torch.vstack([X_test_domain, X_test_BC])
        U_test = torch.vstack([U_test_domain, U_test_BC])

        X_train_P = torch.tensor([[0.0, 0.0, 2.0]])
        P_train = torch.tensor([0.0])
        X_train = torch.concat((X_train, X_train_P), axis=0)
        U_train = torch.concat((U_train, P_train), axis=0)
        
    return X_train.type(tkwargs["dtype"]).to(tkwargs['device']), X_col.type(tkwargs["dtype"]).clone().requires_grad_(True).to(tkwargs['device']), X_test.type(tkwargs["dtype"]).to(tkwargs['device']), U_train.squeeze().type(tkwargs["dtype"]).to(tkwargs['device']), U_test.squeeze().type(tkwargs["dtype"])

def solve_Eikonal(N, epsilon):
    hg = np.array(1/(N+1))
    x_grid = (np.arange(1,N+1,1))*hg
    a1 = np.ones((N,N+1))
    a2 = np.ones((N+1,N))

    # diagonal element of A
    a_diag = np.reshape(a1[:,:N]+a1[:,1:]+a2[:N,:]+a2[1:,:], (1,-1))

    # off-diagonals
    a_super1 = np.reshape(np.append(a1[:,1:N], np.zeros((N,1)), axis = 1), (1,-1))
    a_super2 = np.reshape(a2[1:N,:], (1,-1))

    A = diags([[-a_super2[np.newaxis, :]], [-a_super1[np.newaxis, :]], [a_diag], [-a_super1[np.newaxis, :]], [-a_super2[np.newaxis, :]]], [-N,-1,0,1,N], shape=(N**2, N**2), format = 'csr')
    XX, YY = np.meshgrid(x_grid, x_grid)
    f = np.zeros((N,N))
    f[0,:] = f[0,:] + epsilon**2 / (hg**2)
    f[N-1,:] = f[N-1,:] + epsilon**2 / (hg**2)
    f[:, 0] = f[:, 0] + epsilon**2 / (hg**2)
    f[:, N-1] = f[:, N-1] + epsilon**2 / (hg**2)
    fv = f.flatten()
    fv = fv[:, np.newaxis]

    mtx = identity(N**2)+(epsilon**2)*A/(hg**2)
    sol_v = scipy.sparse.linalg.spsolve(mtx, fv)
    # sol_v, exitCode = scipy.sparse.linalg.cg(mtx, fv)
    # print(exitCode)
    sol_u = -epsilon*np.log(sol_v)
    sol_u = np.reshape(sol_u, (N,N))
    return XX, YY, sol_u 

