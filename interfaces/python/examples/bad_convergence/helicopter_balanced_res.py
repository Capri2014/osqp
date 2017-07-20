# import osqp
import osqppurepy as osqp
import numpy as np
import scipy.sparse as spa

import pickle

# Load one problem
with open('./data/%s.pickle' % 'helicopter_balanced_residuals', 'rb') as f:
    problem = pickle.load(f)


# OSQP settings
osqp_settings = {'verbose': True,
                 'scaling': True,
                 'early_terminate_interval': 1,
                 'rho': 10.0,
                #  'diagonal_rho': False,
                #  'update_rho': False,
                 'polish': False}


# Assign problem data
P = problem['P']
q = problem['q']
A = problem['A']
l = problem['l']
u = problem['u']


# Store mat file
# import scipy.io as sio
# sio.savemat('bad_mpc_problem.mat', problem)


# Scale data?
# norm_scaling = np.linalg.norm(P.todense())
# norm_scaling = 1
# P /= norm_scaling
# q /= norm_scaling

# Solve with OSQP
model = osqp.OSQP()
model.setup(P, q, A,
            l, u, **osqp_settings)
res_osqp = model.solve()
