import osqp
#  import osqppurepy as osqp
import scipy.sparse as sparse
import scipy as sp
import numpy as np
import mathprogbasepy as mpbpy
sp.random.seed(2)

# n = 100
# m = 500
n = 2
m = 3
A = sparse.random(m, n, density=0.9, format='csc')
lA = -sp.rand(m) * 2.
uA = sp.rand(m) * 2.

P = sparse.random(n, n, density=0.9, format='csc')
P = P.dot(P.T)
q = sp.randn(n)


qp = mpbpy.QuadprogProblem(P, q, A, lA, uA)


osqp_opts = {'rho_ineq': 0.1,
            #  'auto_rho': True,
             'sigma': 1e-06,
            #  'eps_rel': 1e-08,
            #  'eps_abs': 1e-08,
             'scaled_termination': True,
             'early_terminate_interval': 1,
             'polish': False,
             'scaling': True,
             'scaling_norm': -1,
             'max_iter': 2500,
             'verbose': True
             }

# qp.solve(solver=GUROBI)
res_purepy = qp.solve(solver=mpbpy.OSQP_PUREPY, **osqp_opts)
# res_osqp = qp.solve(solver=mpbpy.OSQP, **osqp_opts)

model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=lA, u=uA, **osqp_opts)
res_osqp = model.solve()
#
#
# # Store optimal values
# x_opt = res_osqp.x
# y_opt = res_osqp.y
#
# # Warm start with zeros
# model.warm_start(x=np.zeros(n), y=np.zeros(m))
# res_osqp_zero_warmstart = model.solve()
#
# # Warm start with optimal values
# model.warm_start(x=x_opt, y=y_opt)
# res_osqp_opt_warmstart = model.solve()
