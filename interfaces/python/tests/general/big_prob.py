import osqp
import osqppurepy as osqppurepy
import scipy.sparse as sparse
import scipy as sp
import numpy as np
import mathprogbasepy as mpbpy
sp.random.seed(5)

n = 50
m = 100
A = sparse.random(m, n, density=0.4, format='csc')
lA = -sp.rand(m) * 2.
uA = sp.rand(m) * 2.

# Shift bounds
# lA[-1] = lA[-1] + 1e6
# uA[-1] = uA[-1] + 1e6

# n_eq = 5
# # eyeI = sparse.random(n_eq, n_eq, density=0.2)
# eyeI = sparse.eye(n_eq)
# A = sparse.vstack([A,
#                    sparse.hstack([eyeI, sparse.csc_matrix((n_eq, n - n_eq))])
#                    ]).tocsc()
# leq = np.random.rand(n_eq)
# ueq = leq
# lA = np.append(lA, leq)
# uA = np.append(uA, ueq)


# A = sparse.eye(n).tocsc()
# lA = -sp.rand(n)
# uA = lA + 0.1
# uA = np.inf * np.ones(n)

P = sparse.random(n, n, density=0.9, format='csc')
P = P.dot(P.T)
q = sp.randn(n)

# Scaling
# norm_q = np.linalg.norm(q)
# P /= norm_q
# q /= norm_q

# A *= 1e03
# lA *= 1e03
# uA *= 1e03

qp = mpbpy.QuadprogProblem(P, q, A, lA, uA)


osqp_opts = {'rho': 0.1,
             'auto_rho': False,
             'scaled_termination': False,
             'scaling_norm': -1,
             'early_terminate_interval': 1,
             'polish': False,
             'max_iter': 2500,
             'verbose': True,
             }

model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=lA, u=uA, **osqp_opts)
res_osqp = model.solve()

# osqp_opts['line_search'] = True

s = osqppurepy.OSQP()
s.setup(P, q, A, lA, uA, **osqp_opts)
res_purepy = s.solve()


# qp.solve(solver=GUROBI)
# res_purepy = qp.solve(solver=mpbpy.OSQP_PUREPY, **osqp_opts)
# res_osqp = qp.solve(solver=mpbpy.OSQP, **osqp_opts)


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
