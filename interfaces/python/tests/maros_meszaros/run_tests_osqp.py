from __future__ import print_function

import numpy as np
import numpy.linalg as la
import scipy.sparse as spa
import sys
import os


from utils.utils import load_maros_meszaros_problem
import mathprogbasepy as mpbpy

import osqppurepy as osqpurepy
import osqp


def constrain_scaling(s, min_val, max_val):
    s = np.minimum(np.maximum(s, min_val), max_val)
    return s


def scale_cost(problem):
    """
    Normalize cost by linear part of the cost.
    """
    norm_q = np.linalg.norm(problem.q)
    cost_scal = constrain_scaling(norm_q, 1e-03, 1e03)
    if norm_q < 1e-06:  # q is null!
        cost_scal = 1.
    else:
        cost_scal = norm_q

    problem.q /= cost_scal
    problem.P /= cost_scal


def scale_constraints(problem):
    """
    Scale constraints of the problem
    """
    m_constraints = len(problem.l)
    E = np.zeros(m_constraints)
    for i in range(m_constraints):
        abs_l = np.abs(problem.l[i])
        if np.isinf(abs_l) or abs_l > 1e10 or abs_l < 1e-06:
            abs_l = 1.
        else:
            abs_l = constrain_scaling(abs_l, 1e-03, 1e03)

        abs_u = np.abs(problem.u[i])
        if np.isinf(abs_u) or abs_u > 1e10 or abs_l < 1e-06:
            abs_u = 1.
        else:
            abs_u = constrain_scaling(abs_u, 1e-03, 1e03)

        # # Scale using maximum bound
        # max_abs_bnds = np.minimum(abs_l, abs_u)
        # E[i] = 1./max_abs_bnds

        # Scale using both bounds
        # E[i] = 1. / (abs_l * abs_u)

        # Exponentially scale bounds
        log_l = np.log(abs_l)
        log_u = np.log(abs_u)
        E[i] = np.exp((log_l + log_u)/2)

    # Select scaling
    # E = spa.diags(E)
    E = spa.diags(np.ones(m_constraints))

    # New constraints
    problem.l = E.dot(m.l)
    problem.u = E.dot(m.u)
    problem.A = E.dot(m.A).tocsc()


'''
Main script
'''

# Directory of problems
prob_dir = './mat'
lst_probs = os.listdir(prob_dir)

# Count number of problems
n_prob = len([name for name in lst_probs
             if os.path.isfile(prob_dir + "/" + name)])

# Problems index
p = 0

# Number unsolved problems
n_unsolved = 0

# Solve all Maroz Meszaros problems
for f in lst_probs:

    # if f[:-4] == 'QAFIRO':
    if f[:-4] == 'CVXQP1_S':
    # if f[:-4] == 'DUALC1':
    # if f[:-4] == 'CVXQP1_M':
    # if f[:-4] == 'AUG2DCQP':
    # if f[:-4] == 'BOYD1':
    # if f[:-4] == 'AUG2D':
    # if f[:-4] == 'AUG2DC':
    # if f[:-4] == 'CONT-101':
    # if f[:-4] == 'CONT-300':
    # if True:
    # if f[:-4] == 'QPCBOEI2':
    # if f[:-4] == 'AUG3D':
    # if f[:-4] == 'QSHIP04S':

        problem = load_maros_meszaros_problem(prob_dir + "/" + f)  # Load problem

        print("%3i) %s\t" % (p, f[:-4]), end='')

        # Scale cost
        # scale_cost(problem)

        # Scale constraints
        # scale_constraints(problem)

        settings = {'rho': 50.0,
                    'auto_rho': False,
                    'verbose': True,
                    'scaled_termination': True,
                    'polish': False,
                    'early_terminate_interval': 1}

        s = osqp.OSQP()
        s.setup(problem.P, problem.q, problem.A, problem.l, problem.u,
                **settings)
        res = s.solve()

        # Solve with purepy
        s = osqpurepy.OSQP()
        s.setup(problem.P, problem.q, problem.A, problem.l, problem.u,
                **settings)
        res_purepy = s.solve()

        p += 1

        if res.info.status_val == \
            s.constant('OSQP_MAX_ITER_REACHED'):
            n_unsolved += 1
            # import ipdb; ipdb.set_trace()

        print(res.info.status)


print('Number of solved problems %i/%i' % (n_prob - n_unsolved,
                                           n_prob))
