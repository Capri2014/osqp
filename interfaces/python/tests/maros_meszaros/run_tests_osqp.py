from __future__ import print_function

import numpy as np
import numpy.linalg as la
import scipy.sparse as spa
import sys
import os
import time
from multiprocessing import Pool, cpu_count
from itertools import repeat
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


def solve_problem(name, settings):
    """
    Solve single problem called name
    """

    problem = load_maros_meszaros_problem(prob_dir + "/" + name)  # Load prob

    # Scale cost
    # scale_cost(problem)

    # Scale constraints
    # scale_constraints(problem)

    # Solve with OSQP
    # s = osqp.OSQP()
    # s.setup(problem.P, problem.q, problem.A, problem.l, problem.u,
    #         **settings)
    # res = s.solve()

    # Solve with purepy
    s = osqpurepy.OSQP()
    s.setup(problem.P, problem.q, problem.A, problem.l, problem.u,
            **settings)
    res = s.solve()

    if res.info.status_val == \
            s.constant('OSQP_MAX_ITER_REACHED'):
            solved = False
    else:
        solved = True

    print("%s  \t\t%s" % (name, res.info.status))

    return solved, res.info.iter


'''
Main script
'''

# Directory of problems
prob_dir = './mat'
lst_probs = os.listdir(prob_dir)

# Count number of problems
n_prob = len([name for name in lst_probs
             if os.path.isfile(prob_dir + "/" + name)])
problem_names = [f[:-4] for f in lst_probs]

# List of interesting probs
# 'QAFIRO' or name == 'CVXQP1_S':
# 'QAFIRO':
# 'CVXQP1_S':
# 'DUALC1':
# 'PRIMAL4':
# 'CVXQP1_M':
# 'AUG2DCQP':
# 'BOYD1':
# 'AUG2D':
# 'AUG2DC':
# 'CONT-101':
# 'CONT-300':
# 'QPCBOEI2':
# 'AUG3D':
# 'QSHIP04S':

# Solve only few problems
problem_names = ['QAFIRO', 'CVXQP1_S', 'QSHIP04S', 'PRIMAL4']
# problem_names = ['CVXQP1_S']

# Problems index
p = 0

# Number unsolved problems
n_unsolved = 0

# OSQP Settings
settings = {'rho': 0.2,
            'auto_rho': False,
            'verbose': True,
            'scaled_termination': False,
            'polish': False,
            'scaling': True,
            'early_terminate_interval': 1}

parallel = False  # Execute script in parallel

# Results
results = []

start = time.perf_counter()


'''
Solve all Maros-Meszaros problems
'''

if parallel:
    # Parallel
    pool = Pool(processes=cpu_count())
    results = pool.starmap(solve_problem, zip(problem_names, repeat(settings)))
else:
    # Serial
    for name in problem_names:
        # Solve single problem
        results.append(solve_problem(name, settings))

end = time.perf_counter()
elapsed_time = end - start

zipped_results = list(zip(*results))
solved = list(zipped_results[0])
n_iter = list(zipped_results[1])
unsolved = np.invert(solved)

print('Number of solved problems %i/%i' % (n_prob - sum(unsolved),
                                           n_prob))
print("Time elapsed %.2e sec" % elapsed_time)
