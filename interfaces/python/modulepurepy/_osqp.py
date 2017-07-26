"""
OSQP Solver pure python implementation: low level module
"""
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
import pdb   # Debugger

# Solver Constants
OSQP_SOLVED = 1
OSQP_MAX_ITER_REACHED = -2
OSQP_PRIMAL_INFEASIBLE = -3
OSQP_DUAL_INFEASIBLE = -4
OSQP_UNSOLVED = -10

# Printing interval
PRINT_INTERVAL = 100

# OSQP Infinity
OSQP_INFTY = 1e+20

# OSQP Nan
OSQP_NAN = 1e+20  # Just as placeholder. Not real value

# Linear system solver options
SUITESPARSE_LDL = 0

# Scaling
SCALING_REG = 1e-08
#  MAX_SCALING = 1e06
#  MIN_SCALING = 1e-06


class workspace(object):
    """
    OSQP solver workspace

    Attributes
    ----------
    data                   - scaled QP problem
    info                   - solver information
    linsys_solver          - structure for linear system solution
    scaling                - scaling matrices
    settings               - settings structure
    solution               - solution structure


    Additional workspace variables
    -------------------------------
    first_run              - flag to indicate if it is the first run
    timer                  - saved time instant for timing purposes
    x                      - primal iterate
    x_prev                 - previous primal iterate
    xz_tilde               - x_tilde and z_tilde iterates stacked together
    y                      - dual iterate
    z                      - z iterate
    z_prev                 - previous z iterate

    Primal infeasibility related workspace variables
    -----------------------------------------
    delta_y                - difference of consecutive y
    Atdelta_y              - A' * delta_y

    Dual infeasibility related workspace variables
    -----------------------------------------
    delta_x                - difference of consecutive x
    Pdelta_x               - P * delta_x
    Adelta_x               - A * delta_x

    """


class problem(object):
    """
    QP problem of the form
        minimize	1/2 x' P x + q' x
        subject to	l <= A x <= u

    Attributes
    ----------
    P, q
    A, l, u
    """

    def __init__(self, dims, Pdata, Pindices, Pindptr, q,
                 Adata, Aindices, Aindptr,
                 l, u):
        # Set problem dimensions
        (self.n, self.m) = dims

        # Set problem data
        self.P = spspa.csc_matrix((Pdata, Pindices, Pindptr),
                                  shape=(self.n, self.n))
        self.q = q
        self.A = spspa.csc_matrix((Adata, Aindices, Aindptr),
                                  shape=(self.m, self.n))
        self.l = l if l is not None else -np.inf*np.ones(self.m)
        self.u = u if u is not None else np.inf*np.ones(self.m)

    def objval(self, x):
        # Compute quadratic objective value for the given x
        return .5 * np.dot(x, self.P.dot(x)) + np.dot(self.q, x)


class settings(object):
    """
    OSQP solver settings

    Attributes
    ----------
    -> These cannot be changed without running setup
    sigma    [1e-06]           - Regularization parameter for polish
    scaling  [True]            - Prescaling/Equilibration
    scaling_iter [15]          - Number of Steps for Scaling Method
    scaling_norm [2]           - Equilibration scaling norm

    -> These can be changed without running setup
    rho  [1.6]                 - Step in ADMM procedure
    max_iter [5000]                     - Maximum number of iterations
    eps_abs  [1e-05]                    - Absolute tolerance
    eps_rel  [1e-05]                    - Relative tolerance
    eps_prim_inf  [1e-06]                    - Primal infeasibility tolerance
    eps_dual_inf  [1e-06]                    - Dual infeasibility tolerance
    alpha [1.6]                         - Relaxation parameter
    line_search [True]                  - Line search acceleration
    delta [1.0]                         - Regularization parameter for polish
    verbose  [True]                     - Verbosity
    scaled_termination [False]             - Evalute scaled termination criteria
    early_terminate  [True]             - Evalute termination criteria
    early_terminate_interval  [25]    - Interval for evaluating termination criteria
    warm_start [False]                  - Reuse solution from previous solve
    polish  [True]                      - Solution polish
    pol_refine_iter  [3]                - Number of iterative refinement iterations
    auto_rho  [True]                    - Automatic rho computation
    """

    def __init__(self, **kwargs):

        # self.rho = kwargs.pop('rho', 0.1)
        self.sigma = kwargs.pop('sigma', 1e-06)
        self.scaling = kwargs.pop('scaling', True)
        self.scaling_iter = kwargs.pop('scaling_iter', 15)
        self.scaling_norm = kwargs.pop('scaling_norm', 2)
        self.max_iter = kwargs.pop('max_iter', 2500)
        self.eps_abs = kwargs.pop('eps_abs', 1e-3)
        self.eps_rel = kwargs.pop('eps_rel', 1e-3)
        self.eps_prim_inf = kwargs.pop('eps_prim_inf', 1e-4)
        self.eps_dual_inf = kwargs.pop('eps_dual_inf', 1e-4)
        self.alpha = kwargs.pop('alpha', 1.6)
        self.line_search = kwargs.pop('line_search', False)
        self.linsys_solver = kwargs.pop('linsys_solver', SUITESPARSE_LDL)
        self.delta = kwargs.pop('delta', 1e-6)
        self.verbose = kwargs.pop('verbose', True)
        self.scaled_termination = kwargs.pop('scaled_termination', False)
        self.early_terminate = kwargs.pop('early_terminate', True)
        self.early_terminate_interval = kwargs.pop('early_terminate_interval', 25)
        self.warm_start = kwargs.pop('warm_start', True)
        self.polish = kwargs.pop('polish', True)
        self.pol_refine_iter = kwargs.pop('pol_refine_iter', 3)
        self.diagonal_rho = kwargs.pop('diagonal_rho', False)
        self.update_rho = kwargs.pop('update_rho', False)
        self.auto_rho = kwargs.pop('auto_rho', False)
        self.rho_mid = kwargs.pop('rho', False)


class scaling(object):
    """
    Matrices for diagonal scaling

    Attributes
    ----------
    D     - matrix in R^{n \\times n}
    E     - matrix in R^{m \\times n}
    Dinv  - inverse of D
    Einv  - inverse of E
    """
    def __init__(self):
        self.D = None
        self.E = None
        self.Dinv = None
        self.Einv = None
        self.cost_scaling = None


class solution(object):
    """
    Solver solution vectors z, u
    """
    def __init__(self):
        self.x = None
        self.y = None


class info(object):
    """
    Solver information

    Attributes
    ----------
    iter            - number of iterations taken
    status          - status string, e.g. 'Solved'
    status_val      - status as c_int, defined in constants.h
    status_polish   - polish status: successful (1), not (0)
    obj_val         - primal objective
    pri_res         - norm of primal residual
    dua_res         - norm of dual residual
    setup_time      - time taken for setup phase (milliseconds)
    solve_time      - time taken for solve phase (milliseconds)
    polish_time     - time taken for polish phase (milliseconds)
    run_time        - total time  (milliseconds)
    """
    def __init__(self):
        self.iter = 0
        self.status_val = OSQP_UNSOLVED
        self.status_polish = 0
        self.polish_time = 0.0


class pol(object):
    """
    Polishing structure containing active constraints at the solution

    Attributes
    ----------
    ind_low         - indices of lower-active constraints
    ind_upp         - indices of upper-active constraints
    n_low           - number of lower-active constraints
    n_upp           - number of upper-active constraints
    Ared            - Part of A containing only active rows
    x               - polished x
    z               - polished z
    y_red           - polished y (part corresponding to active constraints)
    """
    def __init__(self):
        self.ind_low = None
        self.ind_upp = None
        self.n_low = None
        self.n_upp = None
        self.Ared = None
        self.x = None
        self.z = None
        self.y_red = None


class linsys_solver(object):
    """
    Linear systems solver
    """

    def __init__(self, work):
        """
        Initialize structure for KKT system solution
        """
        # Construct reduced KKT matrix
        rho_inv = work.settings.rho_inv
        KKT = spspa.vstack([
              spspa.hstack([work.data.P + work.settings.sigma *
                            spspa.eye(work.data.n), work.data.A.T]),
              spspa.hstack([work.data.A, -rho_inv])])

        # Initialize structure
        self.kkt_factor = spla.splu(KKT.tocsc())

    def solve(self, rhs):
        """
        Solve linear system with given factorization
        """
        return self.kkt_factor.solve(rhs)


class results(object):
    """
    Results structure

    Attributes
    ----------
    x           - primal solution
    y           - dual solution
    info        - info structure
    """
    def __init__(self, solution, info):
        self.x = solution.x
        self.y = solution.y
        self.info = info


class OSQP(object):
    """OSQP solver lower level interface
    Attributes
    ----------
    work    - workspace
    """
    def __init__(self):
        self._version = "0.1.2"

    @property
    def version(self):
        """Return solver version
        """
        return self._version

    def scale_data(self):
        """
        Perform symmetric diagonal scaling via equilibration
        """
        n = self.work.data.n
        m = self.work.data.m

        # Scale cost
        SCALE_COST_MIN = 1e-08
        q_norm = np.linalg.norm(self.work.data.q)
        q_norm = 1. if q_norm < SCALE_COST_MIN else q_norm
        P_avg_norm = np.mean(spspa.linalg.norm(self.work.data.P, axis=0))
        P_avg_norm = 1. if P_avg_norm < SCALE_COST_MIN else P_avg_norm
        A_avg_norm = np.mean(spspa.linalg.norm(self.work.data.A, axis=0))
        A_avg_norm = 1. if A_avg_norm < SCALE_COST_MIN else A_avg_norm
        cost_scaling = A_avg_norm/(q_norm + P_avg_norm)

        # Scale data (just to check)
        # self.work.data.P /= cost_scaling
        # self.work.data.q /= cost_scaling

        # # Scale constraints
        # for i in range(m):  # Range over all the constraints
        #     # TODO: Continue from here!

        # constraints_scaling  # TODO: Complete!

        scaling_norm = self.work.settings.scaling_norm
        scaling_norm = scaling_norm if scaling_norm == 1 or scaling_norm == 2 \
            else np.inf

        # Initialize scaling
        d = np.ones(n + m)
        d_temp = np.ones(n + m)

        # Define reduced KKT matrix to scale
        KKT = spspa.vstack([
              spspa.hstack([self.work.data.P, self.work.data.A.T]),
              spspa.hstack([self.work.data.A,
                            spspa.csc_matrix((m, m))])]).tocsc()

        # Iterate Scaling
        for i in range(self.work.settings.scaling_iter):

            # Ruiz equilibration
            for j in range(n + m):
                if scaling_norm != 2:
                    norm_col_j = spspa.linalg.norm(KKT[:, j],
                                                   scaling_norm)
                else:
                    # Scipy hasn't implemented that function yet!
                    norm_col_j = np.linalg.norm(KKT[:, j].todense(),
                                                scaling_norm)

                if norm_col_j > SCALING_REG:
                    d_temp[j] = 1./(np.sqrt(norm_col_j))

            S_temp = spspa.diags(d_temp)
            d = np.multiply(d, d_temp)
            KKT = S_temp.dot(KKT.dot(S_temp))

            #  # DEBUG: Check scaling
            #  D = spspa.diags(d[:n])
            #
            #  if m == 0:
            #      # spspa.diags() will throw an error if fed with an empty array
            #      E = spspa.csc_matrix((0, 0))
            #      A = \
            #          E.dot(self.work.data.A.dot(D)).todense()
            #      cond_A = 1.
            #  else:
            #      E = spspa.diags(d[n:])
            #      A = \
            #          E.dot(self.work.data.A.dot(D)).todense()
            #      cond_A = np.linalg.cond(A)
            #  cond_KKT = np.linalg.cond(KKT.todense())
            #  P = \
            #      D.dot(self.work.data.P.dot(D)).todense()
            #  cond_P = np.linalg.cond(P)
            #
            #  # Get rato between columns and rows
            #  n_plus_m = n + m
            #  max_norm_rows = 0.0
            #  min_norm_rows = np.inf
            #  for j in range(n_plus_m):
            #     norm_row_j = la.norm(np.asarray(KKT[j, :].todense()))
            #     max_norm_rows = np.maximum(norm_row_j,
            #                                max_norm_rows)
            #     min_norm_rows = np.minimum(norm_row_j,
            #                                min_norm_rows)
            #
            #  # Compute residuals
            #  res_rows = max_norm_rows / min_norm_rows
            #
            #  np.set_printoptions(suppress=True, linewidth=500, precision=3)
            #  print("\nIter %i" % i)
            #  print("cond(KKT) = %.4e" % cond_KKT)
            #  print("cond(P) = %.4e" % cond_P)
            #  print("cond(A) = %.4e" % cond_A)
            #  print("res_rows = %.4e / %.4e = %.4e" %
            #       (max_norm_rows, min_norm_rows, res_rows))
            #
            #
        # Obtain Scaler Matrices
        D = spspa.diags(d[:self.work.data.n])
        if m == 0:
            # spspa.diags() will throw an error if fed with an empty array
            E = spspa.csc_matrix((0, 0))
        else:
            E = spspa.diags(d[self.work.data.n:])

        # Scale problem Matrices
        P = D.dot(self.work.data.P.dot(D)).tocsc()
        A = E.dot(self.work.data.A.dot(D)).tocsc()
        q = D.dot(self.work.data.q)
        l = E.dot(self.work.data.l)
        u = E.dot(self.work.data.u)

        # Assign scaled problem
        self.work.data = problem((n, m), P.data, P.indices, P.indptr, q,
                                 A.data, A.indices, A.indptr, l, u)

        # Assign scaling matrices
        self.work.scaling = scaling()
        self.work.scaling.D = D
        self.work.scaling.Dinv = \
            spspa.diags(np.reciprocal(D.diagonal()))
        self.work.scaling.E = E
        if m == 0:
            self.work.scaling.Einv = E
        else:
            self.work.scaling.Einv = \
                spspa.diags(np.reciprocal(E.diagonal()))

    def compute_rho(self):
        """
        Automatically compute rho value
        """
        RHO_MIN = 1e-06
        RHO_MAX = 1e06
        RHO_TOL = 1e-04

        if self.work.settings.auto_rho:
            # Norm q
            norm_q = np.linalg.norm(self.work.data.q)
            norm_q = norm_q if norm_q > 1e-6 else 1.

            # Norm P
            # norm_P = np.mean(spspa.linalg.norm(self.work.data.P, axis=0))
            # norm_P = np.linalg.norm(self.work.data.P)
            norm_P = np.sum(self.work.data.P.diagonal())
            norm_P = norm_P if norm_P > 1e-6 else 1.

            # Norm A
            norm_A = np.sum(self.work.data.A.T.dot(self.work.data.A))

            RHO_MID = (norm_q + norm_P)/norm_A

            self.work.settings.rho_mid = RHO_MID
        else:
            RHO_MID = self.work.settings.rho_mid

        m = self.work.data.m
        rho_vec = np.zeros(m)

        # Define index of equality constraints
        ineq_idx = np.zeros(m, dtype=bool)

        # Get lower and upper bounds
        l = self.work.data.l
        u = self.work.data.u

        if not self.work.settings.diagonal_rho:
            rho_vec = RHO_MID * np.ones(m)
            ineq_idx = np.ones(m, dtype=bool)
        else:
            for i in range(m):
                if np.abs(l[i]) >= OSQP_INFTY*1e-06 and np.abs(u[i]) >= OSQP_INFTY*1e-06:
                    # Unconstrained
                    rho_vec[i] = RHO_MIN
                elif np.abs(u[i] - l[i]) >= OSQP_INFTY * 1e-06:
                    # One sided constraint
                    rho_vec[i] = RHO_MID
                    ineq_idx[i] = True
                elif np.abs(u[i] - l[i]) < RHO_TOL:
                    # Equality constraint
                    rho_vec[i] = RHO_MAX
                else:
                    # Range constraint
                    # rho_vec[i] = 1000. / (u[i] - l[i])
                    rho_vec[i] = RHO_MID
                    ineq_idx[i] = True

                # Constrain between maximum and minimum
                rho_vec[i] = np.maximum(np.minimum(rho_vec[i], RHO_MAX), RHO_MIN)

        self.work.settings.rho = spspa.diags(rho_vec)
        self.work.settings.rho_inv = spspa.diags(np.reciprocal(rho_vec))
        self.work.ineq_idx = ineq_idx

    def print_setup_header(self, data, settings):
        """Print solver header
        """
        print("-------------------------------------------------------")
        print("      OSQP v%s  -  Operator Splitting QP Solver" % \
            self.version)
        print("              Pure Python Implementation")
        print("     (c) Bartolomeo Stellato, Goran Banjac")
        print("   University of Oxford  -  Stanford University 2017")
        print("-------------------------------------------------------")

        print("Problem:  variables n = %d, constraints m = %d" % \
            (data.n, data.m))
        print("Settings: ", end='')
        if settings.linsys_solver == SUITESPARSE_LDL:
            print("linear system solver = SuiteSparse LDL")
        print("          eps_abs = %.2e, eps_rel = %.2e," % \
            (settings.eps_abs, settings.eps_rel))
        print("          eps_prim_inf = %.2e, eps_dual_inf = %.2e," % \
            (settings.eps_prim_inf, settings.eps_dual_inf))
        # print("          rho = %.2e " % settings.rho, end='')
        # if settings.auto_rho:
        #     print("(auto)")
        # else:
        #     print("")
        print("          sigma = %.2e, alpha = %.2e," % \
            (settings.sigma, settings.alpha))
        print("          max_iter = %d" % settings.max_iter)
        if settings.scaling:
            print("          scaling: on ", end='')
            if settings.scaling_norm != -1:
                print("(%d-norm), " % settings.scaling_norm, end='')
            else:
                print("(inf-norm), ", end='')
        else:
            print("          scaling: off, ", end='')
        if settings.scaled_termination:
            print("scaled_termination: on")
        else:
            print("scaled_termination: off")
        if settings.warm_start:
            print("          warm_start: on, ", end='')
        else:
            print("          warm_start: off, ", end='')
        if settings.polish:
            print("polish: on")
        else:
            print("polish: off")
        if settings.diagonal_rho:
            print("          diagonal rho")
        if settings.update_rho:
            print("          update rho")
        print("          rho_mid = %.3e" % settings.rho_mid)

        print("")

    def print_header(self):
        """
        Print header before the iterations
        """
        print("Iter    Obj  Val     Pri  Res     Dua  Res       Time")

    def update_status_string(self, status):
        if status == OSQP_SOLVED:
            return "Solved"
        elif status == OSQP_PRIMAL_INFEASIBLE:
            return "Primal infeasible"
        elif status == OSQP_UNSOLVED:
            return "Unsolved"
        elif status == OSQP_DUAL_INFEASIBLE:
            return "Dual infeasible"
        elif status == OSQP_MAX_ITER_REACHED:
            return "Maximum iterations reached"

    def cold_start(self):
        """
        Cold start optimization variables to zero
        """
        self.work.x = np.zeros(self.work.data.n)
        self.work.z = np.zeros(self.work.data.m)
        self.work.y = np.zeros(self.work.data.m)

    def update_xz_tilde(self, x, z, y):
        """
        First ADMM step: update xz_tilde
        """
        n = self.work.data.n
        m = self.work.data.m
        rho_inv = self.work.settings.rho_inv
        sigma = self.work.settings.sigma

        # Preallocate xz_tilde
        xz_tilde = np.zeros(n + m)

        # Compute rhs and store it in xz_tilde
        rhs = np.append(sigma * x - self.work.data.q,
                        z - rho_inv.dot(y))

        # Solve linear system
        sol = self.work.linsys_solver.solve(rhs)

        # Update z_tilde
        x_tilde = sol[:n]
        v = sol[n:]
        z_tilde = z + rho_inv.dot(v - y)

        # Return
        return x_tilde, z_tilde

    def update_x(self, x_tilde, x):
        """
        Update x variable in second ADMM step
        """
        alpha = self.work.settings.alpha

        x_new = alpha * x_tilde + (1. - alpha) * x

        return x_new

    def project(self, z):
        """
        Project z variable in set C (for now C = [l, u])
        """
        return np.minimum(np.maximum(z, self.work.data.l), self.work.data.u)

    def update_z(self, z_tilde, z, y):
        """
        Update z variable in second ADMM step
        """
        alpha = self.work.settings.alpha
        rho_inv = self.work.settings.rho_inv

        # new z
        z_new = alpha * z_tilde + (1. - alpha) * z + rho_inv.dot(y)

        return self.project(z_new)

    def update_y(self, y, z_tilde, z_new, z):
        """
        Third ADMM step: update dual variable y
        """
        rho = self.work.settings.rho
        alpha = self.work.settings.alpha

        # New y
        return y + rho.dot(alpha * z_tilde + (1. - alpha) * z - z_new)

    def compute_pri_res(self, polish, scaled_termination):
        """
        Compute primal residual ||Ax - z||
        """
        if self.work.data.m == 0:  # No constraints
            return 0.
        if polish:
            pri_res = np.maximum(self.work.data.l - self.work.pol.z, 0) + \
                np.maximum(self.work.pol.z - self.work.data.u, 0)
        else:
            pri_res = self.work.data.A.dot(self.work.x) - self.work.z

        if self.work.settings.scaling and not scaled_termination:
            pri_res = self.work.scaling.Einv.dot(pri_res)

        return la.norm(pri_res, np.inf)

    def compute_dua_res(self, polish, scaled_termination):
        """
        Compute dual residual ||Px + q + A'y||
        """
        if polish:
            dua_res = self.work.data.P.dot(self.work.pol.x) + \
                self.work.data.q +\
                self.work.pol.Ared.T.dot(self.work.pol.y_red)
        else:
            dua_res = self.work.data.P.dot(self.work.x) +\
                self.work.data.q +\
                self.work.data.A.T.dot(self.work.y)

        if self.work.settings.scaling and not scaled_termination:
            dua_res = self.work.scaling.Dinv.dot(dua_res)

        return la.norm(dua_res, np.inf)

    def is_primal_infeasible(self):
        """
        Check primal infeasibility
                ||A'*v||_2 = 0
        with v = delta_y/||delta_y||_2 given that following condition holds
            u'*(v)_{+} + l'*(v)_{-} < 0
        """

        # # DEBUG
        # if (self.work.info.iter % PRINT_INTERVAL == 0):
        #     print "\n\nValues with r_pri"
        #     r_pri = self.work.z - self.work.data.A.dot(self.work.x)
        #
        #     lhs = 0.
        #     for i in range(self.work.data.m):
        #         if self.work.data.u[i] < self.constant('OSQP_INFTY')*1e-03:
        #             lhs += self.work.data.u[i] * max(r_pri[i], 0)
        #
        #         if self.work.data.l[i] > -self.constant('OSQP_INFTY')*1e-03:
        #             lhs += self.work.data.l[i] * min(r_pri[i], 0)
        #     # lhs = self.work.data.u.dot(np.maximum(r_pri, 0)) + \
        #     #     self.work.data.l.dot(np.minimum(r_pri, 0))
        #     print("u' * (v)_{+} + l' * v_{-} = %6.2e" % (lhs))
        #     Atr_pri = self.work.data.A.T.dot(r_pri)
        #     print("||A'*v|| = %6.2e" % (la.norm(Atr_pri)))
        #

            # lhsp = self.work.data.u.dot(np.maximum(self.work.delta_y, 0))
            # lhsm = self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
            # print("Values with delta_y")
            # lhs = 0.
            # lhsp = 0.
            # lhsm = 0.
            # for i in range(self.work.data.m):
            #     if self.work.data.u[i] < self.constant('OSQP_INFTY')*1e-05:
            #         lhsp += self.work.data.u[i] * max(self.work.delta_y[i], 0)
            #
            #     if self.work.data.l[i] > -self.constant('OSQP_INFTY')*1e-03:
            #         lhsm += self.work.data.l[i] * min(self.work.delta_y[i], 0)
            # lhs = lhsp + lhsm
            # print("u' * (v_{+}) = %6.2e" % lhsp)
            # print("l' * (v_{-}) = %6.2e" % lhsm)
            # print("u' * (v_{+}) + l' * (v_{-}) = %6.2e" % (lhs))
            # self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
            # print("||A'*v|| = %6.2e" % (la.norm(self.work.Atdelta_y)))
            # pdb.set_trace()

        # Prevent 0 division
        # if la.norm(self.work.delta_y) > eps_prim_inf*eps_prim_inf:
        #     # self.work.delta_y /= la.norm(self.work.delta_y)
        #     # lhs = self.work.data.u.dot(np.maximum(self.work.delta_y, 0)) + \
        #     #     self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
        #     # if  lhs < -eps_prim_inf:
        #     #     self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
        #     #     return la.norm(self.work.Atdelta_y) < eps_prim_inf

        eps_prim_inf = self.work.settings.eps_prim_inf
        norm_delta_y = la.norm(self.work.delta_y, np.inf)
        if norm_delta_y > eps_prim_inf:
            self.work.delta_y /= norm_delta_y
            lhs = self.work.data.u.dot(np.maximum(self.work.delta_y, 0)) + \
                    self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
            if lhs < -eps_prim_inf:
                self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
                if self.work.settings.scaling and not self.work.settings.scaled_termination:
                        self.work.Atdelta_y = self.work.scaling.Dinv.dot(self.work.Atdelta_y)
                return la.norm(self.work.Atdelta_y, np.inf) < eps_prim_inf

        return False

    def is_dual_infeasible(self):
        """
        Check dual infeasibility
            ||P*v||_inf = 0
        with v = delta_x / ||delta_x||_inf given that the following
        conditions hold
            q'* v < 0 and
                        | 0     if l_i, u_i \in R
            (A * v)_i = { >= 0  if u_i = +inf
                        | <= 0  if l_i = -inf
        """
        eps_dual_inf = self.work.settings.eps_dual_inf
        norm_delta_x = la.norm(self.work.delta_x, np.inf)
        # Prevent 0 division
        if norm_delta_x > eps_dual_inf:
            # Normalize delta_x
            self.work.delta_x /= norm_delta_x

            # First check q'* delta_x < 0
            if self.work.data.q.dot(self.work.delta_x) < -eps_dual_inf:

                # Compute P * delta_x
                self.work.Pdelta_x = self.work.data.P.dot(self.work.delta_x)

                # Scale if necessary
                if self.work.settings.scaling and not self.work.settings.scaled_termination:
                    self.work.Pdelta_x = self.work.scaling.Dinv.dot(self.work.Pdelta_x)

                # Check if ||P * delta_x|| = 0
                if la.norm(self.work.Pdelta_x, np.inf) < eps_dual_inf:

                    # Compute A * delta_x
                    self.work.Adelta_x = self.work.data.A.dot(
                        self.work.delta_x)

                    # Scale if necessary
                    if self.work.settings.scaling and not self.work.settings.scaled_termination:
                        self.work.Adelta_x = self.work.scaling.Einv.dot(self.work.Adelta_x)

                    for i in range(self.work.data.m):
                        # De Morgan's Law applied to negate
                        # conditions on A * delta_x
                        if ((self.work.data.u[i] < OSQP_INFTY*1e-06) and
                            (self.work.Adelta_x[i] > eps_dual_inf)) or \
                            ((self.work.data.l[i] > -OSQP_INFTY*1e-06) and
                             (self.work.Adelta_x[i] < -eps_dual_inf)):

                            # At least one condition not satisfied
                            return False

                    # All conditions passed -> dual infeasible
                    return True

        # No all checks managed to pass. Problem not dual infeasible
        return False

    def update_info(self, iter, polish):
        """
        Update information at iterations
        """
        if polish == 1:
            self.work.pol.obj_val = self.work.data.objval(self.work.pol.x)
            self.work.pol.pri_res = self.compute_pri_res(1, self.work.settings.scaled_termination)
            self.work.pol.dua_res = self.compute_dua_res(1, self.work.settings.scaled_termination)
        else:
            self.work.info.iter = iter
            self.work.info.obj_val = self.work.data.objval(self.work.x)
            self.work.info.pri_res = self.compute_pri_res(0, self.work.settings.scaled_termination)
            self.work.info.dua_res = self.compute_dua_res(0, self.work.settings.scaled_termination)
            self.work.info.solve_time = time.time() - self.work.timer

    def print_summary(self):
        """
        Print status summary at each ADMM iteration
        """
        print("%4i %12.4e %12.4e %12.4e %9.2fs" % \
            (self.work.info.iter,
             self.work.info.obj_val,
             self.work.info.pri_res,
             self.work.info.dua_res,
             self.work.info.setup_time + self.work.info.solve_time))

    def print_polish(self):
        """
        Print polish information
        """
        print("PLSH %12.4e %12.4e %12.4e %9.2fs" % \
            (self.work.info.obj_val,
             self.work.info.pri_res,
             self.work.info.dua_res,
             self.work.info.setup_time + self.work.info.solve_time +
             self.work.info.polish_time))

    def check_termination(self):
        """
        Check residuals for algorithm convergence and update solver status
        """
        pri_check = 0
        dua_check = 0
        prim_inf_check = 0
        dual_inf_check = 0

        eps_abs = self.work.settings.eps_abs
        eps_rel = self.work.settings.eps_rel

        if self.work.data.m == 0:  # No constraints -> always  primal feasible
            pri_check = 1
        else:
            # Compute primal tolerance
            if self.work.settings.scaling and not self.work.settings.scaled_termination:
                Einv = self.work.scaling.Einv
                max_rel_eps = np.max([
                    la.norm(Einv.dot(self.work.data.A.dot(self.work.x)), np.inf),
                    la.norm(Einv.dot(self.work.z), np.inf)])
            else:
                max_rel_eps = np.max([
                    la.norm(self.work.data.A.dot(self.work.x), np.inf),
                    la.norm(self.work.z, np.inf)])

            eps_pri = eps_abs + eps_rel * max_rel_eps

            if self.work.info.pri_res < eps_pri:
                pri_check = 1
            else:
                # Check infeasibility
                prim_inf_check = self.is_primal_infeasible()

        # Compute dual tolerance

        if self.work.settings.scaling and not self.work.settings.scaled_termination:
            Dinv = self.work.scaling.Dinv
            max_rel_eps = np.max([
                la.norm(Dinv.dot(self.work.data.A.T.dot(self.work.y)), np.inf),
                la.norm(Dinv.dot(self.work.data.P.dot(self.work.x)), np.inf),
                la.norm(Dinv.dot(self.work.data.q), np.inf)])
        else:
            max_rel_eps = np.max([
                la.norm(self.work.data.A.T.dot(self.work.y), np.inf),
                la.norm(self.work.data.P.dot(self.work.x), np.inf),
                la.norm(self.work.data.q, np.inf)])

        eps_dua = eps_abs + eps_rel * max_rel_eps

        if self.work.info.dua_res < eps_dua:
            dua_check = 1
        else:
            # Check dual infeasibility
            dual_inf_check = self.is_dual_infeasible()

        # Compare residuals and determine solver status
        if pri_check & dua_check:
            self.work.info.status_val = OSQP_SOLVED
            return 1
        elif prim_inf_check:
            self.work.info.status_val = OSQP_PRIMAL_INFEASIBLE
            self.work.info.obj_val = OSQP_INFTY
            return 1
        elif dual_inf_check:
            self.work.info.status_val = OSQP_DUAL_INFEASIBLE
            self.work.info.obj_val = -OSQP_INFTY
            return 1

    def print_footer(self):
        """
        Print footer at the end of the optimization
        """
        print("")  # Add space after iterations
        print("Status: %s" % self.work.info.status)
        if self.work.settings.polish and \
                self.work.info.status_val == OSQP_SOLVED:
                    if self.work.info.status_polish == 1:
                        print("Solution polish: Successful")
                    elif self.work.info.status_polish == -1:
                        print("Solution polish: Unsuccessful")
        print("Number of iterations: %d" % self.work.info.iter)
        if self.work.info.status_val == OSQP_SOLVED:
            print("Optimal objective: %.4f" % self.work.info.obj_val)
        if self.work.info.run_time > 1e-03:
            print("Run time: %.3fs" % (self.work.info.run_time))
        else:
            print("Run time: %.3fms" % (1e03*self.work.info.run_time))

        print("")  # Print last space

    def store_solution(self):
        """
        Store current primal and dual solution in solution structure
        """

        if (self.work.info.status_val is not OSQP_PRIMAL_INFEASIBLE) and \
                (self.work.info.status_val is not OSQP_DUAL_INFEASIBLE):
            self.work.solution.x = self.work.x
            self.work.solution.y = self.work.y

            # Unscale solution
            if self.work.settings.scaling:
                self.work.solution.x = \
                    self.work.scaling.D.dot(self.work.solution.x)
                self.work.solution.y = \
                    self.work.scaling.E.dot(self.work.solution.y)
        else:
            self.work.solution.x = np.array([None] * self.work.data.n)
            self.work.solution.y = np.array([None] * self.work.data.m)

    #
    #   Main Solver API
    #

    def setup(self, dims, Pdata, Pindices, Pindptr, q,
              Adata, Aindices, Aindptr,
              l, u, **stgs):
        """
        Perform OSQP solver setup QP problem of the form
            minimize	1/2 x' P x + q' x
            subject to	l <= A x <= u

        """
        (n, m) = dims
        self.work = workspace()

        # Start timer
        self.work.timer = time.time()

        # Unscaled problem data
        self.work.data = problem((n, m), Pdata, Pindices, Pindptr, q,
                                 Adata, Aindices, Aindptr,
                                 l, u)

        # Initialize workspace variables
        self.work.x = np.zeros(n)
        self.work.z = np.zeros(m)
        self.work.xz_tilde = np.zeros(n + m)
        self.work.x_prev = np.zeros(n)
        self.work.z_prev = np.zeros(m)
        self.work.y_prev = np.zeros(m)
        self.work.y = np.zeros(m)
        self.work.delta_y = np.zeros(m)   # Delta_y for primal infeasibility
        self.work.delta_x = np.zeros(n)   # Delta_x for dual infeasibility

        # Flag indicating first run
        self.work.first_run = 1

        # Settings
        self.work.settings = settings(**stgs)

        # Scale problem
        if self.work.settings.scaling:
            self.scale_data()

        # Compute in case
        self.compute_rho()

        # Factorize KKT
        self.work.linsys_solver = linsys_solver(self.work)

        # Solution
        self.work.solution = solution()

        # Info
        self.work.info = info()

        # Polishing structure
        self.work.pol = pol()

        # End timer
        self.work.info.setup_time = time.time() - self.work.timer

        # Print setup header
        if self.work.settings.verbose:
            self.print_setup_header(self.work.data, self.work.settings)

    '''
    Define methods related to operator T
    '''

    def q_from_xzy(self, x, z, y):
        """
        Obtain q from x, z, y vectors
        """
        return np.append(x, z + self.work.settings.rho_inv.dot(y))

    def xzy_from_q(self, q):
        """
        Obtain x, z, y from q
        """
        n = self.work.data.n
        rho = self.work.settings.rho

        x = q[:n]
        v = q[n:]
        z = self.project(v)
        y = rho.dot(v - z)

        # Return vectors
        return x, z, y

    def operator_T(self, q):
        """
        Perform operator T
        """
        # Get x, z, y from q
        x, z, y = self.xzy_from_q(q)

        # Admm steps
        # First step: update \tilde{x} and \tilde{z}
        x_tilde, z_tilde = self.update_xz_tilde(x, z, y)

        # Second step: update x and z
        x_new = self.update_x(x_tilde, x)
        z_new = self.update_z(z_tilde, z, y)

        # Third step: update y
        y_new = self.update_y(y, z_tilde, z_new, z)

        # Get q_new from x_new, z_new, y_new
        q_new = self.q_from_xzy(x_new, z_new, y_new)

        return q_new

    def single_admm_step(self):
        """
        Run single ADMM step
        """

        # Update x_prev_prev, z_prev_prev, y_prev_prev
        self.work.x_prev_prev = np.copy(self.work.x_prev)
        self.work.z_prev_prev = np.copy(self.work.z_prev)
        self.work.y_prev_prev = np.copy(self.work.y_prev)

        # Update x_prev, z_prev, y_prev
        self.work.x_prev = np.copy(self.work.x)
        self.work.z_prev = np.copy(self.work.z)
        self.work.y_prev = np.copy(self.work.y)

        # Get iterate q
        q = self.q_from_xzy(self.work.x, self.work.z, self.work.y)

        # Perform one operator iteration
        q_next = self.operator_T(q)

        '''
        Convert q back to x, z, y
        '''
        # Get new x, y, z
        self.work.x, self.work.z, self.work.y = self.xzy_from_q(q_next)

        # Get deltas
        self.work.delta_x = self.work.x - self.work.x_prev
        self.work.delta_y = self.work.y - self.work.y_prev


    def alpha_acceleration(self):
        """
        Perform alpha acceleration using the operator q^{k+1} = T q^{k}

        where q^{k} = (x^{k}, z^{k} + y^{k}/rho)
        """
        ALPHA_MAX = 1000
        ALPHA_RED = 0.7
        EPS_ACCELERATION = 1e-05
        EPS_ACTIVATION_ACCELERATION = 1e-05

        # Get current vectors
        x_prev_prev = self.work.x_prev_prev
        y_prev_prev = self.work.y_prev_prev
        z_prev_prev = self.work.z_prev_prev
        x_prev = self.work.x_prev
        y_prev = self.work.y_prev
        z_prev = self.work.z_prev
        x = self.work.x
        y = self.work.y
        z = self.work.z

        # Compute q_prev, q, q_next
        q_prev = self.q_from_xzy(x_prev_prev, z_prev_prev, y_prev_prev)
        q = self.q_from_xzy(x_prev, z_prev, y_prev)
        q_next = self.q_from_xzy(x, z, y)

        # Compute current fixed-point residual (acceleration direction)
        r = q_next - q

        # Check if we need to accelerate or not
        delta_q = q - q_prev
        delta_q_next = q_next - q
        if la.norm(delta_q) > 1e-09 and la.norm(delta_q_next) > 1e-09:
            cos_angle_delta_q = delta_q_next.dot(delta_q) / \
                (la.norm(delta_q) * la.norm(delta_q_next))
            # print("cos_angle_delta_q = %.4f" % cos_angle_delta_q)
        else:
            cos_angle_delta_q = 0.0

        if cos_angle_delta_q > (1 - EPS_ACTIVATION_ACCELERATION):
            # Colinear vectors -> perform acceleration!

            if self.work.settings.verbose:
                print("Perform acceleration   ", end='')

            # Compute next residual (apply operator T again)
            q_next_next_nom = self.operator_T(q_next)
            r_next_nom = q_next_next_nom - q_next

            # # Search for alpha
            # alpha_prev = self.work.settings.alpha
            # alpha = (1. + np.sqrt(1. + 4.*(alpha_prev ** 2)))/2.
            # q_temp_prev = q_next
            # while alpha < ALPHA_MAX:

            alpha = ALPHA_MAX
            while alpha > self.work.settings.alpha:

                # print("alpha = %.4f" % alpha)
                q_temp = q + alpha * r

                # Check new residual
                q_next_next = self.operator_T(q_temp)
                r_next = q_next_next - q_temp

                # print("||r^{k+1}|| = %.4f      " % la.norm(r_next), end='')
                # print("||\\bar{r}^{k+1}|| = %.4f     " %
                    #   la.norm(r_next_nom), end='')
                condition_acc = la.norm(r_next) < (1. - EPS_ACCELERATION) * \
                    la.norm(r_next_nom)
                # print("||r^{k+1}|| < ||\\bar{r}^{k+1}|| : %r" % condition_acc)

                if condition_acc:
                    if self.work.settings.verbose:
                        print("better alpha found! alpha = %.4f" % alpha)
                    q_next = q_temp

                    # Store norm of q for plotting
                    self.work.norm_delta_q.pop()   # Remove last element
                    self.work.norm_delta_q.append(la.norm(q_next - q))
                    break

                # Increase alpha
                alpha = ALPHA_RED * alpha

                # if not condition_acc:
                #     if alpha > self.work.settings.alpha:
                #         print("better alpha found! alpha = %.4f" % alpha_prev)
                #         import ipdb; ipdb.set_trace()
                #     q_next = q_temp_prev
                #     break
                #
                # q_temp_prev = np.copy(q_temp)
                # alpha_prev = np.copy(alpha)
                # alpha = (1. + np.sqrt(1. + 4. * (alpha_prev ** 2))) / 2.

                # alpha = ALPHA_RED * alpha

            if alpha < self.work.settings.alpha and self.work.settings.verbose:
                print("no better alpha found!")
            # import ipdb; ipdb.set_trace()

            # Get new x, y, z
            self.work.x, self.work.z, self.work.y = self.xzy_from_q(q_next)

            # Get deltas
            self.work.delta_x = self.work.x - self.work.x_prev
            self.work.delta_y = self.work.y - self.work.y_prev

    def store_plotting_vars(self):
        """
        Store variables to plot
        """

        # Get variables q, q_prev, q_next
        q_prev = self.q_from_xzy(self.work.x_prev_prev,
                                 self.work.z_prev_prev,
                                 self.work.y_prev_prev)
        q = self.q_from_xzy(self.work.x_prev,
                            self.work.z_prev,
                            self.work.y_prev)
        q_next = self.q_from_xzy(self.work.x,
                                 self.work.z,
                                 self.work.y)

        # Store norm of q for plotting
        self.work.norm_delta_q.append(la.norm(q_next - q))

        # Store cos angle delta_qk and delta_qk+1
        delta_q = q - q_prev
        delta_q_next = q_next - q
        cos_angle_delta_q = 0.0
        if la.norm(delta_q) > 1e-09 and la.norm(delta_q_next) > 1e-09:
            cos_angle_delta_q = delta_q_next.dot(delta_q) / \
                (la.norm(delta_q) * la.norm(delta_q_next))
        self.work.cos_angle_delta_q.append(cos_angle_delta_q)

        # Store diff_delta_q
        self.work.diff_delta_q.append(la.norm(delta_q_next) -
                                      la.norm(delta_q))
        self.work.ratio_delta_q.append(la.norm(delta_q_next) /
                                       (la.norm(delta_q) + 1e-08))

        # Compute primal dual residual ratio
        pri_res = self.compute_pri_res(0, self.work.settings.scaled_termination)
        dua_res = self.compute_dua_res(0, self.work.settings.scaled_termination)
        self.work.residuals_ratio.append(pri_res/dua_res)

    def change_rho(self):
        """
        Check if rho has to be changed and in case change it
        """

        CHANGE_RHO_TOL = 1e-03

        # Get variables q, q_prev, q_next
        q_prev = self.q_from_xzy(self.work.x_prev_prev,
                                 self.work.z_prev_prev,
                                 self.work.y_prev_prev)
        q = self.q_from_xzy(self.work.x_prev,
                            self.work.z_prev,
                            self.work.y_prev)
        q_next = self.q_from_xzy(self.work.x,
                                 self.work.z,
                                 self.work.y)
        # Check cos angle delta_qk and delta_qk+1
        delta_q = q - q_prev
        delta_q_next = q_next - q
        cos_angle_delta_q = 0.0
        if la.norm(delta_q) > 1e-09 and la.norm(delta_q_next) > 1e-09:
            cos_angle_delta_q = delta_q_next.dot(delta_q) / \
                (la.norm(delta_q) * la.norm(delta_q_next))

        # print("norm diff delta = %.2e" % la.norm(la.norm(delta_q_next) - la.norm(delta_q)))
        # if la.norm(la.norm(delta_q_next) - la.norm(delta_q)) < \
        #         CHANGE_RHO_TOL:
            # if cos_angle_delta_q > 1 - CHANGE_RHO_TOL:
        if la.norm(la.norm(delta_q_next) /
                   (la.norm(delta_q) + 1e-08) - 1) < CHANGE_RHO_TOL or \
                cos_angle_delta_q > 1 - CHANGE_RHO_TOL:
            pri_res = self.compute_pri_res(0, True)  # Scaled residual
            dua_res = self.compute_dua_res(0, True)  # Scaled residual

            # Compute residuals ratio
            res_ratio = pri_res / dua_res

            # Change rho only if the residual ratio is big enough
            if res_ratio > 10:
                # Get current mid_rho value
                ineq_idx = self.work.ineq_idx
                m = self.work.data.m
                rho_diag = self.work.settings.rho.diagonal()
                if any(ineq_idx):
                    cur_mid_rho = rho_diag[np.nonzero(ineq_idx)[0][0]]

                    # Compute new rho
                    new_rho_mid = cur_mid_rho * np.sqrt(res_ratio)

                    # # Print
                    if self.work.settings.verbose:
                        print("New rho_mid = %.2e" % new_rho_mid)
                    # print("Updated rho!")

                    # Construct new rho_vec
                    new_rho_vec = np.zeros(m)
                    for i in range(m):
                        if ineq_idx[i]:
                            new_rho_vec[i] = new_rho_mid
                        else:
                            new_rho_vec[i] = rho_diag[i]

                    # Construct new rho matrix
                    new_rho = spspa.diags(new_rho_vec)

                    # Update rho
                    self.update_rho(new_rho)
                    # print("rho = ", end=''); print(new_rho_vec)

    def generate_plots(self):
        import matplotlib.pylab as plt

        # # Plot norm of q
        # fig = plt.figure(1)
        # ax = fig.add_subplot(311)
        # ax.set_ylabel(r'$\|q_{k+1} - q_{k}\|$')
        # plt.semilogy(self.work.norm_delta_q)
        # # ax.set_xlim([0, self.work.settings.max_iter])
        # plt.grid()
        # plt.tight_layout()
        # plt.show(block=False)

        # # Plot diff delta_q delta_qprev
        # fig = plt.figure(1)
        # ax = fig.add_subplot(311)
        # ax.set_ylabel(r'$\|\delta_{q_{k+1}}\| - \|\delta_{q_{k}})\|$')
        # plt.plot(self.work.diff_delta_q)
        # # ax.set_xlim([0, self.work.settings.max_iter])
        # plt.grid()
        # plt.tight_layout()
        # plt.show(block=False)

        # Plot ratio delta_q delta_qprev
        fig = plt.figure(1)
        ax = fig.add_subplot(311)
        ax.set_ylabel(r'$\|\delta_{q_{k+1}}\| / \|\delta_{q_{k}})\|$')
        plt.plot(self.work.ratio_delta_q)
        ax.set_ylim([0., 2.])
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)

        # Plot cos_angle_delta_q
        # fig = plt.figure(2)
        ax = fig.add_subplot(312)
        ax.set_ylabel(r'$\|\cos(\delta_{q_{k+1}}, \delta_{q_{k}})\|$')
        plt.plot(self.work.cos_angle_delta_q)
        # ax.set_xlim([0, self.work.settings.max_iter])
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)


        # Plot residual ratio
        # fig = plt.figure(3)
        ax = fig.add_subplot(313)
        ax.set_ylabel(r'$\|r_{pri} / r_{dua}\|$')
        plt.semilogy(self.work.residuals_ratio)
        # ax.set_xlim([0, self.work.settings.max_iter])
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)

    def solve(self):
        """
        Solve QP problem using OSQP
        """
        # Dimensions for easier notation
        m = self.work.data.m
        n = self.work.data.n

        # Start timer
        self.work.timer = time.time()

        # Print header
        if self.work.settings.verbose:
            self.print_header()

        # Cold start if not warm start
        if not self.work.settings.warm_start:
            self.cold_start()

        # Plotting
        self.work.norm_delta_q = []
        self.work.cos_angle_delta_q = []
        self.work.diff_delta_q = []
        self.work.ratio_delta_q = []
        self.work.residuals_ratio = []

        # ADMM algorithm
        for iter in range(1, self.work.settings.max_iter + 1):
            # Run single admm step
            self.single_admm_step()

            if self.work.settings.line_search:
                # Perform acceleration (if necessary)
                self.alpha_acceleration()

            self.store_plotting_vars()

            # Update rho?
            if self.work.settings.update_rho:
                # Rho update based on the residuals
                self.change_rho()

                # Update using bouncing values
                # if iter % 2 == 0:
                #     new_rho = spspa.diags(1e3 * np.ones(m))
                # else:
                #     new_rho = spspa.diags(1e-3 * np.ones(m))
                # self.update_rho(new_rho)

            # Check algorithm termination
            if self.work.settings.early_terminate:
                # Update info
                self.update_info(iter, 0)

                # Print summary
                if (self.work.settings.verbose) & \
                        ((iter % PRINT_INTERVAL == 0) | (iter == 1)):
                    self.print_summary()

                # Break if converged
                if self.check_termination():
                    break

        if not self.work.settings.early_terminate:
            # Update info
            self.update_info(self.work.settings.max_iter, 0)

            # Print summary
            if (self.work.settings.verbose):
                self.print_summary()

            # Break if converged
            self.check_termination()

        # Print summary for last iteration
        if (self.work.settings.verbose) & (iter % PRINT_INTERVAL != 0):
            self.print_summary()

        # If max iterations reached, update status accordingly
        if iter == self.work.settings.max_iter:
            self.work.info.status_val = OSQP_MAX_ITER_REACHED

        # Update status string
        self.work.info.status = \
            self.update_status_string(self.work.info.status_val)

        # Update solve time
        self.work.info.solve_time = time.time() - self.work.timer

        # Solution polish
        if self.work.settings.polish and \
                self.work.info.status_val == OSQP_SOLVED:
                    self.polish()

        # Update total times
        if self.work.first_run:
            self.work.info.run_time = self.work.info.setup_time + \
                self.work.info.solve_time + self.work.info.polish_time
        else:
            self.work.info.run_time = self.work.info.solve_time + \
                                      self.work.info.polish_time

        # Print footer
        if self.work.settings.verbose:
            self.print_footer()

        # Store solution
        self.store_solution()

        # Eliminate first run flag
        if self.work.first_run:
            self.work.first_run = 0

        '''
        Plotting
        '''
        if self.work.settings.verbose:
            self.generate_plots()

        # Store results structure
        return results(self.work.solution, self.work.info)

    #
    #   Auxiliary API Functions
    #

    def update_lin_cost(self, q_new):
        """
        Update linear cost without requiring factorization
        """
        # Copy cost vector
        self.work.data.q = np.copy(q_new)

        # Scaling
        if self.work.settings.scaling:
            self.work.data.q = self.work.scaling.D.dot(self.work.data.q)

    def update_bounds(self, l_new, u_new):
        """
        Update counstraint bounds without requiring factorization
        """

        # Check if bounds are correct
        if not np.greater_equal(u_new, l_new).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

        # Update vectors
        self.work.data.l = np.copy(l_new)
        self.work.data.u = np.copy(u_new)

        # Scale vectors
        if self.work.settings.scaling:
            self.work.data.l = self.work.scaling.E.dot(self.work.data.l)
            self.work.data.u = self.work.scaling.E.dot(self.work.data.u)

    def update_lower_bound(self, l_new):
        """
        Update lower bound without requiring factorization
        """
        # Update lower bound
        self.work.data.l = l_new

        # Scale vector
        if self.work.settings.scaling:
            self.work.data.l = self.work.scaling.E.dot(self.work.data.l)

        # Check values
        if not np.greater(self.work.data.u, self.work.data.l).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

    def update_upper_bound(self, u_new):
        """
        Update upper bound without requiring factorization
        """
        # Update upper bound
        self.work.data.u = u_new

        # Scale vector
        if self.work.settings.scaling:
            self.work.data.u = self.work.scaling.E.dot(self.work.data.u)

        # Check values
        if not np.greater(self.work.data.u, self.work.data.l).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

    def warm_start(self, x, y):
        """
        Warm start primal and dual variables
        """
        # Update warm_start setting to true
        self.work.settings.warm_start = True

        # Copy primal and dual variables into the iterates
        self.work.x = x
        self.work.y = y

        # Scale iterates
        self.work.x = self.work.scaling.Dinv.dot(self.work.x)
        self.work.y = self.work.scaling.Einv.dot(self.work.y)

        # Update z iterate as well
        self.work.z = self.work.data.A.dot(self.work.x)


    def warm_start_x(self, x):
        """
        Warm start primal variable
        """
        # Update warm_start setting to true
        self.work.settings.warm_start = True

        # Copy primal and dual variables into the iterates
        self.work.x = x

        # Scale iterates
        self.work.x = self.work.scaling.Dinv.dot(self.work.x)

        # Update z iterate as well
        self.work.z = self.work.data.A.dot(self.work.x)

        # Cold start y
        self.work.y = np.zeros(self.work.data.m)

    def warm_start_y(self, y):
        """
        Warm start dual variable
        """
        # Update warm_start setting to true
        self.work.settings.warm_start = True

        # Copy primal and dual variables into the iterates
        self.work.y = y

        # Scale iterates
        self.work.y = self.work.scaling.Einv.dot(self.work.y)

        # Cold start x and z
        self.work.x = np.zeros(self.work.data.n)
        self.work.z = np.zeros(self.work.data.m)


    #
    #   Update Problem Settings
    #
    def update_max_iter(self, max_iter_new):
        """
        Update maximum number of iterations
        """
        # Check that maxiter is positive
        if max_iter_new <= 0:
            raise ValueError("max_iter must be positive")

        # Update max_iter
        self.work.settings.max_iter = max_iter_new

    def update_eps_abs(self, eps_abs_new):
        """
        Update absolute tolerance
        """
        if eps_abs_new <= 0:
            raise ValueError("eps_abs must be positive")

        self.work.settings.eps_abs = eps_abs_new

    def update_eps_rel(self, eps_rel_new):
        """
        Update relative tolerance
        """
        if eps_rel_new <= 0:
            raise ValueError("eps_rel must be positive")

        self.work.settings.eps_rel = eps_rel_new

    def update_rho(self, rho_new):
        """
        Update set-size parameter rho
        """
        if any(rho_new.diagonal() <= 0):
            raise ValueError("rho must be positive")

        # Update rho
        self.work.settings.rho = rho_new
        self.work.settings.rho_inv = spspa.diags(
            np.reciprocal(rho_new.diagonal()))

        # Factorize KKT
        self.work.linsys_solver = linsys_solver(self.work)

    def update_alpha(self, alpha_new):
        """
        Update relaxation parameter alpga
        """
        if not (alpha_new >= 0 | alpha_new <= 2):
            raise ValueError("alpha must be between 0 and 2")

        self.work.settings.alpha = alpha_new

    def update_delta(self, delta_new):
        """
        Update delta parameter for polish
        """
        if delta_new <= 0:
            raise ValueError("delta must be positive")

        self.work.settings.delta = delta_new

    def update_polish(self, polish_new):
        """
        Update polish parameter
        """
        if (polish_new is not True) & (polish_new is not False):
            raise ValueError("polish should be either True or False")

        self.work.settings.polish = polish_new
        self.work.info.polish_time = 0.0

    def update_pol_refine_iter(self, pol_refine_iter_new):
        """
        Update number iterative refinement iterations in polish
        """
        if pol_refine_iter_new < 0:
            raise ValueError("pol_refine_iter must be nonnegative")

        self.work.settings.pol_refine_iter = pol_refine_iter_new

    def update_verbose(self, verbose_new):
        """
        Update verbose parameter
        """
        if (verbose_new is not True) & (verbose_new is not False):
            raise ValueError("verbose should be either True or False")

        self.work.settings.verbose = verbose_new

    def update_scaled_termination(self, scaled_termination_new):
        """
        Update scaled_termination parameter
        """
        if (scaled_termination_new is not True) & (scaled_termination_new is not False):
            raise ValueError("scaled_termination should be either True or False")

        self.work.settings.scaled_termination = scaled_termination_new

    def update_early_terminate(self, early_terminate_new):
        """
        Update early_terminate parameter
        """
        if (early_terminate_new is not True) & (early_terminate_new is not False):
            raise ValueError("early_terminate should be either True or False")

        self.work.settings.early_terminate = early_terminate_new

    def update_early_terminate_interval(self, early_terminate_interval_new):
        """
        Update early_terminate_interval parameter
        """
        if (early_terminate_interval_new is not True) & (early_terminate_interval_new is not False):
            raise ValueError("early_terminate_interval should be either True or False")

        self.work.settings.early_terminate_interval = early_terminate_interval_new

    def update_warm_start(self, warm_start_new):
        """
        Update warm_start parameter
        """
        if (warm_start_new is not True) & (warm_start_new is not False):
            raise ValueError("warm_start should be either True or False")

        self.work.settings.warm_start = warm_start_new

    def constant(self, constant_name):
        """
        Return solver constant
        """
        if constant_name == "OSQP_INFTY":
            return OSQP_INFTY
        if constant_name == "OSQP_NAN":
            return OSQP_NAN
        if constant_name == "OSQP_SOLVED":
            return OSQP_SOLVED
        if constant_name == "OSQP_UNSOLVED":
            return OSQP_UNSOLVED
        if constant_name == "OSQP_PRIMAL_INFEASIBLE":
            return OSQP_PRIMAL_INFEASIBLE
        if constant_name == "OSQP_DUAL_INFEASIBLE":
            return OSQP_DUAL_INFEASIBLE
        if constant_name == "OSQP_MAX_ITER_REACHED":
            return OSQP_MAX_ITER_REACHED

        raise ValueError('Constant not recognized!')


    def iter_refin(self, KKT_factor, z, b):
        """
        Iterative refinement of the solution of a linear system
            1. (K + dK) * dz = b - K*z
            2. z <- z + dz
        """
        for i in range(self.work.settings.pol_refine_iter):
            rhs = b - np.hstack([
                            self.work.data.P.dot(z[:self.work.data.n]) +
                            self.work.pol.Ared.T.dot(z[self.work.data.n:]),
                            self.work.pol.Ared.dot(z[:self.work.data.n])])
            dz = KKT_factor.solve(rhs)
            z += dz
        return z

    def polish(self):
        """
        Solution polish:
        Solve equality constrained QP with assumed active constraints.
        """
        # Start timer
        self.work.timer = time.time()

        # Guess which linear constraints are lower-active, upper-active, free
        self.work.pol.ind_low = np.where(self.work.z -
                                         self.work.data.l < -self.work.y)[0]
        self.work.pol.ind_upp = np.where(self.work.data.u -
                                         self.work.z < self.work.y)[0]
        self.work.pol.n_low = len(self.work.pol.ind_low)
        self.work.pol.n_upp = len(self.work.pol.ind_upp)

        # Form Ared from the assumed active constraints
        self.work.pol.Ared = spspa.vstack([
                                self.work.data.A[self.work.pol.ind_low],
                                self.work.data.A[self.work.pol.ind_upp]])

        # # Terminate if there are no active constraints
        # if self.work.pol.Ared.shape[0] == 0:
        #     return

        # Form and factorize reduced KKT
        KKTred = spspa.vstack([
              spspa.hstack([self.work.data.P + self.work.settings.delta *
                            spspa.eye(self.work.data.n),
                            self.work.pol.Ared.T]),
              spspa.hstack([self.work.pol.Ared, -self.work.settings.delta *
                            spspa.eye(self.work.pol.Ared.shape[0])])])
        KKTred_factor = spla.splu(KKTred.tocsc())

        # Form reduced RHS
        rhs_red = np.hstack([-self.work.data.q,
                             self.work.data.l[self.work.pol.ind_low],
                             self.work.data.u[self.work.pol.ind_upp]])

        # Solve reduced KKT system
        pol_sol = KKTred_factor.solve(rhs_red)

        # Perform iterative refinement to compensate for the reg. error
        if self.work.settings.pol_refine_iter > 0:
            pol_sol = self.iter_refin(KKTred_factor, pol_sol, rhs_red)

        # Store the polished solution
        self.work.pol.x = pol_sol[:self.work.data.n]
        self.work.pol.z = self.work.data.A.dot(self.work.pol.x)
        self.work.pol.y_red = pol_sol[self.work.data.n:]

        # Compute primal and dual residuals of the polished solution
        self.update_info(0, 1)

        # Update polish time
        self.work.info.polish_time = time.time() - self.work.timer

        # Check if polish was successful
        pol_success = (self.work.pol.pri_res < self.work.info.pri_res) and \
                      (self.work.pol.dua_res < self.work.info.dua_res) or \
                      (self.work.pol.pri_res < self.work.info.pri_res) and \
                      (self.work.info.dua_res < 1e-10) or \
                      (self.work.pol.dua_res < self.work.info.dua_res) and \
                      (self.work.info.pri_res < 1e-10)

        if pol_success:
            # Update solver information
            self.work.info.obj_val = self.work.pol.obj_val
            self.work.info.pri_res = self.work.pol.pri_res
            self.work.info.dua_res = self.work.pol.dua_res
            self.work.info.status_polish = 1

            # Update ADMM iterations
            self.work.x = self.work.pol.x
            self.work.z = self.work.pol.z
            self.work.y = np.zeros(self.work.data.m)
            if self.work.pol.Ared.shape[0] > 0:
                self.work.y[self.work.pol.ind_low] = \
                    self.work.pol.y_red[:self.work.pol.n_low]
                self.work.y[self.work.pol.ind_upp] = \
                    self.work.pol.y_red[self.work.pol.n_low:]

            # Print summary
            if self.work.settings.verbose:
                self.print_polish()
        else:
            self.work.info.status_polish = -1
