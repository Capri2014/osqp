"""
MIOSQP Solver pure python implementation: low level module
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
MIOSQP_SOLVED = 1
MIOSQP_MAX_ITER_REACHED = -2
MIOSQP_UNSOLVED = -10

# Printing interval
PRINT_INTERVAL = 100

# OSQP Infinity
MIOSQP_INFTY = 1e+20

# OSQP Nan
MIOSQP_NAN = 1e+20  # Just as placeholder. Not real value


class workspace(object):
    """
    OSQP solver workspace

    Attributes
    ----------
    data                   - scaled QP problem
    info                   - solver information
    priv                   - private structure for linear system solution
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
    int_idx
    """

    def __init__(self, dims, Pdata, Pindices, Pindptr, q,
                 Adata, Aindices, Aindptr,
                 l, u, int_idx):
        # Set problem dimensions
        (self.n, self.m, self.p) = dims

        # Set problem data
        self.P = spspa.csc_matrix((Pdata, Pindices, Pindptr),
                                  shape=(self.n, self.n))
        self.q = q
        self.A = spspa.csc_matrix((Adata, Aindices, Aindptr),
                                  shape=(self.m, self.n))
        self.l = l if l is not None else -np.inf*np.ones(self.m)
        self.u = u if u is not None else np.inf*np.ones(self.m)
        self.int_idx = int_idx

    def objval(self, x):
        # Compute quadratic objective value for the given x
        return .5 * np.dot(x, self.P.dot(x)) + np.dot(self.q, x)


class settings(object):
    """
    MIOSQP solver settings

    Attributes
    ----------
    -> These cannot be changed without running setup
    rho  [1.6]                 - Step in ADMM procedure
    sigma    [1e-01]           - Regularization parameter for polishinging
    scaling  [True]            - Prescaling/Equilibration
    scaling_iter [3]           - Number of Steps for Scaling Method
    scaling_norm [2]           - Scaling norm in SK algorithm

    -> These can be changed without running setup
    max_iter [5000]            - Maximum number of iterations
    eps_abs  [1e-04]           - Integer infeasibility tolerance
    eps_rel  [1e-04]           - Integer infeasibility tolerance
    alpha [1.0]                - Relaxation parameter
    verbose  [True]            - Verbosity
    warm_start [False]         - Reuse solution from previous solve
    """

    def __init__(self, **kwargs):

        self.rho = kwargs.pop('rho', 1.6)
        self.sigma = kwargs.pop('sigma', 1e-01)
        self.scaling = kwargs.pop('scaling', True)
        self.scaling_iter = kwargs.pop('scaling_iter', 3)
        self.scaling_norm = kwargs.pop('scaling_norm', 2)

        self.max_iter = kwargs.pop('max_iter', 5000)
        self.eps_abs = kwargs.pop('eps_abs', 1e-4)
        self.eps_rel = kwargs.pop('eps_rel', 1e-4)
        self.alpha = kwargs.pop('alpha', 1.6)
        self.verbose = kwargs.pop('verbose', True)
        self.warm_start = kwargs.pop('warm_start', False)


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
    obj_val         - primal objective
    pri_res         - norm of primal residual
    setup_time      - time taken for setup phase (milliseconds)
    solve_time      - time taken for solve phase (milliseconds)
    run_time        - total time  (milliseconds)
    """
    def __init__(self):
        self.iter = 0
        self.status_val = MIOSQP_UNSOLVED

class priv(object):
    """
    Linear systems solver
    """

    def __init__(self, work):
        """
        Initialize private structure for KKT system solution
        """
        # Construct reduced KKT matrix
        KKT = spspa.vstack([
              spspa.hstack([work.data.P + work.settings.sigma *
                            spspa.eye(work.data.n), work.data.A.T]),
              spspa.hstack([work.data.A,
                           -1./work.settings.rho * spspa.eye(work.data.m)])])

        # Initialize private structure
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
    def __init__(self, sol, inf):
        self.x = sol.x
        self.y = sol.y
        self.info = inf


class MIOSQP(object):
    """OSQP solver lower level interface
    Attributes
    ----------
    work    - workspace
    """
    def __init__(self):
        self._version = "0.0.0"

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
        p = self.work.data.p
        int_idx = self.work.data.int_idx

        # Initialize scaling
        d = np.ones(n + m)

        # Define reduced KKT matrix to scale
        KKT = spspa.vstack([
              spspa.hstack([self.work.data.P, self.work.data.A.T]),
              spspa.hstack([self.work.data.A,
                            spspa.csc_matrix((m, m))])])

        # Run Scaling
        KKT2 = KKT.copy()
        if self.work.settings.scaling_norm == 2:
            KKT2.data = np.square(KKT2.data)  # Elementwise square
        elif self.work.settings.scaling_norm == 1:
            KKT2.data = np.absolute(KKT2.data)  # Elementwise abs

        # Iterate Scaling
        for i in range(self.work.settings.scaling_iter):
            # Regularize components
            KKT2d = KKT2.dot(d)
            # Prevent division by 0
            d = (n + m)*np.reciprocal(KKT2d + 1e-08)
            # Limit scaling terms
            d = np.maximum(np.minimum(d, 1e+03), 1e-03)

        # Obtain Scaler Matrices
        d = np.power(d, 1./self.work.settings.scaling_norm)
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
        self.work.data = problem((n, m, p), P.data, P.indices, P.indptr, q,
                                 A.data, A.indices, A.indptr, l, u, int_idx)

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

    def print_setup_header(self, data, settings):
        """Print solver header
        """
        print("-------------------------------------------------------")
        print("      MIOSQP v%s  -  Operator Splitting MIQP Solver" % \
            self.version)
        print("              Pure Python Implementation")
        print("     (c) .....,")
        print("   University of Oxford  -  Stanford University 2016")
        print("-------------------------------------------------------")

        print("Problem:  variables n = %d, constraints m = %d, integers p = %d" % \
            (data.n, data.m, data.p))
        print("Settings: eps_abs = %.2e, eps_rel = %.2e," % \
            (settings.eps_abs, settings.eps_rel))
        print("          rho = %.2f, sigma = %.2f, alpha = %.2f," % \
            (settings.rho, settings.sigma, settings.alpha))
        print("          max_iter = %d" % settings.max_iter)
        if settings.scaling:
            print("          scaling: active")
        else:
            print("          scaling: inactive")
        if settings.warm_start:
            print("          warm_start: active")
        else:
            print("          warm_start: inactive")
        print("")

    def print_header(self):
        """
        Print header before the iterations
        """
        print("Iter    Obj  Val     Pri  Res       Time")

    def update_status_string(self, status):
        if status == MIOSQP_SOLVED:
            return "Solved"
        if status == MIOSQP_UNSOLVED:
            return "Unsolved"
        elif status == MIOSQP_MAX_ITER_REACHED:
            return "Maximum Iterations Reached"

    def cold_start(self):
        """
        Cold start optimization variables to zero
        """
        self.work.x = np.zeros(self.work.data.n)
        self.work.z = np.zeros(self.work.data.m)
        self.work.y = np.zeros(self.work.data.m)

    def update_xz_tilde(self):
        """
        First ADMM step: update xz_tilde
        """
        # Compute rhs and store it in xz_tilde
        self.work.xz_tilde[:self.work.data.n] = \
            self.work.settings.sigma * self.work.x - self.work.data.q
        self.work.xz_tilde[self.work.data.n:] = \
            self.work.z_prev - 1./self.work.settings.rho * self.work.y

        # Solve linear system
        self.work.xz_tilde = self.work.priv.solve(self.work.xz_tilde)

        # Update z_tilde
        self.work.xz_tilde[self.work.data.n:] = \
            self.work.z_prev + 1./self.work.settings.rho * \
            (self.work.xz_tilde[self.work.data.n:] - self.work.y)

    def update_x(self):
        """
        Update x variable in second ADMM step
        """
        self.work.x = \
            self.work.settings.alpha * self.work.xz_tilde[:self.work.data.n] +\
            (1. - self.work.settings.alpha) * self.work.x_prev
        self.work.delta_x = self.work.x - self.work.x_prev

    def project_z(self):
        """
        Project z variable in set C (for now C = [l, u])
        """
        # proj_z = np.minimum(np.maximum(self.work.z,
        #                     self.work.data.l), self.work.data.u)
        # set_trace()
        self.work.z = np.minimum(np.maximum(self.work.z,
                                 self.work.data.l), self.work.data.u)

    def update_z(self):
        """
        Update z variable in second ADMM step
        """
        self.work.z = \
            self.work.settings.alpha * self.work.xz_tilde[self.work.data.n:] +\
            (1. - self.work.settings.alpha) * self.work.z_prev +\
            1./self.work.settings.rho * self.work.y

        self.project_z()

    def update_y(self):
        """
        Third ADMM step: update dual variable y
        """
        self.work.delta_y = self.work.settings.rho * \
            (self.work.settings.alpha * self.work.xz_tilde[self.work.data.n:] +
                (1. - self.work.settings.alpha) * self.work.z_prev -
                self.work.z)
        self.work.y += self.work.delta_y

    def compute_pri_res(self):
        """
        Compute primal residual ||Ax - z||
        """
        return la.norm(self.work.data.A.dot(self.work.x) - self.work.z)


    def update_info(self, iter):
        """
        Update information at iterations
        """
        self.work.info.iter = iter
        self.work.info.obj_val = self.work.data.objval(self.work.x)
        self.work.info.pri_res = self.compute_pri_res()
        self.work.info.solve_time = time.time() - self.work.timer

    def print_summary(self):
        """
        Print status summary at each ADMM iteration
        """
        print("%4i %12.4e %12.4e %12.4e %9.2fs" % \
            (self.work.info.iter,
             self.work.info.obj_val,
             self.work.info.pri_res,
             self.work.info.setup_time + self.work.info.solve_time))


    def print_footer(self):
        """
        Print footer at the end of the optimization
        """
        print("")  # Add space after iterations
        print("Status: %s" % self.work.info.status)
        print("Number of iterations: %d" % self.work.info.iter)
        if self.work.info.status_val == MIOSQP_SOLVED:
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

        if (self.work.info.status_val is not MIOSQP_INFEASIBLE) and \
                (self.work.info.status_val is not MIOSQP_UNBOUNDED):
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
              l, u, int_idx, **stgs):
        """
        Perform OSQP solver setup QP problem of the form
            minimize	1/2 x' P x + q' x
            subject to	l <= A x <= u

        """
        (n, m, p) = dims
        self.work = workspace()

        # Start timer
        self.work.timer = time.time()

        # Unscaled problem data
        self.work.data = problem((n, m, p), Pdata, Pindices, Pindptr, q,
                                 Adata, Aindices, Aindptr,
                                 l, u, int_idx)

        # Initialize workspace variables
        self.work.x = np.zeros(n)
        self.work.z = np.zeros(m)
        self.work.xz_tilde = np.zeros(n + m)
        self.work.x_prev = np.zeros(n)
        self.work.z_prev = np.zeros(m)
        self.work.y = np.zeros(m)

        # Flag indicating first run
        self.work.first_run = 1

        # Settings
        self.work.settings = settings(**stgs)

        # Scale problem
        if self.work.settings.scaling:
            self.scale_data()

        # Factorize KKT
        self.work.priv = priv(self.work)

        # Solution
        self.work.solution = solution()

        # Info
        self.work.info = info()

        # End timer
        self.work.info.setup_time = time.time() - self.work.timer

        # Print setup header
        if self.work.settings.verbose:
            self.print_setup_header(self.work.data, self.work.settings)

    def solve(self):
        """
        Solve MIQP problem using ADMM Heuristic
        """
        # Start timer
        self.work.timer = time.time()

        # Print header
        if self.work.settings.verbose:
            self.print_header()

        # Cold start if not warm start
        if not self.work.settings.warm_start:
            self.cold_start()

        # ADMM algorithm
        for iter in range(1, self.work.settings.max_iter + 1):
            # Update x_prev, z_prev
            self.work.x_prev = np.copy(self.work.x)
            self.work.z_prev = np.copy(self.work.z)

            # Admm steps
            # First step: update \tilde{x} and \tilde{z}
            self.update_xz_tilde()

            # Second step: update x and z
            self.update_x()
            self.update_z()

            # Third step: update y
            self.update_y()



            # Check if obtained new solution!


            # Update info
            self.update_info(iter)

            # Print summary
            if (self.work.settings.verbose) & \
                    ((iter % PRINT_INTERVAL == 0) | (iter == 1)):
                self.print_summary()



        # Print summary for last iteration
        if (self.work.settings.verbose) & (iter % PRINT_INTERVAL != 0):
            self.print_summary()


        # Update solve time
        self.work.info.solve_time = time.time() - self.work.timer


        # Update total times
        if self.work.first_run:
            self.work.info.run_time = self.work.info.setup_time + \
                self.work.info.solve_time
        else:
            self.work.info.run_time = self.work.info.solve_time

        # Print footer
        if self.work.settings.verbose:
            self.print_footer()

        # Eliminate first run flag
        if self.work.first_run:
            self.work.first_run = 0

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

    def update_alpha(self, alpha_new):
        """
        Update relaxation parameter alpga
        """
        if not (alpha_new >= 0 | alpha_new <= 2):
            raise ValueError("alpha must be between 0 and 2")

        self.work.settings.alpha = alpha_new

    def update_delta(self, delta_new):
        """
        Update delta parameter for polishing
        """
        if delta_new <= 0:
            raise ValueError("delta must be positive")

        self.work.settings.delta = delta_new

    def update_verbose(self, verbose_new):
        """
        Update verbose parameter
        """
        if (verbose_new is not True) & (verbose_new is not False):
            raise ValueError("verbose should be either True or False")

        self.work.settings.verbose = verbose_new

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
        if constant_name == "MIOSQP_INFTY":
            return MIOSQP_INFTY
        if constant_name == "MIOSQP_NAN":
            return MIOSQP_NAN
        if constant_name == "MIOSQP_SOLVED":
            return MIOSQP_SOLVED
        if constant_name == "MIOSQP_UNSOLVED":
            return MIOSQP_UNSOLVED
        if constant_name == "MIOSQP_MAX_ITER_REACHED":
            return MIOSQP_MAX_ITER_REACHED

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
        Solution polishing:
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

        # Check if polishing was successful
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
                self.print_polishing()
        else:
            self.work.info.status_polish = -1
