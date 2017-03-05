# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
import scipy.sparse as spspa

# Unit Test
import unittest


class unboundedness_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup default options
        """
        self.opts = {'verbose': False,
                     'eps_abs': 1e-05,
                     'eps_rel': 1e-05,
                     'eps_inf': 1e-15,  # Focus only on unboundedness
                     'eps_unb': 1e-6,
                     'scaling': True,
                     'scaling_norm': 2,
                     'scaling_iter': 3,
                     'rho': 1.6,
                     'alpha': 1.6,
                     'max_iter': 2500,
                     'polish': False,
                     'pol_refine_iter': 4}

    def test_unbounded_lp(self):

        # Unbounded example
        self.P = spspa.csc_matrix((2, 2))
        self.q = np.array([2, -1])
        self.A = spspa.eye(2).tocsc()
        self.l = np.array([0., 0.])
        self.u = np.array([np.inf, np.inf])

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem with OSQP
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_UNBOUNDED'))

    def test_unbounded_qp(self):

        # Unbounded example
        self.P = spspa.csc_matrix(np.diag(np.array([4., 0.])))
        self.q = np.array([0, 2])
        self.A = spspa.csc_matrix([[1., 1.], [-1., 1.]])
        self.l = np.array([-np.inf, -np.inf])
        self.u = np.array([2., 3.])

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem with OSQP
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_UNBOUNDED'))

    def test_infeasible_and_unbounded_problem(self):

        self.n = 2
        self.m = 4
        self.P = spspa.csc_matrix((2, 2))
        self.q = np.array([-1., -1.])
        self.A = spspa.csc_matrix([[1., -1.], [-1., 1.], [1., 0.], [0., 1.]])
        self.l = np.array([1., 1., 0., 0.])
        self.u = np.inf * np.ones(self.m)

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Warm start to avoid infeasibility detection at first step
        x0 = 25.*np.ones(self.n)
        y0 = -2.*np.ones(self.m)
        self.model.warm_start(x=x0, y=y0)

        # Solve
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_UNBOUNDED'))
