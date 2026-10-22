"""Physical constraints on raw DKI coefficients; actual constrained QP checks.

The fitted model is log(S)=log(S0)-b D(g)+b^2 Wraw(g)/6.
The optional upper bound enforces d(log S)/db <= 0 through bmax,
so (3/bmax) D(g)-Wraw(g) >= 0. Wraw=MD^2*normalized_W.
"""
import itertools
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import cvxpy as cvx
import numpy as np
from pydesigner.fitting.dwipy import DWI
from pydesigner.fitting import dwidirs

FLAGS = list(itertools.product((0, 1), repeat=3))
EVIDENCE = []


def packed(dwi, D, W, log_s0=0.):
    di = dwi.createTensorOrder(2)[1]
    wi = dwi.createTensorOrder(4)[1]
    return np.r_[log_s0, D[tuple(di.T)], W[tuple(wi.T)]]


def isotropic_w(amplitude):
    eye = np.eye(3)
    return amplitude / 3 * (np.einsum('ij,kl->ijkl', eye, eye)
                            + np.einsum('ik,jl->ijkl', eye, eye)
                            + np.einsum('il,jk->ijkl', eye, eye))


def physical_margins(directions, D, W, bmax, flags):
    d = np.einsum('ni,ij,nj->n', directions, D, directions)
    w = np.einsum('ni,nj,nk,nl,ijkl->n', directions, directions, directions, directions, W)
    blocks = [d, w, 3 / bmax * d - w]
    return np.concatenate([a for flag, a in zip(flags, blocks) if flag]) if any(flags) else np.zeros(0)


class ConstraintPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dwi = DWI.__new__(DWI)
        path = os.environ.get('PYDKE_PILOT_BVEC')
        if path:
            g = np.loadtxt(path).T
        else:
            g = np.r_[np.zeros((1, 3)), np.tile(dwidirs.dirs30, (4, 1))]
        bvals = np.r_[0., np.repeat([.5, 1., 1.5, 2.], 30)]
        cls.dwi.grad = np.c_[g, bvals]
        _, di = cls.dwi.createTensorOrder(2)
        _, wi = cls.dwi.createTensorOrder(4)
        dc, _ = cls.dwi.createTensorOrder(2)
        wc, _ = cls.dwi.createTensorOrder(4)
        cls.design = np.c_[np.ones(121), -bvals[:, None] * dc * np.prod(g[:, di], axis=2),
                          bvals[:, None]**2 / 6 * wc * np.prod(g[:, wi], axis=2)]

    def test_direct_full_tensor_contraction_all_combinations(self):
        rng = np.random.default_rng(7)
        D = rng.normal(size=(3, 3)); D = (D + D.T) / 2
        # Symmetric rank-one sums exercise every fourth-order component.
        v = rng.normal(size=(5, 3)); weights = rng.normal(size=5)
        W = np.einsum('n,ni,nj,nk,nl->ijkl', weights, v, v, v, v)
        coefficients = packed(self.dwi, D, W)
        for flags in FLAGS:
            with self.subTest(flags=flags):
                actual = self.dwi.createConstraints(list(flags)) @ coefficients
                expected = physical_margins(dwidirs.dirs30, D, W, 2., flags)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_antipodal_invariance_all_combinations(self):
        for flags in FLAGS:
            with self.subTest(flags=flags):
                plus = self.dwi.createConstraints(list(flags))
                with patch.object(dwidirs, 'dirs30', -dwidirs.dirs30):
                    minus = self.dwi.createConstraints(list(flags))
                np.testing.assert_allclose(plus, minus, atol=1e-12, rtol=0)

    def test_rotation_covariance_all_combinations(self):
        q, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
        if np.linalg.det(q) < 0: q[:, 0] *= -1
        D = np.diag([.4, .8, 1.2])
        u = np.array([.3, .4, .5]); u /= np.linalg.norm(u)
        W = isotropic_w(.3) + .2 * np.einsum('i,j,k,l->ijkl', u, u, u, u)
        Dr = q @ D @ q.T
        Wr = np.einsum('ia,jb,kc,ld,abcd->ijkl', q, q, q, q, W)
        for flags in FLAGS:
            with self.subTest(flags=flags):
                original = self.dwi.createConstraints(list(flags)) @ packed(self.dwi, D, W)
                with patch.object(dwidirs, 'dirs30', dwidirs.dirs30 @ q.T):
                    rotated = self.dwi.createConstraints(list(flags)) @ packed(self.dwi, Dr, Wr)
                np.testing.assert_allclose(original, rotated, atol=1e-12, rtol=1e-12)

    def test_known_feasible_and_infeasible_tensors(self):
        D = .8 * np.eye(3)
        good = self.dwi.createConstraints([1, 1, 1]) @ packed(self.dwi, D, isotropic_w(.64))
        self.assertGreater(good.min(), 0)
        bad_upper = self.dwi.createConstraints([0, 0, 1]) @ packed(self.dwi, D, isotropic_w(1.92))
        self.assertTrue(np.all(bad_upper < 0))
        bad_D = self.dwi.createConstraints([1, 0, 0]) @ packed(self.dwi, -.1 * np.eye(3), isotropic_w(.64))
        self.assertTrue(np.all(bad_D < 0))
        bad_W = self.dwi.createConstraints([0, 1, 0]) @ packed(self.dwi, D, isotropic_w(-.2))
        self.assertTrue(np.all(bad_W < 0))

    def test_actual_constrained_solves_all_combinations(self):
        for kurtosis in [1., 3.]:
            coeff = packed(self.dwi, .8 * np.eye(3), isotropic_w(.64 * kurtosis))
            signal = np.exp(self.design @ coeff)
            for flags in FLAGS[1:]:
                with self.subTest(kurtosis=kurtosis, flags=flags):
                    C = self.dwi.createConstraints(list(flags))
                    fitted, info = self.dwi.wlls(signal, signal, self.design, cons=C, return_diagnostics=True)
                    margin = float(np.min(C @ fitted))
                    EVIDENCE.append({'kurtosis': kurtosis, 'flags': list(flags), 'margin': margin, **info})
                    self.assertFalse(info['fallback_used'])
                    self.assertIn(info['solver_status'], ('optimal', 'optimal_inaccurate'))
                    self.assertTrue(np.isfinite(fitted).all())
                    self.assertGreaterEqual(margin, -1e-6)
                    self.assertEqual(info['nonfinite_solution_coefficients'], 0)
                    if kurtosis == 1.:
                        np.testing.assert_allclose(fitted, coeff, atol=1e-6, rtol=0)

    def test_explicit_acquisition_constraint_directions(self):
        directions = self.dwi.grad[1:31, :3]
        D = np.diag([.4, .8, 1.2]); W = isotropic_w(.3)
        with patch.object(dwidirs, 'dirs30', directions):
            C = self.dwi.createConstraints([1, 1, 1])
        np.testing.assert_allclose(C @ packed(self.dwi, D, W),
            physical_margins(directions, D, W, 2., [1, 1, 1]), atol=1e-10, rtol=0)

    def test_actual_constrained_signal_scale_invariance(self):
        coeff = packed(self.dwi, .8 * np.eye(3), isotropic_w(.64))
        C = self.dwi.createConstraints([1, 1, 1])
        for scale in (1., 1000.):
            scaled = coeff.copy(); scaled[0] = np.log(scale)
            signal = np.exp(self.design @ scaled)
            fitted, info = self.dwi.wlls(signal, signal, self.design, cons=C, return_diagnostics=True)
            self.assertFalse(info['fallback_used'])
            np.testing.assert_allclose(fitted, scaled, atol=1e-6, rtol=0)
            self.assertGreaterEqual(info['minimum_constraint_margin'], -1e-6)

    def test_high_intensity_noisy_boundary_fit_checks_actual_margin(self):
        # Data-free regression: the unscaled/default-tolerance solve reports
        # optimal but violates a sampled inequality by about 2.7e-6.
        g = np.r_[np.zeros((1, 3)), np.tile(dwidirs.dirs30, (4, 1))]
        b = np.r_[0., np.repeat([.5, 1., 1.5, 2.], 30)]
        dc, di = self.dwi.createTensorOrder(2)
        wc, wi = self.dwi.createTensorOrder(4)
        B = np.c_[np.ones(121), -b[:, None] * dc * np.prod(g[:, di], axis=2),
                  b[:, None]**2 / 6 * wc * np.prod(g[:, wi], axis=2)]
        coeff = packed(self.dwi, np.diag([.2, .8, 1.4]), isotropic_w(.5), np.log(5000.))
        signal = np.exp(B @ coeff + np.random.default_rng(5).normal(0, .08, 121))
        weights = np.exp(B @ (np.linalg.pinv(B) @ np.log(signal)))
        fitted, info = self.dwi.wlls(weights, signal, B,
            cons=self.dwi.createConstraints([1, 1, 1]), return_diagnostics=True)
        self.assertFalse(info['fallback_used'])
        self.assertGreaterEqual(info['minimum_constraint_margin'], -1e-6)
        self.assertTrue(info['valid_constrained_solution'])

    def test_failed_constrained_fit_without_diagnostics_raises(self):
        coeff = packed(self.dwi, .8 * np.eye(3), isotropic_w(.64))
        signal = np.exp(self.design @ coeff)
        with patch.object(cvx.Problem, 'solve', side_effect=cvx.error.SolverError('intentional')):
            with self.assertRaisesRegex(RuntimeError, 'Constrained'):
                self.dwi.wlls(signal, signal, self.design, cons=self.dwi.createConstraints([1, 1, 1]))

    def test_optimal_status_cannot_override_bad_constraint_margin(self):
        coeff = packed(self.dwi, .8 * np.eye(3), isotropic_w(1.92))
        signal = np.exp(self.design @ coeff)
        def pretend_optimal(problem, **kwargs):
            problem._status = 'optimal'
            problem.variables()[0].value = coeff
        with patch.object(cvx.Problem, 'solve', pretend_optimal):
            fitted, info = self.dwi.wlls(signal, signal, self.design,
                cons=self.dwi.createConstraints([1, 1, 1]), return_diagnostics=True)
        self.assertLess(info['minimum_constraint_margin'], -1e-6)
        self.assertFalse(info['valid_constrained_solution'])

    def test_solver_failure_is_reported(self):
        coeff = packed(self.dwi, .8 * np.eye(3), isotropic_w(.64))
        signal = np.exp(self.design @ coeff)
        with patch.object(cvx.Problem, 'solve', side_effect=cvx.error.SolverError('intentional test failure')):
            fitted, info = self.dwi.wlls(signal, signal, self.design,
                cons=self.dwi.createConstraints([1, 1, 1]), return_diagnostics=True)
        self.assertTrue(info['fallback_used'])
        self.assertFalse(info['valid_constrained_solution'])
        self.assertIn('SolverError', info['fallback_reason'])
        self.assertTrue(np.isfinite(fitted).all())

    def test_legacy_array_return_is_preserved(self):
        coeff = packed(self.dwi, .8 * np.eye(3), isotropic_w(.64))
        signal = np.exp(self.design @ coeff)
        fitted = self.dwi.wlls(signal, signal, self.design)
        self.assertIsInstance(fitted, np.ndarray)
        np.testing.assert_allclose(fitted, coeff, atol=1e-10, rtol=0)

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get('PYDKE_CONSTRAINT_EVIDENCE')
        if path:
            Path(path).write_text(json.dumps(EVIDENCE, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': unittest.main()
