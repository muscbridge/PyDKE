"""Independent scalar integration and all-direction constraint regressions."""
import itertools
import unittest
import numpy as np
from scipy.integrate import quad
from pydesigner.fitting.dwipy import DWI
from pydesigner.fitting import dwidirs
from pydesigner.fitting.physical import (D_INDEX,W_INDEX,D_COUNT,W_COUNT,FULL_W_INDEX,
    GRAM_MAP,DNORM_MAP,GRAM_EXP,PhysicalDkiFitter,exact_axial_radial,perpendicular_circle,diffusion_matrix)


def design(g,b):
    return np.c_[np.ones(len(g)),-b[:,None]*D_COUNT*np.prod(g[:,D_INDEX],axis=2),
                 b[:,None]**2/6*W_COUNT*np.prod(g[:,W_INDEX],axis=2)]


def iso_w(amplitude):
    I=np.eye(3)
    return amplitude/3*(np.einsum('ij,kl->ijkl',I,I)+np.einsum('ik,jl->ijkl',I,I)+np.einsum('il,jk->ijkl',I,I))


def pack(D,W,s0=1):return np.r_[np.log(s0),D[tuple(D_INDEX.T)],W[tuple(W_INDEX.T)]]


def normalized(beta):
    dt=beta[1:].copy();dt[6:]/=(np.trace(diffusion_matrix(dt[:6]))/3)**2
    return dt


class ScalarPhysicsTests(unittest.TestCase):
    def test_public_circle_handles_axes_antipodes_and_exact_count(self):
        dwi=DWI.__new__(DWI)
        for axis in (np.array([0.,0.,1.]),np.array([0.,0.,-1.]),np.array([1e-14,0.,1.]),np.array([1.,2.,3.])):
            p=dwi.radialSampling(axis,256)
            self.assertEqual(p.shape,(256,3));self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p@axis,0,atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(p,axis=1),1,atol=1e-12)
            np.testing.assert_allclose(p[:128]+p[128:],0,atol=1e-12)

    def test_public_rk_agrees_with_independent_adaptive_quadrature(self):
        rng=np.random.default_rng(8);dwi=DWI.__new__(DWI)
        for radial in [( .3,.3),(.3,.3+1e-12),(.0001,.3)]:
            for rotate in (False,True):
                Q=np.eye(3) if not rotate else np.linalg.qr(rng.normal(size=(3,3)))[0]
                D=Q@np.diag([1.6,*radial])@Q.T
                v=rng.normal(size=(4,3));W=np.einsum('ni,nj,nk,nl->ijkl',v,v,v,v)*.02
                raw=pack(D,W);dt=normalized(raw);axis=Q[:,0]
                def integrand(theta):
                    g=Q[:,1]*np.cos(theta)+Q[:,2]*np.sin(theta)
                    return np.einsum('ijkl,i,j,k,l',W,g,g,g,g)/(g@D@g)**2
                expected=sum(quad(integrand,k*np.pi/2,(k+1)*np.pi/2,epsabs=1e-9,epsrel=1e-10,limit=300)[0] for k in range(4))/(2*np.pi)
                actual=dwi.dkiTensorParams(axis,dt)[1]
                self.assertTrue(np.isfinite(actual))
                np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-9)
                np.testing.assert_allclose(dwi.dkiTensorParams(-axis,dt)[1],actual,rtol=1e-12,atol=1e-9)

    def test_exact_isotropic_limit(self):
        dt=normalized(pack(.8*np.eye(3),iso_w(.64*.7)))
        for axis in np.eye(3):np.testing.assert_allclose(exact_axial_radial(axis,dt),[.7,.7],atol=1e-12)

    def test_invalid_tensor_does_not_create_finite_blowup(self):
        dwi=DWI.__new__(DWI);dt=normalized(pack(np.diag([1.,.3,-.01]),iso_w(.2)))
        with self.assertRaisesRegex(ValueError,'positive-definite'):exact_axial_radial([1,0,0],dt)
        actual=dwi.dkiTensorParams(np.array([1.,0.,0.]),dt)
        self.assertTrue(np.isnan(actual[0]) and np.isnan(actual[1]))


class SosPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g=np.r_[np.zeros((1,3)),np.tile(dwidirs.dirs30,(4,1))]
        cls.b=np.r_[0.,np.repeat([.5,1.,1.5,2.],30)]
        cls.B=design(cls.g,cls.b)

    def test_independent_gram_and_dnorm_polynomials(self):
        rng=np.random.default_rng(10)
        Q=rng.normal(size=(6,6));Q=Q@Q.T
        coefficients=GRAM_MAP@Q.reshape(36)
        W=(coefficients/W_COUNT)[FULL_W_INDEX]
        D=rng.normal(size=(3,3));D=(D+D.T)/2
        rawD=D[tuple(D_INDEX.T)]
        for g in rng.normal(size=(25,3)):
            v=np.prod(g[None,:]**GRAM_EXP,axis=1)
            np.testing.assert_allclose(v@Q@v,np.einsum('ijkl,i,j,k,l',W,g,g,g,g),rtol=1e-12,atol=1e-10)
            polynomial=np.sum(DNORM_MAP@rawD*np.prod(g[W_INDEX],axis=1))
            np.testing.assert_allclose(polynomial,(g@D@g)*(g@g),rtol=1e-12,atol=1e-10)

    def test_noiseless_recovery_all_requested_flag_combinations(self):
        beta=pack(.8*np.eye(3),iso_w(.32))
        signal=np.exp(self.B@beta)
        for flags in itertools.product((0,1),repeat=3):
            with self.subTest(flags=flags):
                fitter=PhysicalDkiFitter(self.B,2.,flags=flags)
                fit,info=fitter.fit(signal,signal)
                self.assertTrue(info['valid_physical_fit'])
                self.assertEqual(set(info['certificates']),({ 'lower'} if flags[1] else set())|({'upper'} if flags[2] else set()))
                np.testing.assert_allclose(fit,beta,atol=1e-5,rtol=1e-5)

    def test_rotation_reversal_and_intensity_invariance(self):
        rng=np.random.default_rng(12);Q=np.linalg.qr(rng.normal(size=(3,3)))[0]
        D=np.diag([.6,.8,1.2]);W=iso_w(.25)
        beta=pack(D,W)
        for scale in (1.,1e4):
            for rotation in (np.eye(3),Q):
                for sign in (1,-1):
                    g=sign*self.g@rotation.T;B=design(g,self.b)
                    Dr=rotation@D@rotation.T
                    Wr=np.einsum('ia,jb,kc,ld,abcd->ijkl',rotation,rotation,rotation,rotation,W)
                    truth=pack(Dr,Wr,scale);signal=np.exp(B@truth)
                    fit,info=PhysicalDkiFitter(B,2.).fit(signal,signal)
                    self.assertTrue(info['valid_physical_fit'])
                    np.testing.assert_allclose(fit,truth,atol=1e-5,rtol=1e-5)

    def test_public_python_fit_selects_certified_backend(self):
        import tempfile
        from pathlib import Path
        import nibabel as nib
        beta=pack(.8*np.eye(3),iso_w(.32),1000.)
        signal=np.exp(self.B@beta)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            nib.save(nib.Nifti1Image(np.broadcast_to(signal,(2,2,2,121)).copy(),np.eye(4)),root/'dwi.nii')
            nib.save(nib.Nifti1Image(np.ones((2,2,2),np.uint8),np.eye(4)),root/'mask.nii')
            np.savetxt(root/'dwi.bvec',self.g.T);np.savetxt(root/'dwi.bval',self.b[None,:]*1000)
            dwi=DWI(str(root/'dwi.nii'),str(root/'dwi.bvec'),str(root/'dwi.bval'),str(root/'mask.nii'),nthreads=1)
            dwi.fit([1,1,1],constraint_backend='global-sos')
            self.assertEqual(len(dwi.fit_diagnostics),8)
            self.assertTrue(all(row['valid_physical_fit'] for row in dwi.fit_diagnostics))
            md,_,_,_,_,_=dwi.extractDTI()
            np.testing.assert_allclose(md,.8,atol=1e-5,rtol=0)
            from pydesigner.fitting.dwipy import fit_regime
            import json
            output=root/'metrics';output.mkdir()
            fit_regime(str(root/'dwi.nii'),str(output),mask=str(root/'mask.nii'),
                irlls=False,akc=False,nthreads=1,fit_constraints=np.array([1,1,1]),
                constraint_backend='global-sos',scalar_only=True)
            record=json.loads((output/'physical_fit_diagnostics.json').read_text())
            self.assertEqual(record['constraint_flags'],[1,1,1])
            self.assertEqual(len(record['fits']),8)
            self.assertTrue(all(r['valid_physical_fit'] for r in record['fits']))
            self.assertFalse(list(output.glob('*.fib')))
            # A repeated fitting call recomputes fits, even if output maps exist.
            record['backend']='stale-sampled-record'
            (output/'physical_fit_diagnostics.json').write_text(json.dumps(record))
            fit_regime(str(root/'dwi.nii'),str(output),mask=str(root/'mask.nii'),
                irlls=False,akc=False,nthreads=1,fit_constraints=[1,1,1],
                constraint_backend='global-sos',scalar_only=True,min_diffusivity=1e-5)
            record=json.loads((output/'physical_fit_diagnostics.json').read_text())
            self.assertEqual(record['backend'],'global-sos')
            self.assertEqual(record['min_diffusivity'],1e-5)


    def test_configuration_rejects_invalid_scales_and_fixed_s0_order(self):
        for value in (np.nan,np.inf,0.,-1.):
            with self.assertRaises(ValueError):PhysicalDkiFitter(self.B,2.,min_diffusivity=value)
            with self.assertRaises(ValueError):PhysicalDkiFitter(self.B,2.,certificate_tolerance=value)
        with self.assertRaisesRegex(ValueError,'leading b0'):
            PhysicalDkiFitter(self.B[::-1],2.,fit_intercept=False)

    def test_constraint_is_in_fit_not_eigenvalue_repair(self):
        truth=pack(np.diag([-.01,.8,1.2]),iso_w(.3));signal=np.exp(self.B@truth)
        for minimum in (1e-7,1e-6,1e-5):
            fit,info=PhysicalDkiFitter(self.B,2.,min_diffusivity=minimum).fit(signal,signal)
            self.assertTrue(info['valid_physical_fit'])
            self.assertGreater(np.linalg.eigvalsh(diffusion_matrix(fit[1:7]))[0],0)
            self.assertGreaterEqual(info['raw_minimum_eigenvalue'],minimum-info['psd_residual_tolerance'])
            for cert in info['certificates'].values():self.assertGreaterEqual(cert['all_direction_polynomial_lower_bound'],-info['certificate_tolerance'])


if __name__=='__main__':unittest.main()
