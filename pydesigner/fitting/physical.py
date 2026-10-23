"""Opt-in all-direction DKI constraints and stable scalar mathematics.

Wraw(g)=MD**2*Wnormalized(g). A nonnegative ternary quartic is a sum
of squares of quadratics, so PSD Gram coefficient equalities enforce
Wraw>=0 and (3/bmax)*D(g)*||g||**2-Wraw(g)>=0 for every direction.
These are cumulant-model constraints, not proof of propagator positivity.
See Powers et al., math/0405475; Dela Haije et al., NeuroImage209:116405.
"""
from itertools import combinations_with_replacement
from math import factorial
import cvxpy as cvx
import numpy as np

D_INDEX=np.array(list(combinations_with_replacement(range(3),2)),dtype=int)
W_INDEX=np.array(list(combinations_with_replacement(range(3),4)),dtype=int)
D_COUNT=np.array([factorial(2)/np.prod([factorial(int(n)) for n in np.bincount(i,minlength=3)]) for i in D_INDEX])
W_COUNT=np.array([factorial(4)/np.prod([factorial(int(n)) for n in np.bincount(i,minlength=3)]) for i in W_INDEX])
W_EXP=np.array([np.bincount(i,minlength=3) for i in W_INDEX])
GRAM_EXP=np.array([[2,0,0],[0,2,0],[0,0,2],[1,1,0],[1,0,1],[0,1,1]])
GRAM_MAP=np.zeros((15,36))
DNORM_MAP=np.zeros((15,6))
for k,exponent in enumerate(W_EXP):
    for i in range(6):
        for j in range(6):
            if np.array_equal(GRAM_EXP[i]+GRAM_EXP[j],exponent):GRAM_MAP[k,i*6+j]=1
    for i,index in enumerate(D_INDEX):
        for axis in range(3):
            product=np.bincount(index,minlength=3)+2*np.eye(3,dtype=int)[axis]
            if np.array_equal(product,exponent):DNORM_MAP[k,i]+=D_COUNT[i]
W_LOOKUP={tuple(i):k for k,i in enumerate(W_INDEX)}
FULL_W_INDEX=np.array([W_LOOKUP[tuple(sorted(i))] for i in np.ndindex((3,3,3,3))]).reshape(3,3,3,3)


def diffusion_matrix(coefficients):
    a=np.asarray(coefficients)
    return np.array([[a[0],a[1],a[2]],[a[1],a[3],a[4]],[a[2],a[4],a[5]]])


def perpendicular_circle(axis,n):
    axis=np.asarray(axis,dtype=float).reshape(3)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)==0 or n<3:
        raise ValueError('Require a finite nonzero axis and at least three samples')
    axis=axis/np.linalg.norm(axis)
    helper=np.eye(3)[np.argmin(np.abs(axis))]
    u=np.cross(axis,helper);u/=np.linalg.norm(u);v=np.cross(axis,u)
    theta=2*np.pi*np.arange(n)/n
    return np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v


def exact_axial_radial(axis,normalized_dt):
    """Exact AK/RK for a specified axis and positive-definite diffusion tensor.

    The radial plane is diagonalized; square-root formulas remain continuous
    when radial eigenvalues coincide. No ADC/eigenvalue floor is applied.
    """
    dt=np.asarray(normalized_dt,dtype=float).reshape(21)
    D=diffusion_matrix(dt[:6])
    if not np.isfinite(dt).all() or np.linalg.eigvalsh(D)[0]<=0:
        raise ValueError('AK/RK requires a finite positive-definite diffusion tensor')
    axis=np.asarray(axis,dtype=float).reshape(3)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)==0:
        raise ValueError('Require a finite nonzero axis')
    axis=axis/np.linalg.norm(axis)
    plane=perpendicular_circle(axis,4)[:2].T
    eigenvalues,rotation=np.linalg.eigh(plane.T@D@plane)
    plane=plane@rotation
    u,v=np.sqrt(eigenvalues)
    W=dt[6:][FULL_W_INDEX]*(np.trace(D)/3)**2
    first,second=plane[:,0],plane[:,1]
    wa=np.einsum('ijkl,i,j,k,l',W,first,first,first,first)
    wb=np.einsum('ijkl,i,j,k,l',W,second,second,second,second)
    wab=np.einsum('ijkl,i,j,k,l',W,first,first,second,second)
    A=(2*u+v)/(2*u**3*(u+v)**2)
    B=(2*v+u)/(2*v**3*(u+v)**2)
    C=1/(2*u*v*(u+v)**2)
    radial=wa*A+6*wab*C+wb*B
    axial=np.einsum('ijkl,i,j,k,l',W,axis,axis,axis,axis)/(axis@D@axis)**2
    return float(axial),float(radial)


class PhysicalDkiFitter:
    """Reusable serial constrained WLS with PSD/SOS certificates.

    Explicit opt-in estimator; existing sampled-QP defaults are unchanged.
    Flags select D positivity, W positivity, and upper cumulant bound.
    min_diffusivity (um²/ms) is an optimization constraint, not a map repair.
    The caller must report boundary fits and test sensitivity to this scale.
    """
    def __init__(self,design,bmax,flags=(1,1,1),min_diffusivity=1e-6,
                 fit_intercept=True,certificate_tolerance=1e-7):
        self.design=np.asarray(design,dtype=float)
        if self.design.ndim!=2 or self.design.shape[1]!=22 or not np.isfinite(self.design).all():
            raise ValueError('Require a finite Nx22 DKI design')
        self.flags=tuple(flags)
        if len(self.flags)!=3 or any(f not in (0,1) for f in self.flags):raise ValueError('Invalid constraint flags')
        self.flags=tuple(map(int,self.flags))
        if (not np.isfinite(bmax) or bmax<=0 or not np.isfinite(min_diffusivity) or min_diffusivity<=0
                or not np.isfinite(certificate_tolerance) or certificate_tolerance<=0):
            raise ValueError('Physical scales/tolerances must be finite and positive')
        if np.linalg.matrix_rank(self.design)<22:raise ValueError('DKI design must have full rank22')
        if not fit_intercept and not np.allclose(self.design[0],np.r_[1.,np.zeros(21)],atol=1e-12,rtol=0):
            raise ValueError('Fixed-S0 mode requires a leading b0 design row')
        if 'CLARABEL' not in cvx.installed_solvers():
            raise RuntimeError('global-sos requires the pydke[physical] extra with Clarabel PSD support')
        self.bmax=float(bmax);self.min_diffusivity=float(min_diffusivity)
        self.tolerance=float(certificate_tolerance);self.fit_intercept=bool(fit_intercept)
        n=len(self.design)
        self.A=cvx.Parameter((n,22));self.y=cvx.Parameter(n)
        self.x=cvx.Variable(22);self.log_s0=cvx.Parameter()
        D=cvx.bmat([[self.x[1],self.x[2],self.x[3]],
                    [self.x[2],self.x[4],self.x[5]],[self.x[3],self.x[5],self.x[6]]])
        constraints=[]
        if self.flags[0]:constraints.append(D-self.min_diffusivity*np.eye(3)>>0)
        self.grams={}
        for name,enabled,coeff in [('lower',self.flags[1],cvx.multiply(W_COUNT,self.x[7:])),
            ('upper',self.flags[2],3/self.bmax*DNORM_MAP@self.x[1:7]-cvx.multiply(W_COUNT,self.x[7:]))]:
            if enabled:
                Q=cvx.Variable((6,6),symmetric=True)
                constraints.extend([Q>>0,GRAM_MAP@cvx.vec(Q,order='C')==coeff])
                self.grams[name]=Q
        if not self.fit_intercept:constraints.append(self.x[0]==self.log_s0)
        self.problem=cvx.Problem(cvx.Minimize(.5*cvx.sum_squares(self.A@self.x-self.y)),constraints)
        if not self.problem.is_dpp():raise RuntimeError('Physical fitter must support parameter caching')

    def fit(self,weights,signal):
        signal=np.asarray(signal,dtype=float).reshape(-1)
        weights=np.asarray(weights,dtype=float).reshape(-1)
        if len(signal)!=len(self.design) or weights.shape!=signal.shape:
            raise ValueError('Signal/weight/design mismatch')
        if not np.isfinite(signal).all() or not np.isfinite(weights).all() or np.any(weights<0) or weights.max()<=0:
            raise ValueError('Require finite signal and nonnegative nonzero weights')
        floors=int(np.sum(signal<1e-7));safe=np.maximum(signal,1e-7)
        scale=float(weights.max());weights=weights/scale
        self.A.value=weights[:,None]*self.design;self.y.value=weights*np.log(safe)
        self.log_s0.value=float(np.log(safe[0]))
        self.problem.solve(solver=cvx.CLARABEL,tol_gap_abs=1e-10,tol_gap_rel=1e-10,
            tol_feas=1e-10,max_iter=300,verbose=False)
        if self.x.value is None:raise RuntimeError('Physical solve returned no coefficients: '+str(self.problem.status))
        beta=np.asarray(self.x.value).reshape(22).copy()
        if not np.isfinite(beta).all():
            raise RuntimeError('Physical solve returned nonfinite coefficients: '+str(self.problem.status))
        diagnostics={'solver_status':self.problem.status,'solver_iterations':int(self.problem.solver_stats.num_iters or 0),
            'fallback_used':False,'signal_samples_floored':floors,'weight_normalization_factor':scale,
            'raw_minimum_eigenvalue':float(np.linalg.eigvalsh(diffusion_matrix(beta[1:7]))[0]),
            'min_diffusivity':self.min_diffusivity,'constraint_flags':list(self.flags),
            'certificate_tolerance':self.tolerance,'certificates':{},'fit_intercept':self.fit_intercept}
        valid=self.problem.status in ('optimal','optimal_inaccurate') and np.isfinite(beta).all()
        diagnostics['psd_residual_tolerance']=min(self.tolerance,self.min_diffusivity*.01)
        if self.flags[0]:
            valid &= (diagnostics['raw_minimum_eigenvalue']>0 and diagnostics['raw_minimum_eigenvalue']>=self.min_diffusivity-diagnostics['psd_residual_tolerance'])
        for name,Q in self.grams.items():
            if Q.value is None or not np.isfinite(Q.value).all():
                raise RuntimeError('Physical solve returned an invalid Gram certificate')
            gram=np.asarray(Q.value)
            target=W_COUNT*beta[7:] if name=='lower' else 3/self.bmax*DNORM_MAP@beta[1:7]-W_COUNT*beta[7:]
            errors=GRAM_MAP@gram.reshape(36)-target
            smallest=float(np.linalg.eigvalsh(gram)[0])
            # On ||g||=1, 2/3 <= ||[x²,y²,z²,xy,xz,yz]||² <= 1.
            bound=(smallest if smallest<0 else 2/3*smallest)-float(np.sum(np.abs(errors)))
            diagnostics['certificates'][name]={'minimum_gram_eigenvalue':smallest,
                'coefficient_equality_max_error':float(np.max(np.abs(errors))),
                'all_direction_polynomial_lower_bound':bound}
            valid &= bound>=-self.tolerance
        diagnostics['valid_physical_fit']=bool(valid)
        diagnostics['near_diffusivity_boundary']=diagnostics['raw_minimum_eigenvalue']<=10*self.min_diffusivity
        diagnostics['objective']=float(self.problem.value)
        return beta,diagnostics
