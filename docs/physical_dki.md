# Experimental all-direction DKI fitting

Select `--constraint-backend global-sos --fit_constraints 1,1,1 --scalar-only --nthreads 1`
with the usual input, output, mask and preprocessing options. Install the optional
`physical` extra. Do not combine this backend with `--akc` or `--window`.
The default remains the sampled-direction estimator. `--resume` reuses eligible
preprocessing products, but fitting is recomputed, including its diagnostics.
Use a fresh output directory to preserve earlier fitting products.

The three flags independently select positive diffusion, nonnegative raw kurtosis
quartic W, and U=(3/bmax) D(g)||g||²-W(g) >= 0. Positive diffusion is enforced inside
the optimization as D >= dmin I; the default dmin is 1e-6 um²/ms. This is an explicit
estimator choice, not an eigenvalue repair. Inspect sensitivity at 1e-7 and 1e-5,
raw eigenvalues, conditioning and residuals. Solves are serial; IRLLS pattern
models use a bounded eight-entry cache.

A ternary homogeneous quartic is nonnegative iff it is a sum of squares of
quadratics. With v=[x²,y²,z²,xy,xz,yz], each selected quartic is represented by
v'Qv with Q positive semidefinite and exact coefficient equalities. Clarabel solves
the resulting convex weighted least squares problem. Diagnostics retain actual
Gram eigenvalues, equality residuals and a conservative all-direction polynomial
lower bound; acceptance permits numerical error up to 1e-7 in the raw quartic
units. Diffusion residual tolerance is min(1e-7,0.01*dmin). These are cumulant
constraints, not a proof that the diffusion propagator is nonnegative.

For unit g, 2/3 <= ||v||² <= 1. The reported lower bound is therefore
min(lambda_min(Q),2/3*lambda_min(Q))-sum(abs(coefficient residuals)). Solver status
alone does not pass a fit. Invalid/missing certificates raise rather than silently
falling back to an unconstrained result. Near-zero diffusivity can amplify small
quartic errors and must be inspected even when the certificate passes.

RK uses an exact radial-circle integral, with a stable square-root expression for
coincident radial eigenvalues. Axis-aligned sampling no longer divides by zero,
and radialSampling returns exactly the requested number of points. Invalid raw
diffusion tensors produce undefined scalar values rather than absolute eigenvalue
or ADC-floor repairs. Raw eigenvalues and a scalar-validity mask are exported.
Output maps use floating point to preserve undefined values. No map clipping or
spatial repair is applied in the global backend.

The directional bound implies K(g)<=3/(bmax*D(g)), which can exceed three when D
is small. It does not impose a fixed RK ceiling of three. MATLAB DKE's clipped
maps and different weighting/intercept conventions must be distinguished from
raw pyDKE estimates. One control rat cannot establish replacement equivalence.

Mathematical background: Powers et al., *A new proof of Hilbert's theorem on
ternary quartics*, arXiv:math/0405475; Dela Haije, Özarslan and Feragen,
NeuroImage 209 (2020) 116405, DOI 10.1016/j.neuroimage.2019.116405.
Scalar context: Tabesh et al., MRM 65 (2011) 823–836, DOI 10.1002/mrm.22655.
Existing DESIGNER code acknowledgments remain applicable.
