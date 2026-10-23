# Optional noncentral-χ noise-bias correction

This option is an alternative magnitude-noise-bias model to the separate legacy Rician correction. Its primary mode uses a magnitude-mean estimate and an **MP-PCA magnitude standard-deviation map** to estimate both the underlying signal amplitude and effective Gaussian noise level through generalized Koay moment inversion. It does not equate MP-PCA magnitude dispersion with Gaussian component dispersion or require an independent Gaussian calibration file for this mode.

The correction does **not** enable or change a denoising algorithm. Denoising method, patch, shrinkage and worker choices remain those explicitly selected for the existing workflow. The inversion follows the specified statistical model; the accuracy of its input moments and effective coil parameter remains a separate scientific question.

The receiver-noise “SoS” terminology here is separate from the `global-sos` sum-of-squares polynomial fitting constraint; selecting one does not select the other.

## MP-PCA magnitude-noise mode

When denoising is explicitly requested, the correction can use the resulting MP-PCA magnitude noise map directly as its magnitude-dispersion input:

```sh
pydke input.nii.gz -o fresh_output \
  --denoise \
  --sos-noise-correct \
  --sos-effective-coils 4 \
  --sos-input-domain magnitude-mean \
  --sos-sigma-convention magnitude \
  --nthreads 1
```

`magnitude` is the default sigma convention for this correction. No `--sos-sigma` file is needed when the explicitly requested denoising step generates the map. Existing denoising options still select the desired backend and settings; the noise-bias option does not select them. Here, `4` is an explicitly supplied example of an effective coil parameter, **not a default or a value inferred from hardware**.

An externally prepared magnitude-mean estimate can instead use its existing magnitude-noise map without running denoising again:

```sh
pydke denoised_magnitude_mean.nii.gz -o fresh_output \
  --sos-noise-correct \
  --sos-effective-coils 4 \
  --sos-input-domain magnitude-mean \
  --sos-sigma magnitude_noise_map.nii.gz \
  --sos-sigma-convention magnitude \
  --nthreads 1
```

Use `--sos-effective-coils-map effective_coils.nii.gz` instead of `--sos-effective-coils` for a spatial map. The scalar and map alternatives are mutually exclusive. The effective parameter may be noninteger; it describes the statistical approximation, not a receiver inventory. The explicit `--sos-input-domain magnitude-mean` declaration describes the data entering the bias stage, after any explicitly requested denoising. A single raw magnitude observation or an averaged squared-magnitude image is not automatically such a mean estimate.

The correction runs on the native grid, after denoising when requested and before later interpolation. With `--standard`, it replaces the legacy Rician bias stage while retaining existing denoising choices. Explicit `--rician`, `--resume` and nonempty output directories are rejected for this mode. Use a fresh output directory to preserve the earlier result and correction provenance.

## Moment model

Let A ≥ 0 be the noise-free magnitude, σG the Gaussian standard deviation of each real or imaginary component under the effective model, and L the explicitly supplied effective coil parameter. Let μ and σM be the estimated magnitude mean and standard deviation. For σG > 0, define ρ = A/σG and

\[
\beta_L=\sqrt{2}\,\frac{\Gamma(L+1/2)}{\Gamma(L)},
\qquad
f_L(\rho)=\beta_L\,{}_1F_1\!\left(-\tfrac12;L;-\tfrac{\rho^2}{2}\right).
\]

The noncentral-χ moments satisfy

\[
\mu=\sigma_G f_L(\rho),
\qquad
\sigma_M^2=\sigma_G^2 v_L(\rho),
\qquad
v_L(\rho)=\rho^2+2L-f_L(\rho)^2.
\]

For positive σM, generalized Koay inversion solves

\[
\frac{\mu}{\sigma_M}=\frac{f_L(\rho)}{\sqrt{v_L(\rho)}}
\]

and recovers

\[
\widehat{\sigma_G}=\frac{\sigma_M}{\sqrt{v_L(\rho)}},
\qquad
\widehat A=\rho\widehat{\sigma_G}.
\]

L = 1 is the Rician special case. This is **model-based estimation** of σG from magnitude moments, not an independent measurement of receiver noise. Koay and Basser developed the magnitude-moment inversion framework, including its multiple-receiver generalization. [Koay and Basser, 2006](https://doi.org/10.1016/j.jmr.2006.01.016).

The MP-PCA literature distinguishes the magnitude mean/dispersion estimated from magnitude data from the underlying Gaussian parameters. That distinction motivates this inversion rather than direct substitution of the MP-PCA noise map as σG. [Veraart, Fieremans and Novikov, 2016](https://doi.org/10.1002/mrm.26059).

## Boundary behavior and QC

At A = 0, the central-χ mean-to-standard-deviation ratio is

\[
r_0(L)=\frac{\beta_L}{\sqrt{2L-\beta_L^2}}.
\]

An estimated ratio at or below this boundary cannot yield a positive-amplitude interior solution. The amplitude is set to zero and the boundary condition is recorded in QC masks and counts. The boundary Gaussian noise estimate is a projection that matches the supplied dispersion at A = 0; for a subcentral ratio it cannot also match the supplied mean. It remains flagged and is not evidence of independent calibration. Zero supplied dispersion takes the identity path for a valid nonnegative mean estimate.

No absolute-value reflection, random perturbation or positive signal floor is used. Preserve boundary flags and zero outputs when evaluating downstream fit eligibility. They must not be interpreted as confident evidence that tissue signal is absent. Retain the magnitude dispersion input, derived Gaussian noise estimates, correction configuration and QC results with the run.

## Optional supplied-Gaussian mode

An independently supplied Gaussian component-noise map remains supported explicitly:

```sh
pydke denoised_magnitude_mean.nii.gz -o fresh_output \
  --sos-noise-correct \
  --sos-effective-coils 4 \
  --sos-input-domain magnitude-mean \
  --sos-sigma sigma_gaussian.nii.gz \
  --sos-sigma-convention per-component-gaussian \
  --nthreads 1
```

This mode uses the given σG and inverts the first moment μ = σG fL(A/σG) for A. It does not jointly re-estimate σG. At or below the central-χ mean σGβL, the amplitude is zero with a QC flag; sigma zero takes the identity path. A Gaussian-mode file must be supplied explicitly: a generated MP-PCA magnitude map is not silently reinterpreted as Gaussian noise.

Both conventions require finite, nonnegative standard deviations in matching intensity units. L must be finite and in the supported numerical range [1,4096]. Three-dimensional maps must match the native spatial shape and affine; this option does not register or interpolate a noise or effective-coil map. The conventions are not interchangeable with combined complex-noise power or post-denoising residual standard deviation.

## Why raw power subtraction is different

The raw second moment satisfies

\[
E[M^2]=A^2+2L\sigma_G^2,
\qquad
\mu^2=A^2+2L\sigma_G^2-\operatorname{Var}(M).
\]

Thus subtracting 2LσG² from an estimated raw second moment and subtracting it from a squared denoised mean are different operations. Neither automatically implements the magnitude-moment inversion above. This feature does not convert an arbitrary raw second-moment input into a magnitude mean. [Veraart, Fieremans and Novikov, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4879661/).

## Reconstruction and estimation limits

A surface coil describes receive sensitivity; sum of squares describes how receiver images are combined. An acquisition can use both. Spatially varying sensitivity/SNR does not alone establish a spatially varying Gaussian noise variance.

Physical receiver count does not establish effective L. Correlated channels, unequal coil noise and reconstruction can require effective parameters and make the noncentral-χ model approximate. The supplied L scalar/map must therefore retain its provenance or reference-assumption label. [Aja-Fernández and Tristán-Vega, 2012](https://doi.org/10.1002/mrm.23020).

A typical MP-PCA noise map shares one estimated magnitude dispersion across measurements at a voxel. The true magnitude variance can depend on signal level, so that shared estimate approximates per-measurement dispersion when SNR varies. Patch heterogeneity, reconstruction correlations and spatial noise variation add uncertainty. A denoised or eigenvalue-shrunk output is also only an estimate of the magnitude expectation, not a guaranteed exact moment. These limitations apply even when inversion converges and its mathematical forward check passes.

Accordingly, a missing external Gaussian calibration does **not** block the magnitude-moment feature or a bounded pilot. It does mean that the recovered σG is an inferred effective quantity whose sensitivity to L, the supplied/estimated moments and the model should be reported. Do not claim exact bias removal after MP-PCA shrinkage, improved biological accuracy from reduced kurtosis extremes, or scanner calibration from successful execution. [Veraart, Fieremans and Novikov, 2016](https://doi.org/10.1002/mrm.26059).

Apply this option only to magnitude data that have not already received a noise-floor correction. Do not correct earlier Rician/SoS-corrected images again, or reinterpret their residual noise without a justified model.

## Attribution

Credit Cheng Guan Koay and Peter J. Basser for the generalized magnitude-moment inversion framework; Jelle Veraart, Els Fieremans and Dmitry S. Novikov for MP-PCA noise mapping and its connection to magnitude-noise statistics; and Santiago Aja-Fernández and Antonio Tristán-Vega for effective noise models with correlated receiver coils. These citations acknowledge the scientific framework and do not imply that the original authors validated this implementation or any particular acquisition.
