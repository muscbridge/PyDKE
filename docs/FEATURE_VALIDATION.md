# Feature-branch validation criteria

Recorded 2026-10-23. This document defines software gates and the evidence needed
to evaluate the combined preprocessing and fitting changes. It contains no study
inputs, subject identifiers or study results. Numerical tolerances below are
software checks; they are not biological validity thresholds.

## Branch policy and approval

| Branch | Purpose | Integration policy |
| --- | --- | --- |
| `feature/preprocessing-integration` | Original MRtrix MP-PCA plus optional DESIGNER adaptive MP-PCA and RPG Gibbs correction | Keep separate from `master` during validation. |
| `codex/azm-small-animal-validation` | Combined preprocessing code, fitting corrections, optional global PSD/SOS constraints and corrected scalar calculations | Continue testing on this branch; keep its PR in draft against `feature/preprocessing-integration`. |

**Do not merge either branch into `master`, or release it as the production
replacement, without the user's explicit approval.** Passing tests does not grant
merge approval. A later fitting-to-integration merge and a later proposal for
`master` are separate decisions. Record the exact commit used for every run;
earlier evidence must not be relabelled as validation of newer code.

Preserve original inputs, existing MATLAB outputs and earlier experimental
outputs. Use fresh output directories for comparisons. Keep study-specific
records and images in the local validation project; publish only software code,
data-free tests and general documentation unless separately authorized.

## Gates shared by both branches

1. Run the relevant test suite on the exact candidate commit in supported Python
   3.14. Record package versions, source/import paths, platform, test counts,
   warnings and every skip. A skipped test supplies no evidence for its behavior.
2. Build the universal wheel and source distribution; build the optional native
   RPG wheel where supported. Check package metadata and import/test installed
   artifacts outside the source checkout. Test the optional `physical` extra and
   require the native extension when claiming RPG coverage.
3. Preserve input dimensions, voxel sizes, affine/orientation, measurement order,
   gradients and relevant acquisition metadata, except for explicitly requested
   transformations. Confirm b-value units: input s/mm² and fitting ms/µm²;
   report diffusivity in µm²/ms or its stated equivalent.
4. Reject invalid configuration explicitly. Verify method and parameter changes
   cannot silently reuse incompatible preprocessing products. Fitting must be
   recomputed when fitting settings change, even if old scalar maps exist.
5. Record commands, all options/defaults, dependency versions, thread limits,
   input hashes and output provenance. Exclude Apple-generated companion files
   and metadata directories from traversal and parsing; do not delete them.
6. Test bounded operation suitable for the available computer. Start validation
   with one job and one worker; record any increase. Do not infer scalability or
   thermal safety from small test fixtures.

## Preprocessing integration gates

| Area | Required check | Existing test entry point |
| --- | --- | --- |
| Original behavior | Original MRtrix denoising and Gibbs choices remain available and default; verify adapter arguments and metadata. | `tests/test_adaptive_denoising.py`, existing preprocessing tests |
| Adaptive reference | Match the pinned DESIGNER fixture for signal/noise estimates at `rtol=2e-6`; match reference rank values. Preserve constant patches and reject nonfinite data or invalid patch settings. | `tests/test_adaptive_denoising.py` |
| Adaptive determinism | Require identical signal, noise and rank arrays for the existing one-versus-two-thread fixture. | `tests/test_adaptive_denoising.py` |
| RPG execution | Exercise supported Fourier fractions, constants, acquisition-axis changes, thread counts and metadata round trips. Existing thread/orientation/output comparisons use `atol=1e-6`; constant tests use `atol=1e-5`. | `tests/test_rpg.py` |
| Acquisition correctness | Require a verified acquired Fourier fraction and phase-encoding/slice axes; do not substitute acceleration factor or infer missing settings merely to run RPG. Apply RPG before interpolation. | CLI validation and acquisition review |
| Workflow | Verify preprocessing-only requests produce no fitting outputs and resume settings are checked. | Full CLI and resume tests |
| Attribution | Retain pinned upstream provenance, developer credit, notices and component-specific license text in source and distributions. | `ACKNOWLEDGMENTS.md`, `pydesigner/preprocessing/DESIGNER_NOTICE.md`, shipped source |

Synthetic execution and determinism tests are necessary but do not demonstrate
preprocessing efficacy on an acquisition. Where suitable data are available,
inspect denoising residuals for structured anatomical signal, edge behavior,
noise/rank estimates and Gibbs changes. An upstream RPG numerical reference or
an acquisition-realistic partial-Fourier phantom remains useful additional
coverage; the present tests do not provide that independent efficacy reference.

## Fitting and scalar calculation gates

The default remains the sampled-direction estimator. The `global-sos` backend is
an explicit estimator choice, not a post-fit tensor repair. Evaluate only the
constraints selected by the three flags. Global-backend details and units are in
`docs/physical_dki.md` on the fitting branch.

| Check | Acceptance criterion |
| --- | --- |
| Tensor mathematics | Independent full-tensor contractions agree with packed constraints at `atol=rtol=1e-12`. Preserve antipodal invariance and rotation covariance. |
| Sampled constrained fitting | Actual minimum selected constraint margin is at least `-1e-6`; finite coefficients and acceptable solver status alone are insufficient. Failed array-only constrained fits raise; diagnostics must identify any invalid/fallback result. |
| Global diffusion constraint | When requested, raw minimum eigenvalue is positive and at least `dmin - min(1e-7, 0.01*dmin)`, with default `dmin=1e-6 µm²/ms`. |
| Global quartic constraints | For each requested lower/upper quartic, retain actual Gram eigenvalues and coefficient-equality errors. The conservative all-direction polynomial lower bound must be at least `-1e-7` in raw quartic units. Missing/invalid certificates cannot pass. |
| Recovery and invariance | Data-free noiseless recovery, all eight flag combinations, rotations, gradient reversal and signal scaling pass. Existing global coefficient recovery uses `atol=rtol=1e-5`. |
| RK calculation | Exact radial-circle integration agrees with independent adaptive quadrature at `atol=1e-8`, `rtol=1e-9`, including coincident radial eigenvalues and strongly unequal radial diffusivities. Isotropic limits agree at `atol=1e-12`. |
| Axis handling | Perpendicular samples are finite and normalized, lie in the requested plane and return exactly the requested count; test axes, antipodes and near-axis cases. |
| Invalid tensors | Preserve raw eigenvalue diagnostics; invalid raw tensors yield explicit invalid/undefined scalars. Do not manufacture finite scalars through absolute eigenvalues or ADC-floor repair. Preserve undefined values in floating-point outputs. |
| Boundary sensitivity | Inspect residuals and conditioning near the diffusivity boundary; compare `dmin=1e-7`, `1e-6` and `1e-5` without choosing settings to improve agreement with historical maps. |
| Public workflow | Exercise backend selection, exported diagnostics, scalar-only output and repeat fitting with changed settings. Reject incompatible `--akc`/`--window` combinations for the global backend. |

Primary tests are `tests/test_constraints_physics.py`,
`tests/test_physical_dki.py` and `tests/test_fitting_dwipy.py`. Dense directional
sampling can corroborate an SOS certificate but is not an all-direction proof.
The constraints concern the fitted cumulant model and do not prove a globally
nonnegative diffusion propagator or biological plausibility.

**RK above 3 is allowed.** No scalar ceiling of 3, clipping, exclusion rule or
automatic failure may be introduced on that basis. The directional upper
constraint is `K(g) <= 3 / (bmax * D(g))`, whose value can exceed 3. Investigate
measurement consistency, parameter sensitivity and anatomy independently of this
historical display limit.

## Combined-code comparisons

Separate two questions so differences remain interpretable:

1. **Fitting comparison:** use identical frozen preprocessed signal, gradients,
   geometry and support for pyDKE and the historical estimator. Record weighting,
   intercept, constraints, signal-floor, integration and output-clipping
   conventions. Compare raw tensor-derived values where available; label saved
   clipped MATLAB maps as a different comparison target.
2. **Preprocessing comparison:** start from suitable input preceding the tested
   operations, preserve the original pipeline arm and compare each changed
   operation deliberately. Do not denoise an already denoised reference merely
   to exercise the new option.

Retain human fixture regression coverage while prioritizing an independent
small-animal test case chosen without inspecting diffusion outcomes. Inspect
native-grid scalar maps, signed differences, regional bias/absolute error,
residuals, signal consistency, mask inclusion and parameter sensitivity. Include
invalid/missing fit counts and the exact population used for each metric.
Correlation alone, particularly against clipped maps, is not an agreement gate.
Any empirical equivalence margins must be justified by intended use or
repeatability evidence before interpreting the comparison; do not derive a
passing cutoff from observed results.

## Recorded software evidence and open gates

These entries distinguish historical test evidence from outstanding validation.
Rerun affected gates after code, dependency or build changes.

| Date | Software revision | Recorded evidence or scope |
| --- | --- | --- |
| 2026-10-22 | `8bad3cc00558124d0104bdee8af1a80a9a305686` | Preprocessing source and installed native-wheel suites: 151 passed each. Linux: 150 passed, one FSL-dependent full CLI skip. Four distribution metadata checks passed. Recorded in the local software validation manifest. |
| 2026-10-22 | `a6c20776d995ee79ae96ae3376d8d5ec1764780a` | Preprocessing integration merge commit; this is the feature-branch integration, not a merge into `master`. |
| 2026-10-22 | `209982d36cec761565e36da1fd8c13b16f1e9e5d` | Earlier sampled-constraint corrections and solver-budget revision. Its evidence is historical and does not validate the later global backend. |
| 2026-10-22 | `dc0cc74f74aa1b008ccf4bb360eef7f515dec57c` | Combined source revision introducing the optional global backend and stable raw tensor scalars; installed-suite record reports 173 passed. Confirm current candidate and rerun affected tests before integration. |
| 2026-10-23 | `dc0cc74f74aa1b008ccf4bb360eef7f515dec57c` | Installed native-wheel physical fitting on the existing five-voxel human excerpt, existing sampled regression, adaptive synthetic execution and compiled RPG axis-invariance execution passed outside the checkout; imported source hashes matched. This is bounded regression coverage. |
| Pending | Current candidate | Reviewed independent combined-code case comparisons and full CLI coverage appropriate to the candidate environment; preprocessing efficacy remains separate from numerical execution. |

Known limits remain explicit: the human regression fixture is a small image
excerpt, not whole-human validation; the Linux workflow lacks FSL for one full
CLI test; the universal-wheel smoke test does not itself exercise physical fitting
or native RPG; current RPG tests do not establish partial-Fourier efficacy against
an independent reference; and a single study case cannot establish performance
across populations, protocols or pathology. Passing numerical gates allows
further testing. Readiness to replace MATLAB DKE requires reviewed comparison
evidence and the user's separate approval to merge into `master`.
