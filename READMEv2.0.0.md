# pyDKE

pyDKE is a hands-free diffusion MRI preprocessing and modeling pipeline for DTI, DKI, FBI, FBWM, and related diffusion MRI analyses.

## Version

Current release: `2.0.0`

## Requirements

pyDKE 2.0.0 requires:

- Python `>=3.14`
- MRtrix3 available on your `PATH`
- FSL available on your `PATH`
- uv for Python environment and package installation

The pyDKE wheel is a pure-Python universal wheel:

```text
pydke-2.0.0-py3-none-any.whl
```

## Denoising choices

The original MRtrix MP-PCA remains the default, including with `--standard`:

```sh
pydke input.nii --denoise --denoise-method mppca --extent 5,5,5 -o original
```

DESIGNER-v2 adaptive patch MP-PCA with Frobenius eigenvalue shrinkage is available
as an alternative preprocessing option:

```sh
pydke input.nii --denoise-method adaptive --extent 7,7,7 --denoise-patch-size 274 --nthreads 3 -o adaptive
```

Selecting `adaptive` enables denoising and also works with `--standard`.
`--extent` is the odd-sized search window (default 5,5,5 for both methods).
Adaptive patch size defaults to floor(80% of the window's voxel count), and must
be greater than the number of volumes and smaller than the window. For large
acquisitions, increase the window and patch size explicitly. The adaptive
backend uses Cordero-Grande noise estimation and DESIGNER wrap padding. It
processes real 4D magnitude data; it does not perform complex/phase denoising.
It preserves the gradient table and acquisition properties and produces the
same `noisemap.nii` used by downstream Rician correction and QC. Rician correction
is still a separate option. No absolute-value rectification is added.

Adaptive processing uses `--nthreads` workers (one if omitted), with one BLAS
thread per worker. `denoising.json` records method, extent and patch size.
Changing them requires a new output directory; adaptive resume requires this
record to prevent reuse of outputs from another method. Historical original
MP-PCA runs without the record retain their existing resume behavior.

The DESIGNER implementation is pinned and vendored with its copyright and
non-commercial research license; see
[pinned source and modifications](pydesigner/preprocessing/DESIGNER_NOTICE.md).
This option does not change fitting or eddy correction.

## RPG Gibbs correction (optional compiled backend)

Original MRtrix Gibbs correction remains the default for `--degibbs`. To enable
DESIGNER-v2 RPG, build with FFTW development libraries and a C++ compiler:

```sh
# Debian/Ubuntu: sudo apt-get install g++ libfftw3-dev
# macOS: brew install fftw
PYDKE_BUILD_RPG=1 FFTW_PREFIX=/usr/local python -m pip install .
```

For Apple Silicon Homebrew set `FFTW_PREFIX=/opt/homebrew`; on Debian use
`FFTW_PREFIX=/usr`. A default Python-only wheel does not include the backend.
Build a native wheel with `PYDKE_BUILD_RPG=1 python -m build`; FFTW shared
libraries must be available at runtime. Cross-platform native wheel bundling is
not provided. Build isolation installs pybind11 automatically.

```sh
pydke input.nii -o corrected --gibbs-method rpg \
  --rpg-pf 0.75 --rpg-pe-axis 1 --rpg-slice-axis 2 --nthreads 2
```

The example is only for a 6/8 acquisition with PE along voxel axis 1 and slices
along axis 2. Supply your actual values in the input NIfTI voxel grid. Supported
Fourier fractions are 0.625, 0.75, 0.875 and 1.0; an acceleration factor is not a
Fourier fraction. Unsupported fractions are rejected rather than rounded.
Axes must differ and in-plane dimensions must be at least 16 voxels.
Do not apply RPG twice to an already corrected image. This adapter supports
real magnitude images, using the upstream absolute-value handling internally.
RPG runs after denoising and before distortion correction; `--reslice` is rejected
in the same invocation because interpolation must follow RPG.
`gibbs.json` records acquisition choices and protects against incompatible resumes.
Selecting RPG enables Gibbs correction, including in the standard pipeline, and
bypasses the original partial-Fourier exclusion only for this backend.

See [acknowledgments and source licenses](ACKNOWLEDGMENTS.md) for developer
credits, publications, pinned versions and local changes.
