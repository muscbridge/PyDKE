# Acknowledgments and third-party preprocessing

pyDKE's optional adaptive MP-PCA and RPG choices build on the work of the
DESIGNER-v2 developers at NYU Diffusion MRI and their collaborators:
https://github.com/NYU-DiffusionMRI/DESIGNER-v2.

## Adaptive MP-PCA

Benjamin Ades-Aron authored the vendored MP implementation, copyright 2016
New York University. Noise estimation builds on Jelle Veraart, Els Fieremans,
Dmitry Novikov and colleagues; the Cordero-Grande estimator and Gavish–Donoho
Frobenius shrinkage are used in the adaptive option. Source provenance, local
changes and the complete non-commercial research conditions are in
`pydesigner/preprocessing/DESIGNER_NOTICE.md` and `_designer_mppca.py`.

- Veraart, Fieremans & Novikov (2016), doi:10.1002/mrm.26059.
- Cordero-Grande et al. (2019), doi:10.1016/j.neuroimage.2019.03.061.
- Gavish & Donoho (2017), doi:10.1109/TIT.2017.2653801.
- Chen et al. (2024), Optimization and validation of the DESIGNER preprocessing
  pipeline for clinical diffusion MRI in white matter aging, Imaging Neuroscience.

## RPG Gibbs correction

The RPG core credits Elias Kellner (original subvoxel unringing) and Hong-Hsi
Lee (partial-Fourier strategy), Ricardo Coronado-Leija (MATLAB-to-C++ and CLI
adaptation), and Benjamin Ades-Aron (pybind11 and multithreading). The original
license remains in the shipped C++ source; the software remains the property
of University Medical Center Freiburg and is licensed for non-commercial use.

- Kellner, Dhital, Kiselev & Reisert, Gibbs-ringing artifact removal based on
  local subvoxel-shifts, doi:10.1002/mrm.26054.
- Lee, Novikov & Fieremans (2021), Removal of partial Fourier-induced Gibbs
  (RPG) ringing artifacts in MRI, Magnetic Resonance in Medicine 86:2733–2750,
  doi:10.1002/mrm.28830.

Pinned source: DESIGNER-v2 commit d4979b1cb4e4235bf8341224af3d794d5ac15021,
`rpg_cpp/unring_rpg_pybind.cpp`. Original Git blob:
35ebacddede9e8a388e971dfcea9dafc117535da.
Local changes rename the extension to `_rpg` and expose a positive thread limit,
bounded by slice/volume count. The correction algorithms are retained. The
Python adapter requires explicit acquisition axes and Fourier fraction, supplies
a private contiguous input copy, restores metadata and guards resume reuse.
All original and amended RPG source is included in source and wheel distributions.

Please cite these methods and DESIGNER when using these preprocessing options,
as well as pyDKE. Their licenses are component-specific and are not replaced by
pyDKE's project license.

## Optional SoS/noncentral-chi magnitude-bias model

The separate magnitude-noise-bias stage credits Cheng Guan Koay and Peter J.
Basser for the generalized magnitude-moment inversion framework, including the
multiple-receiver case; Jelle Veraart, Els Fieremans and Dmitry S. Novikov for
MP-PCA noise mapping and the distinction between magnitude dispersion and
underlying Gaussian noise; and Santiago Aja-Fernández and Antonio Tristán-Vega
for effective noise models with correlated receiver coils. André and colleagues
provide related first-/second-moment noise-correction analysis in DKI.

- Koay & Basser (2006), doi:10.1016/j.jmr.2006.01.016.
- André et al. (2014), doi:10.1371/journal.pone.0094531.
- Veraart, Fieremans & Novikov (2016), doi:10.1002/mrm.26059.
- Aja-Fernández & Tristán-Vega (2012), doi:10.1002/mrm.23020.

The new numerical implementation is local pyDKE code and does not replace or
modify the credited original/adaptive MP-PCA algorithms. Its primary magnitude
mode applies generalized Koay mean/variance inversion to the denoised magnitude
mean and MP-PCA magnitude-noise profile. It infers an effective Gaussian
component standard deviation under that model; an independent Gaussian
calibration file is not required. An optional supplied-Gaussian convention
retains first-moment inversion when component sigma is explicitly available.

Both conventions require an explicitly supplied effective coil parameter
(scalar or spatial map, supported numerical range 1 to 4096, fractions allowed).
The parameter is not inferred from physical receiver count. This is an optional
alternative to the separate legacy Rician bias stage, not a change to denoising.
The citations acknowledge the scientific framework and do not imply that the
original authors validated this implementation or a particular acquisition.
See `docs/sos_noise.md` for model assumptions, provenance and diagnostics.
