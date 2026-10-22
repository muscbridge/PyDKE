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
