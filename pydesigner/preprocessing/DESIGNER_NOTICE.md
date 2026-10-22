# DESIGNER adaptive MP-PCA provenance

`_designer_mppca.py` contains the MP class from
https://github.com/NYU-DiffusionMRI/DESIGNER-v2/blob/8588f44c46dc4662ba9b477ac0b5893320d1e863/lib/mpdenoise_adaptive.py
(upstream commit 8588f44c46dc4662ba9b477ac0b5893320d1e863).
The original author, copyright, permission notice and research-use conditions
are retained in the source. This component is for non-commercial research,
not clinical care; see the complete notice in that file.

Local changes: omit upstream process/CLI convenience wrappers; remove global
warning suppression; default SVD attempts to one (fail rather than add random
noise); return unchanged signal and zero sigma/rank for constant patches.
The adaptive patch selection, wrap padding, noise estimator and Frobenius
shrinkage are otherwise retained. The pyDKE adapter selects Cordero-Grande
noise estimation, accepts only real magnitude data, validates patch sizes,
and runs bounded batches with explicitly limited workers and BLAS threads.

References:
- Veraart, Fieremans & Novikov (2016), doi:10.1002/mrm.26059.
- Cordero-Grande et al. (2019), doi:10.1016/j.neuroimage.2019.03.061.
- Gavish & Donoho (2017), doi:10.1109/TIT.2017.2653801.
- Chen et al. (2024), Optimization and validation of the DESIGNER preprocessing
  pipeline for clinical diffusion MRI in white matter aging, Imaging Neuroscience.

Python 3.14 / NumPy 2 compatibility: `np.matrix` multiplications were replaced
with equivalent ndarray `@` operations, checked against pinned reference data.
