"""DESIGNER-v2 RPG adapter. Core: Kellner, Lee; C++: Coronado-Leija;
pybind11: Ades-Aron. See ACKNOWLEDGMENTS.md and rpg_backend source license.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import nibabel as nib
import numpy as np


def degibbs_array(data, pf, pe_axis, slice_axis, nthreads=1):
    """Correct real magnitude data using explicit axes in the input voxel grid."""
    data = np.asarray(data)
    if data.ndim != 4 or np.iscomplexobj(data) or not np.isfinite(data).all():
        raise ValueError("RPG requires finite real 4D magnitude data")
    if pf not in (0.625, 0.75, 0.875, 1.0):
        raise ValueError("RPG supports Fourier fractions 5/8, 6/8, 7/8 and 1")
    if pe_axis not in (0, 1, 2) or slice_axis not in (0, 1, 2) or pe_axis == slice_axis:
        raise ValueError("Distinct spatial PE and slice axes are required")
    if not isinstance(nthreads, int) or nthreads < 1:
        raise ValueError("nthreads must be positive")
    read_axis = next(i for i in range(3) if i not in (pe_axis, slice_axis))
    if min(data.shape[read_axis], data.shape[pe_axis]) < 16:
        raise ValueError("RPG requires at least 16 voxels in each in-plane dimension")
    try:
        from . import _rpg
    except ImportError as exc:
        raise RuntimeError("RPG backend is not installed. Build with PYDKE_BUILD_RPG=1 and FFTW; see READMEv2.0.0.md") from exc
    order = (3, slice_axis, pe_axis, read_axis)
    # Backend modifies input (absolute value); always supply a private C-order copy.
    arranged = np.array(data.transpose(order), dtype=np.float64, order='C', copy=True)
    corrected, _ = _rpg.unring(arranged, pfv=pf, pfdimf=True, phase_flag=False, nthreads=nthreads)
    result = np.asarray(corrected).transpose(np.argsort(order)).astype(np.float32)
    if result.shape != data.shape or not np.isfinite(result).all():
        raise ValueError("RPG produced invalid output")
    return result


def degibbs_mif(input, output, pf, pe_axis, slice_axis, nthreads, force, verbose):
    from ..system.models import input_path_validator, output_path_validator
    source = input_path_validator(input, '.mif')
    destination = output_path_validator(output, '.mif')
    if Path(destination).exists() and not force:
        raise FileExistsError(destination)
    options = [] if verbose else ['-quiet']
    if nthreads is not None:
        options += ['-nthreads', str(nthreads)]
    with TemporaryDirectory(prefix='pydke-rpg-', dir=Path(destination).parent) as tmp:
        tmp = Path(tmp)
        raw, corrected = tmp/'input.nii', tmp/'corrected.nii'
        grad, props = tmp/'grad.txt', tmp/'properties.json'
        subprocess.run(['mrconvert', source, str(raw), '-export_grad_mrtrix', str(grad),
                        '-json_export', str(props), *options], check=True)
        image = nib.load(raw)
        result = degibbs_array(np.asarray(image.dataobj), pf, pe_axis, slice_axis,
                               1 if nthreads is None else nthreads)
        header = image.header.copy(); header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(result, image.affine, header), corrected)
        subprocess.run(['mrconvert', str(corrected), destination, '-grad', str(grad),
                        '-json_import', str(props), *(['-force'] if force else []), *options], check=True)
