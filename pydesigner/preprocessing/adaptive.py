"""Magnitude MP-PCA adapter for the pinned DESIGNER-v2 implementation."""
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import nibabel as nib
import numpy as np
from threadpoolctl import threadpool_limits

from ._designer_mppca import MP


def denoise_array(data, extent="5,5,5", patch_size=None, nthreads=None):
    """Return signal, sigma and rank; use adaptive patches and Frobenius shrinkage.

    Spatial boundaries use DESIGNER's wrap padding. No magnitude rectification
    or Rician bias correction is performed here.
    """
    data = np.asarray(data)
    if data.ndim != 4 or data.shape[-1] < 2 or np.iscomplexobj(data):
        raise ValueError("Adaptive denoising requires real 4D data with at least two volumes")
    if not np.isfinite(data).all():
        raise ValueError("Adaptive denoising requires finite input")
    try:
        kernel = tuple(int(x) for x in extent.split(","))
    except (ValueError, AttributeError) as exc:
        raise ValueError("extent must contain three odd integers >= 3") from exc
    if len(kernel) != 3 or any(x < 3 or x % 2 == 0 for x in kernel):
        raise ValueError("extent must contain three odd integers >= 3")
    count = int(np.prod(kernel))
    size = int(0.8 * count) if patch_size is None else patch_size
    if not isinstance(size, int) or not data.shape[-1] < size < count:
        raise ValueError("Adaptive patch size must exceed volume count and be smaller than the search window; increase --extent")
    threads = 1 if nthreads is None else nthreads
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("nthreads must be a positive integer")
    mp = MP(data.astype(np.float32), kernel, 'adaptive', size, 'frob', 'cordero-grande', 0)
    signal = np.empty(data.shape, dtype=np.float32)
    sigma = np.empty(data.shape[:3], dtype=np.float32)
    rank = np.empty(data.shape[:3], dtype=np.float32)
    offset = np.asarray(kernel) // 2

    def one(coord):
        result, noise, npars = mp.denoise(np.asarray(coord) + offset)
        return coord, np.asarray(result).ravel(), noise, npars

    # Bounded batches avoid queuing a Future for every voxel in a full volume.
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=threads) as pool:
        coords = np.ndindex(data.shape[:3])
        while True:
            batch = list(islice(coords, 256))
            if not batch:
                break
            for coord, result, noise, npars in pool.map(one, batch):
                signal[coord], sigma[coord], rank[coord] = result, noise, npars
    if not all(np.isfinite(x).all() for x in (signal, sigma, rank)):
        raise ValueError("Adaptive denoising produced nonfinite output")
    return signal, sigma, rank


def denoise_mif(input, output, noisemap, extent, nthreads, force, verbose, patch_size):
    """Convert through NIfTI and restore gradients and acquisition properties."""
    from ..system.models import input_path_validator, output_path_validator
    source = input_path_validator(input, ".mif")
    destination = output_path_validator(output, ".mif")
    if not isinstance(noisemap, bool):
        raise TypeError("noisemap must be a boolean")
    noise_path = Path(output).parent / 'noisemap.nii'
    if not force and (Path(output).exists() or (noisemap and noise_path.exists())):
        raise FileExistsError("Denoising output exists; use force to overwrite")
    opts = [] if verbose else ['-quiet']
    if nthreads is not None:
        opts += ['-nthreads', str(nthreads)]
    with TemporaryDirectory(prefix='pydke-denoise-', dir=Path(output).parent) as temp:
        temp = Path(temp)
        raw, denoised = temp/'input.nii', temp/'denoised.nii'
        grad, props = temp/'grad.txt', temp/'properties.json'
        subprocess.run(['mrconvert', str(source), str(raw), '-export_grad_mrtrix', str(grad),
                        '-json_export', str(props), *opts], check=True)
        image = nib.load(raw)
        signal, sigma, _ = denoise_array(np.asarray(image.dataobj), extent, patch_size, nthreads)
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(signal, image.affine, header), denoised)
        # Preserve phase encoding, gradient table and other MRtrix header properties.
        subprocess.run(['mrconvert', str(denoised), str(destination), '-grad', str(grad),
                        '-json_import', str(props), *(['-force'] if force else []), *opts], check=True)
        if noisemap:
            nib.save(nib.Nifti1Image(sigma, image.affine, header), noise_path)
