"""Generate BUAN point assignments and native-space scalar profiles."""
import csv
import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from dipy.io.streamline import load_tractogram
from dipy.stats.analysis import assignment_map


def profile(model_path, common_path, native_path, metric_paths, mask_path, out_dir, disks=100):
    """Save point-level observations and unweighted disk means for QC.

    Values with incomplete mask support, out-of-image coordinates, or
    non-finite interpolation support are missing, never filled with zero.
    """
    if disks < 2:
        raise ValueError('At least two disks are required')
    model, common, native = [load_tractogram(str(p), 'same', bbox_valid_check=False).streamlines
                             for p in (model_path, common_path, native_path)]
    if not len(model) or not len(common) or len(common) != len(native):
        raise ValueError('Missing or mismatched bundle streamlines')
    lengths = np.array([len(s) for s in native])
    if not np.array_equal(lengths, [len(s) for s in common]):
        raise ValueError('Native and common bundles must match point-for-point')
    result = assignment_map(common, model, disks)
    assignment = result[1] if isinstance(result, tuple) else result
    assignment = np.asarray(assignment)
    if len(assignment) != lengths.sum() or np.any((assignment < 0) | (assignment >= disks)):
        raise ValueError('Assignment has unexpected size or multiple centroids')
    mask = nib.load(str(mask_path))
    vox = nib.affines.apply_affine(np.linalg.inv(mask.affine), native._data)
    coords = vox.T
    support = map_coordinates((np.asarray(mask.dataobj) > 0).astype(float), coords,
                              order=1, mode='constant', cval=0, prefilter=False)
    stream_ids = np.repeat(np.arange(len(native)), lengths)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for metric_path in metric_paths:
        img = nib.load(str(metric_path))
        if img.shape != mask.shape or not np.allclose(img.affine, mask.affine, atol=1e-4, rtol=0):
            raise ValueError(f'Metric grid does not match native mask: {metric_path}')
        data = np.asarray(img.dataobj, dtype=float)
        finite = np.isfinite(data)
        weights = map_coordinates(finite.astype(float), coords, order=1, mode='constant', cval=0, prefilter=False)
        values = map_coordinates(np.where(finite, data, 0), coords, order=1, mode='constant', cval=0, prefilter=False)
        valid = (support >= 0.999) & (weights >= 0.999)
        values[~valid] = np.nan
        name = Path(metric_path).name.removesuffix('.gz').removesuffix('.nii')
        with gzip.open(out / f'{name}_points.csv.gz', 'wt', newline='') as f:
            w = csv.writer(f); w.writerow(['streamline', 'disk', 'value', 'valid'])
            w.writerows(zip(stream_ids, assignment + 1, values, valid.astype(int)))
        with (out / f'{name}_disks.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['disk', 'n_points', 'n_valid', 'mean'])
            for disk in range(disks):
                chosen = assignment == disk
                ok = chosen & valid
                w.writerow([disk + 1, int(chosen.sum()), int(ok.sum()), float(values[ok].mean()) if ok.any() else ''])
