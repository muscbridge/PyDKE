"""Participant-level experimental tractography and BUAN preparation.

This module deliberately requires an explicit tracking cutoff: diffusion ODF
amplitudes are not calibrated like MRtrix CSD FODs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(odf, mask, reference, metrics, exclude_extremes=False):
    images = [nib.load(str(p)) for p in [odf, mask, reference, *metrics]]
    sh = images[0]
    if len(sh.shape) != 4 or sh.shape[-1] not in (1, 6, 15, 28, 45, 66, 91):
        raise ValueError('ODF must contain a valid even-order symmetric SH coefficient count')
    for image in images[1:]:
        if image.shape[:3] != sh.shape[:3] or not np.allclose(image.affine, sh.affine, atol=1e-4, rtol=0):
            raise ValueError('Input image grids or affines do not match; register before running')
        if len(image.shape) != 3:
            raise ValueError('Mask, reference and metrics must be 3D')
    mask_data = np.asarray(images[1].dataobj)
    if not np.isfinite(mask_data).all() or not np.any(mask_data > 0):
        raise ValueError('Mask must be finite and nonempty')
    selected = mask_data > 0
    data = np.asarray(sh.dataobj)
    if not np.isfinite(data[selected]).all():
        raise ValueError('Nonfinite SH coefficients inside mask')
    extreme = np.max(np.abs(data), axis=-1) > 1000
    if np.any(extreme & selected) and not exclude_extremes:
        raise ValueError('Extreme SH coefficients (>1000) inside mask; review ODF before tracking')
    for name, image in zip(metrics, images[3:]):
        if not np.isfinite(np.asarray(image.dataobj)[selected]).all():
            raise ValueError(f'Nonfinite scalar values inside mask: {name}')
    return {'shape': list(sh.shape), 'affine': sh.affine.tolist(),
            'mask_voxels': int(selected.sum()), 'extreme_voxels': int((extreme & selected).sum()),
            'c00_percentiles': np.percentile(data[..., 0][selected], [0, 5, 50, 95, 100]).tolist()}


def run_command(cmd, log, env):
    with open(log, 'w') as out:
        out.write('$ ' + ' '.join(map(str, cmd)) + '\n')
        out.flush()
        subprocess.run(list(map(str, cmd)), stdout=out, stderr=subprocess.STDOUT, env=env, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--odf', type=Path, required=True, help='pyDKE MRtrix-basis SH ODF NIfTI')
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True, help='native 3D diffusion reference')
    p.add_argument('--bvec', type=Path, required=True, help='gradient file actually used to fit pyDKE')
    p.add_argument('--gradient-mif', type=Path, required=True, help='matching preprocessed MRtrix DWI with scanner gradients')
    p.add_argument('--exclude-extreme-voxels', action='store_true', help='Experimental QC: exclude coefficients exceeding 1000, never clip them')
    p.add_argument('--metric', type=Path, action='append', default=[])
    p.add_argument('--atlas-wholebrain', type=Path, required=True)
    p.add_argument('--atlas-bundle', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cutoff', type=float, required=True, help='experimental ODF-amplitude tracking cutoff')
    p.add_argument('--count', type=int, default=250000)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--basis', choices=['pydke-tournier'], required=True,
                   help='explicitly confirm provenance; no basis conversion is performed')
    p.add_argument('--coordinate-provenance-verified', action='store_true', help='Attest that export version and physical orientation have been independently validated')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(argv)
    if a.cutoff <= 0 or a.count <= 0 or a.threads <= 0:
        p.error('cutoff, count and threads must be positive')
    paths = [a.odf, a.mask, a.reference, a.bvec, a.gradient_mif, *a.metric, a.atlas_wholebrain, *a.atlas_bundle]
    for path in paths:
        if not path.is_file():
            p.error(f'Input not found: {path}')
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out / 'manifest.json'
    if manifest.exists():
        raise FileExistsError('Output already contains a manifest; choose a new directory')
    checks = inspect(a.odf, a.mask, a.reference, a.metric, a.exclude_extreme_voxels)
    info = {'state': 'preflight_passed', 'basis_assertion': a.basis,
            'inputs': {str(x.resolve()): sha256(x) for x in paths}, 'inspection': checks,
            'cutoff': a.cutoff, 'count': a.count, 'threads': a.threads,
            'warning': 'Exploratory DKI diffusion ODF, not a CSD fiber ODF; scientific QC required.'}
    def save():
        manifest.write_text(json.dumps(info, indent=2) + '\n')
    save()
    if a.preflight_only:
        return 0
    if not a.coordinate_provenance_verified:
        info['state'] = 'blocked_coordinate_validation'; save()
        raise RuntimeError('Validate pyDKE export version and physical orientation before tracking; see documentation')
    from io import StringIO
    from pydesigner.buan.sh import pydke_to_mrtrix, rotate_mrtrix_sh, gradient_frame
    gradients = np.loadtxt(StringIO(subprocess.check_output(['mrinfo', str(a.gradient_mif), '-dwgrad'], text=True)))
    bvec = np.loadtxt(a.bvec).T
    matrix, residual = gradient_frame(bvec, gradients)
    info['gradient_source_to_world'] = matrix.tolist(); info['gradient_max_error'] = residual
    source = nib.load(str(a.odf))
    raw = np.asarray(source.dataobj)
    extreme = np.max(np.abs(raw), axis=-1) > 1000
    native_mask = nib.load(str(a.mask))
    cleaned_mask = (np.asarray(native_mask.dataobj) > 0) & ~extreme
    if not np.any(cleaned_mask):
        raise ValueError('QC exclusions removed the entire mask')
    clean_mask_path = a.out / 'tracking_mask.nii.gz'
    nib.save(nib.Nifti1Image(cleaned_mask.astype(np.uint8), native_mask.affine), str(clean_mask_path))
    safe = np.where(cleaned_mask[..., None], raw, 0)
    converted = rotate_mrtrix_sh(pydke_to_mrtrix(safe), matrix)
    corrected = a.out / 'pydke_sh_mrtrix.nii.gz'
    nib.save(nib.Nifti1Image(converted.astype(np.float32), source.affine, source.header), str(corrected))
    info['converted_sh'] = str(corrected.resolve()); info['converted_sha256'] = sha256(corrected); save()
    env = os.environ.copy()
    env.update({'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str((a.out / 'mpl').resolve()),
                'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': str(a.threads)})
    out = a.out.resolve()
    def execute(name, cmd):
        info['state'] = name
        save()
        run_command(cmd, out / (name + '.log'), env)
    try:
        for executable in ['tckgen', 'dipy_convert_tractogram', 'dipy_slr', 'dipy_recobundles', 'dipy_labelsbundles']:
            if not shutil.which(executable):
                raise RuntimeError(f'Missing executable: {executable}')
        native = out / 'native.tck'
        execute('tracking', ['tckgen', corrected.resolve(), native, '-algorithm', 'iFOD2',
                 '-seed_image', clean_mask_path.resolve(), '-mask', clean_mask_path.resolve(), '-select', str(a.count),
                 '-minlength', '20', '-maxlength', '250', '-cutoff', str(a.cutoff), '-nthreads', str(a.threads)])
        execute('convert', ['dipy_convert_tractogram', native, '--reference', a.reference.resolve(),
                 '--out_dir', out, '--out_tractogram', 'native.trk'])
        slr = out / 'slr'
        slr.mkdir()
        execute('slr', ['dipy_slr', a.atlas_wholebrain.resolve(), out / 'native.trk',
                 '--num_threads', str(a.threads), '--out_dir', slr, '--out_moved', 'moved.trk'])
        if not (slr / 'moved.trk').is_file():
            raise RuntimeError('SLR produced no moved tractogram; inspect slr.log')
        rec = out / 'rec_bundles'
        org = out / 'org_bundles'
        rec.mkdir(); org.mkdir()
        # One invocation per model avoids positional pairing by independent sorted lists.
        info['bundles'] = {}
        for bundle in a.atlas_bundle:
            name = bundle.stem
            execute('recognize_' + name, ['dipy_recobundles', slr / 'moved.trk', bundle.resolve(),
                    '--mix_names', '--out_dir', rec, '--out_recognized_transf', 'recognized.trk'])
            labels = list(rec.glob('*' + name + '*labels*.npy'))
            if len(labels) != 1:
                raise RuntimeError(f'Expected exactly one label file for {name}, got {labels}')
            label_data = np.load(labels[0])
            info['bundles'][name] = {'streamlines': int(label_data.size)}; save()
            if label_data.size < 6:
                info['bundles'][name]['status'] = 'insufficient streamlines for profile'; save()
                continue
            execute('native_' + name, ['dipy_labelsbundles', out / 'native.trk', labels[0],
                    '--mix_names', '--out_dir', org, '--out_bundle', 'recognized_orig.trk'])
            recognized = list(rec.glob('*' + name + '*recognized*.trk'))
            original = list(org.glob('*' + name + '*recognized*.trk'))
            if len(recognized) != 1 or len(original) != 1:
                raise RuntimeError(f'Could not uniquely pair native and recognized bundle: {name}')
            from dipy.io.streamline import load_tractogram
            whole = load_tractogram(str(out / 'native.trk'), 'same', bbox_valid_check=False).streamlines
            subset = load_tractogram(str(original[0]), 'same', bbox_valid_check=False).streamlines
            if len(subset) != len(label_data) or not all(
                np.allclose(whole[int(index)], subset[i], atol=1e-4)
                for i, index in enumerate(label_data)
            ):
                raise ValueError('Native bundle streamlines do not match recognized labels')
            if a.metric:
                from pydesigner.buan.profiles import profile
                info['state'] = 'profile_' + name; save()
                profile(bundle, recognized[0], original[0], a.metric, clean_mask_path, out / 'profiles' / name)
        info['state'] = 'complete' if all(v['streamlines'] >= 6 for v in info['bundles'].values()) else 'complete_with_missing_bundles'
        info['note'] = 'Exploratory point assignments and disk summaries; visual and scientific QC required.'
        save()
    except BaseException as exc:
        info['state'] = 'failed'; info['error'] = repr(exc); save()
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
