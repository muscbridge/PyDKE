import numpy as np
import nibabel as nib
import pytest
from dipy.reconst.shm import real_sh_tournier

from pydesigner.buan.pipeline import inspect
from pydesigner.buan.sh import pydke_to_mrtrix
from pydesigner.tractography.odf import shbasis


def test_sh_conversion_preserves_evaluated_function():
    rng = np.random.default_rng(452)
    theta = np.arccos(rng.uniform(-1, 1, 200))
    phi = rng.uniform(0, 2 * np.pi, 200)
    old = shbasis(np.arange(7), theta, phi, 'tournier').real
    indices = [j for l in range(0, 7, 2) for j in range(l*l, (l+1)*(l+1))]
    new = real_sh_tournier(6, theta, phi, legacy=False)[0]
    coeff = rng.normal(size=28)
    np.testing.assert_allclose(old[:, indices] @ coeff, new @ pydke_to_mrtrix(coeff), atol=1e-12)


def test_inspect_rejects_grid_mismatch_and_extremes(tmp_path):
    affine = np.eye(4)
    def save(name, data, aff=affine):
        path = tmp_path / name
        nib.save(nib.Nifti1Image(data, aff), path)
        return path
    data = np.zeros((3, 3, 3, 6), np.float32)
    sh = save('sh.nii', data)
    mask = save('mask.nii', np.ones((3, 3, 3), np.uint8))
    ref = save('ref.nii', np.ones((3, 3, 3), np.float32))
    assert inspect(sh, mask, ref, [])['mask_voxels'] == 27
    wrong = save('wrong.nii', np.ones((3, 3, 3), np.float32), np.diag([2,1,1,1]))
    with pytest.raises(ValueError, match='grids or affines'):
        inspect(sh, mask, wrong, [])
    data[1, 1, 1, 2] = 1001
    bad = save('bad.nii', data)
    with pytest.raises(ValueError, match='Extreme SH'):
        inspect(bad, mask, ref, [])


def test_gradient_frame_rotation_preserves_world_function():
    from pydesigner.buan.sh import gradient_frame, rotate_mrtrix_sh
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(83)
    vec = rng.normal(size=(100, 3)); vec /= np.linalg.norm(vec, axis=1)[:, None]
    matrix = Rotation.from_euler('xyz', [20, -35, 12], degrees=True).as_matrix()
    matrix[0] *= -1  # reflection, as well as an oblique rotation
    gradients = np.column_stack((vec @ matrix, np.full(len(vec), 1000)))
    estimated, error = gradient_frame(vec, gradients)
    assert error < 1e-12
    coeff = rng.normal(size=28)
    converted = rotate_mrtrix_sh(coeff, estimated)
    world = rng.normal(size=(200, 3)); world /= np.linalg.norm(world, axis=1)[:, None]
    def basis(d):
        return real_sh_tournier(6, np.arccos(d[:, 2]), np.arctan2(d[:, 1], d[:, 0]), legacy=False)[0]
    np.testing.assert_allclose(basis(world) @ converted, basis(world @ matrix.T) @ coeff, atol=1e-11)
    gradients[3, 0] += 0.1
    with pytest.raises(ValueError, match='orthogonal coordinate frame'):
        gradient_frame(vec, gradients)


def test_mrtrix_peaks_match_known_world_directions(tmp_path):
    import shutil
    import subprocess
    from scipy.spatial.transform import Rotation
    from pydesigner.buan.sh import rotate_mrtrix_sh
    if not shutil.which('sh2peaks'):
        pytest.skip('MRtrix sh2peaks not available')
    rng = np.random.default_rng(42)
    dirs = rng.normal(size=(1000, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    pybasis = shbasis(np.arange(7), np.arccos(dirs[:, 2]), np.arctan2(dirs[:, 1], dirs[:, 0]), 'tournier').real
    indices = [j for l in range(0, 7, 2) for j in range(l*l, (l+1)*(l+1))]
    source = np.array([[1, 0, 0], [0, 1, 0], [1, 2, 3]], float)
    source /= np.linalg.norm(source, axis=1)[:, None]
    coeff = np.stack([np.linalg.lstsq(pybasis[:, indices], np.exp(4*((dirs @ v)**2-1)), rcond=None)[0] for v in source])
    frame = Rotation.from_euler('xyz', [20, -15, 30], degrees=True).as_matrix()
    frame[0] *= -1
    converted = rotate_mrtrix_sh(pydke_to_mrtrix(coeff), frame)
    image = np.tile(converted[:, None, None, :], (1, 2, 2, 1)).astype(np.float32)
    affine = np.eye(4); affine[:3, :3] = Rotation.from_euler('xyz', [10, 25, -20], degrees=True).as_matrix() * 2
    shfile = tmp_path / 'sh.nii.gz'; peaks = tmp_path / 'peaks.nii.gz'
    nib.save(nib.Nifti1Image(image, affine), shfile)
    subprocess.run(['sh2peaks', str(shfile), str(peaks), '-num', '1', '-quiet'], check=True)
    actual = nib.load(peaks).get_fdata()[:, 0, 0, :]
    expected = source @ frame
    cos = abs(np.sum(actual * expected, axis=1)) / np.linalg.norm(actual, axis=1)
    assert np.all(cos > 0.999)


def test_profile_uses_point_assignments_and_masks_unsupported(tmp_path):
    import csv
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram
    from dipy.tracking.streamline import Streamlines
    from pydesigner.buan.profiles import profile
    grid = np.ones((8, 8, 8), np.float32)
    mask = grid.copy(); mask[6:] = 0
    mask_path = tmp_path/'mask.nii.gz'; metric = tmp_path/'metric.nii.gz'
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    values = np.broadcast_to(np.arange(8, dtype=np.float32)[:, None, None], grid.shape).copy()
    nib.save(nib.Nifti1Image(values, np.eye(4)), metric)
    reference = nib.load(mask_path)
    tracks = Streamlines([np.array([[x, 3 + dy, 3] for x in np.linspace(1, 7, 20)], dtype=float)
                          for dy in (-.1, 0, .1)])
    files = [tmp_path / f'{name}.trk' for name in ['model', 'common', 'native']]
    for path in files:
        save_tractogram(StatefulTractogram(tracks, reference, Space.RASMM), str(path))
    output = tmp_path/'profiles'
    profile(*files, [metric], mask_path, output, disks=10)
    with (output/'metric_disks.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert any(int(r['n_valid']) > 0 for r in rows)
    assert any(int(r['n_valid']) < int(r['n_points']) for r in rows)
    for row in rows:
        if int(row['n_valid']):
            assert float(row['mean']) < 6
