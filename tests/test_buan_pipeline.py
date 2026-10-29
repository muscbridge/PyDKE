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
