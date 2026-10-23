import os.path as op
import pytest
import numpy as np
from conftest import load_data

from pydesigner.fitting.dwipy import DWI
from pydesigner.fitting.dwidirs import dirs30

DATA = load_data(type="hifi")
PATH_DWI = DATA["nifti"]
PATH_BVEC = DATA["bvec"]
PATH_BVAL = DATA["bval"]
PATH_MASK = DATA["mask"]


def is_all_none(array):
    """Check if all elements in an array are None"""
    return not np.all(np.vectorize(lambda x: x is None)(array))


def test_dwi_init_image_nonexistent(tmp_path):
    input_nii = str(tmp_path / "nonexistent.nii")
    with pytest.raises(FileNotFoundError) as exc:
        dwi = DWI(input_nii, PATH_BVEC, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert f"Input image ({input_nii}) not found" in str(exc.value)


def test_dwi_init_bvec_invalid(tmp_path):
    with pytest.raises(TypeError) as exc:
        dwi = DWI(PATH_DWI, 50, PATH_BVAL, PATH_MASK)
    assert "Input file path (input=50) is not a string type." in str(exc.value)


def test_dwi_init_bvec_nonexistent(tmp_path):
    input_bvec = str(tmp_path / "nonexistent.bvec")
    with pytest.raises(FileNotFoundError) as exc:
        dwi = DWI(PATH_DWI, input_bvec, PATH_BVAL, PATH_MASK)
    assert f"Input file path ({input_bvec}) does not exist." in str(exc.value)


def test_dwi_init_bval_invalid(tmp_path):
    with pytest.raises(TypeError) as exc:
        dwi = DWI(PATH_DWI, PATH_BVEC, 50, PATH_MASK)
    assert "Input file path (input=50) is not a string type" in str(exc.value)


def test_dwi_init_bval_nonexistent(tmp_path):
    input_bval = str(tmp_path / "nonexistent.bval")
    with pytest.raises(OSError) as exc:
        dwi = DWI(PATH_DWI, PATH_BVEC, input_bval, PATH_MASK)
    assert f"Input file path ({input_bval}) does not exist." in str(exc.value)


def test_dwi_init_mask_nonexistent(tmp_path, capsys):
    input_mask = str(tmp_path / "nonexistent.nii")
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, input_mask)
    captured = capsys.readouterr()
    assert "No brain mask supplied" in captured.out


def test_dwi_init_success(capsys):
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    captured = capsys.readouterr()
    assert dwi.hdr.header.get_data_shape() == (2, 2, 2, 337)
    assert np.shape(dwi.grad) == (337, 4)
    assert f"Image {op.basename(PATH_DWI)} loaded successfully" in captured.out


def test_dwi_get_bvals():
    """Tests whether function returns correct bvals"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    bvals = dwi.getBvals()
    assert bvals.dtype == np.float64
    assert np.shape(bvals) == (337,)
    assert 0 in bvals
    assert 1 in bvals
    assert 2 in bvals
    assert 8 in bvals


def test_dwi_get_bvecs():
    """Tests whether function returns correct bvecs"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    bvecs = dwi.getBvecs()
    assert bvecs.dtype == np.float64
    assert bvecs.shape == (337, 3)


def test_dwi_max_bval():
    """Tests whether function returns correct max bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxBval()
    assert isinstance(val, float)
    assert val == 8


def test_dwi_max_dti_bval():
    """Tests whether function returns correct max DTI bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxDTIBval()
    assert isinstance(val, float)
    assert val == 1


def test_dwi_max_dki_bval():
    """Tests whether function returns correct max DKI bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxDKIBval()
    assert isinstance(val, float)
    assert val == 2


def test_max_fbi_bval():
    """Tests whether function returns correct max FBI bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxFBIBval()
    assert isinstance(val, float)
    assert val == 8


def test_dwi_idx_b0():
    """Tests whether function returns correct index of b0"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxb0()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 21


def test_dwi_idx_dti():
    """Tests whether function returns correct index of DTI b-values"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxdti()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 51


def test_dwi_idx_dki():
    """Tests whether function returns correct index of DKI b-values"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxdki()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 81


def test_idx_fbi():
    """Tests whether function returns correct index of FBI b-values"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxfbi()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 256


def test_dwi_n_dirs():
    """Tests whether function returns correct number of directions"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.getndirs() == 30


def test_dwi_tensor_type():
    """Tests whether function returns correct tensor type"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    tensor = dwi.tensorType()
    assert isinstance(tensor, list)
    assert "dti" in tensor
    assert "dki" in tensor
    assert "fbi" in tensor
    assert "fbwm" in tensor


def test_dwi_is_dti():
    """Tests whether function returns correct boolean for DTI dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isdti() is True


def test_dwi_is_dki():
    """Tests whether function returns correct boolean for DKI dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isdki() is True


def test_dwi_is_fbi():
    """Tests whether function returns correct boolean for FBI dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isfbi() is True


def test_dwi_is_fbwm():
    """Tests whether function returns correct boolean for FBWM dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isfbwm() is True


def test_dwi_tensor_order_invalid_order():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    with pytest.raises(ValueError):
        cnt, ind = dwi.createTensorOrder(5)


def test_dwi_tensor_order_valid_order():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    cnt, ind = dwi.createTensorOrder(2)
    assert len(cnt) == 6
    assert np.shape(ind) == (6, 2)


def test_dwi_tensor_order_auto_detect():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    cnt, ind = dwi.createTensorOrder()
    assert len(cnt) == 15
    assert np.shape(ind) == (15, 4)


def test_dwi_fibonacci_sphere_invalid_samples():
    """Tests whether function returns correct response from invalid samples type"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    with pytest.raises(TypeError):
        dwi.fibonacciSphere(samples=5.2)


def test_dwi_fibonacci_sphere_success():
    """Tests whether function returns correct response"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    sphere = dwi.fibonacciSphere(samples=5)
    assert sphere.dtype == np.float64
    assert np.shape(sphere) == (5, 3)


def test_dwi_radial_sampling():
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    samples = 128
    dirs = dwi.radialSampling(dirs30[0], samples)
    assert dirs.dtype == np.float64
    assert np.shape(dirs) == (samples, 3)
    np.testing.assert_allclose(dirs @ dirs30[0], 0, atol=1e-12)


def test_dwi_constraints_invalid():
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    with pytest.raises(ValueError) as exc:
        val = dwi.createConstraints([1, 2, 3])
    assert "Invalid contraints" in str(exc.value)


@pytest.mark.parametrize(
    "constraints",
    (
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
    ),
)
def test_dwi_constraints_success(constraints):
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.createConstraints(constraints)
    if sum(constraints) == 0:
        shape = (0, 22)
    elif sum(constraints) == 1:
        shape = (30, 22)
    elif sum(constraints) == 2:
        shape = (60, 22)
    elif sum(constraints) == 3:
        shape = (90, 22)
    assert val.dtype == np.float64
    assert np.shape(val) == shape


def test_dwi_fit_constrained(capsys):
    """Tests whether constrained fitting works normally"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    dwi.fit([0, 1, 0])
    captured = capsys.readouterr()
    assert hasattr(dwi, "dt")
    assert np.shape(dwi.dt) == (21, 5)
    assert "Constrained Tensor Fit" in captured.err


def test_dwi_fit_unconstrained(capsys):
    """Tests whether unconstrained fitting works normally"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    dwi.fit()
    captured = capsys.readouterr()
    assert hasattr(dwi, "dt")
    assert np.shape(dwi.dt) == (21, 5)
    assert "Unconstrained Tensor Fit" in captured.err


def test_dwi_dti_dki_params(capsys):
    """Tests whether function returns correct DTI values"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    dwi.fit([0, 1, 0])
    md, rd, ad, fa, fe, trace = dwi.extractDTI()
    mk, rk, ak, kfa, mkt, trace = dwi.extractDKI()
    awf, eas_ad, eas_rd, eas_tort, ias_da = dwi.extractWMTI()

    captured = capsys.readouterr()

    assert "Constrained Tensor Fit" in captured.err
    assert "DTI Parameters" in captured.err
    assert "DKI Parameters" in captured.err
    assert "Extracting AWF" in captured.err
    assert "Extracting EAS and IAS" in captured.err

    assert md.dtype == np.float64
    assert rd.dtype == np.float64
    assert ad.dtype == np.float64
    assert fa.dtype == np.float64
    assert fe.dtype == np.float64
    assert trace.dtype == np.float64

    assert np.shape(md) == (2, 2, 2)
    assert np.shape(rd) == (2, 2, 2)
    assert np.shape(ad) == (2, 2, 2)
    assert np.shape(fa) == (2, 2, 2)
    assert np.shape(fe) == (2, 2, 2, 3)
    assert np.shape(trace) == (2, 2, 2, 61)

    assert np.nanmean(md) > 0.40 and np.nanmean(md) < 0.60
    assert np.nanmean(rd) > 0.15 and np.nanmean(rd) < 0.40
    assert np.nanmean(ad) > 1.00 and np.nanmean(ad) < 1.30
    assert np.nanmean(fa) > 0.70 and np.nanmean(fa) < 0.80
    assert np.nanmean(fe) > 0.15 and np.nanmean(fe) < 0.40
    assert np.nanmean(trace) > 0.15 and np.nanmean(trace) < 0.40

    assert mk.dtype == np.float64
    assert rk.dtype == np.float64
    assert ak.dtype == np.float64
    assert kfa.dtype == np.float64
    assert mkt.dtype == np.float64
    assert trace.dtype == np.float64

    assert np.shape(mk) == (2, 2, 2)
    assert np.shape(rk) == (2, 2, 2)
    assert np.shape(ak) == (2, 2, 2)
    assert np.shape(kfa) == (2, 2, 2)
    assert np.shape(mkt) == (2, 2, 2)
    assert np.shape(trace) == (2, 2, 2, 61)

    assert np.nanmean(mk) > 0.60 and np.nanmean(mk) < 0.80
    assert np.nanmean(rk) > 1.20 and np.nanmean(rk) < 1.70
    assert np.nanmean(ak) > 0.15 and np.nanmean(ak) < 0.50
    assert np.nanmean(kfa) > 0.20 and np.nanmean(kfa) < 0.50
    assert np.nanmean(mkt) > 0.40 and np.nanmean(mkt) < 0.60
    assert np.nanmean(trace) > 0.15 and np.nanmean(trace) < 0.40

    assert awf.dtype == np.float64
    assert eas_ad.dtype == np.float64
    assert eas_rd.dtype == np.float64
    assert eas_tort.dtype == np.float64
    assert ias_da.dtype == np.float64

    assert np.shape(awf) == (2, 2, 2)
    assert np.shape(eas_ad) == (2, 2, 2)
    assert np.shape(eas_rd) == (2, 2, 2)
    assert np.shape(eas_tort) == (2, 2, 2)
    assert np.shape(ias_da) == (2, 2, 2)

    assert np.nanmean(awf) > 0.20 and np.nanmean(awf) < 0.40
    assert np.nanmean(eas_ad) > 1.50 and np.nanmean(eas_ad) < 1.75
    assert np.nanmean(eas_rd) > 0.35 and np.nanmean(eas_rd) < 0.55
    assert np.nanmean(eas_tort) > 2.40 and np.nanmean(eas_tort) < 2.70
    assert np.nanmean(ias_da) > 0.80 and np.nanmean(ias_da) < 0.90


def test_dwi_fbi_without_fbwm(capsys):
    """Tests whether FBI fitting works normally"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    dwi.fit([0, 1, 0])
    (
        zeta,
        faa,
        sph,
        sph_mrtrix,
        min_awf,
        Da,
        De_mean,
        De_ax,
        De_rad,
        De_fa,
        min_cost,
        min_cost_fn,
    ) = dwi.fbi(fbwm=False)
    captured = capsys.readouterr()

    assert "Constrained Tensor Fit" in captured.err
    assert "FBI Fit" in captured.err

    assert zeta.dtype == np.float64
    assert faa.dtype == np.float64
    assert sph.dtype == np.complex128
    assert sph_mrtrix.dtype == np.complex128
    assert min_awf.dtype == np.dtype("O")
    assert Da.dtype == np.dtype("O")
    assert De_mean.dtype == np.dtype("O")
    assert De_ax.dtype == np.dtype("O")
    assert De_rad.dtype == np.dtype("O")
    assert De_fa.dtype == np.dtype("O")
    assert min_cost.dtype == np.dtype("O")
    assert min_cost_fn.dtype == np.dtype("O")

    assert np.shape(zeta) == (2, 2, 2)
    assert np.shape(faa) == (2, 2, 2)
    assert np.shape(sph) == (2, 2, 2, 28)
    assert np.shape(sph_mrtrix) == (2, 2, 2, 28)
    assert np.shape(min_awf) == (2, 2, 2)
    assert np.shape(Da) == (2, 2, 2)
    assert np.shape(De_mean) == (2, 2, 2)
    assert np.shape(De_ax) == (2, 2, 2)
    assert np.shape(De_rad) == (2, 2, 2)
    assert np.shape(De_fa) == (2, 2, 2)
    assert np.shape(min_cost) == (2, 2, 2)
    assert np.shape(min_cost_fn) == (2, 2, 2)

    assert np.nanmean(zeta) > 0.20 and np.nanmean(zeta) < 0.30
    assert np.nanmean(faa) > 0.50 and np.nanmean(faa) < 0.55
    assert is_all_none(min_awf)
    assert is_all_none(Da)
    assert is_all_none(De_mean)
    assert is_all_none(De_ax)
    assert is_all_none(De_rad)
    assert is_all_none(De_fa)
    assert is_all_none(min_cost)
    assert is_all_none(min_cost_fn)


def test_symmetric_diffusion_tensor_eigenvectors_real():
    """Repeated and rotated eigenvalues must retain real orthogonal vectors."""
    q,_ = np.linalg.qr(np.random.default_rng(61).normal(size=(3,3)))
    for eigenvalues in ([1.,1.,1.],[.2,.3,1.5]):
        tensor=q@np.diag(eigenvalues)@q.T
        values,vectors=DWI.dtiTensorParams(None,tensor)
        assert not np.iscomplexobj(vectors)
        np.testing.assert_allclose(vectors@np.diag(values)@vectors.T,tensor,atol=1e-14)
        np.testing.assert_allclose(vectors.T@vectors,np.eye(3),atol=1e-14)
        assert np.all(np.diff(values)<=0)
