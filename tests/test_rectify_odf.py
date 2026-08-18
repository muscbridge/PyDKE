import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pydesigner.tractography.rectify_odf import main


MRTRIX_COMMANDS = (
    "mrconvert",
    "sh2amp",
    "amp2sh",
)

pytestmark = pytest.mark.skipif(
    any(shutil.which(cmd) is None for cmd in MRTRIX_COMMANDS),
    reason="MRtrix3 is required for rectify-odf integration tests",
)


def run_command(args):
    subprocess.run(
        [str(x) for x in args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def make_affine():
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 2.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0

    affine[:3, 3] = (
        -20.0,
        -30.0,
        -40.0,
    )

    return affine


def write_test_fod(path, affine):
    """
    Create a tiny lmax=8 MRtrix SH image.

    45 coefficients corresponds to even-order
    real SH coefficients through lmax=8.

    A positive c0 with all higher-order terms zero
    gives a positive isotropic FOD, which is sufficient
    for testing image geometry and stride handling.
    """
    data = np.zeros(
        (3, 4, 2, 45),
        dtype=np.float32,
    )

    data[..., 0] = 1.0

    img = nib.Nifti1Image(
        data,
        affine,
    )

    nib.save(img, str(path))


def write_test_mask(path, affine):
    data = np.ones(
        (3, 4, 2),
        dtype=np.uint8,
    )

    img = nib.Nifti1Image(
        data,
        affine,
    )

    nib.save(img, str(path))


def convert_to_mif(
    source,
    destination,
    strides,
):
    run_command([
        "mrconvert",
        source,
        destination,
        "-strides",
        strides,
        "-force",
    ])


def convert_to_canonical_nifti(
    source,
    destination,
    ndim,
):
    strides = (
        "1,2,3,4"
        if ndim == 4
        else "1,2,3"
    )

    run_command([
        "mrconvert",
        source,
        destination,
        "-strides",
        strides,
        "-force",
    ])


def run_rectify(
    monkeypatch,
    input_path,
    mask_path,
    output_path,
):
    argv = [
        "rectify-odf",
        str(input_path),
        str(output_path),
        "--mask",
        str(mask_path),
        "--resolution",
        "low",
        "--validation-resolution",
        "low",
        "--iterations",
        "4",
        "--bisection-tolerance",
        "1e-8",
        "--min-mass-stability-ratio",
        "0.20",
        "--validation-tolerance",
        "1e-4",
        "--batch-voxels",
        "32",
        "--nthreads",
        "1",
        "--force",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )

    return main()


def test_equivalent_physical_grid_with_different_strides(
    tmp_path,
    monkeypatch,
):
    """
    Regression test for MRtrix stride canonicalization.

    Physically identical images with:

        FOD  strides = -1,2,3,4
        mask strides =  1,2,3

    must be accepted and must produce the same result
    as canonical positive-stride inputs.
    """

    affine = make_affine()

    fod_nii = tmp_path / "fod.nii.gz"
    mask_nii = tmp_path / "mask.nii.gz"

    write_test_fod(
        fod_nii,
        affine,
    )
    write_test_mask(
        mask_nii,
        affine,
    )

    fod_negative_stride = (
        tmp_path / "fod_negative_stride.mif"
    )
    fod_positive_stride = (
        tmp_path / "fod_positive_stride.mif"
    )
    mask_positive_stride = (
        tmp_path / "mask_positive_stride.mif"
    )

    convert_to_mif(
        fod_nii,
        fod_negative_stride,
        "-1,2,3,4",
    )

    convert_to_mif(
        fod_nii,
        fod_positive_stride,
        "1,2,3,4",
    )

    convert_to_mif(
        mask_nii,
        mask_positive_stride,
        "1,2,3",
    )

    output_negative = (
        tmp_path / "rectified_negative.mif"
    )
    output_positive = (
        tmp_path / "rectified_positive.mif"
    )

    assert (
        run_rectify(
            monkeypatch,
            fod_negative_stride,
            mask_positive_stride,
            output_negative,
        )
        == 0
    )

    assert (
        run_rectify(
            monkeypatch,
            fod_positive_stride,
            mask_positive_stride,
            output_positive,
        )
        == 0
    )

    negative_nii = (
        tmp_path / "negative_result.nii.gz"
    )
    positive_nii = (
        tmp_path / "positive_result.nii.gz"
    )

    convert_to_canonical_nifti(
        output_negative,
        negative_nii,
        ndim=4,
    )

    convert_to_canonical_nifti(
        output_positive,
        positive_nii,
        ndim=4,
    )

    negative_img = nib.load(
        str(negative_nii)
    )
    positive_img = nib.load(
        str(positive_nii)
    )

    negative_data = np.asarray(
        negative_img.dataobj,
        dtype=np.float64,
    )
    positive_data = np.asarray(
        positive_img.dataobj,
        dtype=np.float64,
    )

    assert negative_img.shape == positive_img.shape

    assert np.allclose(
        negative_img.affine,
        positive_img.affine,
        rtol=0.0,
        atol=1e-6,
    )

    assert np.allclose(
        negative_data,
        positive_data,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


def test_genuine_affine_mismatch_is_rejected(
    tmp_path,
    monkeypatch,
):
    """
    The stride fix must not disable genuine geometry
    validation.

    A physically displaced mask must still fail.
    """

    fod_affine = make_affine()

    mask_affine = fod_affine.copy()
    mask_affine[0, 3] += 10.0

    fod_nii = tmp_path / "fod.nii.gz"
    mask_nii = tmp_path / "mask_shifted.nii.gz"

    write_test_fod(
        fod_nii,
        fod_affine,
    )

    write_test_mask(
        mask_nii,
        mask_affine,
    )

    fod_mif = tmp_path / "fod.mif"
    mask_mif = tmp_path / "mask_shifted.mif"

    convert_to_mif(
        fod_nii,
        fod_mif,
        "1,2,3,4",
    )

    convert_to_mif(
        mask_nii,
        mask_mif,
        "1,2,3",
    )

    output = tmp_path / "should_not_exist.mif"

    with pytest.raises(
        ValueError,
        match="Mask and ODF affines do not match",
    ):
        run_rectify(
            monkeypatch,
            fod_mif,
            mask_mif,
            output,
        )
