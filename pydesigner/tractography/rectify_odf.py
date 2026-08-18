from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from . import sphericalsampling
from .rectify import (
    rectify_odf_samples,
    rectify_normalized_odf_samples,
)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required MRtrix command not found on PATH: {name}"
        )


def run(cmd: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("+", " ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def infer_lmax(ncoeff: int) -> int:
    # Antipodally symmetric even-order SH:
    # n = (lmax + 1)(lmax + 2) / 2
    root = math.sqrt(1 + 8 * ncoeff)
    lmax = int(round((-3 + root) / 2))

    expected = (lmax + 1) * (lmax + 2) // 2

    if (
        expected != ncoeff
        or lmax < 0
        or lmax % 2
    ):
        raise ValueError(
            f"{ncoeff} coefficients do not correspond to an "
            "even antipodally-symmetric SH series"
        )

    return lmax


def spherical_grid(resolution: str):
    result = sphericalsampling.odfgrid(resolution)

    grid = np.asarray(result[0], dtype=np.float64)
    area = np.asarray(result[3], dtype=np.float64).reshape(-1)

    if grid.ndim != 2 or grid.shape[1] < 2:
        raise ValueError(
            f"Unexpected spherical grid shape: {grid.shape}"
        )

    phi = grid[:, 0]
    theta = grid[:, 1]

    directions = np.column_stack(
        (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        )
    )

    return directions, area


def write_directions(path: Path, directions: np.ndarray) -> None:
    np.savetxt(
        path,
        directions,
        fmt="%.12g",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Rectify an MRtrix spherical-harmonic ODF using the "
            "pyDKE bisection algorithm."
        ),
    )

    p.add_argument(
        "input",
        type=Path,
        help="Input MRtrix SH FOD/ODF image.",
    )

    p.add_argument(
        "output",
        type=Path,
        help="Rectified output SH FOD/ODF image.",
    )

    p.add_argument(
        "--mask",
        type=Path,
        help=(
            "Optional spatial mask. Outside-mask coefficients are "
            "set to zero. If omitted, nonzero l=0 voxels are used."
        ),
    )

    p.add_argument(
        "--resolution",
        choices=("low", "med", "high"),
        default="med",
        help="pyDKE spherical sampling grid used for rectification.",
    )

    p.add_argument(
        "--validation-resolution",
        choices=("low", "med", "high"),
        default="high",
        help="Independent grid used after SH refitting for QC.",
    )

    p.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum bisection iterations.",
    )

    p.add_argument(
        "--bisection-tolerance",
        type=float,
        default=1e-8,
        help="Bisection convergence tolerance.",
    )

    p.add_argument(
        "--validation-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Amplitudes below negative this value are counted as "
            "residual negative samples after SH refitting."
        ),
    )

    p.add_argument(
        "--batch-voxels",
        type=int,
        default=512,
        help="Number of masked voxels rectified per Python batch.",
    )

    p.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="MRtrix thread count.",
    )

    p.add_argument(
        "--qc",
        type=Path,
        help="Optional JSON QC report.",
    )

    p.add_argument(
        "--min-mass-stability-ratio",
        type=float,
        default=0.20,
        help=(
            "Minimum accepted angular mass-stability ratio "
            "R=(P-N)/(P+N). Positive-mass fODFs below this "
            "threshold are treated as unstable/artifactual "
            "support and excluded before rectification. "
            "(default: 0.20)"
        ),
    )

    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return nonzero status if the refitted SH ODF retains "
            "samples below --validation-tolerance."
        ),
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    for command in (
        "mrconvert",
        "sh2amp",
        "amp2sh",
    ):
        require_command(command)

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if args.mask is not None:
        mask_path = args.mask.expanduser().resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
    else:
        mask_path = None

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"{output_path} exists; use --force to overwrite"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rectify_dirs, area = spherical_grid(
        args.resolution
    )

    validation_dirs, _ = spherical_grid(
        args.validation_resolution
    )

    qc = {
        "input": str(input_path),
        "output": str(output_path),
        "mask": str(mask_path) if mask_path else None,
        "resolution": args.resolution,
        "validation_resolution": args.validation_resolution,
        "iterations": args.iterations,
        "bisection_tolerance": args.bisection_tolerance,
        "validation_tolerance": args.validation_tolerance,
        "rectification_directions": int(rectify_dirs.shape[0]),
        "validation_directions": int(validation_dirs.shape[0]),
        "quadrature_area_sum": float(np.sum(area)),
    }

    with tempfile.TemporaryDirectory(
        prefix="rectify_odf_"
    ) as tmp_name:

        tmp = Path(tmp_name)

        input_nii = tmp / "input_sh.nii.gz"

        # Canonicalize spatial storage before loading with nibabel.
        # MRtrix .mif images may represent the same physical grid with
        # different signed stride conventions. NIfTI cannot preserve all
        # MRtrix stride representations equivalently, so explicitly use
        # positive strides for all temporary NIfTI intermediates.
        run([
            "mrconvert",
            str(input_path),
            str(input_nii),
            "-strides", "1,2,3,4",
            "-datatype", "float32",
            "-force",
        ])

        source_img = nib.load(str(input_nii))

        if len(source_img.shape) != 4:
            raise ValueError(
                "Expected a 4D SH image; got "
                f"shape {source_img.shape}"
            )

        nx, ny, nz, ncoeff = source_img.shape
        lmax = infer_lmax(ncoeff)

        qc["shape"] = list(source_img.shape)
        qc["ncoeff"] = int(ncoeff)
        qc["lmax"] = int(lmax)

        if mask_path is not None:
            mask_nii = tmp / "mask.nii.gz"

            run([
                "mrconvert",
                str(mask_path),
                str(mask_nii),
                "-strides", "1,2,3",
                "-datatype", "uint8",
                "-force",
            ])

            mask_img = nib.load(str(mask_nii))

            if mask_img.shape[:3] != source_img.shape[:3]:
                raise ValueError(
                    "Mask shape differs from ODF shape: "
                    f"{mask_img.shape[:3]} vs "
                    f"{source_img.shape[:3]}"
                )

            if not np.allclose(
                mask_img.affine,
                source_img.affine,
                atol=1e-4,
            ):
                raise ValueError(
                    "Mask and ODF affines do not match."
                )

            mask = (
                np.asanyarray(mask_img.dataobj) > 0
            )

        else:
            source_data_for_mask = np.asanyarray(
                source_img.dataobj
            )

            l0 = source_data_for_mask[..., 0]

            mask = (
                np.isfinite(l0)
                & (l0 != 0)
            )

        rectify_dir_file = (
            tmp / "rectify_directions.txt"
        )
        validation_dir_file = (
            tmp / "validation_directions.txt"
        )

        write_directions(
            rectify_dir_file,
            rectify_dirs,
        )
        write_directions(
            validation_dir_file,
            validation_dirs,
        )

        output_coeff = np.zeros(
            source_img.shape,
            dtype=np.float32,
        )

        total_voxels = int(np.count_nonzero(mask))
        voxels_requiring_rectification = 0

        negative_samples_before = 0
        negative_samples_after = 0

        voxels_negative_before = 0
        voxels_negative_after = 0

        total_fit_samples = 0
        total_validation_samples = 0

        min_before = math.inf
        min_after = math.inf

        offsets_all = []

        # Mass-stability support QC.
        positive_mass_voxels = 0
        nonpositive_mass_voxels = 0
        invalid_mass_voxels = 0
        unstable_mass_voxels = 0
        valid_mass_stability_voxels = 0

        # Post-amp2sh positivity refinement QC.
        positivity_refinement_voxels = 0
        positivity_shifts_all = []
        positivity_scales_all = []
        maximum_absolute_c0_error = 0.0

        negative_samples_final = 0
        voxels_negative_final = 0
        total_final_validation_samples = 0
        min_final = math.inf

        # MRtrix real SH isotropic basis amplitude:
        # Y00 = 1 / sqrt(4*pi) = 0.282094791773...
        mrtrix_y00 = 1.0 / math.sqrt(4.0 * math.pi)

        # Small positive margin on the dense validation grid.
        # This is deliberately much smaller than the default
        # validation tolerance (1e-4).
        positivity_margin = 1e-6

        for z in range(nz):
            slab_mask = mask[:, :, z].reshape(-1)

            if not np.any(slab_mask):
                continue

            print(
                f"[{z + 1:03d}/{nz:03d}] "
                f"slice {z}: "
                f"{np.count_nonzero(slab_mask)} voxels"
            )

            slab_sh = tmp / f"sh_{z:03d}.nii.gz"
            slab_amp = tmp / f"amp_{z:03d}.nii.gz"
            slab_rect_amp = (
                tmp / f"amp_rect_{z:03d}.nii.gz"
            )
            slab_rect_sh = (
                tmp / f"sh_rect_{z:03d}.nii.gz"
            )
            slab_validation = (
                tmp / f"validation_{z:03d}.nii.gz"
            )

            slab_img = source_img.slicer[
                :,
                :,
                z:z + 1,
                :,
            ]

            nib.save(
                slab_img,
                str(slab_sh),
            )

            run([
                "sh2amp",
                str(slab_sh),
                str(rectify_dir_file),
                str(slab_amp),
                "-datatype", "float32",
                "-nthreads", str(args.nthreads),
                "-force",
            ], quiet=True)

            amp_img = nib.load(str(slab_amp))
            amp = np.asanyarray(
                amp_img.dataobj,
                dtype=np.float64,
            )

            ndir = amp.shape[-1]
            flat = amp.reshape(-1, ndir)

            selected_indices = np.flatnonzero(
                slab_mask
            )

            rectified_flat = np.zeros_like(
                flat,
                dtype=np.float64,
            )

            for start in range(
                0,
                selected_indices.size,
                args.batch_voxels,
            ):
                indices = selected_indices[
                    start:
                    start + args.batch_voxels
                ]

                values = flat[indices, :]

                # ------------------------------------------------
                # Mass-stability support classification.
                #
                # P = positive angular mass
                # N = negative angular mass magnitude
                # M = P - N   signed angular mass
                # A = P + N   absolute angular mass
                # R = M / A   mass-stability ratio
                # ------------------------------------------------

                finite_rows = np.all(
                    np.isfinite(values),
                    axis=1,
                )

                safe_values = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                positive_mass_batch = np.sum(
                    np.maximum(
                        safe_values,
                        0.0,
                    )
                    * area[None, :],
                    axis=1,
                )

                negative_mass_batch = np.sum(
                    np.maximum(
                        -safe_values,
                        0.0,
                    )
                    * area[None, :],
                    axis=1,
                )

                signed_mass_batch = (
                    positive_mass_batch
                    - negative_mass_batch
                )

                absolute_mass_batch = (
                    positive_mass_batch
                    + negative_mass_batch
                )

                ratio_defined = (
                    finite_rows
                    & np.isfinite(signed_mass_batch)
                    & np.isfinite(absolute_mass_batch)
                    & (absolute_mass_batch > 0.0)
                )

                positive_mass_mask = (
                    ratio_defined
                    & (signed_mass_batch > 0.0)
                )

                nonpositive_mass_mask = (
                    ratio_defined
                    & (signed_mass_batch <= 0.0)
                )

                invalid_mass_mask = ~ratio_defined

                mass_stability_ratio = np.zeros(
                    values.shape[0],
                    dtype=np.float64,
                )

                mass_stability_ratio[
                    ratio_defined
                ] = (
                    signed_mass_batch[ratio_defined]
                    / absolute_mass_batch[ratio_defined]
                )

                stable_mass_mask = (
                    positive_mass_mask
                    & (
                        mass_stability_ratio
                        >= args.min_mass_stability_ratio
                    )
                )

                unstable_mass_mask = (
                    positive_mass_mask
                    & ~stable_mass_mask
                )

                positive_mass_voxels += int(
                    np.count_nonzero(
                        positive_mass_mask
                    )
                )

                nonpositive_mass_voxels += int(
                    np.count_nonzero(
                        nonpositive_mass_mask
                    )
                )

                invalid_mass_voxels += int(
                    np.count_nonzero(
                        invalid_mass_mask
                    )
                )

                unstable_mass_voxels += int(
                    np.count_nonzero(
                        unstable_mass_mask
                    )
                )

                valid_mass_stability_voxels += int(
                    np.count_nonzero(
                        stable_mass_mask
                    )
                )

                negative = values < 0.0

                voxels_requiring_rectification += int(
                    np.count_nonzero(
                        np.any(negative, axis=1)
                    )
                )

                negative_samples_before += int(
                    np.count_nonzero(negative)
                )

                voxels_negative_before += int(
                    np.count_nonzero(
                        np.any(
                            values
                            < -args.validation_tolerance,
                            axis=1,
                        )
                    )
                )

                total_fit_samples += int(
                    values.size
                )

                if values.size:
                    min_before = min(
                        min_before,
                        float(np.min(values)),
                    )

                rectified, offsets = (
                    rectify_normalized_odf_samples(
                        values,
                        area,
                        iterations=args.iterations,
                        tolerance=args.bisection_tolerance,
                        target_hemisphere_mass=0.5,
                        min_mass_stability_ratio=(
                            args.min_mass_stability_ratio
                        ),
                        mass_conservation_tolerance=1e-6,
                        return_offsets=True,
                    )
                )

                rectified_flat[
                    indices,
                    :
                ] = rectified

                offset_vector = np.asarray(
                    offsets,
                    dtype=np.float64,
                ).reshape(-1)

                # Rejected / unstable rows receive zero offsets
                # from the wrapper. Exclude them from EPS summary
                # statistics so EPS describes actual rectification.
                offsets_all.append(
                    offset_vector[
                        stable_mass_mask
                    ]
                )

            rectified_amp = (
                rectified_flat
                .reshape(amp.shape)
                .astype(np.float32)
            )

            rect_header = amp_img.header.copy()
            rect_header.set_data_dtype(
                np.float32
            )

            nib.save(
                nib.Nifti1Image(
                    rectified_amp,
                    amp_img.affine,
                    rect_header,
                ),
                str(slab_rect_amp),
            )

            run([
                "amp2sh",
                str(slab_rect_amp),
                str(slab_rect_sh),
                "-directions",
                str(rectify_dir_file),
                "-lmax",
                str(lmax),
                "-nthreads",
                str(args.nthreads),
                "-force",
            ], quiet=True)

            rect_sh_img = nib.load(
                str(slab_rect_sh)
            )

            rect_sh = np.asanyarray(
                rect_sh_img.dataobj,
                dtype=np.float32,
            )

            # ----------------------------------------------------
            # Determine actual retained support after normalized
            # mass-stability filtering + bisection.
            # ----------------------------------------------------

            rect_sh_flat = rect_sh.reshape(
                -1,
                rect_sh.shape[-1],
            )

            slab_mask_flat = np.asarray(
                slab_mask,
                dtype=bool,
            ).reshape(-1)

            rectified_support_flat = (
                slab_mask_flat
                & np.any(
                    rectified_flat != 0.0,
                    axis=1,
                )
            )

            # Explicitly zero all SH coefficients outside the
            # retained mass-stability support.
            rect_sh_flat[
                ~rectified_support_flat,
                :
            ] = 0.0

            rect_sh = rect_sh_flat.reshape(
                rect_sh.shape
            )

            # ----------------------------------------------------
            # First dense validation:
            # unconstrained amp2sh reconstruction.
            # ----------------------------------------------------

            rect_sh_for_validation = (
                tmp / f"sh_validation_input_{z:03d}.nii.gz"
            )

            validation_header = (
                rect_sh_img.header.copy()
            )

            nib.save(
                nib.Nifti1Image(
                    rect_sh,
                    rect_sh_img.affine,
                    validation_header,
                ),
                str(rect_sh_for_validation),
            )

            run([
                "sh2amp",
                str(rect_sh_for_validation),
                str(validation_dir_file),
                str(slab_validation),
                "-datatype", "float32",
                "-nthreads", str(args.nthreads),
                "-force",
            ], quiet=True)

            val_img = nib.load(
                str(slab_validation)
            )

            val = np.asanyarray(
                val_img.dataobj,
                dtype=np.float64,
            )

            val_flat = val.reshape(
                -1,
                val.shape[-1],
            )

            selected_val = val_flat[
                rectified_support_flat,
                :
            ]

            if selected_val.size:
                min_after = min(
                    min_after,
                    float(np.min(selected_val)),
                )

                negative_after = (
                    selected_val
                    < -args.validation_tolerance
                )

                negative_samples_after += int(
                    np.count_nonzero(
                        negative_after
                    )
                )

                voxels_negative_after += int(
                    np.count_nonzero(
                        np.any(
                            negative_after,
                            axis=1,
                        )
                    )
                )

                total_validation_samples += int(
                    selected_val.size
                )

            # ----------------------------------------------------
            # Positivity-preserving SH refinement.
            #
            # amp2sh is an unconstrained least-squares projection
            # and can reintroduce Gibbs-like negative lobes.
            #
            # For each retained fODF:
            #
            #   1. Determine its minimum amplitude on the dense
            #      validation grid.
            #   2. If negative, add a purely isotropic c0 term
            #      sufficient to raise the minimum above zero.
            #   3. Positively rescale every SH coefficient so c0
            #      exactly returns to the original input c0.
            #
            # The resulting angular transformation is
            #
            #       f_final(Omega) = a * [f_fit(Omega) + b]
            #
            # with a > 0 and b >= 0. Therefore angular peak
            # directions are unchanged by this refinement.
            # ----------------------------------------------------

            if selected_val.size:
                retained_rows = np.flatnonzero(
                    rectified_support_flat
                )

                minimum_by_voxel = np.min(
                    selected_val,
                    axis=1,
                )

                needs_refinement = (
                    minimum_by_voxel < 0.0
                )

                if np.any(needs_refinement):
                    refined_rows = retained_rows[
                        needs_refinement
                    ]

                    amplitude_shift = (
                        -minimum_by_voxel[
                            needs_refinement
                        ]
                        + positivity_margin
                    )

                    delta_c0 = (
                        amplitude_shift
                        / mrtrix_y00
                    )

                    coeff = np.asarray(
                        rect_sh_flat[
                            refined_rows,
                            :
                        ],
                        dtype=np.float64,
                    ).copy()

                    # Original input MRtrix c0 for this z-slab.
                    # Restoring this value restores the original
                    # SH angular integral exactly up to float32
                    # storage precision.
                    source_c0_slab = np.asanyarray(
                        source_img.dataobj[
                            :,
                            :,
                            z:z + 1,
                            0,
                        ],
                        dtype=np.float64,
                    ).reshape(-1)

                    target_c0 = source_c0_slab[
                        refined_rows
                    ]

                    if (
                        np.any(~np.isfinite(target_c0))
                        or np.any(target_c0 <= 0.0)
                    ):
                        raise RuntimeError(
                            "Retained fODF has non-finite or "
                            "non-positive original c0 during "
                            "positivity refinement."
                        )

                    coeff[:, 0] += delta_c0

                    shifted_c0 = coeff[:, 0]

                    if (
                        np.any(~np.isfinite(shifted_c0))
                        or np.any(shifted_c0 <= 0.0)
                    ):
                        raise RuntimeError(
                            "Invalid shifted c0 during positivity "
                            "refinement."
                        )

                    positivity_scale = (
                        target_c0
                        / shifted_c0
                    )

                    if (
                        np.any(~np.isfinite(positivity_scale))
                        or np.any(positivity_scale <= 0.0)
                    ):
                        raise RuntimeError(
                            "Invalid positive SH rescaling factor."
                        )

                    coeff *= positivity_scale[:, None]

                    rect_sh_flat[
                        refined_rows,
                        :
                    ] = coeff.astype(
                        np.float32
                    )

                    stored_c0 = np.asarray(
                        rect_sh_flat[
                            refined_rows,
                            0,
                        ],
                        dtype=np.float64,
                    )

                    slab_c0_error = np.max(
                        np.abs(
                            stored_c0
                            - target_c0
                        )
                    )

                    maximum_absolute_c0_error = max(
                        maximum_absolute_c0_error,
                        float(slab_c0_error),
                    )

                    positivity_refinement_voxels += int(
                        refined_rows.size
                    )

                    positivity_shifts_all.append(
                        amplitude_shift.astype(
                            np.float64
                        )
                    )

                    positivity_scales_all.append(
                        positivity_scale.astype(
                            np.float64
                        )
                    )

            # Keep rejected support explicitly zero after all
            # coefficient operations.
            rect_sh_flat[
                ~rectified_support_flat,
                :
            ] = 0.0

            rect_sh = rect_sh_flat.reshape(
                rect_sh.shape
            )

            # This is the FINAL coefficient set written to output.
            output_coeff[
                :,
                :,
                z:z + 1,
                :,
            ] = rect_sh

            # ----------------------------------------------------
            # Second dense validation:
            # validate the actual final SH coefficients after the
            # isotropic correction and c0 restoration.
            # ----------------------------------------------------

            nib.save(
                nib.Nifti1Image(
                    rect_sh,
                    rect_sh_img.affine,
                    validation_header,
                ),
                str(rect_sh_for_validation),
            )

            run([
                "sh2amp",
                str(rect_sh_for_validation),
                str(validation_dir_file),
                str(slab_validation),
                "-datatype", "float32",
                "-nthreads", str(args.nthreads),
                "-force",
            ], quiet=True)

            final_val_img = nib.load(
                str(slab_validation)
            )

            final_val = np.asanyarray(
                final_val_img.dataobj,
                dtype=np.float64,
            )

            final_val_flat = final_val.reshape(
                -1,
                final_val.shape[-1],
            )

            selected_final_val = final_val_flat[
                rectified_support_flat,
                :
            ]

            if selected_final_val.size:
                min_final = min(
                    min_final,
                    float(
                        np.min(
                            selected_final_val
                        )
                    ),
                )

                negative_final = (
                    selected_final_val
                    < -args.validation_tolerance
                )

                negative_samples_final += int(
                    np.count_nonzero(
                        negative_final
                    )
                )

                voxels_negative_final += int(
                    np.count_nonzero(
                        np.any(
                            negative_final,
                            axis=1,
                        )
                    )
                )

                total_final_validation_samples += int(
                    selected_final_val.size
                )

        if total_voxels == 0:
            raise RuntimeError(
                "ODF mask contains no voxels."
            )

        output_nii = (
            tmp / "rectified_full.nii.gz"
        )

        output_header = (
            source_img.header.copy()
        )
        output_header.set_data_dtype(
            np.float32
        )

        nib.save(
            nib.Nifti1Image(
                output_coeff,
                source_img.affine,
                output_header,
            ),
            str(output_nii),
        )

        run([
            "mrconvert",
            str(output_nii),
            str(output_path),
            "-datatype", "float32",
            "-force",
        ])

        if offsets_all:
            offsets = np.concatenate(
                offsets_all
            )
        else:
            offsets = np.array(
                [],
                dtype=np.float64,
            )

        if positivity_shifts_all:
            positivity_shifts = np.concatenate(
                positivity_shifts_all
            )
        else:
            positivity_shifts = np.array(
                [],
                dtype=np.float64,
            )

        if positivity_scales_all:
            positivity_scales = np.concatenate(
                positivity_scales_all
            )
        else:
            positivity_scales = np.array(
                [],
                dtype=np.float64,
            )

        # ------------------------------------------------
        # Mass-QC bookkeeping invariants.
        #
        # Every selected voxel must be exactly one of:
        #   positive signed mass
        #   non-positive signed mass
        #   invalid / undefined mass
        #
        # Every positive-mass voxel must then be exactly
        # one of:
        #   stable
        #   unstable
        # ------------------------------------------------

        mass_partition_total = (
            positive_mass_voxels
            + nonpositive_mass_voxels
            + invalid_mass_voxels
        )

        if mass_partition_total != total_voxels:
            raise RuntimeError(
                "Mass QC bookkeeping mismatch: "
                f"positive={positive_mass_voxels}, "
                f"nonpositive={nonpositive_mass_voxels}, "
                f"invalid={invalid_mass_voxels}, "
                f"partition_total={mass_partition_total}, "
                f"mask_total={total_voxels}."
            )

        positive_partition_total = (
            valid_mass_stability_voxels
            + unstable_mass_voxels
        )

        if (
            positive_partition_total
            != positive_mass_voxels
        ):
            raise RuntimeError(
                "Mass-stability QC bookkeeping mismatch: "
                f"stable={valid_mass_stability_voxels}, "
                f"unstable={unstable_mass_voxels}, "
                f"partition_total="
                f"{positive_partition_total}, "
                f"positive_mass="
                f"{positive_mass_voxels}."
            )

        def _fraction(
            numerator: int,
            denominator: int,
        ) -> float:
            if denominator == 0:
                return 0.0
            return float(numerator / denominator)

        positive_mass_fraction = _fraction(
            positive_mass_voxels,
            total_voxels,
        )
        nonpositive_mass_fraction = _fraction(
            nonpositive_mass_voxels,
            total_voxels,
        )
        invalid_mass_fraction = _fraction(
            invalid_mass_voxels,
            total_voxels,
        )
        stable_mass_fraction_mask = _fraction(
            valid_mass_stability_voxels,
            total_voxels,
        )
        stable_mass_fraction_positive = _fraction(
            valid_mass_stability_voxels,
            positive_mass_voxels,
        )
        unstable_mass_fraction_mask = _fraction(
            unstable_mass_voxels,
            total_voxels,
        )
        unstable_mass_fraction_positive = _fraction(
            unstable_mass_voxels,
            positive_mass_voxels,
        )
        rectification_fraction = _fraction(
            voxels_requiring_rectification,
            total_voxels,
        )
        positivity_refinement_fraction = _fraction(
            positivity_refinement_voxels,
            total_voxels,
        )

        qc.update({
            "mask_voxels": total_voxels,
            "mass_stability_threshold":
                args.min_mass_stability_ratio,
            "positive_mass_voxels":
                positive_mass_voxels,
            "nonpositive_mass_voxels":
                nonpositive_mass_voxels,
            "invalid_mass_voxels":
                invalid_mass_voxels,
            "unstable_mass_voxels":
                unstable_mass_voxels,
            "valid_mass_stability_voxels":
                valid_mass_stability_voxels,
            "mass_partition_total":
                mass_partition_total,
            "positive_mass_partition_total":
                positive_partition_total,
            "positive_mass_fraction_of_mask":
                positive_mass_fraction,
            "nonpositive_mass_fraction_of_mask":
                nonpositive_mass_fraction,
            "invalid_mass_fraction_of_mask":
                invalid_mass_fraction,
            "valid_mass_stability_fraction_of_mask":
                stable_mass_fraction_mask,
            "valid_mass_stability_fraction_of_positive_mass":
                stable_mass_fraction_positive,
            "unstable_mass_fraction_of_mask":
                unstable_mass_fraction_mask,
            "unstable_mass_fraction_of_positive_mass":
                unstable_mass_fraction_positive,
            "rectification_fraction_of_mask":
                rectification_fraction,
            "positivity_refinement_fraction_of_mask":
                positivity_refinement_fraction,
            "hemisphere_target_mass": 0.5,
            "mrtrix_y00":
                mrtrix_y00,
            "positivity_refinement_method":
                "isotropic_c0_shift_then_restore_original_c0",
            "positivity_margin":
                positivity_margin,
            "positivity_refinement_voxels":
                positivity_refinement_voxels,
            "positivity_shift_median":
                float(np.median(positivity_shifts))
                if positivity_shifts.size else 0.0,
            "positivity_shift_p95":
                float(np.percentile(
                    positivity_shifts,
                    95,
                ))
                if positivity_shifts.size else 0.0,
            "positivity_shift_max":
                float(np.max(positivity_shifts))
                if positivity_shifts.size else 0.0,
            "positivity_scale_median":
                float(np.median(positivity_scales))
                if positivity_scales.size else 1.0,
            "positivity_scale_p05":
                float(np.percentile(
                    positivity_scales,
                    5,
                ))
                if positivity_scales.size else 1.0,
            "positivity_scale_min":
                float(np.min(positivity_scales))
                if positivity_scales.size else 1.0,
            "maximum_absolute_c0_error":
                maximum_absolute_c0_error,
            "voxels_requiring_rectification":
                voxels_requiring_rectification,
            "voxels_negative_before":
                voxels_negative_before,
            "voxels_negative_after_refit":
                voxels_negative_after,
            "negative_samples_before":
                negative_samples_before,
            "negative_samples_after_refit":
                negative_samples_after,
            "total_fit_samples":
                total_fit_samples,
            "total_validation_samples":
                total_validation_samples,
            "minimum_amplitude_before":
                None if not math.isfinite(min_before)
                else min_before,
            "minimum_amplitude_after_refit":
                None if not math.isfinite(min_after)
                else min_after,
            "minimum_amplitude_final":
                None if not math.isfinite(min_final)
                else min_final,
            "voxels_negative_final":
                voxels_negative_final,
            "negative_samples_final":
                negative_samples_final,
            "total_final_validation_samples":
                total_final_validation_samples,
            "median_eps":
                float(np.median(offsets))
                if offsets.size else 0.0,
            "eps_p95":
                float(np.percentile(offsets, 95))
                if offsets.size else 0.0,
            "maximum_eps":
                float(np.max(offsets))
                if offsets.size else 0.0,
            "negative_validation_fraction_after_refit":
                (
                    negative_samples_after
                    / total_validation_samples
                    if total_validation_samples
                    else 0.0
                ),
            "negative_validation_fraction":
                (
                    negative_samples_final
                    / total_final_validation_samples
                    if total_final_validation_samples
                    else 0.0
                ),
        })

        print()
        print("========================================")
        print("rectify-odf QC")
        print("========================================")
        print(f"lmax                         : {lmax}")
        print(f"mask voxels                  : {total_voxels}")
        print(
            "positive-mass voxels         : "
            f"{positive_mass_voxels} "
            f"({100.0 * positive_mass_fraction:.4f}% of mask)"
        )
        print(
            "nonpositive-mass voxels      : "
            f"{nonpositive_mass_voxels} "
            f"({100.0 * nonpositive_mass_fraction:.4f}% of mask)"
        )
        print(
            "invalid-mass voxels          : "
            f"{invalid_mass_voxels} "
            f"({100.0 * invalid_mass_fraction:.4f}% of mask)"
        )
        print(
            "voxels requiring rectification: "
            f"{voxels_requiring_rectification} "
            f"({100.0 * rectification_fraction:.4f}% of mask)"
        )
        print(
            "minimum amplitude before      : "
            f"{qc['minimum_amplitude_before']}"
        )
        print(
            "minimum amplitude after refit : "
            f"{qc['minimum_amplitude_after_refit']}"
        )
        print(
            "negative samples before       : "
            f"{negative_samples_before}"
        )
        print(
            "negative samples after refit  : "
            f"{negative_samples_after}"
        )
        print(
            "negative samples final        : "
            f"{negative_samples_final}"
        )
        print(
            "voxels negative final         : "
            f"{voxels_negative_final}"
        )
        print(
            "minimum amplitude final       : "
            f"{qc['minimum_amplitude_final']}"
        )
        print(
            "valid mass-stability voxels   : "
            f"{valid_mass_stability_voxels} "
            f"({100.0 * stable_mass_fraction_mask:.4f}% "
            f"of mask; "
            f"{100.0 * stable_mass_fraction_positive:.4f}% "
            f"of positive mass)"
        )
        print(
            "unstable mass voxels          : "
            f"{unstable_mass_voxels} "
            f"({100.0 * unstable_mass_fraction_mask:.4f}% "
            f"of mask; "
            f"{100.0 * unstable_mass_fraction_positive:.4f}% "
            f"of positive mass)"
        )
        print(
            "positivity-refined voxels     : "
            f"{positivity_refinement_voxels} "
            f"({100.0 * positivity_refinement_fraction:.4f}% "
            f"of mask)"
        )
        print(
            "maximum absolute c0 error     : "
            f"{maximum_absolute_c0_error:.8g}"
        )
        print(
            "negative validation fraction  : "
            f"{qc['negative_validation_fraction']:.8g}"
        )
        print(
            "median EPS                    : "
            f"{qc['median_eps']:.8g}"
        )
        print(
            "95th percentile EPS           : "
            f"{qc['eps_p95']:.8g}"
        )
        print(
            "maximum EPS                   : "
            f"{qc['maximum_eps']:.8g}"
        )

        if args.qc is not None:
            qc_path = args.qc.expanduser().resolve()
            qc_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with qc_path.open(
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    qc,
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")

            print(f"QC report: {qc_path}")

        print(f"Output: {output_path}")

        if args.strict and negative_samples_final:
            print(
                "[ERROR] Significant negative amplitudes remain "
                "after final SH positivity refinement."
            )
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
