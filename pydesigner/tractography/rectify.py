from __future__ import annotations

import numpy as np


def rectify_odf_samples(
    fodf,
    sh_area,
    *,
    iterations: int = 100,
    tolerance: float = 1e-8,
    return_offsets: bool = False,
):
    """
    Rectify sampled ODF amplitudes using the pyDKE bisection algorithm.

    Parameters
    ----------
    fodf : array_like
        ODF amplitudes. The final dimension contains spherical samples.
        Input may be shape (n_directions,) or (n_voxels, n_directions).
    sh_area : array_like
        Spherical quadrature weights corresponding to the final dimension.
    iterations : int
        Maximum number of bisection iterations.
    tolerance : float
        Bisection convergence tolerance and numerical zero threshold.
        The default (1e-2) reproduces the historical pyDKE minZero scale.
    return_offsets : bool
        If True, also return the per-voxel bisection offset EPS.

    Returns
    -------
    rectified : ndarray
        Nonnegative sampled ODF amplitudes.
    offsets : ndarray, optional
        Per-voxel EPS values when return_offsets=True.
    """
    values = np.asarray(fodf, dtype=np.float64)
    weights = np.asarray(sh_area, dtype=np.float64).reshape(-1)

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")

    original_shape = values.shape
    squeeze = values.ndim == 1

    if squeeze:
        values = values[None, :]
    elif values.ndim != 2:
        values = values.reshape(-1, values.shape[-1])

    if values.shape[1] != weights.size:
        raise ValueError(
            "ODF direction dimension and sh_area length differ: "
            f"{values.shape[1]} != {weights.size}"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError("sh_area contains non-finite values")

    if np.any(weights < 0):
        raise ValueError("sh_area contains negative values")

    if not np.any(weights > 0):
        raise ValueError("sh_area contains no positive weights")

    f = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    out = np.array(f, copy=True)
    offsets = np.zeros(f.shape[0], dtype=np.float64)

    # Do not perturb ODFs that are already nonnegative.
    needs_rectification = np.any(f < 0.0, axis=1)

    # An ODF with no positive support cannot be recovered by a positive shift.
    no_positive_support = np.max(f, axis=1) <= 0.0
    if np.any(no_positive_support):
        out[no_positive_support, :] = 0.0

    active = needs_rectification & ~no_positive_support

    if np.any(active):
        a = f[active, :]

        lower = np.zeros(a.shape[0], dtype=np.float64)
        upper = np.max(a, axis=1)
        eps = np.zeros(a.shape[0], dtype=np.float64)

        def objective(level):
            level = level[:, None]
            return np.sum(
                (
                    np.abs(a - level)
                    - a
                    - level
                )
                * weights[None, :],
                axis=1,
            )

        f_lower = objective(lower)
        live = np.ones(a.shape[0], dtype=bool)

        for _ in range(iterations):
            midpoint = (lower + upper) / 2.0
            f_midpoint = objective(midpoint)

            converged = live & (
                np.isclose(f_midpoint, 0.0, atol=np.finfo(float).eps)
                | (((upper - lower) / 2.0) < tolerance)
            )

            eps[converged] = midpoint[converged]

            update = live & ~converged
            if not np.any(update):
                live[:] = False
                break

            same_sign = (
                np.sign(f_midpoint)
                == np.sign(f_lower)
            )

            move_lower = update & same_sign
            move_upper = update & ~same_sign

            lower[move_lower] = midpoint[move_lower]
            f_lower[move_lower] = f_midpoint[move_lower]
            upper[move_upper] = midpoint[move_upper]

            live = update

        # Robust fallback if max iterations is reached.
        if np.any(live):
            eps[live] = (
                lower[live] + upper[live]
            ) / 2.0

        rectified = 0.5 * (
            np.abs(a - eps[:, None])
            + a
            - eps[:, None]
        )

        rectified[
            np.abs(rectified) < tolerance
        ] = 0.0

        # Numerical safety.
        rectified[rectified < 0.0] = 0.0

        out[active, :] = rectified
        offsets[active] = eps

    if squeeze:
        out = out[0]
        offsets = offsets[0]

    if return_offsets:
        return out, offsets

    return out


def rectify_normalized_odf_samples(
    fodf,
    sh_area,
    *,
    iterations=100,
    tolerance=1e-8,
    target_hemisphere_mass=0.5,
    min_mass_stability_ratio=0.20,
    mass_conservation_tolerance=1e-6,
    return_offsets=False,
):
    """
    Rectify sampled antipodally symmetric fODFs after unit-integral
    normalization while preserving the original angular mass.

    This wrapper is intended for sampled CSD fODFs. It does NOT modify the
    behavior of ``rectify_odf_samples`` used elsewhere.

    For each sampled fODF f_i with quadrature weights w_i:

        P = sum_i w_i max(f_i, 0)
        N = sum_i w_i max(-f_i, 0)
        M = P - N
        A = P + N
        R = M / A

    A voxel is considered valid when:

        * all sampled amplitudes are finite;
        * M > 0;
        * A > 0; and
        * R >= min_mass_stability_ratio.

    Valid fODFs are scaled such that the sampled hemisphere integral equals
    ``target_hemisphere_mass``. For an antipodally symmetric fODF,
    target_hemisphere_mass=0.5 corresponds to a full-sphere integral of 1.

    The normalized samples are passed through ``rectify_odf_samples`` and then
    rescaled to their original signed angular mass M.

    Invalid / unstable rows are returned as zeros.

    Parameters
    ----------
    fodf : array_like
        Sampled fODFs with shape (n_voxels, n_directions), or a single
        (n_directions,) fODF.
    sh_area : array_like
        Non-negative quadrature weights for the sampled hemisphere.
    iterations : int
        Maximum bisection iterations.
    tolerance : float
        Numerical bisection convergence tolerance. Default 1e-8.
    target_hemisphere_mass : float
        Angular mass of the internally normalized hemisphere distribution.
        Default 0.5.
    min_mass_stability_ratio : float
        Minimum accepted R=(P-N)/(P+N). Default 0.20.
    mass_conservation_tolerance : float
        Maximum allowed absolute error in normalized hemisphere mass after
        bisection. Default 1e-6.
    return_offsets : bool
        If True, also return normalized-space bisection offsets.

    Returns
    -------
    out : ndarray
        Rectified samples restored to the original angular mass.
    eps : ndarray, optional
        Bisection offsets in normalized fODF amplitude units.
    """

    import numpy as np

    x = np.asarray(fodf, dtype=np.float64)
    w = np.asarray(sh_area, dtype=np.float64).reshape(-1)

    input_was_1d = x.ndim == 1

    if input_was_1d:
        x = x[np.newaxis, :]

    if x.ndim != 2:
        raise ValueError(
            "fodf must have shape (n_voxels, n_directions) "
            "or (n_directions,)"
        )

    if x.shape[1] != w.size:
        raise ValueError(
            f"fODF directions ({x.shape[1]}) do not match "
            f"quadrature weights ({w.size})"
        )

    if not np.all(np.isfinite(w)):
        raise ValueError("quadrature weights must all be finite")

    if np.any(w < 0):
        raise ValueError("quadrature weights must be non-negative")

    if not np.any(w > 0):
        raise ValueError("at least one quadrature weight must be positive")

    if not np.isfinite(target_hemisphere_mass) or target_hemisphere_mass <= 0:
        raise ValueError("target_hemisphere_mass must be finite and > 0")

    if (
        not np.isfinite(min_mass_stability_ratio)
        or min_mass_stability_ratio < 0
        or min_mass_stability_ratio > 1
    ):
        raise ValueError(
            "min_mass_stability_ratio must be finite and within [0, 1]"
        )

    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")

    if mass_conservation_tolerance <= 0:
        raise ValueError("mass_conservation_tolerance must be > 0")

    # Output is deliberately zero for rejected rows.
    out = np.zeros_like(x, dtype=np.float64)
    eps_out = np.zeros(x.shape[0], dtype=np.float64)

    finite = np.all(np.isfinite(x), axis=1)

    # Use a safe copy for mass calculations. Non-finite rows remain invalid.
    safe = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    positive_mass = np.sum(
        np.maximum(safe, 0.0) * w[np.newaxis, :],
        axis=1,
    )

    negative_mass = np.sum(
        np.maximum(-safe, 0.0) * w[np.newaxis, :],
        axis=1,
    )

    signed_mass = positive_mass - negative_mass
    absolute_mass = positive_mass + negative_mass

    ratio = np.zeros(x.shape[0], dtype=np.float64)

    ratio_defined = (
        finite
        & np.isfinite(signed_mass)
        & np.isfinite(absolute_mass)
        & (absolute_mass > 0)
    )

    ratio[ratio_defined] = (
        signed_mass[ratio_defined]
        / absolute_mass[ratio_defined]
    )

    valid = (
        ratio_defined
        & (signed_mass > 0)
        & (ratio >= min_mass_stability_ratio)
    )

    if np.any(valid):
        m = signed_mass[valid]

        # For an antipodally symmetric fODF:
        #
        # target hemisphere mass = 0.5
        #             <=>
        # full-sphere integral = 1.0
        #
        # More generally:
        #
        # f_norm = f * target / M
        #
        scale_to_normalized = target_hemisphere_mass / m

        normalized = (
            safe[valid]
            * scale_to_normalized[:, np.newaxis]
        )

        rectified_norm, eps = rectify_odf_samples(
            normalized,
            w,
            iterations=iterations,
            tolerance=tolerance,
            return_offsets=True,
        )

        # Verify the property that motivated normalization:
        # bisection must preserve angular mass.
        mass_after = rectified_norm @ w

        mass_error = np.abs(
            mass_after - target_hemisphere_mass
        )

        max_mass_error = float(np.max(mass_error))

        if max_mass_error > mass_conservation_tolerance:
            raise RuntimeError(
                "normalized fODF bisection failed angular-mass "
                "conservation QC: "
                f"max error={max_mass_error:.12g}, "
                f"allowed={mass_conservation_tolerance:.12g}"
            )

        # Restore each voxel's original signed CSD angular mass.
        scale_from_normalized = (
            m / target_hemisphere_mass
        )

        out[valid] = (
            rectified_norm
            * scale_from_normalized[:, np.newaxis]
        )

        eps_out[valid] = np.asarray(
            eps,
            dtype=np.float64,
        ).reshape(-1)

    if input_was_1d:
        out = out[0]
        eps_out = eps_out[0]

    if return_offsets:
        return out, eps_out

    return out

