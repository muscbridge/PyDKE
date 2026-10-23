"""First-moment bias inversion for denoised root-sum-of-squares magnitude MRI.

``sigma_component`` is the standard deviation of the underlying Gaussian noise
in EACH real/imaginary component. An MP-PCA magnitude fluctuation map is not that
parameter at low SNR. The explicit joint mean/variance path can infer a Gaussian
component SD under an effective noncentral-chi model; this is not a calibration.
Effective coil counts require reconstruction-specific justification.

The returned amplitude is a nonnegative model estimate, not a claim of unbiased
finite-sample recovery. No second-moment subtraction or absolute-value repair is
used. Reference: Andre et al., PLoS ONE 9:e94531 (2014), equations 3 and 4;
Koay and Basser, J Magn Reson 179:317-322 (2006).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, hyp1f1

ALGORITHM_VERSION = "ncchi-denoised-mean-v1"
MAX_EFFECTIVE_COILS = 4096.0  # Declared numerical model range, not a hardware inference.


def _real_array(value, name):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(name + " must be real numeric data")
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _beta(n_eff):
    if np.any(np.asarray(n_eff) > MAX_EFFECTIVE_COILS):
        raise ValueError("n_eff exceeds the supported numerical model range [1,4096]")
    # gammaln avoids overflow from separately evaluating Gamma(L+1/2)/Gamma(L).
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp(.5*np.log(2.) + gammaln(n_eff+.5)-gammaln(n_eff))
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise RuntimeError("Noncentral-chi mean factor could not be evaluated")
    df = 2*np.asarray(n_eff)
    tolerance = 128*np.finfo(float).eps*np.maximum(1., df)
    if np.any(result**2 < df-1-tolerance) or np.any(result**2 >= df):
        raise RuntimeError("Noncentral-chi central mean/variance failed numerical sanity bounds")
    return result


def _mean_and_derivative(amplitude, n_eff, beta):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        argument = -.5*amplitude*amplitude
        mean = beta*hyp1f1(-.5, n_eff, argument)
        derivative = (beta/(2*n_eff))*amplitude*hyp1f1(.5, n_eff+1, argument)
    if not np.isfinite(mean).all() or not np.isfinite(derivative).all():
        raise RuntimeError("Noncentral-chi mean/derivative is outside the numerical range")
    return mean, derivative


def ncchi_mean(noncentrality, n_eff):
    """Return E[M] for sigma=1 and 2*n_eff degrees of freedom (float64).

    ``noncentrality`` is A/sigma, not its square. Inputs broadcast by NumPy rules;
    amplitudes must be finite/nonnegative and effective coil counts in [1,4096].
    The hypergeometric argument is -A**2/2, including the factor one half.
    """
    amplitude = _real_array(noncentrality, "noncentrality")
    coils = _real_array(n_eff, "n_eff")
    if np.any(amplitude < 0) or np.any(coils < 1):
        raise ValueError("Require noncentrality >=0 and n_eff >=1")
    try:
        a, L = np.broadcast_arrays(np.asarray(amplitude, float), np.asarray(coils, float))
    except ValueError as exc:
        raise ValueError("Noncentrality and n_eff must broadcast") from exc
    result, _ = _mean_and_derivative(a, L, _beta(L))
    return result


def _parameter(value, name, spatial_shape, full_shape, allow_full):
    array = _real_array(value, name)
    if array.ndim == 0:
        return array, "scalar"
    if array.shape == spatial_shape:
        return array, "spatial"
    if allow_full and array.shape == full_shape:
        return array, "full"
    permitted = "scalar, exact spatial 3D map" + (", or exact 4D array" if allow_full else "")
    raise ValueError(name + " must be a " + permitted)


def _chunk_parameter(array, mode, start, stop, spatial_index):
    if mode == "scalar":
        return np.full(stop-start, float(array), dtype=np.float64)
    if mode == "spatial":
        # flatiter indexing handles non-contiguous maps without copying all data.
        return np.asarray(array.flat[spatial_index], dtype=np.float64)
    return np.asarray(array.flat[start:stop], dtype=np.float64)


def _inverse(mean, coils, max_iterations, rtol, atol):
    """Safeguarded vector Newton solve, restricted to means above the floor."""
    beta = _beta(coils)
    if np.any(mean <= beta):
        raise RuntimeError("Internal inverse requires means above the central-chi mean")
    lower, upper = np.zeros_like(mean), mean.copy()
    # Stable sqrt(mean**2-beta**2); close to the root at both low and high SNR.
    estimate = mean*np.sqrt(np.maximum(0., 1.-(beta/mean)**2))
    active = np.arange(mean.size)
    epsilon = np.finfo(np.float64).eps
    for iteration in range(1, max_iterations+1):
        x, L = estimate[active], coils[active]
        predicted, derivative = _mean_and_derivative(x, L, beta[active])
        residual = predicted-mean[active]
        high = residual >= 0
        upper[active[high]] = x[high]
        lower[active[~high]] = x[~high]
        tolerance = atol+rtol*np.maximum(1., np.abs(x))
        # The roundoff term matters near zero, where inversion is ill-conditioned.
        finished = ((upper[active]-lower[active] <= tolerance) |
                    (np.abs(residual) <= derivative*tolerance + 4*epsilon*np.maximum(1., mean[active])))
        if finished.all():
            return estimate, iteration
        remaining = ~finished
        indices = active[remaining]
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = x[remaining]-residual[remaining]/derivative[remaining]
        inside = (np.isfinite(candidate) & (candidate > lower[indices]) & (candidate < upper[indices]))
        middle = lower[indices]+.5*(upper[indices]-lower[indices])
        estimate[indices] = np.where(inside, candidate, middle)
        active = indices
    raise RuntimeError(f"Noncentral-chi mean inversion did not converge within {max_iterations} iterations")


def correct_ncchi_mean(mean_signal, sigma_component, n_eff, mask=None,
                       return_diagnostics=False, *, chunk_size=65536,
                       max_iterations=64, rtol=1e-10, atol=1e-12):
    """Invert the noncentral-chi first moment of denoised magnitude estimates.

    Args:
        mean_signal: Finite real 4D array approximating E[M], not raw magnitudes.
        sigma_component: Calibrated underlying Gaussian SD per real/imaginary
            component: scalar, exact spatial 3D map, or exact full 4D array.
        n_eff: Effective coil count in [1,4096], scalar or exact spatial 3D map. Fractional
            counts are supported. This is not inferred from physical coil count.
        mask: Optional exact spatial 3D binary mask; outside it values are copied.
        return_diagnostics: Return ``(corrected, info)`` when True.

    Returns:
        A new float64 amplitude array. Selected estimates at/below the central
        mean, including negative denoising estimates, return the explicit A=0
        boundary. Sigma=0 preserves nonnegative estimates; negative ones give 0.
        Diagnostics include full 4D boundary, zero-noise and negative-estimate
        masks and counts. No input is modified; no spatial map is regridded.

    Solver tolerances apply to dimensionless amplitude A/sigma. Chunking bounds
    temporary working arrays; the output (and requested diagnostic masks) occupy
    full image size. Numerical failure raises; there is no silent fallback.
    """
    signal = _real_array(mean_signal, "mean_signal")
    if signal.ndim != 4 or any(size == 0 for size in signal.shape):
        raise ValueError("mean_signal must be a nonempty real 4D array")
    sigma, sigma_mode = _parameter(sigma_component, "sigma_component", signal.shape[:3], signal.shape, True)
    coils, coil_mode = _parameter(n_eff, "n_eff", signal.shape[:3], signal.shape, False)
    if np.any(sigma < 0) or np.any(coils < 1):
        raise ValueError("Require sigma_component >=0 and n_eff >=1")
    _beta(np.asarray(coils, float))
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise ValueError("return_diagnostics must be boolean")
    if (not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, (bool, np.bool_)) or chunk_size < 1
            or not isinstance(max_iterations, (int, np.integer)) or isinstance(max_iterations, (bool, np.bool_)) or max_iterations < 1):
        raise ValueError("chunk_size and max_iterations must be positive integers")
    if not np.isscalar(rtol) or not np.isscalar(atol) or not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0 or atol < 0:
        raise ValueError("Require finite positive rtol and nonnegative atol")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != signal.shape[:3] or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("mask must be a finite binary exact spatial 3D map")
    output = np.array(signal, dtype=np.float64, copy=True, order="C")
    flat = output.reshape(-1)
    names = ("boundary", "zero_noise", "negative_estimate")
    masks = {name: np.zeros(signal.shape, dtype=bool) for name in names} if return_diagnostics else {}
    counts = {name: 0 for name in names}
    selected_count, solved_count, most_iterations = 0, 0, 0
    nvol = signal.shape[-1]
    for start in range(0, signal.size, int(chunk_size)):
        stop = min(start+int(chunk_size), signal.size)
        spatial_index = np.arange(start, stop)//nvol
        selected = np.ones(stop-start, bool) if mask is None else np.asarray(mask.flat[spatial_index], bool)
        selected_count += int(selected.sum())
        chunk = flat[start:stop]
        s = _chunk_parameter(sigma, sigma_mode, start, stop, spatial_index)
        L = _chunk_parameter(coils, coil_mode, start, stop, spatial_index)
        negative = selected & (chunk < 0)
        zero_noise = selected & (s == 0)
        candidates = selected & (s > 0) & (chunk > 0)
        boundary = selected & (chunk <= 0)
        if candidates.any():
            selected_index = np.flatnonzero(candidates)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                normalized = chunk[selected_index]/s[selected_index]
            if not np.isfinite(normalized).all():
                raise RuntimeError("Signal/sigma is outside the normalized numerical range")
            floor = _beta(L[selected_index])
            at_boundary = normalized <= floor
            boundary[selected_index[at_boundary]] = True
            solve = selected_index[~at_boundary]
            if solve.size:
                amplitude, iterations = _inverse(normalized[~at_boundary], L[solve], int(max_iterations), float(rtol), float(atol))
                with np.errstate(over="ignore", invalid="ignore"):
                    chunk[solve] = amplitude*s[solve]
                if not np.isfinite(chunk[solve]).all():
                    raise RuntimeError("Corrected amplitude is outside the numerical range")
                solved_count += int(solve.size)
                most_iterations = max(most_iterations, iterations)
        chunk[boundary] = 0.
        for name, flags in (("boundary", boundary), ("zero_noise", zero_noise), ("negative_estimate", negative)):
            counts[name] += int(flags.sum())
            if return_diagnostics:
                masks[name].reshape(-1)[start:stop] = flags
    if not return_diagnostics:
        return output
    info = {"algorithm_version": ALGORITHM_VERSION,
        "input_domain": "denoised_magnitude_mean", "output_domain": "nonnegative_underlying_amplitude",
        "sigma_convention": "underlying_Gaussian_SD_per_real_or_imaginary_component",
        "sigma_mode": sigma_mode, "n_eff_mode": coil_mode,
        "output_dtype": "float64", "selected_samples": selected_count,
        "solved_samples": solved_count, "boundary_samples": counts["boundary"],
        "zero_noise_samples": counts["zero_noise"], "negative_estimate_samples": counts["negative_estimate"],
        "converged": True, "maximum_iterations_used": most_iterations,
        "max_iterations": int(max_iterations), "normalized_rtol": float(rtol), "normalized_atol": float(atol),
        "chunk_size": int(chunk_size), "boundary_mask": masks["boundary"],
        "zero_noise_mask": masks["zero_noise"], "negative_estimate_mask": masks["negative_estimate"]}
    return output, info


def _unit_magnitude_moments(amplitude, coils, beta):
    """Mean, variance and their derivatives with sigma_component=1.

    Direct variance subtraction is ill-conditioned at high SNR. The expansion
    below retains terms through rho^-6 and switches only when rho/sqrt(2L)>=32;
    this keeps omitted terms small while avoiding cancellation at rho=1e5.
    """
    mean, mean_derivative = _mean_and_derivative(amplitude, coils, beta)
    df = 2*coils
    variance = df+(amplitude-mean)*(amplitude+mean)
    variance_derivative = 2*(amplitude-mean*mean_derivative)
    large = amplitude >= 32*np.sqrt(df)
    if np.any(large):
        a, k = amplitude[large], df[large]
        inverse = 1/a
        t = inverse**2
        c2 = -(k-1)/2
        c4 = (k-1)*(k-3)/2
        c6 = (k-1)*(k-3)*(19-4*k)/8
        variance[large] = 1+c2*t+c4*t**2+c6*t**3
        variance_derivative[large] = -2*inverse*(c2*t+2*c4*t**2+3*c6*t**3)
    if not np.isfinite(variance).all() or np.any(variance <= 0) or np.any(variance > 1+1e-8):
        raise RuntimeError("Noncentral-chi variance is outside the valid numerical range")
    return mean, variance, mean_derivative, variance_derivative


def ncchi_variance(noncentrality, n_eff):
    """Return magnitude variance for unit component Gaussian SD (float64).

    This is NOT the variance after denoising. It describes the magnitude noise
    profile whose fluctuations MP-PCA estimates under its local model.
    """
    amplitude = _real_array(noncentrality, "noncentrality")
    coils = _real_array(n_eff, "n_eff")
    if np.any(amplitude < 0) or np.any(coils < 1):
        raise ValueError("Require noncentrality >=0 and n_eff >=1")
    try:
        a, L = np.broadcast_arrays(np.asarray(amplitude, float), np.asarray(coils, float))
    except ValueError as exc:
        raise ValueError("Noncentrality and n_eff must broadcast") from exc
    shape = a.shape
    _, variance, _, _ = _unit_magnitude_moments(a.reshape(-1), L.reshape(-1), _beta(L.reshape(-1)))
    return variance.reshape(shape)


def _joint_inverse(ratio, coils, max_iterations, rtol, atol):
    beta = _beta(coils)
    central_variance = 2*coils-beta**2
    central_ratio = beta/np.sqrt(central_variance)
    if np.any(ratio <= central_ratio):
        raise RuntimeError("Internal joint inverse requires an above-central mean/SD ratio")
    lower, upper = np.zeros_like(ratio), ratio.copy()
    excess = ratio/central_ratio-1
    # r-r0 is order rho^4 near zero: the quadratic terms cancel.
    small = (16*coils*(coils+1)*central_variance*excess)**.25
    large = ratio*np.sqrt(np.maximum(0., 1-(central_ratio/ratio)**2))
    estimate = np.minimum(np.where(excess < .05, small, large), upper)
    active = np.arange(ratio.size)
    epsilon = np.finfo(np.float64).eps
    for iteration in range(1, max_iterations+1):
        x, L = estimate[active], coils[active]
        mean, variance, dm, dv = _unit_magnitude_moments(x, L, beta[active])
        root_variance = np.sqrt(variance)
        predicted = mean/root_variance
        derivative = dm/root_variance-.5*mean*dv/(variance*root_variance)
        residual = predicted-ratio[active]
        high = residual >= 0
        upper[active[high]] = x[high]
        lower[active[~high]] = x[~high]
        tolerance = atol+rtol*np.maximum(1., np.abs(x))
        finished = ((upper[active]-lower[active] <= tolerance) |
            (np.abs(residual) <= np.maximum(derivative, 0)*tolerance
             + 8*epsilon*np.maximum(1., ratio[active])))
        if finished.all():
            _, final_variance, _, _ = _unit_magnitude_moments(estimate, coils, beta)
            return estimate, final_variance, iteration
        remaining = ~finished
        indices = active[remaining]
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = x[remaining]-residual[remaining]/derivative[remaining]
        inside = ((derivative[remaining] > 0) & np.isfinite(candidate)
                  & (candidate > lower[indices]) & (candidate < upper[indices]))
        middle = lower[indices]+.5*(upper[indices]-lower[indices])
        estimate[indices] = np.where(inside, candidate, middle)
        active = indices
    raise RuntimeError(f"Joint noncentral-chi mean/variance inversion did not converge within {max_iterations} iterations")


def correct_ncchi_magnitude(mean_signal, sigma_magnitude, n_eff, mask=None,
                            return_diagnostics=False, *, chunk_size=65536,
                            max_iterations=64, rtol=1e-10, atol=1e-12):
    """Generalized Koay joint mean/variance inversion of magnitude noise profiles.

    ``mean_signal`` is a denoised magnitude mean estimate. ``sigma_magnitude``
    is the ORIGINAL magnitude-fluctuation SD estimated by magnitude MP-PCA, not
    residual SD after denoising and not component Gaussian SD. Given effective L,
    solve mu/sigma_M=f_L(rho)/sqrt(v_L(rho)), infer sigma_G=sigma_M/sqrt(v), and
    return A=rho*sigma_G. Both denoising methods remain separate and unchanged.

    Shapes, masks, dtype and chunking follow ``correct_ncchi_mean``. Outside a
    mask amplitudes are copied and inferred sigma is undefined (NaN). At/below
    the central mean/SD ratio, the explicit A=0 boundary uses sigma_G inferred
    from sigma_M and the central variance. Zero sigma_M preserves nonnegative
    signal with sigma_G=0; negative estimates give zero and remain flagged.

    Near-zero inversion is ill-conditioned (ratio excess is O(rho^4)). The
    diagnostic ill_conditioned_mask marks relative ratio excess <=1e-6, without
    imposing a positive signal floor or excluding those values. Inferred Gaussian
    sigma is MODEL-BASED, not a calibration. A shared MP map approximates a local
    magnitude SD that can be signal dependent across volumes at low SNR; inferred
    per-sample sigma does not establish that this approximation is exact.
    """
    signal = _real_array(mean_signal, "mean_signal")
    if signal.ndim != 4 or any(size == 0 for size in signal.shape):
        raise ValueError("mean_signal must be a nonempty real 4D array")
    sigma, sigma_mode = _parameter(sigma_magnitude, "sigma_magnitude", signal.shape[:3], signal.shape, True)
    coils, coil_mode = _parameter(n_eff, "n_eff", signal.shape[:3], signal.shape, False)
    if np.any(sigma < 0) or np.any(coils < 1):
        raise ValueError("Require sigma_magnitude >=0 and n_eff >=1")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise ValueError("return_diagnostics must be boolean")
    if (not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, (bool, np.bool_)) or chunk_size < 1
            or not isinstance(max_iterations, (int, np.integer)) or isinstance(max_iterations, (bool, np.bool_)) or max_iterations < 1):
        raise ValueError("chunk_size and max_iterations must be positive integers")
    if not np.isscalar(rtol) or not np.isscalar(atol) or not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0 or atol < 0:
        raise ValueError("Require finite positive rtol and nonnegative atol")
    # Validate effective-L numerical range even for an empty mask/noiseless data.
    _beta(np.asarray(coils, float))
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != signal.shape[:3] or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("mask must be a finite binary exact spatial 3D map")
    output = np.array(signal, dtype=np.float64, copy=True, order="C")
    flat = output.reshape(-1)
    names = ("boundary", "zero_noise", "negative_estimate", "ill_conditioned")
    masks = {name: np.zeros(signal.shape, dtype=bool) for name in names} if return_diagnostics else {}
    inferred = np.full(signal.shape, np.nan, dtype=np.float64) if return_diagnostics else None
    counts = {name: 0 for name in names}
    selected_count, solved_count, most_iterations = 0, 0, 0
    for start in range(0, signal.size, int(chunk_size)):
        stop = min(start+int(chunk_size), signal.size)
        spatial_index = np.arange(start, stop)//signal.shape[-1]
        selected = np.ones(stop-start, bool) if mask is None else np.asarray(mask.flat[spatial_index], bool)
        selected_count += int(selected.sum())
        chunk = flat[start:stop]
        s = _chunk_parameter(sigma, sigma_mode, start, stop, spatial_index)
        L = _chunk_parameter(coils, coil_mode, start, stop, spatial_index)
        negative = selected & (chunk < 0)
        zero_noise = selected & (s == 0)
        boundary = selected & (chunk <= 0)
        ill = np.zeros(stop-start, bool)
        gaussian = np.full(stop-start, np.nan)
        gaussian[zero_noise] = 0.
        candidates = selected & (s > 0)
        if candidates.any():
            index = np.flatnonzero(candidates)
            beta = _beta(L[index])
            central_variance = 2*L[index]-beta**2
            central_ratio = beta/np.sqrt(central_variance)
            # Nonpositive estimates are below the central ratio without division.
            ratio = np.zeros(index.size)
            positive = chunk[index] > 0
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                ratio[positive] = chunk[index[positive]]/s[index[positive]]
            if not np.isfinite(ratio).all():
                raise RuntimeError("Magnitude mean/SD ratio is outside the numerical range")
            at_boundary = ratio <= central_ratio
            boundary[index[at_boundary]] = True
            ill[index] = (ratio/central_ratio-1) <= 1e-6
            gaussian[index[at_boundary]] = s[index[at_boundary]]/np.sqrt(central_variance[at_boundary])
            solve = index[~at_boundary]
            if solve.size:
                rho, variance, iterations = _joint_inverse(ratio[~at_boundary], L[solve], int(max_iterations), float(rtol), float(atol))
                with np.errstate(over="ignore", invalid="ignore"):
                    gaussian[solve] = s[solve]/np.sqrt(variance)
                    chunk[solve] = rho*gaussian[solve]
                if not np.isfinite(chunk[solve]).all():
                    raise RuntimeError("Joint corrected amplitude is outside the numerical range")
                solved_count += int(solve.size)
                most_iterations = max(most_iterations, iterations)
            if not np.isfinite(gaussian[index]).all():
                raise RuntimeError("Inferred Gaussian sigma is outside the numerical range")
        chunk[boundary] = 0.
        for name, flags in (("boundary", boundary), ("zero_noise", zero_noise),
                            ("negative_estimate", negative), ("ill_conditioned", ill)):
            counts[name] += int(flags.sum())
            if return_diagnostics:
                masks[name].reshape(-1)[start:stop] = flags
        if return_diagnostics:
            inferred.reshape(-1)[start:stop] = gaussian
    if not return_diagnostics:
        return output
    info = {"algorithm_version": "ncchi-magnitude-joint-koay-v1",
        "input_domain": "denoised_magnitude_mean", "output_domain": "nonnegative_underlying_amplitude",
        "sigma_convention": "original_magnitude_fluctuation_SD",
        "inferred_sigma_convention": "model_inferred_Gaussian_SD_per_real_or_imaginary_component",
        "inferred_sigma_is_calibration": False, "sigma_mode": sigma_mode, "n_eff_mode": coil_mode,
        "output_dtype": "float64", "selected_samples": selected_count, "solved_samples": solved_count,
        "boundary_samples": counts["boundary"], "zero_noise_samples": counts["zero_noise"],
        "negative_estimate_samples": counts["negative_estimate"], "ill_conditioned_samples": counts["ill_conditioned"],
        "ill_conditioned_relative_central_ratio_excess": 1e-6,
        "converged": True, "maximum_iterations_used": most_iterations, "max_iterations": int(max_iterations),
        "normalized_rtol": float(rtol), "normalized_atol": float(atol), "chunk_size": int(chunk_size),
        "inferred_sigma_component": inferred, "boundary_mask": masks["boundary"],
        "zero_noise_mask": masks["zero_noise"], "negative_estimate_mask": masks["negative_estimate"],
        "ill_conditioned_mask": masks["ill_conditioned"]}
    return output, info
