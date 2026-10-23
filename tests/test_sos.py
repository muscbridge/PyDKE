"""Independent distribution checks for opt-in denoised-magnitude mean inversion."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import ncx2, rice
from scipy.special import i0e, i1e

from pydesigner.preprocessing.sos import correct_ncchi_mean, ncchi_mean


def test_forward_matches_independent_rician_implementation():
    amplitude = np.array([0., 1e-4, .1, 1., 3., 10.])
    np.testing.assert_allclose(ncchi_mean(amplitude, 1), rice.mean(amplitude), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("coils", [1.5, 4., 16.])
@pytest.mark.parametrize("amplitude", [0., .5, 4., 15.])
def test_forward_matches_independent_noncentral_chisquare_quadrature(coils, amplitude):
    # M² has an ncx2 law; integrate its density, independently of the mean formula.
    upper = ncx2.ppf(1.-1e-12, 2*coils, amplitude**2)
    expected = quad(lambda x: np.sqrt(x)*ncx2.pdf(x, 2*coils, amplitude**2),
                    0., upper, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
    np.testing.assert_allclose(ncchi_mean(amplitude, coils), expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("coils", [1., 1.5, 4., 16.])
def test_inverse_recovery_scaling_and_determinism(coils):
    amplitude = np.geomspace(1e-4, 1e5, 19).reshape(1, 1, 1, -1)
    for scale in (1e-3, 1., 1e3):
        mean = ncchi_mean(amplitude, coils)*scale
        first, info = correct_ncchi_mean(mean, scale, coils, return_diagnostics=True, chunk_size=7)
        second = correct_ncchi_mean(mean, scale, coils, chunk_size=3)
        np.testing.assert_allclose(first/scale, amplitude, rtol=3e-6, atol=3e-10)
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12*scale)
        assert info["converged"] and info["maximum_iterations_used"] <= 64
        assert info["solved_samples"] == mean.size
        assert not info["boundary_mask"].any()


def test_boundaries_zero_noise_negative_estimates_and_nonmutation():
    central = float(ncchi_mean(0., 4.))
    data = np.array([-1., 0., central*.8, central, central*2.]).reshape(1, 1, 1, 5)
    sigma = np.array([1., 1., 1., 1., 0.]).reshape(data.shape)
    data_copy, sigma_copy = data.copy(), sigma.copy()
    out, info = correct_ncchi_mean(data, sigma, 4., return_diagnostics=True)
    np.testing.assert_array_equal(out, np.array([0., 0., 0., 0., central*2.]).reshape(data.shape))
    assert info["boundary_samples"] == 4 and info["negative_estimate_samples"] == 1
    assert info["zero_noise_samples"] == 1
    np.testing.assert_array_equal(data, data_copy); np.testing.assert_array_equal(sigma, sigma_copy)
    zero = correct_ncchi_mean(np.array([-1., 2.]).reshape(1, 1, 1, 2), 0., 1.)
    np.testing.assert_array_equal(zero, np.array([0., 2.]).reshape(1, 1, 1, 2))


def test_spatial_maps_full_sigma_mask_and_noncontiguous_arrays():
    amplitude = np.arange(1., 25.).reshape(2, 2, 2, 3)/3
    coils = np.array([1., 1.5, 4., 16., 2., 3., 1.2, 8.]).reshape(2, 2, 2)
    sigma = np.linspace(.2, 2., 8).reshape(2, 2, 2)
    mean = ncchi_mean(amplitude, coils[..., None])*sigma[..., None]
    mask = np.ones((2, 2, 2), bool); mask[0, 0, 0] = False
    original = [x.copy() for x in (mean, coils, sigma, mask)]
    out, info = correct_ncchi_mean(mean, sigma, coils, mask, True, chunk_size=5)
    np.testing.assert_allclose(out[mask], (amplitude*sigma[..., None])[mask], rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(out[~mask], mean[~mask])
    assert info["selected_samples"] == 21
    full = np.broadcast_to(sigma[..., None], mean.shape)
    other = correct_ncchi_mean(mean[:, ::-1], full[:, ::-1], coils[:, ::-1], mask[:, ::-1], chunk_size=2)
    np.testing.assert_allclose(other[:, ::-1], out, atol=1e-9, rtol=1e-9)
    for first, second in zip((mean, coils, sigma, mask), original):
        np.testing.assert_array_equal(first, second)


def test_high_snr_mean_has_expected_first_moment_limit():
    amplitude = np.array([1e3, 1e4, 1e5])
    for coils in (1., 1.5, 4., 16.):
        # Independent large-A expansion: E[M] = A+(2L-1)/(2A)+O(A^-3).
        expected = amplitude+(2*coils-1)/(2*amplitude)
        np.testing.assert_allclose(ncchi_mean(amplitude, coils), expected, rtol=1e-10, atol=2e-7)


def test_inversion_does_not_subtract_second_moment_from_mean():
    mean = float(ncchi_mean(1., 4.))
    recovered = correct_ncchi_mean(np.full((1, 1, 1, 2), mean), 1., 4.)
    np.testing.assert_allclose(recovered, 1., atol=1e-9)
    assert not np.isclose(np.sqrt(max(mean**2-8., 0.)), 1.)


@pytest.mark.parametrize("name,value", [("sigma_component", -1.), ("sigma_component", np.nan),
    ("n_eff", .5), ("n_eff", np.inf), ("sigma_component", np.ones((2, 1))),
    ("n_eff", np.ones((1, 1, 1, 2))), ("mask", np.full((1, 1, 1), 2)),
    ("mask", np.ones((2, 1, 1))), ("chunk_size", 0), ("max_iterations", 0),
    ("rtol", 0.), ("atol", -1.)])
def test_invalid_parameters_raise(name, value):
    kwargs = dict(mean_signal=np.ones((1, 1, 1, 2)), sigma_component=1., n_eff=4.)
    kwargs[name] = value
    with pytest.raises(ValueError):
        correct_ncchi_mean(**kwargs)


@pytest.mark.parametrize("signal", [np.ones((2, 2, 2)), np.full((1, 1, 1, 2), np.nan),
                                    np.ones((1, 1, 1, 2), complex)])
def test_invalid_signal_raises(signal):
    with pytest.raises(ValueError):
        correct_ncchi_mean(signal, 1., 4.)


def test_nonconvergence_raises_without_returning_partial_result():
    with pytest.raises(RuntimeError, match="did not converge"):
        correct_ncchi_mean(np.full((1, 1, 1, 2), ncchi_mean(.4, 4.)), 1., 4., max_iterations=1)


def test_forward_rejects_invalid_parameters():
    for amplitude, coils in ((-1., 1.), (np.nan, 4.), (1., .5), (1j, 1.)):
        with pytest.raises(ValueError):
            ncchi_mean(amplitude, coils)


def test_rician_scaled_bessel_reference_remains_finite_at_high_snr():
    amplitude = np.array([0., 1e-4, 1., 10., 100., 1e3, 1e5])
    x = amplitude**2/4
    expected = np.sqrt(np.pi/2)*((1+2*x)*i0e(x)+2*x*i1e(x))
    np.testing.assert_allclose(ncchi_mean(amplitude, 1.), expected, rtol=2e-12, atol=2e-12)


def test_negative_estimate_boundary_does_not_divide_by_tiny_sigma():
    signal = np.full((1, 1, 1, 2), -1e300)
    out, info = correct_ncchi_mean(signal, 1e-300, 4., return_diagnostics=True)
    np.testing.assert_array_equal(out, 0.)
    assert info['negative_estimate_samples'] == 2 and info['boundary_samples'] == 2


from pydesigner.preprocessing.sos import correct_ncchi_magnitude, ncchi_variance


@pytest.mark.parametrize("coils", [1., 1.5, 4., 16.])
@pytest.mark.parametrize("amplitude", [0., .1, 2., 15.])
def test_joint_variance_matches_independent_centered_ncx2_quadrature(coils, amplitude):
    lower, upper = ncx2.ppf([1e-12, 1-1e-12], 2*coils, amplitude**2)
    # Center at amplitude to avoid subtracting two large raw moments.
    mean_delta = quad(lambda x: (np.sqrt(x)-amplitude)*ncx2.pdf(x, 2*coils, amplitude**2),
                      lower, upper, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
    second_delta = quad(lambda x: (np.sqrt(x)-amplitude)**2*ncx2.pdf(x, 2*coils, amplitude**2),
                        lower, upper, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
    expected = second_delta-mean_delta**2
    np.testing.assert_allclose(ncchi_variance(amplitude, coils), expected, rtol=3e-9, atol=3e-9)


@pytest.mark.parametrize("coils", [1., 1.5, 4., 16.])
def test_joint_recovers_signal_and_component_sigma_across_snr_and_scale(coils):
    rho = np.array([.1, .3, 1., 3., 10., 100., 1e3, 1e4, 1e5]).reshape(1, 1, 1, -1)
    for scale in (1e-3, 1., 1e3):
        mean, sigma = ncchi_mean(rho, coils)*scale, np.sqrt(ncchi_variance(rho, coils))*scale
        output, info = correct_ncchi_magnitude(mean, sigma, coils, return_diagnostics=True, chunk_size=4)
        np.testing.assert_allclose(output/scale, rho, rtol=3e-5, atol=5e-7)
        np.testing.assert_allclose(info['inferred_sigma_component']/scale, 1., rtol=3e-7, atol=3e-7)
        assert info['converged'] and not info['inferred_sigma_is_calibration']
        repeat = correct_ncchi_magnitude(mean, sigma, coils, chunk_size=2)
        np.testing.assert_allclose(output, repeat, rtol=1e-12, atol=1e-12*scale)


def test_joint_central_boundary_and_zero_noise_are_explicit():
    beta, variance = float(ncchi_mean(0., 4.)), float(ncchi_variance(0., 4.))
    signal = np.array([-1., 0., beta*.8, beta, 7.]).reshape(1, 1, 1, -1)
    sigma = np.array([np.sqrt(variance)]*4+[0.]).reshape(signal.shape)
    original = signal.copy(), sigma.copy()
    output, info = correct_ncchi_magnitude(signal, sigma, 4., return_diagnostics=True)
    np.testing.assert_array_equal(output, np.array([0., 0., 0., 0., 7.]).reshape(signal.shape))
    np.testing.assert_allclose(info['inferred_sigma_component'], np.array([1., 1., 1., 1., 0.]).reshape(signal.shape))
    assert info['boundary_samples'] == 4 and info['ill_conditioned_samples'] == 4
    assert info['zero_noise_samples'] == 1 and info['negative_estimate_samples'] == 1
    np.testing.assert_array_equal(signal, original[0]); np.testing.assert_array_equal(sigma, original[1])


def test_joint_spatial_sigma_coil_maps_mask_and_nonmutation():
    shape = (2, 2, 2, 3)
    coils = np.linspace(1., 4., 8).reshape(shape[:3])
    rho = np.linspace(.5, 20., 8).reshape(shape[:3])
    component = np.linspace(.1, 3., 8).reshape(shape[:3])
    mean = np.repeat((ncchi_mean(rho, coils)*component)[..., None], 3, axis=3)
    magnitude = np.sqrt(ncchi_variance(rho, coils))*component
    mask = np.ones(shape[:3], bool); mask[0, 0, 0] = False
    original = mean.copy(), magnitude.copy(), coils.copy(), mask.copy()
    output, info = correct_ncchi_magnitude(mean[:, ::-1], magnitude[:, ::-1], coils[:, ::-1],
                                          mask[:, ::-1], True, chunk_size=5)
    output, inferred = output[:, ::-1], info['inferred_sigma_component'][:, ::-1]
    expected = np.repeat((rho*component)[..., None], 3, axis=3)
    np.testing.assert_allclose(output[mask], expected[mask], rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(output[~mask], mean[~mask])
    np.testing.assert_allclose(inferred[mask], np.repeat(component[..., None], 3, axis=3)[mask], rtol=1e-7, atol=1e-8)
    assert np.isnan(inferred[~mask]).all()
    for first, second in zip((mean, magnitude, coils, mask), original):
        np.testing.assert_array_equal(first, second)


def test_joint_near_zero_snr_is_flagged_without_positive_floor():
    rho = np.array([0., .001, .01, .05]).reshape(1, 1, 1, -1)
    mean, sd = ncchi_mean(rho, 4.), np.sqrt(ncchi_variance(rho, 4.))
    output, info = correct_ncchi_magnitude(mean, sd, 4., return_diagnostics=True)
    assert np.isfinite(output).all() and (output >= 0).all()
    assert info['ill_conditioned_mask'].all()
    assert output.ravel()[0] == 0.
    # Ratios below double-precision resolution cannot identify tiny positive A.
    np.testing.assert_allclose(output.ravel()[-1], .05, rtol=2e-3, atol=2e-5)


def test_joint_variance_high_snr_is_stable_and_tends_to_one():
    for coils in (1., 1.5, 4., 16.):
        rho = np.array([1e3, 1e4, 1e5])
        variance = ncchi_variance(rho, coils)
        assert np.isfinite(variance).all() and (variance > 0).all() and (variance < 1).all()
        np.testing.assert_allclose(variance, 1-(2*coils-1)/(2*rho**2), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("sigma,coils", [(-1., 4.), (np.inf, 4.), (np.nan, 4.), (1., .5), (1., np.inf), (1., 1e20)])
def test_joint_invalid_noise_or_coil_parameters_raise(sigma, coils):
    with pytest.raises(ValueError):
        correct_ncchi_magnitude(np.ones((1, 1, 1, 2)), sigma, coils)


def test_joint_nonconvergence_raises():
    rho = .4
    with pytest.raises(RuntimeError, match="did not converge"):
        correct_ncchi_magnitude(np.full((1, 1, 1, 2), ncchi_mean(rho, 4.)),
                                float(np.sqrt(ncchi_variance(rho, 4.))), 4., max_iterations=1)


@pytest.mark.parametrize('coils', [1., 1.5, 4., 16.])
def test_joint_expansion_seam_is_continuous_and_ratio_monotone(coils):
    seam = 32*np.sqrt(2*coils)
    rho = seam*np.array([1-1e-8, 1., 1+1e-8])
    mean, variance = ncchi_mean(rho, coils), ncchi_variance(rho, coils)
    ratio = mean/np.sqrt(variance)
    assert (np.diff(mean) > 0).all() and (np.diff(ratio) > 0).all()
    assert np.max(np.abs(np.diff(variance))) < 1e-8
    output = correct_ncchi_magnitude(mean.reshape(1, 1, 1, -1),
                                    np.sqrt(variance).reshape(1, 1, 1, -1), coils)
    np.testing.assert_allclose(output.ravel(), rho, rtol=2e-9, atol=2e-8)


@pytest.mark.parametrize('coils', [1.5, 16.])
@pytest.mark.parametrize('relative_seam', [.8, 1.2])
def test_centered_variance_quadrature_on_both_sides_of_asymptotic_seam(coils, relative_seam):
    rho = 32*np.sqrt(2*coils)*relative_seam
    lower, upper = ncx2.ppf([1e-12, 1-1e-12], 2*coils, rho**2)
    first = quad(lambda x: (np.sqrt(x)-rho)*ncx2.pdf(x, 2*coils, rho**2),
                 lower, upper, epsabs=1e-9, epsrel=1e-9, limit=200)[0]
    second = quad(lambda x: (np.sqrt(x)-rho)**2*ncx2.pdf(x, 2*coils, rho**2),
                  lower, upper, epsabs=1e-9, epsrel=1e-9, limit=200)[0]
    np.testing.assert_allclose(ncchi_variance(rho, coils), second-first**2, rtol=2e-8, atol=2e-8)


def test_maximum_effective_coil_range_has_independent_central_mean_sanity():
    from scipy.special import poch
    coils = 4096.
    expected_beta = np.sqrt(2)*poch(coils, .5)
    np.testing.assert_allclose(ncchi_mean(0., coils), expected_beta, rtol=2e-11, atol=1e-10)
    expected_variance = .5-1/(16*coils)-1/(64*coils**2)+5/(1024*coils**3)
    np.testing.assert_allclose(ncchi_variance(0., coils), expected_variance, rtol=3e-7, atol=1e-7)
    with pytest.raises(ValueError, match='model range'):
        ncchi_mean(0., np.nextafter(coils, np.inf))
