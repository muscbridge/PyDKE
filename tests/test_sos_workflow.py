"""SoS mean-bias workflow contracts; real MRtrix I/O, no research datasets.

The forward fields below are expectations of a noncentral-chi magnitude,
computed independently with SciPy's hypergeometric expression. They represent
externally denoised magnitude means, not random individual magnitude samples.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy.special import gammaln, hyp1f1


AFFINE = np.array([[.3, 0, 0, -3.6], [0, .4, 0, -4.8],
                   [0, 0, 1., -1.], [0, 0, 0, 1.]])
SHAPE = (4, 5, 3, 6)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    return subprocess.run([str(arg) for arg in args], check=True,
                          capture_output=True, text=True, timeout=30)


@pytest.fixture
def adapter():
    if not all(shutil.which(tool) for tool in ('mrconvert', 'mrinfo')):
        pytest.skip('MRtrix is required for the SoS MIF workflow')
    from pydesigner.preprocessing.sos_mif import correct_sos_mif
    return correct_sos_mif


@pytest.fixture
def cli_runtime():
    if not all(shutil.which(tool) for tool in ('mrconvert', 'mrinfo', 'dwidenoise', 'fsl')):
        pytest.skip('Current full CLI imports require MRtrix and FSL')


def save_nifti(path, array, affine=AFFINE):
    image = nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
    return path


def forward_mean(amplitude, sigma, effective_coils):
    """Independent E[M] for known amplitude, Gaussian component sigma and L."""
    amplitude, sigma, coils = np.broadcast_arrays(amplitude, sigma, effective_coils)
    result = np.array(amplitude, dtype=np.float64, copy=True)
    noisy = sigma > 0
    lam = amplitude[noisy] / sigma[noisy]
    L = coils[noisy]
    result[noisy] = (sigma[noisy] * np.sqrt(2.) *
        np.exp(gammaln(L + .5) - gammaln(L)) *
        hyp1f1(-.5, L, -.5 * lam**2))
    return result


def forward_magnitude_sd(amplitude, sigma, effective_coils):
    """Independent magnitude variance from its exact first/second moments."""
    mean = forward_mean(amplitude, sigma, effective_coils)
    variance = np.asarray(amplitude)**2 + 2*np.asarray(effective_coils)*np.asarray(sigma)**2 - mean**2
    return np.sqrt(np.maximum(variance, 0.))


def make_mif(tmp_path, data, name='input'):
    raw = save_nifti(tmp_path / (name + '.nii'), data)
    grad = tmp_path / (name + '_grad.txt')
    np.savetxt(grad, [[0, 0, 0, 0], [1, 0, 0, 1000], [0, 1, 0, 1000],
                     [0, 0, 1, 1000], [1, 0, 0, 1000], [0, 1, 0, 1000]])
    mif = tmp_path / (name + '.mif')
    command('mrconvert', raw, mif, '-grad', grad,
            '-set_property', 'PhaseEncodingDirection', 'j-',
            '-set_property', 'PartialFourier', '1',
            '-set_property', 'sos_test_property', 'retain-me',
            '-nthreads', '1', '-quiet')
    return mif


def export_mif(path, destination):
    command('mrconvert', path, destination, '-nthreads', '1', '-quiet')
    return nib.load(destination)


def property_value(path, property_name):
    return command('mrinfo', path, '-property', property_name).stdout.strip()


def assert_reproducible_fingerprint(summary):
    configuration = {key: summary[key] for key in (
        'algorithm_version', 'input_domain', 'sigma_convention', 'input_sha256',
        'sigma_sha256', 'effective_coils', 'effective_coils_map_sha256')}
    encoded = json.dumps(configuration, sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == summary['parameter_fingerprint']
    assert summary['input_domain'] == 'magnitude-mean'
    assert summary['sigma_convention'] in ('magnitude', 'per-component-gaussian')


def assert_retained_cli_bias_output(output, record):
    """CLI provenance identifies the retained bias stage, not a removed MIF."""
    assert_reproducible_fingerprint(record)
    retained = Path(record['output'])
    assert retained.is_file()
    assert retained.parent.resolve() == (output/'intermediate_nifti').resolve()
    assert retained.suffix == '.nii'
    assert digest(retained) == record['output_sha256']
    assert Path(record['transient_mif_output']).suffix == '.mif'
    stage_properties = json.loads(retained.with_suffix('.json').read_text())
    final_properties = json.loads((output/'dwi_preprocessed.json').read_text())
    for properties in (stage_properties, final_properties):
        assert properties['pydke_sos_noise_correction'] == record['parameter_fingerprint']


def check_qc(output, expected_shape=SHAPE):
    qc = output.parent / (output.stem + '_sos_qc')
    summary = json.loads((qc / 'summary.json').read_text())
    assert isinstance(summary, dict) and summary
    assert_reproducible_fingerprint(summary)
    masks = {}
    for name in ('boundary_mask', 'zero_noise_mask', 'negative_estimate_mask'):
        image = nib.load(qc / (name + '.nii.gz'))
        values = np.asarray(image.dataobj)
        assert image.shape == expected_shape
        assert image.get_data_dtype() == np.dtype('uint8')
        assert np.isin(values, (0, 1)).all()
        np.testing.assert_allclose(image.affine, AFFINE, atol=1e-6, rtol=0)
        masks[name] = values.astype(bool)
    return summary, masks


def test_sos_mif_recovers_known_mean_field_and_preserves_metadata(tmp_path, adapter):
    amplitude = np.linspace(5., 24., np.prod(SHAPE)).reshape(SHAPE)
    sigma = np.full(SHAPE[:3], 2.)
    source = make_mif(tmp_path, forward_mean(amplitude, sigma[..., None], 4.))
    sigma_file = save_nifti(tmp_path / 'gaussian_sigma.nii.gz', sigma)
    before = {path: digest(path) for path in (source, sigma_file)}
    output = tmp_path / 'corrected.mif'
    summary = adapter(str(source), str(output), str(sigma_file), n_eff=4., nthreads=1,
                      sigma_convention='per-component-gaussian')
    assert Path(summary['qc_directory']) == tmp_path / 'corrected_sos_qc'
    result = export_mif(output, tmp_path / 'result.nii')
    np.testing.assert_allclose(result.get_fdata(), amplitude, atol=5e-4, rtol=5e-5)
    np.testing.assert_allclose(result.affine, AFFINE, atol=1e-6, rtol=0)
    assert np.issubdtype(result.get_data_dtype(), np.floating)
    assert command('mrinfo', source, '-dwgrad').stdout == command('mrinfo', output, '-dwgrad').stdout
    for name in ('PhaseEncodingDirection', 'PartialFourier', 'sos_test_property'):
        assert property_value(source, name) == property_value(output, name)
    assert property_value(output, 'pydke_sos_noise_correction')
    _, masks = check_qc(output)
    assert not any(mask.any() for mask in masks.values())
    assert {path: digest(path) for path in before} == before


def test_sos_mif_supports_volume_sigma_and_spatial_effective_coils(tmp_path, adapter):
    amplitude = np.linspace(6., 20., np.prod(SHAPE)).reshape(SHAPE)
    sigma = np.broadcast_to(np.linspace(1., 2.5, SHAPE[-1]), SHAPE).copy()
    coils = np.linspace(1., 4., np.prod(SHAPE[:3])).reshape(SHAPE[:3])
    source = make_mif(tmp_path, forward_mean(amplitude, sigma, coils[..., None]))
    sigma_file = save_nifti(tmp_path / 'sigma_4d.nii.gz', sigma)
    coils_file = save_nifti(tmp_path / 'effective_coils.nii.gz', coils)
    before = {path: digest(path) for path in (source, sigma_file, coils_file)}
    output = tmp_path / 'map_corrected.mif'
    adapter(str(source), str(output), str(sigma_file),
            n_eff_map=str(coils_file), nthreads=1,
            sigma_convention='per-component-gaussian')
    result = export_mif(output, tmp_path / 'map_result.nii')
    np.testing.assert_allclose(result.get_fdata(), amplitude, atol=5e-4, rtol=5e-5)
    check_qc(output)
    assert {path: digest(path) for path in before} == before


@pytest.mark.parametrize('effective_coils', [1., 4.])
def test_sos_mif_joint_magnitude_profile_recovers_amplitude_and_component_sigma(tmp_path, adapter, effective_coils):
    # The supplied profile is magnitude dispersion, the convention intended
    # for MP-PCA. Gaussian component sigma is recovered rather than supplied.
    amplitude = np.linspace(5., 20., np.prod(SHAPE)).reshape(SHAPE)
    component_sigma = np.broadcast_to(np.linspace(1., 2.5, SHAPE[-1]), SHAPE).copy()
    magnitude_mean = forward_mean(amplitude, component_sigma, effective_coils)
    magnitude_sd = forward_magnitude_sd(amplitude, component_sigma, effective_coils)
    source = make_mif(tmp_path, magnitude_mean)
    sigma_file = save_nifti(tmp_path/'magnitude_sigma.nii.gz', magnitude_sd)
    before = {path: digest(path) for path in (source, sigma_file)}
    output = tmp_path/'joint_corrected.mif'
    summary = adapter(str(source), str(output), str(sigma_file), n_eff=effective_coils,
                      sigma_convention='magnitude', nthreads=1)
    result = export_mif(output, tmp_path/'joint_result.nii')
    np.testing.assert_allclose(result.get_fdata(), amplitude, atol=5e-4, rtol=5e-5)
    inferred = nib.load(Path(summary['qc_directory'])/'inferred_sigma_component.nii.gz')
    assert inferred.shape == SHAPE
    np.testing.assert_allclose(inferred.affine, AFFINE, atol=1e-6, rtol=0)
    np.testing.assert_allclose(inferred.get_fdata(), component_sigma, atol=5e-4, rtol=5e-5)
    assert summary['sigma_convention'] == 'magnitude'
    check_qc(output)
    assert {path: digest(path) for path in before} == before


def test_sos_mif_lossless_map_axis_flip_is_aligned_without_interpolation(tmp_path, adapter):
    amplitude = np.linspace(5., 20., np.prod(SHAPE)).reshape(SHAPE)
    sigma = np.linspace(1., 2., np.prod(SHAPE[:3])).reshape(SHAPE[:3])
    coils = np.linspace(1., 4., np.prod(SHAPE[:3])).reshape(SHAPE[:3])
    source = make_mif(tmp_path, forward_mean(amplitude, sigma[..., None], coils[..., None]))
    flip = np.eye(4)
    flip[0, 0] = -1
    flip[0, 3] = SHAPE[0] - 1
    flipped_affine = AFFINE @ flip
    sigma_file = save_nifti(tmp_path / 'sigma_flipped.nii.gz', sigma[::-1], flipped_affine)
    coils_file = save_nifti(tmp_path / 'coils_flipped.nii.gz', coils[::-1], flipped_affine)
    output = tmp_path / 'aligned.mif'
    adapter(str(source), str(output), str(sigma_file),
            n_eff_map=str(coils_file), nthreads=1,
            sigma_convention='per-component-gaussian')
    result = export_mif(output, tmp_path / 'aligned_result.nii')
    np.testing.assert_allclose(result.get_fdata(), amplitude, atol=5e-4, rtol=5e-5)
    np.testing.assert_allclose(result.affine, AFFINE, atol=1e-6, rtol=0)


@pytest.mark.parametrize('mismatch', ['sigma_translation', 'sigma_shape', 'coils_translation'])
def test_sos_mif_rejects_nonmatching_physical_grid(tmp_path, adapter, mismatch):
    source = make_mif(tmp_path, np.full(SHAPE, 20.))
    shifted = AFFINE.copy()
    shifted[0, 3] += .1  # Fractional voxel shift cannot be repaired by an axis flip.
    sigma_shape = (SHAPE[0]+1, *SHAPE[1:3]) if mismatch == 'sigma_shape' else SHAPE[:3]
    sigma_file = save_nifti(tmp_path / 'sigma.nii.gz', np.ones(sigma_shape),
                            shifted if mismatch == 'sigma_translation' else AFFINE)
    coils_file = save_nifti(tmp_path / 'coils.nii.gz', np.full(SHAPE[:3], 4.),
                            shifted if mismatch == 'coils_translation' else AFFINE)
    output = tmp_path / 'must_not_exist.mif'
    before = digest(source)
    with pytest.raises(ValueError):
        adapter(str(source), str(output), str(sigma_file),
                n_eff_map=str(coils_file), nthreads=1,
                sigma_convention='per-component-gaussian')
    assert not output.exists()
    assert digest(source) == before


def test_sos_mif_zero_noise_identity_and_negative_estimate_diagnostics(tmp_path, adapter):
    # Zero-noise identity is tested separately from signed denoising estimates.
    amplitude = np.linspace(0., 20., np.prod(SHAPE)).reshape(SHAPE).astype(np.float32)
    source = make_mif(tmp_path, amplitude)
    sigma_file = save_nifti(tmp_path / 'zero_sigma.nii.gz', np.zeros(SHAPE[:3]))
    output = tmp_path / 'identity.mif'
    adapter(str(source), str(output), str(sigma_file), n_eff=4., nthreads=1,
            sigma_convention='per-component-gaussian')
    result = export_mif(output, tmp_path / 'identity_result.nii')
    np.testing.assert_array_equal(result.get_fdata(), amplitude)
    _, masks = check_qc(output)
    assert masks['zero_noise_mask'].all()
    assert not masks['negative_estimate_mask'].any()

    estimates = np.full(SHAPE, 20., dtype=np.float32)
    estimates[0, 0, 0, 0] = -3.
    estimates[0, 0, 0, 1] = 0.
    signed_source = make_mif(tmp_path, estimates, name='signed_denoised_mean')
    positive_sigma = save_nifti(tmp_path / 'positive_sigma.nii.gz', np.full(SHAPE[:3], 2.))
    signed_before = digest(signed_source)
    signed_output = tmp_path / 'signed_corrected.mif'
    adapter(str(signed_source), str(signed_output), str(positive_sigma), n_eff=4., nthreads=1,
            sigma_convention='per-component-gaussian')
    corrected = export_mif(signed_output, tmp_path / 'signed_result.nii').get_fdata()
    _, masks = check_qc(signed_output)
    assert corrected[0, 0, 0, 0] == 0.  # Never reflect a negative estimate with abs().
    assert masks['negative_estimate_mask'][0, 0, 0, 0]
    assert int(masks['negative_estimate_mask'].sum()) == 1
    assert masks['boundary_mask'][0, 0, 0, 0]
    assert masks['boundary_mask'][0, 0, 0, 1]
    assert digest(signed_source) == signed_before


def test_sos_mif_rejects_repeat_inplace_and_existing_output(tmp_path, adapter):
    source = make_mif(tmp_path, np.full(SHAPE, 20.))
    sigma_file = save_nifti(tmp_path / 'sigma.nii.gz', np.ones(SHAPE[:3]))
    before = digest(source)
    with pytest.raises(ValueError):
        adapter(str(source), str(source), str(sigma_file), n_eff=4., nthreads=1, force=True,
                sigma_convention='per-component-gaussian')
    assert digest(source) == before
    output = tmp_path / 'corrected.mif'
    adapter(str(source), str(output), str(sigma_file), n_eff=4., nthreads=1,
            sigma_convention='per-component-gaussian')
    corrected_before = digest(output)
    with pytest.raises(ValueError):
        adapter(str(output), str(tmp_path/'twice.mif'), str(sigma_file), n_eff=4., nthreads=1,
                sigma_convention='per-component-gaussian')
    assert digest(output) == corrected_before
    with pytest.raises(FileExistsError):
        adapter(str(source), str(output), str(sigma_file), n_eff=4., nthreads=1,
                sigma_convention='per-component-gaussian')
    assert digest(output) == corrected_before


def run_cli(arguments, tmp_path):
    environment = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
        environment[key] = '1'
    # Reuse the established font cache across subprocesses. An isolated cache
    # per parameterized CLI case needlessly rebuilt the entire font inventory.
    environment['MPLCONFIGDIR'] = os.environ.get('MPLCONFIGDIR', '/private/tmp/pydke-mpl')
    return subprocess.run([sys.executable, '-m', 'pydesigner.main', *map(str, arguments)],
                          capture_output=True, text=True, env=environment, timeout=60)


@pytest.mark.parametrize('case,expected', [
    ('missing_sigma', 'sigma'), ('missing_coils', 'coil'),
    ('both_coils', 'coil'), ('invalid_coils', 'coil'),
    ('excessive_coils', 'coil'), ('zero_threads', 'thread'),
    ('wrong_convention', 'convention'), ('missing_domain', 'domain'),
    ('legacy_rician', 'rician'), ('resume', 'resume'),
    ('nonempty_output', 'output'),
])
def test_sos_cli_invalid_configuration_fails_before_conversion(tmp_path, cli_runtime, case, expected):
    # A nonexistent DWI ensures these errors are caught before parsing/conversion.
    source = tmp_path/'must_not_be_read.nii'
    output = tmp_path/'output'
    sigma = save_nifti(tmp_path/'sigma.nii.gz', np.ones(SHAPE[:3]))
    coils_map = save_nifti(tmp_path/'coils.nii.gz', np.full(SHAPE[:3], 4.))
    flags = ['--sos-noise-correct', '--sos-sigma', str(sigma),
             '--sos-effective-coils', '4',
             '--sos-sigma-convention', 'per-component-gaussian',
             '--sos-input-domain', 'magnitude-mean', '--nofit', '--noqc']
    if case == 'missing_sigma':
        start = flags.index('--sos-sigma'); del flags[start:start+2]
    elif case == 'missing_coils':
        start = flags.index('--sos-effective-coils'); del flags[start:start+2]
    elif case == 'both_coils':
        flags += ['--sos-effective-coils-map', str(coils_map)]
    elif case == 'invalid_coils':
        flags[flags.index('--sos-effective-coils')+1] = '-1'
    elif case == 'excessive_coils':
        flags[flags.index('--sos-effective-coils')+1] = '4097'
    elif case == 'zero_threads':
        flags += ['--nthreads', '0']
    elif case == 'wrong_convention':
        flags[flags.index('--sos-sigma-convention')+1] = 'magnitude-standard-deviation'
    elif case == 'missing_domain':
        start = flags.index('--sos-input-domain'); del flags[start:start+2]
    elif case == 'legacy_rician':
        flags += ['--rician']
    elif case == 'resume':
        flags += ['--resume']
    sentinel = None
    if case in ('nonempty_output', 'resume'):
        output.mkdir()
        sentinel = output/'keep.txt'
        sentinel.write_text('Untouched prior work.\n')
    result = run_cli([source, '-o', output, *flags], tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'unrecognized arguments' not in result.stderr
    assert expected in result.stderr.lower(), result.stderr
    assert not (output/'working.mif').exists()
    assert not (output/'dwi_raw.nii').exists()
    if sentinel:
        assert sentinel.read_text() == 'Untouched prior work.\n'


def test_sos_cli_external_magnitude_mean_does_not_run_denoising(tmp_path, cli_runtime):
    # Below the central-chi mean: boundary output proves the supplied sigma is
    # used. No thermal denoising is requested or implicitly enabled.
    source = save_nifti(tmp_path/'external_mean.nii', np.full(SHAPE, 3.))
    np.savetxt(tmp_path/'external_mean.bval', [[0, 1000, 1000, 1000, 1000, 1000]])
    np.savetxt(tmp_path/'external_mean.bvec',
               [[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0]])
    (tmp_path/'external_mean.json').write_text(json.dumps(dict(
        PhaseEncodingDirection='j-', PartialFourier=1, EchoTime=.034,
        TotalReadoutTime=.02)))
    sigma = save_nifti(tmp_path/'explicit_gaussian_sigma.nii.gz', np.full(SHAPE[:3], 2.))
    before = {path: digest(path) for path in (source, sigma)}
    output = tmp_path/'output'
    result = run_cli([source, '-o', output, '--sos-noise-correct',
        '--sos-sigma', sigma, '--sos-effective-coils', '4',
        '--sos-sigma-convention', 'per-component-gaussian',
        '--sos-input-domain', 'magnitude-mean', '--nthreads', '1',
        '--nofit', '--noqc'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    preprocessed = nib.load(output/'dwi_preprocessed.nii')
    np.testing.assert_array_equal(preprocessed.get_fdata(), np.zeros(SHAPE))
    np.testing.assert_allclose(preprocessed.affine, AFFINE, atol=1e-6, rtol=0)
    assert not (output/'metrics').exists()
    assert not (output/'denoising.json').exists()
    assert not (output/'noisemap.nii').exists()
    assert not list((output/'intermediate_nifti').glob('*denoised*'))
    qc_directories = list(output.glob('*_sos_qc'))
    assert len(qc_directories) == 1
    qc = qc_directories[0]
    assert json.loads((qc/'summary.json').read_text())
    boundary = nib.load(qc/'boundary_mask.nii.gz')
    assert boundary.shape == SHAPE
    assert np.asarray(boundary.dataobj).all()
    assert 'pydke_sos_noise_correction' in json.loads((output/'dwi_preprocessed.json').read_text())
    record = json.loads((output/'sos_noise.json').read_text())
    assert record['sigma_convention'] == 'per-component-gaussian'
    assert_retained_cli_bias_output(output, record)
    assert {path: digest(path) for path in before} == before


def make_cli_input(tmp_path, data, name):
    source = save_nifti(tmp_path/(name+'.nii'), data)
    np.savetxt(tmp_path/(name+'.bval'), [[0, 1000, 1000, 1000, 1000, 1000]])
    np.savetxt(tmp_path/(name+'.bvec'),
               [[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0]])
    (tmp_path/(name+'.json')).write_text(json.dumps(dict(
        PhaseEncodingDirection='j-', PartialFourier=1, EchoTime=.034,
        TotalReadoutTime=.02)))
    return source


def test_sos_cli_defaults_to_joint_magnitude_noise_profile(tmp_path, cli_runtime):
    amplitude, component_sigma, coils = 12., 2., 4.
    mean = float(forward_mean(amplitude, component_sigma, coils))
    magnitude_sd = float(forward_magnitude_sd(amplitude, component_sigma, coils))
    source = make_cli_input(tmp_path, np.full(SHAPE, mean), 'known_magnitude_mean')
    sigma_file = save_nifti(tmp_path/'known_magnitude_sd.nii.gz',
                            np.full(SHAPE[:3], magnitude_sd))
    before = {path: digest(path) for path in (source, sigma_file)}
    output = tmp_path/'joint_output'
    # No convention flag: the intended CLI default consumes magnitude SD.
    result = run_cli([source, '-o', output, '--sos-noise-correct',
        '--sos-sigma', sigma_file, '--sos-effective-coils', str(coils),
        '--sos-input-domain', 'magnitude-mean', '--nthreads', '1',
        '--nofit', '--noqc'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = nib.load(output/'dwi_preprocessed.nii').get_fdata()
    np.testing.assert_allclose(actual, amplitude, atol=5e-4, rtol=5e-5)
    record = json.loads((output/'sos_noise.json').read_text())
    assert record['sigma_convention'] == 'magnitude'
    assert_retained_cli_bias_output(output, record)
    np.testing.assert_allclose(nib.load(Path(record['qc_directory'])/
        'inferred_sigma_component.nii.gz').get_fdata(), component_sigma,
        atol=5e-4, rtol=5e-5)
    assert not (output/'denoising.json').exists()
    assert not (output/'noisemap.nii').exists()
    assert not (output/'metrics').exists()
    assert {path: digest(path) for path in before} == before


@pytest.mark.parametrize('backend', ['mppca', 'adaptive'])
def test_sos_cli_uses_selected_mppca_profile_without_changing_denoiser(tmp_path, cli_runtime, backend):
    # A deterministic raw four-coil SoS phantom supplies magnitude samples;
    # the explicitly selected existing denoiser estimates their magnitude mean
    # and profile. Bias correction must consume that generated profile without
    # changing either MP-PCA implementation or silently selecting another one.
    rng = np.random.default_rng(90210)
    real = rng.normal(0., 2., (*SHAPE, 4))
    imaginary = rng.normal(0., 2., (*SHAPE, 4))
    real[..., 0] += np.linspace(18., 24., np.prod(SHAPE)).reshape(SHAPE)
    observed = np.sqrt(np.sum(real**2 + imaginary**2, axis=-1))
    source = make_cli_input(tmp_path, observed, 'raw_sos')
    source_before = digest(source)
    backend_flags = ['--denoise-method', backend, '--extent', '3,3,3']
    if backend == 'mppca':
        backend_flags += ['--denoise']
    else:
        backend_flags += ['--denoise-patch-size', '20']
    baseline = tmp_path/('baseline_'+backend)
    corrected = tmp_path/('corrected_'+backend)
    common = ['--nthreads', '1', '--nofit', '--noqc', *backend_flags]
    baseline_result = run_cli([source, '-o', baseline, *common], tmp_path)
    assert baseline_result.returncode == 0, baseline_result.stdout + baseline_result.stderr
    corrected_result = run_cli([source, '-o', corrected, *common,
        '--sos-noise-correct', '--sos-effective-coils', '4',
        '--sos-input-domain', 'magnitude-mean'], tmp_path)
    assert corrected_result.returncode == 0, corrected_result.stdout + corrected_result.stderr
    for output in (baseline, corrected):
        assert json.loads((output/'denoising.json').read_text())['method'] == backend
    assert json.loads((baseline/'denoising.json').read_text()) == json.loads((corrected/'denoising.json').read_text())
    record = json.loads((corrected/'sos_noise.json').read_text())
    assert record['sigma_convention'] == 'magnitude'
    assert_retained_cli_bias_output(corrected, record)
    assert Path(record['sigma_path']).resolve() == (corrected/'noisemap.nii').resolve()
    assert record['sigma_sha256'] == digest(corrected/'noisemap.nii')
    np.testing.assert_array_equal(nib.load(baseline/'noisemap.nii').get_fdata(),
                                  nib.load(corrected/'noisemap.nii').get_fdata())
    original_denoised = list((baseline/'intermediate_nifti').glob('*_dwi_denoised.nii'))
    new_denoised = list((corrected/'intermediate_nifti').glob('*_dwi_denoised.nii'))
    assert len(original_denoised) == len(new_denoised) == 1
    np.testing.assert_array_equal(nib.load(original_denoised[0]).get_fdata(),
                                  nib.load(new_denoised[0]).get_fdata())
    assert np.isfinite(nib.load(corrected/'dwi_preprocessed.nii').get_fdata()).all()
    assert (Path(record['qc_directory'])/'inferred_sigma_component.nii.gz').is_file()
    assert not (corrected/'metrics').exists()
    assert digest(source) == source_before
