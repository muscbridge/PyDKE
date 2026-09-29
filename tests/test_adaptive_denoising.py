"""Small synthetic tests: no participant data or large worker pools."""
import shutil
import subprocess
import numpy as np
import nibabel as nib
import pytest
from pydesigner.preprocessing.adaptive import denoise_array, denoise_mif
from pydesigner.preprocessing import mrpreproc


def phantom():
    rng = np.random.default_rng(14)
    return (100 + np.arange(6)[None, None, None, :] * 10 + rng.normal(0, 3, (4, 4, 4, 6))).astype('float32')


def test_deterministic_across_threads():
    first = denoise_array(phantom(), '3,3,3', 20, 1)
    second = denoise_array(phantom(), '3,3,3', 20, 2)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
    assert np.all(first[1] >= 0)
    assert np.mean((first[0] - (100 + np.arange(6)*10))**2) < np.mean((phantom() - (100 + np.arange(6)*10))**2)


@pytest.mark.parametrize('value', [0, 100])
def test_constant_patch(value):
    signal, sigma, rank = denoise_array(np.full((3,3,3,6), value), '3,3,3', 20)
    np.testing.assert_array_equal(signal, value)
    np.testing.assert_array_equal(sigma, 0)
    np.testing.assert_array_equal(rank, 0)


@pytest.mark.parametrize('kwargs', [{'extent':'4,3,3'}, {'patch_size':6}, {'patch_size':27}, {'nthreads':0}])
def test_invalid_options(kwargs):
    options = dict(extent='3,3,3', patch_size=20, nthreads=1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        denoise_array(phantom(), **options)


def test_nonfinite():
    data=phantom(); data[0,0,0,0]=np.nan
    with pytest.raises(ValueError, match='finite'):
        denoise_array(data)


def test_original_backend_unchanged(tmp_path, monkeypatch):
    source=tmp_path/'in.mif'; source.touch()
    calls=[]
    class Success:
        returncode=0
    monkeypatch.setattr(mrpreproc.subprocess, 'run', lambda args: (calls.append(args), Success())[1])
    mrpreproc.denoise(str(source), str(tmp_path/'out.mif'), nthreads=2)
    assert calls == [['dwidenoise','-force','-quiet','-nthreads','2','-noise',str(tmp_path/'noisemap.nii'),'-extent','5,5,5',str(source),str(tmp_path/'out.mif')]]


@pytest.mark.skipif(not shutil.which('mrconvert'), reason='MRtrix not installed')
def test_mif_metadata_roundtrip(tmp_path):
    raw=tmp_path/'raw.nii'; source=tmp_path/'in.mif'; out=tmp_path/'out.mif'
    affine=np.diag([2.,2.,2.,1.])
    nib.save(nib.Nifti1Image(phantom(), affine),raw)
    grad=tmp_path/'grad.txt'
    np.savetxt(grad, [[0,0,0,0]]+[[1,0,0,1000]]*5)
    subprocess.run(['mrconvert',str(raw),str(source),'-grad',str(grad),'-set_property','PhaseEncodingDirection','j-','-quiet'],check=True)
    denoise_mif(str(source),str(out),True,'3,3,3',2,False,False,20)
    result=tmp_path/'result.nii'
    subprocess.run(['mrconvert',str(out),str(result),'-quiet'],check=True)
    np.testing.assert_allclose(nib.load(result).affine,affine)
    np.testing.assert_allclose(nib.load(result).get_fdata(),denoise_array(phantom(),'3,3,3',20,1)[0])
    assert nib.load(tmp_path/'noisemap.nii').shape == (4,4,4)
    for flag in ['-dwgrad','-property']:
        extra=['PhaseEncodingDirection'] if flag=='-property' else []
        a=subprocess.check_output(['mrinfo',str(source),flag,*extra])
        b=subprocess.check_output(['mrinfo',str(out),flag,*extra])
        assert a == b
    with pytest.raises(FileExistsError):
        denoise_mif(str(source),str(out),True,'3,3,3',1,False,False,20)


def test_pinned_designer_reference():
    import json
    from pathlib import Path
    reference=json.loads((Path(__file__).parent/'data/designer_adaptive_reference.json').read_text())
    data=np.random.default_rng(reference['seed']).normal(100,5,(4,4,4,6)).astype('float32')
    signal,sigma,rank=denoise_array(data,'3,3,3',20,1)
    for item in reference['results']:
        c=tuple(item['coord'])
        np.testing.assert_allclose(signal[c],item['signal'],rtol=2e-6)
        np.testing.assert_allclose(sigma[c],item['sigma'],rtol=2e-6)
        assert rank[c] == item['rank']


def test_resume_method_and_parameters(tmp_path):
    import json
    from pydesigner.preprocessing.denoise_config import check_resume
    record=tmp_path/'denoising.json'
    original=dict(method='mppca',extent='5,5,5',patch_size=None)
    adaptive=dict(method='adaptive',extent='7,7,7',patch_size=274)
    check_resume(record, original)
    with pytest.raises(ValueError):
        check_resume(record, adaptive)
    record.write_text(json.dumps(adaptive))
    check_resume(record,adaptive)
    for changed in [original,dict(adaptive,extent='9,9,9'),dict(adaptive,patch_size=275)]:
        with pytest.raises(ValueError):
            check_resume(record,changed)
