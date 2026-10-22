import numpy as np
import pytest
from pydesigner.preprocessing.rpg import degibbs_array


def image():
    x,y=np.mgrid[:32,:32]
    step=((x>8)&(x<24)&(y>7)&(y<25)).astype(float)
    spectrum=np.fft.fftshift(np.fft.fft2(step))
    spectrum[:8]=0; spectrum[24:]=0
    spectrum[:,:8]=0; spectrum[:,24:]=0
    ringing=np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    return np.repeat(ringing[:,:,None,None],2,axis=2)


@pytest.mark.parametrize('pf',[0.625,0.75,0.875,1.0])
def test_native_rpg_threads_and_geometry(pf):
    pytest.importorskip('pydesigner.preprocessing._rpg')
    data=image(); original=data.copy()
    a=degibbs_array(data,pf,1,2,1)
    b=degibbs_array(data,pf,1,2,2)
    np.testing.assert_array_equal(data,original)
    np.testing.assert_allclose(a,b,atol=1e-6)
    assert a.shape==data.shape and np.isfinite(a).all()
    assert not np.allclose(a,data)
    # Same image reoriented to another voxel plane must give the same correction.
    c=degibbs_array(data.transpose(2,0,1,3),pf,2,0,1).transpose(1,2,0,3)
    np.testing.assert_allclose(a,c,atol=1e-6)


def test_constant():
    pytest.importorskip('pydesigner.preprocessing._rpg')
    for value in (0.,100.):
        np.testing.assert_allclose(degibbs_array(np.full((32,32,1,1),value),0.75,1,2),value,atol=1e-5)


@pytest.mark.parametrize('pf,pe,sl,threads',[(0.66,1,2,1),(0.75,2,2,1),(0.75,1,2,0)])
def test_bad_acquisition(pf,pe,sl,threads):
    with pytest.raises(ValueError):
        degibbs_array(image(),pf,pe,sl,threads)


def test_mif_metadata(tmp_path):
    import shutil
    import subprocess
    import nibabel as nib
    from pydesigner.preprocessing.mrpreproc import degibbs
    pytest.importorskip('pydesigner.preprocessing._rpg')
    if not shutil.which('mrconvert'):
        pytest.skip('MRtrix required')
    source=tmp_path/'input.nii'; mif=tmp_path/'input.mif'; output=tmp_path/'output.mif'
    data=np.repeat(image(),6,axis=3)
    affine=np.diag([.2,.2,1.,1.])
    nib.save(nib.Nifti1Image(data,affine),source)
    grad=tmp_path/'grad.txt'; np.savetxt(grad,[[0,0,0,0]]+[[1,0,0,1000]]*5)
    subprocess.run(['mrconvert',str(source),str(mif),'-grad',str(grad),'-set_property','PhaseEncodingDirection','j-','-quiet'],check=True)
    degibbs(str(mif),str(output),method='rpg',pf=.75,pe_axis=1,slice_axis=2,nthreads=1)
    result=tmp_path/'result.nii'
    subprocess.run(['mrconvert',str(output),str(result),'-quiet'],check=True)
    np.testing.assert_allclose(nib.load(result).affine,affine)
    np.testing.assert_allclose(nib.load(result).get_fdata(),degibbs_array(data,.75,1,2),atol=1e-6)
    for flag,extra in [('-dwgrad',[]),('-property',['PhaseEncodingDirection'])]:
        assert subprocess.check_output(['mrinfo',str(mif),flag,*extra]) == subprocess.check_output(['mrinfo',str(output),flag,*extra])
