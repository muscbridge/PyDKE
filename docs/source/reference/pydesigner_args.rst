List of Flags
=============

pyDKE is extremely flexible when it comes to dMRI processing.
Users can easily enable or disable various preprocessing steps without
changing the overall sequence.

The list below covers all these flags.

IO Control
---------------

These flags allow control of the pipeline's I/O handling

-o DIR, --output DIR    pyDKE output directory


Preprocessing Control
---------------------

Preprocessing contol flags allow users to tweak certain parts of the
preprocessing pipeline, to accomodate all types of datasets.


-s, --standard      Runs the recommended preprocessing pipeline in order: denoise, degibbs, undistort, brain mask, smooth, rician

-n, --denoise       Denoises input DWI

--extent            Shape of denoising extent matrix, defaults to 5,5,5

--reslice xyz       Reslices DWI to voxel resolution specified in miliimeters (mm) or output dimensions. Performinh reslicing will skip plotting of SNR curves. Dimensions less than 9 will reslice in mm, image dimension otherwise. Specify argument as `--reslice x,y,z`.

--interp CHOICE     Set the interpolation to use when reslicing. Choices are linear (default), nearest, cubic, and sinc

-g, --degibbs       Corrects Gibbs ringing

-u, --undistort     Undistorts image using a suite of EPI distortion correction, eddy current correction, and co-registration. Does not run EPI correction if reverse phase encoding DWI is absent.

--rpe_pairs N       Speeds up TOPUP if a reverse PE is present; specify the number (integer) of reverse PE direction B0 pairs to use.

--mask              Computes a brain mask at 0.20 threshold by default

--maskthr           Specify FSL bet fractional intensity threshold for brain masking, defaults to 0.20

--user_mask         Provide path to user-generated brain mask in NifTi (.nii) format

-cf, --csf_fsl      Compute a CSF mask for CSF-excluded smoothing to minimize partial volume effects using FSL fit_constraints

-cd, --csf_adc      Compute CSF mask for CSF-excluded smoothing to minimize partial volume effects by thresholding a pseudo-ADC map computed as ln(S0/S1000)/b1000. N is the ADC thresold to use (default: 2).

-z, --smooth        Smooths DWI data at a default FWHM of 1.25

--fwhm              Specify the FWHM at which to smooth, defaults to 1.25

-r, --rician        Corrects Rician bias

-te, --multite      Enable multi-TE support. This mode preprocesses all concatenated DWIs together, but performs tensor fitting separately.


Diffusion Tensor Control
------------------------

Users may also tweak computations in estimating DTI or DKI parameters
with the following flags.

--nofit             Performs preprocessing only, disables DTI/DKI parameter extraction

--noakc             Performs brute forced kurtosis tensor outlier rejection

--nooutliers        Disables IRLLS outlier detection

--fit_constraints   Specify fitting constraints to use, defaults to 0,1,0

--noqc              Disables saving of quality control (QC) metrics

--median            Performs post processing median filter of final DTI/DKI maps. **WARNING: Use on a case-by-case basis for bad data only. When applied, the filter alters the values of most voxels, so it should be used with caution and avoided when data quality is otherwise adequate. While maps appear visually soother with this flag on, they may nonetheless be less accurate**

Fiber Ball Imaging (FBI) Control
--------------------------------

FBI parameters may be fine-tuned with the following flags.

--l_max n     Maximum spherical harmonic degree used in spherical harmonic expansion for fODF calculation

--no_rectify  Disable rectification of FBI fODFs in instances where it does more harm than good. In rare instances, fODFs computed from FBI acquisitions can be degraded from rectification - this flag disables rectification for such datasets.

Pipeline Control
----------------

These are more general pipeline flags that interface directly with the
user or machine.

--nthreads n    Specify number of CPU workers to use in processing, defaults to all physically available workers

--resume        Resumes preprocessing from an aborted or partial previous run

--force         Forces overwrite of existing output files

--verbose       Displays console output

--adv           Diables safety check to force run certain preprocessing steps **WARNING: This flag is for advanced users only who fully understand the MRI system and its outputs. Running with this flag could potentially yield inaccuracies in resulting DTI/DKI metrics**


Optional SoS magnitude-bias correction
------------------------------------

``--sos-noise-correct`` selects a noncentral-chi noise-bias model instead of the
separate legacy ``--rician`` stage. It neither enables nor changes denoising.
Declare ``--sos-input-domain magnitude-mean``: the input to this bias stage must
estimate the magnitude mean, not a raw squared-magnitude moment or an image
already corrected for its noise floor.

``--sos-sigma-convention magnitude`` is the default when this option is enabled.
It uses generalized Koay joint mean/variance inversion to infer signal amplitude
and effective Gaussian component noise from the magnitude mean and original
magnitude-noise standard deviation. The MP-PCA noise map generated by explicitly
selected denoising can supply this profile directly; an independent Gaussian
calibration file is not required. This model-based inference does not identify
MP-PCA magnitude dispersion with Gaussian component dispersion.

The flags are:

* ``--sos-effective-coils L`` supplies an explicit effective coil parameter in
  the supported numerical range **1 to 4096**, including fractional values.
* ``--sos-effective-coils-map PATH`` supplies its spatial 3D map instead. Choose
  exactly one of the scalar and map alternatives. Effective L is not inferred
  from physical receiver count; retain its provenance or reference assumption.
* ``--sos-sigma PATH`` supplies a 3D shared or matching 4D noise-profile NIfTI.
  In magnitude mode, omitting this flag requires explicitly selected denoising
  to generate the MP-PCA profile. A supplied mean/profile pair does not cause
  denoising to run again.
* ``--sos-sigma-convention per-component-gaussian`` instead inverts the first
  moment using an explicitly supplied Gaussian standard deviation per real or
  imaginary component. This convention requires ``--sos-sigma``; a generated
  magnitude MP-PCA map is not silently reinterpreted as Gaussian sigma.

For example, keep the existing denoising choice and use its magnitude profile::

    pydke input.nii.gz -o fresh_output --denoise --sos-noise-correct \
      --sos-effective-coils 4 --sos-input-domain magnitude-mean \
      --sos-sigma-convention magnitude --nthreads 1

Here 4 is an explicitly supplied example, not a default or a hardware inference.
Alternatively, use an existing magnitude-mean estimate and magnitude-noise map
without requesting denoising::

    pydke denoised_mean.nii.gz -o fresh_output --sos-noise-correct \
      --sos-effective-coils 4 --sos-input-domain magnitude-mean \
      --sos-sigma magnitude_noise.nii.gz --sos-sigma-convention magnitude \
      --nthreads 1

The native-grid bias stage runs after explicitly requested denoising and before
later spatial operations. With ``--standard``, it replaces the legacy Rician
bias stage while retaining existing denoising choices. Use fresh output:
``--resume``, nonempty output reuse and simultaneous explicit ``--rician`` are
rejected. Parameter maps must occupy the same physical grid; the adapter permits
only verified lossless signed axis permutations, with no registration or
interpolation.

QC records central-noise boundary, negative-estimate, zero-noise and
ill-conditioned results, together with configuration/profile provenance. The
magnitude mode also exports inferred Gaussian component sigma as a model-based
quantity. No absolute-value reflection or positive signal floor is applied.
A shared MP-PCA noise profile and a denoised mean remain estimates; successful
inversion is not proof of exact bias removal, calibration or biological accuracy.
The SoS noise model is unrelated to the ``global-sos`` tensor-fitting backend.
