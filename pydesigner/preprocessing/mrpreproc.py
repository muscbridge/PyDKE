#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""Utilities for running various MRtrix3's DWI preprocessing tools"""

import os
import os.path as op
import subprocess
from shutil import copyfile, which

import numpy as np

from ..preprocessing import mrinfoutil, rician, smoothing
from ..system.errors import MRTrixError
from ..system.models import input_path_validator, modelmrtrix, output_path_validator


def miftonii(
    input: str,
    output: str,
    nthreads: int = None,
    force: bool = True,
    verbose: bool = False,
) -> None:
    """Converts input `.mif` images to output `.nii` images

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .nii file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file

    See Also
    --------
    niitomif
    """
    opts = modelmrtrix(
        input=input_path_validator(input, ".mif"),
        output=output_path_validator(output, ".nii"),
        nthreads=nthreads,
        force=force,
        verbose=verbose,
    )
    input_path_validator(opts.input, ".mif")
    output_path_validator(opts.output, ".nii")
    arg = ["mrconvert"]
    if opts.force:
        arg.append("-force")
    if not opts.verbose:
        arg.append("-quiet")
    if opts.nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    arg.extend(
        [
            "-export_grad_fsl",
            op.splitext(opts.output)[0] + ".bvec",
            op.splitext(opts.output)[0] + ".bval",
        ]
    )
    arg.extend(["-json_export", op.splitext(output)[0] + ".json"])
    arg.extend([opts.input, opts.output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        msg = f"Conversion from .mif to .nii failed. Return code: {completion.returncode}"
        msg += f"\nCommand: {' '.join(arg)}"
        msg += f"\nMRtrix3 error: {completion.stderr}"
        raise MRTrixError(msg)


def niitomif(
    input: str,
    output: str,
    nthreads: int = None,
    force: bool = True,
    verbose: bool = False,
) -> None:
    """Converts input `.nii` images to output `.mif` images provided that
    all BVEC, BVAL and JSON files are provided and named same as input .nii

    Parameters
    ----------
    input : str
        Path to input .nii file
    output : str
        Path to output .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file

    See Also
    --------
    miftonii
    """
    opts = modelmrtrix(
        input=input_path_validator(input, ".nii"),
        output=output_path_validator(output, ".mif"),
        nthreads=nthreads,
        force=force,
        verbose=verbose,
    )
    input_path_validator(opts.input, ".nii")
    path_bvec = input_path_validator(op.splitext(opts.input)[0] + ".bvec", ".bvec")
    path_bval = input_path_validator(op.splitext(opts.input)[0] + ".bval", ".bval")
    path_json = input_path_validator(op.splitext(opts.input)[0] + ".json", ".json")
    output_path_validator(opts.output, ".mif")
    arg = ["mrconvert"]
    if opts.force:
        arg.append("-force")
    if not opts.verbose:
        arg.append("-quiet")
    if opts.nthreads is not None:
        arg.extend(["-nthreads", str(opts.nthreads)])
    arg.extend(["-fslgrad", path_bvec, path_bval])
    arg.extend(["-json_import", path_json])
    arg.extend([opts.input, opts.output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        msg = f"Conversion from .nii to .mif failed. Return code: {completion.returncode}"
        msg += f"\nCommand: {' '.join(arg)}"
        msg += f"\nMRtrix3 error: {completion.stderr}"
        raise MRTrixError(msg)


def stride_match(
    target: str,
    moving: str,
    output: str,
    nthreads: int = None,
    force: bool = True,
    verbose: bool = False,
) -> None:
    """Matches strides on inputs target and moving by converting strides
    on moving image to those of target image.

    Parameters
    ----------
    target : str
        Path to target image .nii or .mif file
    moving : str
        Path to moving image .nii or .mif file
    output : str
        Path to output .nii or .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    opts = modelmrtrix(output=output, nthreads=nthreads, force=force, verbose=verbose)
    target = input_path_validator(target)
    moving = input_path_validator(moving)

    arg = ["mrconvert"]
    if opts.force:
        arg.append("-force")
    if not opts.verbose:
        arg.append("-quiet")
    if opts.nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    arg.extend(["-strides", target, moving, opts.output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        msg = f"Stride matching failed. Return code: {completion.returncode}"
        msg += f"\nCommand: {' '.join(arg)}"
        msg += f"\nMRtrix3 error: {completion.stderr}"
        raise MRTrixError(msg)


def denoise(
    input: str,
    output: str,
    noisemap: bool = True,
    extent: str = "5,5,5",
    nthreads: int = None,
    force: bool = True,
    verbose: bool = False,
    method: str = "mppca",
    patch_size: int = None,
) -> None:
    """Run original MRtrix MP-PCA or DESIGNER-v2 adaptive MP-PCA.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    noisemap : bool, optional
        Specify whether or not to save the noisemap as a
        nifti file (Default: True)
    extent : str, optional
        Set the window size of the denoising filter.
        (Default: '5,5,5')
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    method : str, optional
        "mppca" (default) or "adaptive" (DESIGNER patches and shrinkage).
    patch_size : int, optional
        Adaptive voxel count; default floor(0.8 * product(extent)).
        Must exceed volume count and be smaller than the search window.

    Returns
    -------
    None; writes out file
    """
    if method not in ("mppca", "adaptive"):
        raise ValueError("Denoising method must be mppca or adaptive")
    if method == "adaptive":
        from .adaptive import denoise_mif
        return denoise_mif(input, output, noisemap, extent, nthreads, force, verbose, patch_size)
    if patch_size is not None:
        raise ValueError("patch_size requires adaptive denoising")
    opts = modelmrtrix(
        input=input_path_validator(input, ".mif"),
        output=output_path_validator(output, ".mif"),
        nthreads=nthreads,
        force=force,
        verbose=verbose,
    )
    if not isinstance(noisemap, bool):
        raise TypeError("Please specify whether noisemap generation is True or False.")
    noisemap_path = op.join(op.dirname(opts.output), "noisemap.nii")
    arg = ["dwidenoise"]
    if opts.force:
        arg.append("-force")
    if not opts.verbose:
        arg.append("-quiet")
    if opts.nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    if noisemap:
        arg.extend(["-noise", noisemap_path])
    if extent is not None:
        arg.extend(["-extent", extent])
    arg.extend([opts.input, opts.output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        msg = f"Dwidenoise failed. Return code: {completion.returncode}"
        msg += f"\nCommand: {' '.join(arg)}"
        msg += f"\nMRtrix3 error: {completion.stderr}"
        raise MRTrixError(msg)


def degibbs(
    input: str,
    output: str,
    nthreads: int = None,
    force: bool = False,
    verbose: bool = False,
    method: str = "mrtrix",
    pf: float = None,
    pe_axis: int = None,
    slice_axis: int = None,
) -> None:
    """Runs MRtrix3's `mrdegibbs` command with optimal parameters for
    PyDesigner.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if method == "rpg":
        from .rpg import degibbs_mif
        return degibbs_mif(input, output, pf, pe_axis, slice_axis, nthreads, force, verbose)
    if method != "mrtrix":
        raise ValueError("Gibbs method must be mrtrix or rpg")
    opts = modelmrtrix(
        input=input_path_validator(input, ".mif"),
        output=output_path_validator(output, ".mif"),
        nthreads=nthreads,
        force=force,
        verbose=verbose,
    )
    arg = ["mrdegibbs"]
    if opts.force:
        arg.append("-force")
    if not opts.verbose:
        arg.append("-quiet")
    if opts.nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    arg.extend([opts.input, opts.output])
    completion = subprocess.run(arg, capture_output=True, text=True)
    if completion.returncode != 0:
        msg = f"Mrdegibbs failed. Return code: {completion.returncode}"
        msg += f"\nCommand: {' '.join(arg)}"
        msg += f"\nMRtrix3 error: {completion.stderr}"
        raise MRTrixError(msg)


def undistort(
    input,
    output,
    rpe="rpe_header",
    epib0=1,
    qc=None,
    nthreads=None,
    force=False,
    verbose=False,
):
    """Runs MRtrix3's distortion correction command with optimal
    parameters for PyDesigner.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    rpe : str, {'rpe_header', 'rpe-pair', 'rpe_none, 'rpe_all'}, optional
        Reverse phase encoding of the dataset. (Default: 'rpe_header')
    epib0 : int
        Number of reverse PE dir B0 pairs to use in TOPUP correction
        (Default: 1)
    qc : str
        Specify path to QC directior. No QC metrics generated if None
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    opts = modelmrtrix(
        input=input_path_validator(input, ".mif"),
        output=output_path_validator(output, ".mif"),
        nthreads=nthreads,
        force=force,
        verbose=verbose,
    )

    if rpe not in ["rpe_none", "rpe_pair", "rpe_all", "rpe_header"]:
        msg = "Entered RPE selection is not valid. Please choose either "
        msg += "'rpe_none', 'rpe_pair' 'rpe_all', or 'rpe_header'"
        raise ValueError(msg)
    if not isinstance(epib0, int):
        msg = "Number of TOPUP B0s need to be specified as a positive integer."
        raise ValueError(msg)
    if qc is not None:
        if not isinstance(qc, str):
            msg = "Please specify QC directory as a string"
            raise TypeError(msg)
        if not op.exists(qc):
            msg = "Specified QC directory does not exist. Please ensure that this is a valid directory."
            raise OSError(msg)
    rpe = "-" + rpe
    # Get output directory
    outdir = op.dirname(opts.output)
    # Extract BVEC and BVALS for shell sampling deduction
    arg_extract = ["mrinfo"]
    arg_extract.extend(
        [
            "-export_grad_fsl",
            op.join(outdir, "dwiec.bvec"),
            op.join(outdir, "dwiec.bval"),
        ]
    )
    arg_extract.append(opts.input)
    completion = subprocess.run(arg_extract)
    if completion.returncode != 0:
        msg = "Extraction of FSL BVEC and BVAL gradients failed. Return code: {completion.returncode}"
        msg += f"\nCommand: {' '.join(arg_extract)}"
        msg += f"\nMRtrix3 error: {completion.stderr}"
        raise MRTrixError(msg)
    # Form main undistortion argument
    arg = []
    if which("dwipreproc") is None:
        arg.append("dwifslpreproc")
    else:
        arg.append("dwipreproc")
    if opts.force:
        arg.append("-force")
    if not opts.verbose:
        arg.append("-quiet")
    if opts.nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    # Determine whether half or full sphere sampling
    repol_string = "--repol "
    if mrinfoutil.is_fullsphere(input):
        # is full, add appropriate dwifslpreproc option
        repol_string += "--data_is_shelled"
    else:
        # half
        repol_string += "--slm=linear"
    if epib0 > 0:
        try:
            epi_path = op.join(outdir, "B0_EPI.mif")
            epiboost(
                input=input,
                output=epi_path,
                num=epib0,
                nthreads=nthreads,
                force=force,
                verbose=verbose,
            )
            arg.extend(["-se_epi", epi_path])
        except:  # noqa: E722
            print("[WARNING] Unable to apply TOPUPBOOST because DWI consists of single PE direction.")
            # Remove the B0_ALL.mif file that is created when epiboost
            # function fails
            try:
                os.remove(op.join(outdir, "B0_ALL.mif"))
            except OSError:
                pass
    arg.extend(["-eddy_options", repol_string])
    arg.append(rpe)
    if qc is not None:
        arg.extend(["-eddyqc_all", qc])
    arg.extend([opts.input, opts.output])
    completion = subprocess.run(arg, capture_output=True, text=True)

    if completion.returncode != 0:
        msg = f"Dwifslpreproc failed. Return code: {completion.returncode}"
        msg += f"\nCommand: {' '.join(arg)}"
        msg += f"\nMRtrix3 stdout:\n{completion.stdout}"
        msg += f"\nMRtrix3 stderr:\n{completion.stderr}"
        raise MRTrixError(msg)
    
    # Remove temporarily generated files
    os.remove(op.join(outdir, "dwiec.bvec"))
    os.remove(op.join(outdir, "dwiec.bval"))
    if epib0 > 0:
        try:
            os.remove(epi_path)
        except:  # noqa: E722
            print("[Warning] unable to remove {} because it does not exist".format(epi_path))


def brainmask(input, output, thresh=0.25, nthreads=None, force=False, verbose=False):
    """Creates a brainmask using FSL's Brain Extraction Tool (BET) and
    MRtrix3's file manipulation tools.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .nii brainmask file
    thresh : float
        BET threshold ranging from 0 to 1 (Default: 0.25)
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    opts = modelmrtrix(input=input, output=output, nthreads=nthreads, force=force, verbose=verbose)
    if (thresh < 0) or (thresh > 1):
        raise ValueError("BET Threshold needs to be within 0 to 1 range.")
    # Read FSL NifTi output format and change it if not '.nii'
    fsl_suffix = os.getenv("FSLOUTPUTTYPE")
    if fsl_suffix is None:
        msg = "Unable to determine system environment variable 'FSF_OUTPUT_FORMAT'. "
        msg += "Ensure that FSL is installed correctly."
        raise OSError(msg)
    if fsl_suffix == "NIFTI_GZ":
        os.environ["FSLOUTPUTTYPE"] = "NIFTI"
    outdir = op.dirname(opts.output)
    B0_nan = op.join(outdir, "B0_nan.nii")
    mask = op.join(outdir, "brain")
    tmp_brain = op.join(outdir, "brain.nii")
    # Extract averaged B0 from DWI
    extractmeanbzero(
        input=opts.input,
        output=B0_nan,
        nthreads=opts.nthreads,
        force=opts.force,
        verbose=opts.verbose,
    )
    # Compute brain mask
    arg_mask = ["bet", B0_nan, mask, "-m", "-f", str(thresh)]
    completion = subprocess.run(arg_mask)
    if completion.returncode != 0:
        msg = rf"Unable to compute brain mask from B0. Return code: {completion.returncode}"
        msg += rf"\nCommand: {' '.join(arg_mask)}"
        msg += rf"\FSL error: {completion.stderr}"
        raise MRTrixError(msg)
    # Remove intermediary file
    os.remove(B0_nan)
    os.remove(tmp_brain)
    os.rename(op.join(outdir, mask + "_mask.nii"), opts.output)


def csfmask(
    input,
    output,
    method="fsl",
    coeff=2,
    thresh=0.25,
    nthreads=None,
    force=False,
    verbose=False,
):
    """Creates a cerebral spinal fluid (CSF) mask from FSL's FAST tool.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .nii CSF mask file
    method : str, optional
        Define method to use for computing a CSF mask. `'fsl'` relies
        on FSL FAST segmentation, and `adc` uses pseudo-diffusion
        coefficient more than 2 (default) to compute a mask
    coeff : float, optional
        Diffusion coefficient to use in thresholding a pseudo-diffusion
        map to estimate CSF (Default: 2)
    thresh : float, optional
        BET threshold ranging from 0 to 1 (Default: 0.25)
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if (op.splitext(output))[-1] != ".nii":
        raise IOError("Output filename {} must be specified as a NifTi (.nii) file.")
    if (thresh < 0) or (thresh > 1):
        raise ValueError("BET Threshold needs to be within 0 to 1 range.")
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    outdir = op.dirname(output)
    if "fsl" in method:
        # Read FSL NifTi output format and change it if not '.nii'
        fsl_suffix = os.getenv("FSLOUTPUTTYPE")
        if fsl_suffix is None:
            raise OSError(
                "Unable to determine system environment variable "
                "FSF_OUTPUT_FORMAT. Ensure that FSL is installed "
                "correctly."
            )
        if fsl_suffix == "NIFTI_GZ":
            os.environ["FSLOUTPUTTYPE"] = "NIFTI"
        f_suffix = ".nii"
        B0_nan = op.join(outdir, "B0_nan" + f_suffix)
        path_brain = op.join(outdir, "brain")
        path_tissue = op.join(outdir, "tissue")

        # Extract averaged B0 from DWI
        extractmeanbzero(input=input, output=B0_nan, nthreads=nthreads, force=force, verbose=verbose)
        # Compute brain mask
        arg_mask = ["bet", B0_nan, path_brain, "-m", "-f", str(thresh)]
        completion = subprocess.run(arg_mask)
        if completion.returncode != 0:
            raise Exception("Unable to compute brain mask from B0. See above for errors")
        arg = ["fast"]
        if verbose:
            arg.append("-v")
        arg.extend(["-n", "4", "-t", "2", "-o", path_tissue, path_brain + f_suffix])
        completion = subprocess.run(arg)
        if completion.returncode != 0:
            raise Exception("FSL FAST segmentation of brain tissue failed. See above for errors.")
        csfclass = []
        for i in range(4):
            arg = [
                "fslmaths",
                path_tissue + "_pve_" + str(i) + f_suffix,
                "-thr",
                "0.95",
                "-bin",
                path_tissue + "_pve_thr_" + str(i) + f_suffix,
            ]
            completion = subprocess.run(arg)
            if completion.returncode != 0:
                raise Exception("FSLMATHS tissue thresholding failed. See above for errors.")
            arg = [
                "fslstats",
                path_brain + ".nii",
                "-k",
                path_tissue + "_pve_thr_" + str(i) + f_suffix,
                "-P",
                "95",
            ]
            completion = subprocess.run(arg, stdout=subprocess.PIPE)
            if completion.returncode != 0:
                raise Exception("FSLSTATS tissue thresholding failed. See above for errors.")
            console = str(completion.stdout).split("\\n")[0]
            console = console.split("b")[-1]
            console = console.replace("'", "")
            csfclass.append(float(console))
        csfind = np.argmax(csfclass)
        arg = [
            "fslmaths",
            path_tissue + "_pve_" + str(csfind) + f_suffix,
            "-thr",
            "0.7",
            "-bin",
            # '-mul', '-1',
            # '-add', '1',
            "-mul",
            path_brain + "_mask" + f_suffix,
            output,
        ]
        completion = subprocess.run(arg)
        if completion.returncode != 0:
            raise Exception("Unable to create CSF mask. See above for errors.")
        # Remove intermediate files
        os.remove(B0_nan)
        os.remove(op.join(outdir, path_brain + f_suffix))
        for i in range(4):
            os.remove(path_tissue + "_pve_" + str(i) + f_suffix)
            os.remove(path_tissue + "_pve_thr_" + str(i) + f_suffix)
        os.remove(path_tissue + "_mixeltype" + f_suffix)
        os.remove(path_tissue + "_pveseg" + f_suffix)
        os.remove(path_tissue + "_seg" + f_suffix)
    if "adc" in method:
        # Get list of b-values
        bvals = mrinfoutil.shells(input)
        # Find index of shell closest to b=1000
        idx = min(range(len(bvals)), key=lambda i: abs(bvals[i] - 1000))
        # Specify file paths
        path_b0 = op.join(outdir, "S0.mif")
        path_shell = op.join(outdir, "S1000.mif")
        # Extract mean B
        # Extract averaged B0 from DWI
        extractmeanbzero(input=input, output=path_b0, nthreads=nthreads, force=force, verbose=verbose)
        # Extract mean of indexed b=1000 shell
        extractmeanshell(
            input=input,
            output=path_shell,
            shell=bvals[idx],
            nthreads=nthreads,
            force=force,
            verbose=verbose,
        )
        # Use the formula D_pseudo = ln(S0/S1000)/B1000 > 2 to compute
        # brain mask based on pseudo-ADC
        arg = ["mrcalc"]
        if force:
            arg.append("-force")
        if not verbose:
            arg.append("-quiet")
        if nthreads is not None:
            arg.extend(["-nthreads", str(nthreads)])
        arg.extend(
            [
                path_b0,
                path_shell,
                "-div",
                "-log",
                str(bvals[idx] / 1000),
                "-div",
                str(coeff),
                "-gt",
                output,
            ]
        )
        completion = subprocess.run(arg)
        if completion.returncode != 0:
            raise Exception("Unable to compute pseudo ADC. See above for errors.")
        os.remove(path_b0)
        os.remove(path_shell)


def smooth(input, output, csfname=None, fwhm=1.25, size=5):
    """Performs Gaussian smoothing on input .mif image

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    csfname : str
        Path to CSF mask file in .nii format
    fwhm : float
        The full width half max in voxels to be smoothed
        (Default: 1.25)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if csfname is not None:
        if not op.exists(csfname):
            raise OSError("Path to CSF mask does not exist. Please ensure that the file specified exists.")
    if fwhm < 0:
        raise Exception("FWHM cannot be less than zero.")
    if size < 0:
        raise Exception("Size cannot be less than zero. Please specify size as a positive integer.")
    # Convert input .mif to .nii
    outdir = op.dirname(output)
    nii_path = op.join(outdir, "dwism.nii")
    miftonii(input=input, output=nii_path)
    # Perform smoothing
    smoothing.smooth_image(nii_path, csfname=csfname, outname=nii_path, width=fwhm)
    # Convert .nii to .mif
    niitomif(input=nii_path, output=output)
    # Remove converted files
    os.remove(nii_path)
    os.remove(op.splitext(nii_path)[0] + ".bvec")
    os.remove(op.splitext(nii_path)[0] + ".bval")
    os.remove(op.splitext(nii_path)[0] + ".json")


def riciancorrect(input, output, noise=None):
    """Performs Rician correction on input .mif

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    noise : str
        Path to noise map from dwidenoise in .nii format (Default: None)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if noise is not None:
        if not op.exists(noise):
            raise OSError("Input noisemap {} does not exist.".format(noise))
        if op.splitext(noise)[-1] != ".nii":
            raise OSError("Noisemap needs to be in NifTi format.")
    else:
        raise Exception("Rician correction cannot be performed without a noisemap.")
    # Convert input .mif to .nii
    outdir = op.dirname(output)
    nii_path = op.join(outdir, "dwirc.nii")
    miftonii(input=input, output=nii_path)
    # Perform Rician correction
    rician.rician_img_correct(nii_path, noise, outpath=nii_path)
    # Convert .nii to .mif
    niitomif(input=nii_path, output=output)
    # Remove converted files
    os.remove(nii_path)
    os.remove(op.splitext(nii_path)[0] + ".bvec")
    os.remove(op.splitext(nii_path)[0] + ".bval")
    os.remove(op.splitext(nii_path)[0] + ".json")


def extractbzero(input, output, nthreads=None, force=False, verbose=False):
    """Extracts only bzero shells from an input mif file.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    arg = ["dwiextract"]
    if force:
        arg.append("-force")
    if not verbose:
        arg.append("-quiet")
    if nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    arg.extend(["-bzero", input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception("Unable to extract B0s from DWI for computation of brain mask. See above for errors.")


def extractmeanbzero(input, output, nthreads=None, force=False, verbose=False):
    """Extracts average B0 from all B0 shells, with NaNs removed.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif or .nii file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    outdir = op.dirname(output)
    fname_bzero = op.join(outdir, "B0_ALL.mif")
    fname_mean = op.join(outdir, "B0_MEAN.mif")
    # Extract all B0s
    extractbzero(input, fname_bzero, nthreads=nthreads, force=force, verbose=verbose)
    arg_mean = ["mrmath", "-axis", "3", fname_bzero, "mean", fname_mean]
    completion = subprocess.run(arg_mean)
    if completion.returncode != 0:
        raise Exception("Unable to compute mean of B0s. See above forerrors.")
    arg_nan = ["mrcalc", fname_mean, "-finite", fname_mean, "0", "-if", output]
    completion = subprocess.run(arg_nan)
    if completion.returncode != 0:
        raise Exception("Unable to remove NaNs from averaged B0. See above for errors.")
    # Remove non-essential files
    os.remove(fname_bzero)
    os.remove(fname_mean)


def extractnonbzero(input, output, nthreads=None, force=False, verbose=False):
    """Extracts only non-bzero shells from an input mif file.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    arg = ["dwiextract"]
    if force:
        arg.append("-force")
    if not verbose:
        arg.append("-quiet")
    if nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    arg.extend(["-no_bzero", input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception("Unable to extract B0s from DWI for computation of brain mask. See above for errors.")


def extractshell(input, output, shell, nthreads=None, force=False, verbose=False):
    """Extracts specified shell from an input mif file.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    shell : int
        Approximate b-value to extract
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if not isinstance(shell, int):
        raise Exception("Please specify the shell to extract as an integer.")
    if shell < 0:
        raise Exception("Please specify the shell to extract as a positive (more than 0) integer.")
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    arg = ["dwiextract"]
    if force:
        arg.append("-force")
    if not verbose:
        arg.append("-quiet")
    if nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    arg.extend(["-no_bzero", "-singleshell", "-shell", str(shell), input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception("Unable to extract specified shells from DWI. See above for errors.")


def extractmeanshell(input, output, shell, nthreads=None, force=False, verbose=False):
    """Extracts mean of specified from an input mif file.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    shell : int
        Approximate b-value to extract
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if not isinstance(shell, int):
        raise Exception("Please specify the shell to extract as an integer.")
    if shell < 0:
        raise Exception("Please specify the shell to extract as a positive (more than 0) integer.")
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    outdir = op.dirname(output)
    fname_shell = op.join(outdir, "b" + str(shell) + "_ALL.mif")
    fname_mean = op.join(outdir, "b" + str(shell) + "_MEAN.mif")
    # Extract all specified shells
    extractshell(input, fname_shell, shell=shell, nthreads=nthreads, force=force, verbose=verbose)
    # Compute mean
    arg_mean = ["mrmath", "-axis", "3", fname_shell, "mean", fname_mean]
    completion = subprocess.run(arg_mean)
    if completion.returncode != 0:
        raise Exception("Unable to compute mean of B0s. See above forerrors.")
    arg_nan = ["mrcalc", fname_mean, "-finite", fname_mean, "0", "-if", output]
    completion = subprocess.run(arg_nan)
    if completion.returncode != 0:
        raise Exception("Unable to remove NaNs from averaged shell image. See above for errors.")
    # Remove non-essential files
    os.remove(fname_shell)
    os.remove(fname_mean)


def epiboost(input, output, num=1, nthreads=None, force=False, verbose=False):
    """Analyzes an input .mif's PE direction to split into two different
    phase encoding (PE) DWIs. B0s from opposing PE are then extracted
    and concatenated with the DWI. This reduces the number of B0s used
    in undistortion for a better and speedier estimation of the
    distortion field.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    num : int
        Number of B0s pairs to use in EPI correction (Default: 1)
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    print("Applying EPIBOOST")
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if op.splitext(output)[-1] != ".mif":
        raise OSError("Output should be specified as a .mif file.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if not isinstance(num, int):
        raise Exception("Number of B0s to use needs to be specified as an integer.")
    if not num > 0:
        raise Exception("Number of B0s to use needs to be a positive integer greater than 0.")
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    outdir = op.dirname(output)
    fname_bzero = op.join(outdir, "B0_ALL.mif")
    # Extract all B0s
    extractbzero(input, fname_bzero, nthreads=nthreads, force=force, verbose=verbose)
    # Start by figuring out whether DWI is composed of multiple PE dirs
    # or single PE dirs. If all PE dirs are the same, the dataset likely
    # comes with matching phase encoding and slice timing train,
    # indicating that it has a single PE direction.
    dw_scheme = np.array(mrinfoutil.dwscheme(fname_bzero), dtype=int)[:, -1]
    pe_scheme = np.array(mrinfoutil.pescheme(fname_bzero))
    if len(pe_scheme) != len(dw_scheme):
        raise Exception(
            "It appears that the input volume possesses a "
            "dw_scheme of length {}, and pe_scheme of length "
            "{}. These number need to match. Please check "
            "your dataset or contact us on GitHub".format(len(dw_scheme), len(pe_scheme))
        )
    uPE, indPE, iPE = np.unique(pe_scheme, axis=0, return_index=True, return_inverse=True)
    nPE = len(uPE)
    if nPE < 2:
        raise Exception("DWI consists of just one PE direction. Unable to extract B0s.")
    # Index unique PE directions
    bval = []
    bind = []
    iteridx = np.unique(iPE)
    for i, val in enumerate(iteridx):
        bind.append(np.where(iPE == val)[0].tolist())
        bval.append(dw_scheme[np.where(iPE == val)].tolist())
    # Check whether number of B0s to extract exceed those in DWI
    if num > min([len(x) for x in bval]):
        raise Exception(
            "Specified number of B0s pairs to extract "
            "({}) exceed those physically present in DWI "
            "({}), please ensure that variable `num` "
            "suitably represents the number of B0s in DWI.".format(num, min([len(x) for x in bval]))
        )
    # Extract the first `num` pairs from each PE direction
    num = np.arange(0, num, dtype=int).tolist()
    idx_extract = []
    for idx, val in enumerate(bind):
        idx_extract.extend([val[i] for i in num])
    # Extract EPI volume
    str_extract = [str(x) for x in idx_extract]
    arg_epi = ["mrconvert"]
    if force:
        arg_epi.append("-force")
    if not verbose:
        arg_epi.append("-quiet")
    if nthreads is not None:
        arg_epi.extend(["-nthreads", str(nthreads)])
    arg_epi.extend(["-coord", "3", ",".join(str_extract)])
    arg_epi.extend([fname_bzero, output])
    completion = subprocess.run(arg_epi)
    if completion.returncode != 0:
        raise Exception("EPIBOOST: failed to extract specified TOPUP B0 indices. See above for errors.")
    # Remove temp files
    os.remove(fname_bzero)


def reslice(input, output, size, interp="linear", nthreads=None, force=False, verbose=False):
    """Reslices input image to target voxel size

    Parameters
    ----------
    input : str
        Path to input file; .mif or .nii
    output : str
        Path to output file; .mif or .nii
    size : tuple of float
        x, y, z voxel size in mm or output dimensions.
    interp : str, {'linear', 'nearest', 'cubic' , 'sinc'}, optional
        set the interpolation method to use when resizing (Default:
        'linear')
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file

    Notes
    -----
    If any of the axes in ``size`` is specified to be over 9 mm, this
    functions reslices to defined output dimensions, instead of voxel
    size. This is done to automatically reslice with minimal user
    input, and also because voxel size beyond 9 mm in unrealistic.

    Additionally, if target resolution is the same as input file's
    resolution, reslicing is skipped but the output file is still
    generated.
    """
    dim_str = "-voxel"
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if not isinstance(size, str):
        raise Exception("Voxel size needs to be defined as a string  of three values")
    if len(size.split(",")) != 3:
        raise Exception(
            "Please specify voxel size for each axis or "
            'single digit for all axes i.e. "3,3,3" for '
            '3 mm isotropic or "42,42,130" for output '
            "dimensions of 42, 42, 130 voxels."
        )
    if max([float(x) for x in size.split(",")]) > 9:
        dim_str = "-size"
    if not isinstance(interp, str):
        raise Exception("Interpolation method needs to be specified as a string")
    if interp not in ("linear", "nearest", "cubic", "sinc"):
        raise Exception("User specified interpoaltion method {} is not a valid option".format(interp))
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    if dim_str == "-voxel":
        current_size = [round(float(x), 2) for x in (mrinfoutil.spacing(input))][0:3]
    elif dim_str == "-size":
        current_size = [round(float(x), 2) for x in mrinfoutil.size(input)][0:3]
    specified_size = [round(float(x), 2) for x in size.split(",")]
    if specified_size == current_size:
        print(
            "[WARNING] target reslicing dimensions {} are the same "
            "as input image dimensions {}, writing file without "
            "reslicing".format(specified_size, current_size)
        )
        copyfile(input, output)
        return
    arg = []
    if which("mrresize") is None:
        arg.extend(["mrgrid", input, "regrid", dim_str, size, "-interp", interp, output])
    else:
        arg.extend(["mrresize", dim_str, size, "-interp", interp, input, output])
    if force:
        arg.append("-force")
    if not verbose:
        arg.append("-quiet")
    if nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception("Failed to reslice. See above for errors.")


def dwiextract(input, output, start, end, nthreads=None, force=False, verbose=False):
    """Extracts a range of volumes from input .mif file in the start:end
    range. Take note that the first volume starts with 0.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output file, usually .mif or .nii
    start : int
        Starting index, inclusive
    end : int
        Ending index, inclusive
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that the folder or file specified exists.")
    if not isinstance(start, int):
        raise Exception("Starting index is needs to be an integer.")
    if not isinstance(end, int):
        raise Exception("Ending index is needs to be an integer.")
    if nthreads is not None:
        if not isinstance(nthreads, int):
            raise Exception("Please specify the number of threads as an integer.")
    if not isinstance(force, bool):
        raise Exception("Please specify whether forced overwrite is True or False.")
    if not isinstance(verbose, bool):
        raise Exception("Please specify whether verbose is True or False.")
    fname, ext = op.splitext(output)
    if ext == ".gz":
        fname, ext = op.splitext(fname)
        ext = ext + ".gz"
    arg = ["mrconvert"]
    if force:
        arg.append("-force")
    if not verbose:
        arg.append("-quiet")
    if nthreads is not None:
        arg.extend(["-nthreads", str(nthreads)])
    arg.extend([input, output, "-coord", "3", str(start) + ":" + str(end)])
    if ".mif" not in ext:
        arg.extend(
            [
                "-json_export",
                fname + ".json",
                "-export_grad_fsl",
                fname + ".bvec",
                fname + ".bval",
            ]
        )
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception("Failed to extract indexed DWI volumes. See above for errors.")
