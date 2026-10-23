"""Runs the PyDesigner pipeline"""

import argparse  # ArgumentParser, add_argument
import glob  # recursive file search
import json
import logging
import os  # mkdir
import os.path as op  # path
import shutil  # which, rmtree
import subprocess  # subprocess
import sys as sys
import textwrap  # dedent

import numpy as np  # array, ndarray

from . import __version__
from .fitting import dwipy as dp
from .plotting import motionplot, snrplot
from .postprocessing import filters
from .preprocessing import mrinfoutil, mrpreproc, util

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
log.addHandler(console_handler)

DWIFile = util.DWIFile
DWIParser = util.DWIParser

# Locate mrtrix3 via which-ing dwidenoise
dwidenoise_location = shutil.which("dwidenoise")
if dwidenoise_location is None:
    raise OSError("Cannot find mrtrix3, please see https://github.com/m-ama/PyDesigner/wiki to troubleshoot.")

# Extract mrtrix3 path from dwidenoise_location
mrtrix3path = op.dirname(dwidenoise_location)

# Locate FSL via which-ing fsl
fsl_location = shutil.which("fsl")
if fsl_location is None:
    raise OSError("Cannot find FSL, please see https://github.com/m-ama/PyDesigner/wiki to troubleshoot.")

# Extract FSL path from fsl_location
fslpath = op.dirname(fsl_location)


def main():
    # -----------------------------------------------------------------
    # Parse Arguments
    # -----------------------------------------------------------------
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(
        prog="pydesigner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
    Appendix
    --------
    Filename note:
        Use the base name without the extension. This makes it easy to program
        in automatic .bvec/.bval detection for Niftis and makes your shell
        easier to read by others. The program will automatically search image
        filenames for .nii and .nii.gz extensions. If you use the --dicom
        option, then the program will assume that the entire directory
        consists of dicom files, and will warn you of any files which fail to
        be read in as dicoms.

    Example usage:
        In order to process in the standard way:
        python3 pydesigner.py \\
                --standard \\
                <dwi>

        In order to process in a custom pipeline with denoising, eddy, reverse
        phase encoding, and smoothing, but no diffusion metrics:
        python3 pydesigner.py \\
                --denoise \\
                --undistort \\
                --smooth \\
                --nofit \\
                <dwi>

        In order to just do denoising, eddy with reverse phase encode, and
        diffusion metrics:
        python3 pydesigner.py \\
                --denoise \\
                --undistort \\
                <dwi>

    Standard pipeline steps:
        1. dwidenoise (thermal denoising)
        2. mrdegibbs (gibbs unringing)
        3. topup + eddy (undistortion)
        4. rician bias correction
        5. normalization to white matter in first b0 image
        6. IRWLLS, CWLLS DKI fit
        7. Outlier detection and removal

    See also:
        GitHub      https://github.com/m-ama/PyDesigner
        mrtrix3     https://www.mrtrix.org/
        fsl         https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

                                    """
        ),
    )

    # Specify arguments below
    # Mandatory
    parser.add_argument(
        "dwi",
        nargs="+",
        help="The diffusion dataset you would like to process. ",
        type=str,
    )

    preprocessing_control = parser.add_argument_group(
        title="Preprocessing controls",
        description="Fine-tune DWI preprocessing",
    )
    estimation_control = parser.add_argument_group(
        title="DTI/DKI/WMTI/FBI/FBWM controls",
        description="Control estimation of diffusion estimation",
    )
    qc_control = parser.add_argument_group(
        title="QC controls",
        description="Controls for QC metrics",
    )
    tractography_control = parser.add_argument_group(
        title="Tractography controls", description="Fine-tune tractography"
    )
    program_control = parser.add_argument_group(
        title="PyDesigner CLI controls", description="Change PyDesigner app behavior"
    )

    # Optional
    preprocessing_control.add_argument(
        "-o",
        "--output",
        metavar="directory",
        help="Output location. Default: same path as dwi.",
        type=str,
    )
    preprocessing_control.add_argument(
        "-s",
        "--standard",
        action="store_true",
        default=False,
        help="Standard preprocessing, bypasses most other "
        "options. See Appendix:Standard pipeline steps "
        "for more information. ",
    )
    preprocessing_control.add_argument(
        "-n",
        "--denoise",
        action="store_true",
        default=False,
        help="Run thermal denoising using --denoise-method.",
    )
    preprocessing_control.add_argument(
        "--denoise-method", choices=("mppca", "adaptive"), default="mppca",
        help="MP-PCA backend: original MRtrix (default) or DESIGNER adaptive patches with shrinkage.",
    )
    preprocessing_control.add_argument(
        "--denoise-patch-size", type=int, default=None,
        help="Adaptive patch voxel count; default is 80%% of the search window.",
    )
    preprocessing_control.add_argument(
        "--extent",
        metavar="n,n,n",
        default="5,5,5",
        help="Denoising extent formatted n,n,n (forces  denoising. Default: 5,5,5.",
    )
    preprocessing_control.add_argument(
        "--gibbs-method", choices=("mrtrix", "rpg"), default="mrtrix",
        help="Gibbs correction backend; rpg enables DESIGNER partial-Fourier correction.",
    )
    preprocessing_control.add_argument("--rpg-pf", type=float, choices=(0.625, 0.75, 0.875, 1.0),
                                       help="Explicit acquired Fourier fraction for RPG (not acceleration).")
    preprocessing_control.add_argument("--rpg-pe-axis", type=int, choices=(0, 1, 2),
                                       help="Phase-encoding axis in the input NIfTI voxel grid.")
    preprocessing_control.add_argument("--rpg-slice-axis", type=int, choices=(0, 1, 2),
                                       help="Slice axis in the input NIfTI voxel grid.")
    preprocessing_control.add_argument(
        "-g",
        "--degibbs",
        action="store_true",
        default=False,
        help="Perform gibbs unringing. Only perform if you "
        "have full Fourier encoding. The program will check "
        "for you if you have a .json sidecar.",
    )
    preprocessing_control.add_argument(
        "-u",
        "--undistort",
        action="store_true",
        default=False,
        help="Run FSL eddy to perform image undistortion. NOTE: needs a --topup to run.",
    )
    preprocessing_control.add_argument(
        "--rpe_pairs",
        default=0,
        type=int,
        metavar="n",
        help="Number of reverse phase encoded B0 "
        "pairs to use in TOPUP. Using less pairs "
        "results in faster TOPUP correction. "
        "Specfying 0 results in using all B0 pairs."
        "We recommend using just one pair. Default: 0",
    )
    preprocessing_control.add_argument(
        "-z",
        "--smooth",
        action="store_true",
        default=False,
        help="Perform smoothing on the DWI data.",
    )
    preprocessing_control.add_argument(
        "--fwhm",
        type=float,
        default=1.25,
        metavar="n",
        help="The FWHM to use as a multiple of voxel size. Default 1.25",
    )
    preprocessing_control.add_argument(
        "-r",
        "--rician",
        action="store_true",
        default=False,
        help="Perform Rician noise correction on the data (requires --denoise to generate a noisemap).",
    )
    preprocessing_control.add_argument(
        "--nofit",
        action="store_true",
        default=False,
        help="Do not fit DTI or DKI tensors.",
    )
    preprocessing_control.add_argument(
        "--akc",
        action="store_true",
        default=False,
        help="Brute force K tensor outlier rejection. Applies "
        "a median filter to tensor voxels that exhibit AKC "
        "values of less than 2 and more than 10.",
    )
    preprocessing_control.add_argument(
        "--nooutliers",
        action="store_true",
        default=False,
        help="Do not perform outlier correction on kurtosis fitting metrics.",
    )
    preprocessing_control.add_argument(
        "-m",
        "--mask",
        action="store_true",
        default=False,
        help="Compute a brain mask prior to tensor fitting "
        "to strip skull and improve efficiency. Optionally, "
        "use --maskthr to specify a threshold manually.",
    )
    preprocessing_control.add_argument(
        "--maskthr",
        metavar="n",
        default=0.25,
        help="FSL bet threshold used for brain masking. Default: 0.25",
    )
    preprocessing_control.add_argument(
        "--user_mask",
        metavar="path",
        help="Path to user-supplied brain mask.",
        type=str,
    )
    preprocessing_control.add_argument(
        "-cf",
        "--csf_fsl",
        action="store_true",
        default=False,
        help="Compute a CSF mask for CSF-excluded smoothing to minimize partial volume effects using FSL FAST.",
    )
    preprocessing_control.add_argument(
        "-cd",
        "--csf_adc",
        metavar="n",
        default=False,
        help="Compute a CSF mask for CSF-excluded "
        "smoothing to minimize partial volume "
        "effects using thresholding a pseudo-ADC map "
        "computed as ln(S0/S1000)/b1000. Default: 2",
    )
    preprocessing_control.add_argument(
        "--reslice",
        metavar="x,y,z",
        help="Relices DWI to voxel resolution "
        "specified in millimeters (mm) or output "
        "dimensions. Performing reslicing will skip "
        "plotting of SNR curves. Providing dimensions "
        "greater than 9 will switch from mm voxel "
        "reslicing to output image reslicing.",
    )
    preprocessing_control.add_argument(
        "--interp",
        action="store_true",
        default="linear",
        help="Set the interpolation to use when reslicing. Choices are linear (default), nearest, cubic, and sinc.",
    )
    preprocessing_control.add_argument(
        "--median",
        action="store_true",
        default=False,
        help="Performs postprocessing median filtering of "
        "final maps. WARNING: Use on a case-by-case "
        "basis for bad data only. When applied, the "
        "filter alters the values of most voxels, so "
        "it should be used with caution and avoided "
        "when data quality is otherwise adequate. "
        "While maps appear visually soother with "
        "this flag on, they may nonetheless be less "
        "accurate.",
    )
    preprocessing_control.add_argument(
        "-te",
        "--multite",
        action="store_true",
        default=False,
        help="Specify whether input DWI consists of "
        "multiple TEs. PyDesigner will preprocess all "
        "TEs together, then extract metric values of "
        "each TE separately.",
    )
    preprocessing_control.add_argument(
        "-w", "--window", action="store_true", default=False, help="Window final metric maps within biological ranges"
    )
    estimation_control.add_argument("--scalar-only", action="store_true",
        help="Write tensor/scalar maps without WMTI, ODF, FBI or tractography outputs.")
    estimation_control.add_argument("--constraint-backend", choices=("sampled", "global-sos"),
        default="sampled", help="Opt-in all-direction PSD/SOS constraints; requires pydke[physical].")
    estimation_control.add_argument("--min-diffusivity", type=float, default=1e-6,
        help="Global backend's in-fit minimum diffusivity in um^2/ms; default 1e-6.")
    estimation_control.add_argument(
        "--fit_constraints",
        default="0,1,0",
        metavar="D>0,K>0,K < 3/(b*D)",
        help="Constrain the WLLS fit. Default: 0,1,0.",
    )
    estimation_control.add_argument(
        "--l_max",
        default=6,
        type=int,
        metavar="n",
        help="Maximum spherical harmonic degree for FBI spherical harmonic expansion",
    )
    estimation_control.add_argument(
        "--no_rectify",
        action="store_true",
        default=False,
        help="Disable rectification of FBI fODF. Use "
        "only when rectification of excellent "
        "acquisitions results in degradation of FBI "
        "or FBWM metric maps",
    )
    tractography_control.add_argument(
        "--t_res",
        type=str,
        default="med",
        help="Resolution of directions for ODF calculation. "
        "Higher resolution implies slower computation. Choose "
        'between "low", "med", or "high". Default: "med"',
    )
    tractography_control.add_argument(
        "--t_fibers",
        type=int,
        default=5,
        help="The maximum number ODF maxima to extract per voxel for tractography. Default: 5",
    )
    qc_control.add_argument(
        "--noqc",
        action="store_true",
        default=False,
        help="Disable QC saving of QC metrics",
    )

    program_control.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help="Number of threads to use for computation. Note that using too many threads will cause a slow-down.",
    )
    program_control.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an aborted or partial previous run of pydesigner.",
    )
    program_control.add_argument(
        "--force",
        action="store_true",
        help="Force overwrites of existing files. Otherwise, there will be an error at runtime.",
    )
    program_control.add_argument(
        "--verbose",
        action="store_true",
        help="Print out all output. This is a very messy "
        "option. We recommend piping output to a text file "
        "if you use this option.",
    )
    program_control.add_argument(
        "--adv",
        action="store_true",
        help="Disables safety checks for advanced users who "
        "want to force a preprocessing step. WARNING: "
        "THIS FLAG IS FOR ADVANCED USERS ONLY WHO FULLY "
        "UNDERSTAND THE MRI SYSTEM AND ITS OUTPUTS. "
        "RUNNING WITH THIS FLAG COULD POTENTIALLY "
        "RESULT IN IMPRECISE AND INACCURATE RESULTS.",
    )
    program_control.add_argument("-v", "--version", action="version", version=__version__)

    # Use argument specification to actually get args
    args = parser.parse_args()
    print(args)

    # -----------------------------------------------------------------
    # Parse Input Image
    # -----------------------------------------------------------------
    image = DWIParser(args.dwi)
    # Variable fType indicates the extension to raw_dwi.X, where X take the
    # place of known dMRI file extensions (.mif, .nii, .nii.gz). This allows
    # easy switching based on any scenario for testing.
    fType = ".mif"
    multi_echo = False
    if not args.output:
        outpath = image.getPath()
    else:
        outpath = args.output

    outpath = op.abspath(args.output)
    os.makedirs(outpath, exist_ok=True)

    image.cat(
        path=outpath,
        ext=fType,
        verbose=args.verbose,
        force=args.force,
        resume=args.resume,
    )
    working_path = op.abspath(op.join(outpath, "working" + fType))
    # Create index of DWI volumes with different TEs
    if not args.resume and np.unique(image.echotime).size > 1:
        multi_echo = True
        multi_echo_start = [0]
        multi_echo_end = [image.vols[0] - 1]
        for idx, vols in enumerate(image.vols[1:]):
            multi_echo_start.append(multi_echo_start[-1] + vols)
            multi_echo_end.append(multi_echo_end[-1] + vols)
        multi_echo_start = [int(x) for x in multi_echo_start]
        multi_echo_end = [int(x) for x in multi_echo_end]

    # Make an initial conversion to nifti
    init_nii = op.abspath(op.join(outpath, "dwi_raw.nii"))
    if not (args.resume and op.exists(init_nii)):
        mrpreproc.miftonii(
            input=working_path,
            output=init_nii,
            nthreads=args.nthreads,
            force=args.force,
            verbose=args.verbose,
        )

    # -----------------------------------------------------------------
    # Validate Arguments
    # -----------------------------------------------------------------
    errmsg = ""
    warningmsg = ""
    msgstart = "Incompatible arguments: "
    override = "; overriding with "
    # Warn if --standard and cherry-picking
    if args.standard:
        stdmsg = "--standard but cherry-picking "
        override = "; overriding with standard pipeline.\n"
        if args.denoise:
            warningmsg += msgstart + stdmsg + "--denoise" + override
        if args.undistort:
            warningmsg += msgstart + stdmsg + "--eddy" + override
        if args.smooth:
            warningmsg += msgstart + stdmsg + "--smooth" + override
        # Coerce all of the above to be true
        args.denoise = True
        args.undistort = True
        args.smooth = True
        args.csf_adc = 2
        args.mask = True
        args.degibbs = True
        args.rician = True

    # An explicit preprocessing-only request must never enable tensor fitting.
    if args.nofit and (args.akc or args.nooutliers):
        warningmsg += "Fitting-specific options are ignored with --nofit.\n"

    if args.denoise_method == "adaptive":
        args.denoise = True
    elif args.denoise_patch_size is not None:
        parser.error("--denoise-patch-size requires --denoise-method adaptive")

    if args.gibbs_method == "rpg":
        if None in (args.rpg_pf, args.rpg_pe_axis, args.rpg_slice_axis):
            parser.error("RPG requires --rpg-pf, --rpg-pe-axis and --rpg-slice-axis")
        if args.rpg_pe_axis == args.rpg_slice_axis:
            parser.error("RPG phase-encoding and slice axes must differ")
        if args.reslice:
            parser.error("RPG must run before interpolation; run --reslice in a later invocation")
        args.degibbs = True
    elif any(x is not None for x in (args.rpg_pf, args.rpg_pe_axis, args.rpg_slice_axis)):
        parser.error("--rpg-* options require --gibbs-method rpg")

    # (Extent or Degibbs) and no Denoise
    if not args.denoise:
        stdmsg = "No --denoise but "
        if args.extent != "5,5,5":
            warningmsg += stdmsg + "--extent given; overriding with --denoise\n"
            args.denoise = True
        if args.rician:
            warningmsg += stdmsg + "--rician given; overriding with --denoise\n"
            args.denoise = True

    # Cannot run --user_mask and --mask at the same time
    if args.user_mask and args.mask:
        errmsg += "Cannot run with both --mask and --user_mask; "
        errmsg += "--mask if you do not have a custom brain mask and --user_mask if you want to supply a mask."

    # Cannot run --csf_fsl and --csf_adc at the same time
    if args.csf_fsl and args.csf_adc:
        errmsg += "Cannot run with both --csf_fsl and --csf_adc; "
        errmsg += "please supply only one option."

    # Check to make sure brain mask exists if given
    if args.user_mask:
        if not op.exists(args.user_mask):
            errmsg += "--user_mask file " + args.user_mask + " not found\n"
        # Then check if it's a nifti file
        if ".nii" not in op.splitext(args.user_mask)[-1]:
            errmsg += "User supplied mask if not in NifTi (.nii) format."

    # Check output directory exists if given
    if args.output:
        if not op.exists(args.output):
            try:
                os.makedirs(args.output, exist_ok=True)
            except:  # noqa: E722
                errmsg += "Cannot find or create output directory"

    if args.constraint_backend == "global-sos":
        if args.akc or args.window:
            parser.error("global-sos cannot be combined with --akc or --window; preserve raw fits")
        if not np.isfinite(args.min_diffusivity) or args.min_diffusivity <= 0:
            parser.error("--min-diffusivity must be finite and positive")

    # Check that --fit_constraints can be converted to int array
    fit_constraints = np.fromstring(args.fit_constraints, dtype=int, sep=",")
    for i in fit_constraints:
        if i < 0 or i > 1:
            errmsg += "Invalid --fit_constraints value, should be 0 or 1\n"
            break

    # Ensure l_max is an even integer
    if args.l_max % 2 != 0:
        errmsg += "User provided l_max = {} is not an even integer.".format(args.l_max)

    # --force and --resume given
    if args.resume and args.force:
        errmsg += msgstart + "--continue and --force\n"

    if args.output:
        if not op.isdir(args.output):
            try:
                os.makedirs(args.output, exist_ok=True)
            except:  # noqa: E722
                errmsg += "Output directory does not exist and cannot be made."

    # Check whether tractography variables are parsed correctly
    if args.t_res not in ["low", "med", "high"]:
        warningmsg += "Specified ODF resolution not understoord. Defaulting to med\n"

    # Print warnings
    if warningmsg != "":
        log.warn(warningmsg)

    # If things are unsalvageable, point out all errors and quit
    if errmsg != "":
        log.error(errmsg)

    # Begin keeping track of nifti files
    filetable = {"dwi": DWIFile(init_nii)}
    if not filetable["dwi"].isAcquisition():
        raise FileNotFoundError("Input dwi does not have .bval/.bvec pair")

    # Begin composing command history
    cmdtable = {"HEAD": "none"}
    cmdtable["input"] = mrinfoutil.commandhistory(working_path)

    # Check to make sure no partial fourier if --degibbs given
    if args.degibbs and (args.adv or args.gibbs_method == "rpg"):
        args.degibbs = True
    else:
        if args.degibbs and filetable["dwi"].isPartialFourier():
            log.warn(
                "[WARNING] Given DWI is partial fourier, overriding "
                "--degibbs; no unringing correction will be done to "
                'avoid artifacts.Use the "--adv" flag to run forced '
                "corrections."
            )
            args.degibbs = False

    # Handle FBI rectification
    fbi_rectify = True
    if args.no_rectify:
        fbi_rectify = False

    # -----------------------------------------------------------------
    # Path Handling
    # -----------------------------------------------------------------
    qcpath = op.join(outpath, "metrics_qc")
    eddyqcpath = op.join(qcpath, "eddy")
    fitqcpath = op.join(qcpath, "fitting")
    metricpath = op.join(outpath, "metrics")
    intermediatepath = op.join(outpath, "intermediate_nifti")
    if not args.nofit:
        if op.exists(metricpath):
            if args.force:
                shutil.rmtree(metricpath)
            elif not args.resume:
                raise Exception(
                    "Running fitting would cause an overwrite. "
                    "In order to run this please delete the "
                    "files, use --force, use --resume, or "
                    "change output destination."
                )
        else:
            os.makedirs(metricpath, exist_ok=True)
    if not args.noqc:
        if op.exists(qcpath):
            if args.force:
                shutil.rmtree(qcpath)
            elif not args.resume:
                raise Exception(
                    "Running QCing would cause an overwrite. "
                    "In order to run this please delete the "
                    "files, use --force, use --resume, or "
                    "change output destination."
                )
        else:
            os.makedirs(qcpath, exist_ok=True)
        if op.exists(eddyqcpath) and args.undistort:
            if args.force:
                shutil.rmtree(eddyqcpath)
            elif not args.resume:
                raise Exception(
                    "Running dwidenoise would cause an "
                    "overwrite. "
                    "In order to run this please delete the "
                    "files, use --force, or change output "
                    "destination."
                )
        if op.exists(fitqcpath) and not args.nofit:
            if args.force:
                shutil.rmtree(fitqcpath)
            elif not args.resume:
                raise Exception(
                    "Running fitting would cause an "
                    "overwrite. "
                    "In order to run this please delete the "
                    "files, use --force, or change output "
                    "destination."
                )
        if args.undistort:
            os.makedirs(eddyqcpath, exist_ok=True)
        if not args.nofit:
            os.makedirs(fitqcpath, exist_ok=True)

    if not op.exists(intermediatepath):
        os.makedirs(intermediatepath, exist_ok=True)

    # TODO: add non-json RPE support, additional RPE type support

    # Get naming and location information
    filetable["dwi"].getName()
    if not args.output:
        outpath = filetable["dwi"].getPath()
    else:
        outpath = args.output
    filetable["outpath"] = outpath

    # Make the pipeline point to dwi as the last file since it's the only one
    # so far
    filetable["HEAD"] = filetable["dwi"]

    if args.nthreads and args.verbose:
        print("Using " + str(args.nthreads) + " threads.")

    # Create processing step variable to count preprocessing stage
    step_count = 0

    # -----------------------------------------------------------------
    # Run Denoising
    # -----------------------------------------------------------------
    if args.denoise:
        step_count += 1
        denoised_name = "dwi_denoised"
        # hardcoding this to be the initial file per dwidenoise
        # recommmendation
        # file names
        denoised_name_full = str(step_count) + "_" + denoised_name
        nii_denoised = op.join(intermediatepath, denoised_name_full + ".nii")
        mif_denoised = op.join(outpath, denoised_name_full + ".mif")
        # output the noise map even without user permission, space is cheap
        noisemap_name = "noisemap.nii"
        nii_noisemap = op.join(outpath, noisemap_name)
        # check to see if this already exists
        denoise_config = {"method": args.denoise_method, "extent": args.extent,
                          "patch_size": args.denoise_patch_size}
        denoise_record = op.join(outpath, "denoising.json")
        if args.resume:
            from .preprocessing.denoise_config import check_resume
            check_resume(denoise_record, denoise_config)
        if not (args.resume and op.exists(nii_denoised) and op.exists(nii_noisemap)):
            # run denoise function
            mrpreproc.denoise(
                input=working_path,
                output=mif_denoised,
                noisemap=True,
                method=args.denoise_method,
                patch_size=args.denoise_patch_size,
                extent=args.extent,
                nthreads=args.nthreads,
                force=args.force,
                verbose=args.verbose,
            )
            mrpreproc.miftonii(
                input=mif_denoised,
                output=nii_denoised,
                nthreads=args.nthreads,
                force=args.force,
                verbose=False,
            )
            # remove old working.mif and replace with new corrected .mif
            os.replace(mif_denoised, working_path)
            with open(denoise_record, "w") as f:
                json.dump(denoise_config, f, indent=2)
            # update command history
            cmdtable["denoise"] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable["HEAD"] = cmdtable["denoise"]
        # update nifti file tracking
        filetable["denoised"] = DWIFile(nii_denoised)
        filetable["HEAD"] = filetable["denoised"]
        filetable["noisemap"] = DWIFile(nii_noisemap)

    # -----------------------------------------------------------------
    # Run Reslicing
    # -----------------------------------------------------------------
    if args.reslice:
        step_count += 1
        reslice_name = "dwi_reslice"
        noise_name = "noisemap_resliced"
        # file names
        reslice_name_full = str(step_count) + "_" + reslice_name
        nii_reslice = op.join(intermediatepath, reslice_name_full + ".nii")
        mif_reslice = op.join(outpath, reslice_name_full + ".mif")
        nii_noise = op.join(outpath, noise_name + ".nii")
        # check to see if this already exists
        if not (args.resume and op.exists(nii_reslice)):
            # run reslice function on both DWI and noisemap
            mrpreproc.reslice(
                input=working_path,
                output=mif_reslice,
                size=args.reslice,
                interp=args.interp,
                nthreads=args.nthreads,
                force=args.force,
                verbose=args.verbose,
            )
            mrpreproc.reslice(
                input=nii_noisemap,
                output=nii_noise,
                size=args.reslice,
                interp=args.interp,
                nthreads=args.nthreads,
                force=True,
                verbose=args.verbose,
            )
            mrpreproc.miftonii(
                input=mif_reslice,
                output=nii_reslice,
                nthreads=args.nthreads,
                force=args.force,
                verbose=False,
            )
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_reslice, working_path)
            os.rename(nii_noise, nii_noisemap)
            # update command history
            cmdtable["reslice"] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable["HEAD"] = cmdtable["reslice"]
        # update nifti file tracking
        filetable["reslice"] = DWIFile(nii_reslice)
        filetable["HEAD"] = filetable["reslice"]

    # -----------------------------------------------------------------
    # Run Gibbs Unringing
    # -----------------------------------------------------------------
    if args.degibbs:
        step_count += 1
        degibbs_name = "dwi_degibbs"
        # file names
        degibbs_name_full = str(step_count) + "_" + degibbs_name
        nii_degibbs = op.join(intermediatepath, degibbs_name_full + ".nii")
        mif_degibbs = op.join(outpath, degibbs_name_full + ".mif")
        # check to see if this already exists
        from .preprocessing.denoise_config import check_resume
        gibbs_config = dict(method=args.gibbs_method, pf=args.rpg_pf,
                            pe_axis=args.rpg_pe_axis, slice_axis=args.rpg_slice_axis)
        gibbs_record = op.join(outpath, "gibbs.json")
        if args.resume:
            check_resume(gibbs_record, gibbs_config, legacy_method="mrtrix")
        if not (args.resume and op.exists(nii_degibbs)):
            # run degibbs function
            mrpreproc.degibbs(
                method=args.gibbs_method, pf=args.rpg_pf,
                pe_axis=args.rpg_pe_axis, slice_axis=args.rpg_slice_axis,
                input=working_path,
                output=mif_degibbs,
                nthreads=args.nthreads,
                force=args.force,
                verbose=args.verbose,
            )
            mrpreproc.miftonii(
                input=mif_degibbs,
                output=nii_degibbs,
                nthreads=args.nthreads,
                force=args.force,
                verbose=False,
            )
            # remove old working.mif and replace with new corrected .mif
            os.replace(mif_degibbs, working_path)
            with open(gibbs_record, "w") as f:
                json.dump(gibbs_config, f, indent=2)
            # update command history
            cmdtable["degibbs"] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable["HEAD"] = cmdtable["degibbs"]
        # update nifti file tracking
        filetable["unrung"] = DWIFile(nii_degibbs)
        filetable["HEAD"] = filetable["unrung"]

    # -----------------------------------------------------------------
    # Undistort
    # -----------------------------------------------------------------
    if args.undistort:
        step_count += 1
        undistorted_name = "dwi_undistorted"
        # file names
        undistorted_name_full = str(step_count) + "_" + undistorted_name
        nii_undistorted = op.join(intermediatepath, undistorted_name_full + ".nii")
        mif_undistorted = op.join(outpath, undistorted_name_full + ".mif")
        if args.noqc:
            eddyqcpath = None
        # check to see if this already exists
        if not (args.resume and op.exists(nii_undistorted)):
            # run undistort function
            mrpreproc.undistort(
                input=working_path,
                output=mif_undistorted,
                rpe="rpe_header",
                qc=eddyqcpath,
                epib0=args.rpe_pairs,
                nthreads=args.nthreads,
                force=args.force,
                verbose=args.verbose,
            )
            mrpreproc.miftonii(
                input=mif_undistorted,
                output=nii_undistorted,
                nthreads=args.nthreads,
                force=args.force,
                verbose=False,
            )
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_undistorted, working_path)
            # update command history
            cmdtable["undistort"] = mrinfoutil.commandhistory(working_path)
            cmdtable["HEAD"] = cmdtable["undistort"]
        # update nifti file tracking
        filetable["undistorted"] = DWIFile(nii_undistorted)
        filetable["HEAD"] = filetable["undistorted"]

        # Plot head motion
        if not args.noqc:
            plot_path_full = op.join(qcpath, "head_motion.png")
            motionplot.plot(
                op.join(eddyqcpath, "eddy_restricted_movement_rms"),
                plot_path_full,
                voxel=mrinfoutil.spacing(working_path),
            )

    # -----------------------------------------------------------------
    # Create CSF Mask
    # -----------------------------------------------------------------
    csfmask_name = "csf_mask.nii"
    csfmask_out = op.join(outpath, csfmask_name)
    # FSL Method
    if args.csf_fsl:
        mrpreproc.csfmask(
            input=working_path,
            output=csfmask_out,
            method="fsl",
            thresh=args.maskthr,
            nthreads=args.nthreads,
            force=args.force,
            verbose=args.verbose,
        )
        filetable["csfmask"] = DWIFile(csfmask_out)
    # ADC Method
    if args.csf_adc:
        mrpreproc.csfmask(
            input=working_path,
            output=csfmask_out,
            method="adc",
            coeff=args.csf_adc,
            nthreads=args.nthreads,
            force=args.force,
            verbose=args.verbose,
        )
        filetable["csfmask"] = DWIFile(csfmask_out)

    # -----------------------------------------------------------------
    # Create Brain Mask
    # -----------------------------------------------------------------
    if args.mask:
        brainmask_name = "brain_mask.nii"
        brainmask_out = op.join(outpath, brainmask_name)
        mrpreproc.brainmask(
            input=working_path,
            output=brainmask_out,
            thresh=args.maskthr,
            nthreads=args.nthreads,
            force=args.force,
            verbose=args.verbose,
        )
        filetable["mask"] = DWIFile(brainmask_out)

    if args.user_mask:
        # Rotates user mask to same orientation as PyDesigner's working
        # file to prevent incorrect masking
        brainmask_name = "brain_mask.nii"
        brainmask_out = op.join(outpath, "brain_mask.nii")
        mrpreproc.stride_match(
            target=working_path,
            moving=args.user_mask,
            output=brainmask_out,
            nthreads=args.nthreads,
            force=args.force,
            verbose=args.verbose,
        )
        filetable["mask"] = DWIFile(brainmask_out)

    # -----------------------------------------------------------------
    # Multiply Brain Mask with CSF Mask if both present
    # -----------------------------------------------------------------
    if (args.mask or args.user_mask) and (args.csf_fsl or args.csf_adc):
        cmd = ["mrcalc", "-force", brainmask_out, csfmask_out, "-mult", csfmask_out]
        completion = subprocess.run(cmd)
        if completion.returncode != 0:
            raise Exception("Unable to multiply CSF mask with brain mask. See above for errors.")
    # -----------------------------------------------------------------
    # Smooth
    # -----------------------------------------------------------------
    if args.smooth:
        csfname = None
        if "csfmask" in filetable:
            csfname = filetable["csfmask"].getFull()
        step_count += 1
        smoothing_name = "dwi_smoothed"
        # file names
        smoothing_name_full = str(step_count) + "_" + smoothing_name
        nii_smoothing = op.join(intermediatepath, smoothing_name_full + ".nii")
        mif_smoothing = op.join(outpath, smoothing_name_full + ".mif")
        # check to see if this already exists
        if not (args.resume and op.exists(nii_smoothing)):
            mrpreproc.smooth(
                input=working_path,
                csfname=csfname,
                output=mif_smoothing,
                fwhm=args.fwhm,
            )

            mrpreproc.miftonii(
                input=mif_smoothing,
                output=nii_smoothing,
                nthreads=args.nthreads,
                force=args.force,
                verbose=False,
            )
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_smoothing, working_path)
            # update command history
            cmdtable["smooth"] = [
                "designer.preprocessing.mrpreproc.smooth(input={}, output={}, fwhm={}".format(
                    working_path, mif_smoothing, args.fwhm
                )
            ]
            cmdtable["smooth"].append(mrinfoutil.commandhistory(working_path)[-1])
            cmdtable["HEAD"] = cmdtable["smooth"]
        # update nifti file tracking
        filetable["smoothed"] = DWIFile(nii_smoothing)
        filetable["HEAD"] = filetable["smoothed"]

    # -----------------------------------------------------------------
    # Rician Noise Correction
    # -----------------------------------------------------------------
    if args.rician:
        step_count += 1
        rician_name = "dwi_rician"
        # file names
        rician_name = str(step_count) + "_" + rician_name
        nii_rician = op.join(intermediatepath, rician_name + ".nii")
        mif_rician = op.join(outpath, rician_name + ".mif")
        # check to see if this already exists
        if not (args.resume and op.exists(nii_rician)):
            mrpreproc.riciancorrect(
                input=working_path,
                output=mif_rician,
                noise=filetable["noisemap"].getFull(),
            )
            nii_rician_name = "r" + filetable["HEAD"].getName() + ".nii"
            op.join(outpath, nii_rician_name)
            mrpreproc.miftonii(
                input=mif_rician,
                output=nii_rician,
                nthreads=args.nthreads,
                force=args.force,
                verbose=False,
            )
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_rician, working_path)
            # update command history
            cmdtable["rician"] = [
                "designer.preprocessing.mrpreproc.riciancorrect(input={}, output={}, noise={})".format(
                    working_path, mif_rician, filetable["noisemap"].getFull()
                )
            ]
            cmdtable["rician"].append(mrinfoutil.commandhistory(working_path)[-1])
            cmdtable["HEAD"] = cmdtable["rician"]
        # update nifti file tracking
        filetable["rician_corrected"] = DWIFile(nii_rician)
        filetable["HEAD"] = filetable["rician_corrected"]

    print("[DEBUG] working_path for B0/shell extraction:", working_path)
    print("[DEBUG] shells:", mrinfoutil.shells(working_path))
    # -----------------------------------------------------------------
    # Extract averaged B0
    # -----------------------------------------------------------------
    # file names
    b0_name = "B0"
    nii_b0 = op.join(outpath, b0_name + ".nii")
    # check to see if this already exists
    if not (args.resume and op.exists(nii_b0)):
        # extract mean B0
        mrpreproc.extractmeanbzero(
            input=working_path,
            output=nii_b0,
            nthreads=args.nthreads,
            force=args.force,
            verbose=args.verbose,
        )
        # update command history
        cmdtable["B0"] = mrinfoutil.commandhistory(working_path)[-1]
        cmdtable["HEAD"] = cmdtable["B0"]
    # update nifti file tracking
    filetable["B0"] = DWIFile(nii_b0)
    filetable["HEAD"] = filetable["B0"]

    # -----------------------------------------------------------------
    # Extract averaged non-B0 shells
    # -----------------------------------------------------------------
    # get non B0 shells
    b_shells = [x for x in mrinfoutil.shells(working_path) if x != 0]
    # remove
    # file names
    b_names = ["B" + str(x) for x in b_shells]
    b_paths = [op.join(outpath, x + ".nii") for x in b_names]
    # check to see if this already exists
    for b_value, b_nii in zip(b_shells, b_paths):
        if not (args.resume and op.exists(b_nii)):
            # extract mean shells
            mrpreproc.extractmeanshell(
                input=working_path,
                output=b_nii,
                shell=b_value,
                nthreads=args.nthreads,
                force=args.force,
                verbose=args.verbose,
            )
            # update command history
            cmdtable["B" + str(b_value)] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable["HEAD"] = cmdtable["B" + str(b_value)]
            # update nifti file tracking
            filetable["B" + str(b_value)] = DWIFile(b_nii)
            filetable["HEAD"] = filetable["B" + str(b_value)]

    # -----------------------------------------------------------------
    # Make preprocessed file
    # -----------------------------------------------------------------
    preprocessed = op.join(outpath, "dwi_preprocessed.nii")
    if not (args.resume and op.exists(preprocessed)):
        mrpreproc.miftonii(
            input=working_path,
            output=preprocessed,
            nthreads=args.nthreads,
            force=args.force,
            verbose=False,
        )
    filetable["preprocessed"] = DWIFile(preprocessed)
    filetable["HEAD"] = filetable["preprocessed"]

    # -----------------------------------------------------------------
    # Compute SNR
    # -----------------------------------------------------------------
    if (args.denoise and not args.reslice) and not args.noqc:
        files = []
        files.append(init_nii)
        files.append(filetable["HEAD"].getFull())
        try:
            if "mask" in filetable:
                snr = snrplot.makesnr(
                    dwilist=files,
                    noisepath=nii_noisemap,
                    maskpath=filetable["mask"].getFull(),
                )
            else:
                snr = snrplot.makesnr(
                    dwilist=files,
                    noisepath=filetable["noisemap"].getFull(),
                    maskpath=None,
                )
            snr.makeplot(path=qcpath, smooth=True, smoothfactor=3)
        except:  # noqa: E722
            print("[WARNING] SNR plotting failed, see above. Proceeding with processing.")

    # -----------------------------------------------------------------
    # Write logs
    # -----------------------------------------------------------------
    with open(op.join(outpath, "log_command.json"), "w") as fp:
        json.dump(cmdtable, fp, indent=2)

    # -----------------------------------------------------------------
    # Handle multi-echo data
    # -----------------------------------------------------------------
    imPath = filetable["HEAD"].getFull()
    if multi_echo and args.multite:
        imPath = []
        for i in range(len(image.echotime)):
            echo_out = op.join(outpath, "TE" + str(image.echotime[i]) + "_dwi_preprocessed.nii")
            mrpreproc.dwiextract(
                working_path,
                echo_out,
                start=multi_echo_start[i],
                end=multi_echo_end[i],
                nthreads=args.nthreads,
                force=args.force,
                verbose=False,
            )
            imPath.append(echo_out)

    # Remove working.mif
    os.remove(working_path)

    # -----------------------------------------------------------------
    # Tensor Fitting
    # -----------------------------------------------------------------
    ext = ".nii"
    fit_mask = None
    if "mask" in filetable:
        fit_mask = filetable["mask"].getFull()
    if not args.nofit:
        # create dwi fitting object
        if multi_echo and args.multite:
            for path, echo in zip(imPath, image.echotime):
                # qcpath = op.join(op.dirname(path), 'qc_fitting')
                os.makedirs(qcpath, exist_ok=True)
                dp.fit_regime(
                    input=path,
                    output=metricpath,
                    prefix="TE" + str(echo) + "_",
                    suffix=None,
                    ext=ext,
                    irlls=not args.nooutliers,
                    akc=args.akc,
                    l_max=args.l_max,
                    res=args.t_res,
                    n_fibers=args.t_fibers,
                    rectify=fbi_rectify,
                    qcpath=fitqcpath,
                    fit_constraints=fit_constraints,
                    constraint_backend=args.constraint_backend,
                    min_diffusivity=args.min_diffusivity,
                    scalar_only=args.scalar_only,
                    mask=fit_mask,
                    window=args.window,
                    nthreads=args.nthreads,
                )
        else:
            dp.fit_regime(
                input=imPath,
                output=metricpath,
                prefix=None,
                suffix=None,
                ext=ext,
                irlls=not args.nooutliers,
                akc=args.akc,
                l_max=args.l_max,
                rectify=fbi_rectify,
                res=args.t_res,
                n_fibers=args.t_fibers,
                qcpath=fitqcpath,
                fit_constraints=fit_constraints,
                constraint_backend=args.constraint_backend,
                min_diffusivity=args.min_diffusivity,
                    scalar_only=args.scalar_only,
                mask=fit_mask,
                window=args.window,
                nthreads=args.nthreads,
            )

    # -----------------------------------------------------------------
    # Post Processing
    # -----------------------------------------------------------------
    if args.median:
        f_metrics = glob.glob(op.join(metricpath, "*" + ext))
        f_metrics = [x for x in f_metrics if not x.endswith("DT.nii")]
        f_metrics = [x for x in f_metrics if not x.endswith("KT.nii")]
        f_metrics = [x for x in f_metrics if not x.endswith("fodf.nii")]
        for f in f_metrics:
            filters.median(f, f)


if __name__ == "__main__":
    main()
