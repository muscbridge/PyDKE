"""Opt-in noncentral-chi magnitude-bias correction with an explicit noise profile.

This adapter neither denoises nor measures a noise profile. Magnitude-domain
profiles (including MP-PCA) use joint mean/variance inversion. Independently
supplied Gaussian component profiles use first-moment inversion.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from . import sos
from .sos import MAX_EFFECTIVE_COILS, correct_ncchi_mean

MARKER = "pydke_sos_noise_correction"
APPLE_METADATA = {".DS_Store", ".AppleDouble", "__MACOSX", ".Spotlight-V100",
                  ".Trashes", ".fseventsd", ".TemporaryItems"}


def _substantive(path):
    return not any(p.startswith("._") or p in APPLE_METADATA for p in Path(path).parts)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def add_sos_arguments(group):
    group.add_argument("--sos-noise-correct", action="store_true",
        help="Use noncentral-chi magnitude-bias correction with an MP-PCA or supplied noise profile instead of --rician; does not enable denoising.")
    group.add_argument("--sos-input-domain", choices=("magnitude-mean",),
        help="Declare that the bias-stage input estimates mean magnitude, not a raw second moment.")
    coils = group.add_mutually_exclusive_group()
    coils.add_argument("--sos-effective-coils", type=float,
        help="Explicit effective coil count in [1,4096], possibly fractional; not inferred from physical channels.")
    coils.add_argument("--sos-effective-coils-map",
        help="Explicit effective-coil 3D NIfTI map in the same physical grid.")
    group.add_argument("--sos-sigma",
        help="Optional supplied 3D/4D noise-profile NIfTI map; otherwise use the selected MP-PCA magnitude-noise map.")
    group.add_argument("--sos-sigma-convention", choices=("magnitude", "per-component-gaussian"),
        help="Noise profile convention: magnitude SD (default when enabled, including MP-PCA) or supplied Gaussian SD per real/imaginary component.")


def validate_sos_arguments(parser, args):
    """Validate before conversion or output creation; resume is not supported."""
    parameters = (args.sos_input_domain, args.sos_effective_coils,
                  args.sos_effective_coils_map, args.sos_sigma, args.sos_sigma_convention)
    if not args.sos_noise_correct:
        if any(value is not None for value in parameters):
            parser.error("--sos-* parameters require --sos-noise-correct")
        return
    if args.rician:
        parser.error("--sos-noise-correct and --rician are alternative bias models; choose one")
    if args.resume:
        parser.error("--sos-noise-correct requires a fresh output directory; --resume is unsupported")
    if args.sos_input_domain != "magnitude-mean":
        parser.error("--sos-noise-correct requires --sos-input-domain magnitude-mean")
    if args.sos_sigma_convention is None:
        args.sos_sigma_convention = "magnitude"
    will_denoise = (args.denoise or args.denoise_method == "adaptive" or args.standard or args.extent != "5,5,5")
    if not args.sos_sigma and (args.sos_sigma_convention != "magnitude" or not will_denoise):
        parser.error("Supply --sos-sigma, or explicitly select magnitude denoising to generate the noise profile")
    if (args.sos_effective_coils is None) == (args.sos_effective_coils_map is None):
        parser.error("Supply exactly one of --sos-effective-coils or --sos-effective-coils-map")
    if args.sos_effective_coils is not None and (
            not np.isfinite(args.sos_effective_coils) or not 1 <= args.sos_effective_coils <= MAX_EFFECTIVE_COILS):
        parser.error("--sos-effective-coils must be finite and in [1,4096]")
    if args.nthreads is not None and args.nthreads < 1:
        parser.error("--nthreads must be a positive integer")
    for name in (args.sos_sigma, args.sos_effective_coils_map):
        if name is not None and (not _substantive(name) or not Path(name).is_file()):
            parser.error("Noise calibration must be a substantive existing NIfTI file: " + name)
    if not args.output:
        parser.error("--sos-noise-correct requires an explicit fresh --output directory")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(_substantive(p.name) for p in output.iterdir())):
        parser.error("--sos-noise-correct requires a fresh empty output directory, even with --force")


def _run(command):
    subprocess.run([str(value) for value in command], check=True, capture_output=True, text=True)


def _calibration_on_grid(path, reference, allow_volume):
    image = nib.load(path)
    allowed = (3, 4) if allow_volume else (3,)
    if image.ndim not in allowed:
        raise ValueError("Calibration map must have an exact spatial 3D or matching 4D shape: " + str(path))
    source_units = image.header.get_xyzt_units()[0]
    reference_units = reference.header.get_xyzt_units()[0]
    if source_units != "unknown" and reference_units != "unknown" and source_units != reference_units:
        raise ValueError("Calibration spatial units differ from the reference image")
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine), nib.orientations.io_orientation(reference.affine))
    index_map = nib.orientations.inv_ornt_aff(orientation, image.shape[:3])
    restored_affine = image.affine @ index_map
    values = nib.orientations.apply_orientation(np.asarray(image.dataobj), orientation)
    if (values.shape[:3] != reference.shape[:3] or
            not np.allclose(restored_affine, reference.affine, atol=2e-5, rtol=0)):
        raise ValueError("Calibration affine/physical grid mismatch; no interpolation or header repair is allowed")
    if values.ndim == 4 and values.shape != reference.shape:
        raise ValueError("4D sigma must match every input volume; use a 3D map for shared sigma")
    return values, {"source_shape": list(image.shape), "aligned_shape": list(values.shape),
        "target_affine": reference.affine.tolist(), "source_affine": image.affine.tolist(),
        "target_to_source_voxel_index_map": index_map.tolist(),
        "operation": "lossless signed axis permutation only; no interpolation"}


def correct_sos_mif(input, output, sigma, n_eff=None, n_eff_map=None, nthreads=1, force=False, *,
                    sigma_convention):
    """Invert magnitude bias using an explicitly interpreted noise profile.

    The caller must supply magnitude-mean estimates, not already bias-corrected
    data. Magnitude sigma uses generalized Koay mean/variance inversion; a
    supplied Gaussian component sigma uses mean inversion. No denoising,
    measured profile estimation, averaging, spatial repair or positive
    fitting floor is performed. The returned summary describes fresh QC outputs.
    """
    if sigma_convention not in ("magnitude", "per-component-gaussian"):
        raise ValueError("Unknown sigma convention")
    source, destination, sigma_path = [Path(p).expanduser().resolve() for p in (input, output, sigma)]
    if any(not _substantive(p) for p in (source, destination, sigma_path)):
        raise ValueError("Apple-generated metadata cannot be image/calibration input")
    if source == destination or (destination.exists() and source.samefile(destination)):
        raise ValueError("In-place SoS correction is prohibited; use a separate output")
    if source.suffix != ".mif" or destination.suffix != ".mif":
        raise ValueError("Input and output must be .mif images")
    if not source.is_file() or not sigma_path.is_file() or not destination.parent.is_dir():
        raise FileNotFoundError("Input, supplied sigma or output parent does not exist")
    if not isinstance(nthreads, int) or isinstance(nthreads, bool) or nthreads < 1:
        raise ValueError("nthreads must be a positive integer")
    if (n_eff is None) == (n_eff_map is None):
        raise ValueError("Supply exactly one effective-coil scalar or map")
    if destination.exists() and not force:
        raise FileExistsError(destination)
    qc = destination.parent / (destination.stem + "_sos_qc")
    if qc.exists():
        raise FileExistsError("Choose a fresh QC/output location: " + str(qc))
    watched = {str(source): _sha(source), str(sigma_path): _sha(sigma_path)}
    coils_path = Path(n_eff_map).expanduser().resolve() if n_eff_map is not None else None
    if coils_path is not None:
        if not _substantive(coils_path) or not coils_path.is_file():
            raise ValueError("Effective-coil map must be a substantive existing image")
        watched[str(coils_path)] = _sha(coils_path)
    with TemporaryDirectory(prefix="pydke-sos-", dir=destination.parent) as temp:
        temp = Path(temp)
        raw, grad, properties = temp/"mean.nii", temp/"grad.txt", temp/"properties.json"
        options = ["-nthreads", str(nthreads), "-quiet"]
        _run(["mrconvert", source, raw, "-export_grad_mrtrix", grad,
              "-json_export", properties, *options])
        props = json.loads(properties.read_text())
        if MARKER in props:
            raise ValueError("Input already has an SoS/noncentral-chi bias-correction tag; refusing double correction")
        image = nib.load(raw)
        if image.ndim != 4:
            raise ValueError("Magnitude-mean input must be 4D")
        signal = np.asarray(image.dataobj)
        sigma_values, sigma_proof = _calibration_on_grid(sigma_path, image, True)
        if coils_path is None:
            coils, coils_proof = n_eff, None
        else:
            coils, coils_proof = _calibration_on_grid(coils_path, image, False)
        if sigma_convention == "magnitude":
            corrected, info = sos.correct_ncchi_magnitude(signal, sigma_values, coils, return_diagnostics=True)
        else:
            corrected, info = correct_ncchi_mean(signal, sigma_values, coils, return_diagnostics=True)
        # Store float64 so extreme finite model estimates cannot overflow a
        # float32 export. Later fit validity remains explicit and independent.
        header = image.header.copy(); header.set_data_dtype(np.float64)
        corrected_path = temp/"corrected.nii"
        nib.save(nib.Nifti1Image(corrected, image.affine, header), corrected_path)
        config = {"algorithm_version": info["algorithm_version"], "input_domain": "magnitude-mean",
            "sigma_convention": sigma_convention, "input_sha256": watched[str(source)],
            "sigma_sha256": watched[str(sigma_path)], "effective_coils": float(n_eff) if n_eff is not None else None,
            "effective_coils_map_sha256": watched.get(str(coils_path)) if coils_path else None}
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        command = ["mrconvert", corrected_path, destination, "-grad", grad,
            "-bvalue_scaling", "no", "-json_import", properties,
            "-set_property", MARKER, fingerprint, *options]
        if force:
            command.append("-force")
        _run(command)
        # Verify both gradients and every voxel's physical position after the
        # format round trip, allowing only a proved signed axis permutation.
        check, check_grad = temp/"checked.nii", temp/"checked_grad.txt"
        _run(["mrconvert", destination, check, "-export_grad_mrtrix", check_grad, *options])
        restored, output_proof = _calibration_on_grid(check, image, True)
        if not np.allclose(restored, corrected, atol=1e-12, rtol=1e-12):
            raise RuntimeError("MIF round trip changed corrected samples")
        if not np.array_equal(np.loadtxt(grad), np.loadtxt(check_grad)):
            raise RuntimeError("MIF round trip changed gradient rows")
        if any(_sha(path) != value for path, value in watched.items()):
            raise RuntimeError("Input or calibration changed during correction")
        qc.mkdir()
        snapshots = {}
        for label, path in (("sigma_profile", sigma_path), ("effective_coils", coils_path)):
            if path is None:
                continue
            target = qc/(label + "_input" + "".join(path.suffixes))
            shutil.copy2(path, target)
            if _sha(target) != watched[str(path)]:
                raise RuntimeError("Calibration snapshot hash changed")
            snapshots[label] = str(target)
        inferred_sigma = info.pop("inferred_sigma_component", None)
        inferred_path = None
        if inferred_sigma is not None:
            inferred_path = qc/"inferred_sigma_component.nii.gz"
            inferred_header = image.header.copy(); inferred_header.set_data_dtype(np.float64)
            nib.save(nib.Nifti1Image(inferred_sigma, image.affine, inferred_header), inferred_path)
        mask_paths = {}
        for name in tuple(key for key, value in info.items() if key.endswith("_mask") and isinstance(value, np.ndarray)):
            flags = info.pop(name).astype(np.uint8)
            path = qc/(name + ".nii.gz")
            mask_header = image.header.copy(); mask_header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(flags, image.affine, mask_header), path)
            mask_paths[name] = str(path)
        # Keep the fingerprinted CLI configuration reproducible while recording
        # the core model's longer semantic labels separately.
        info["core_sigma_convention"] = info.pop("sigma_convention")
        info["core_input_domain"] = info.pop("input_domain")
        summary = {**info, **config, "qc_directory": str(qc), "mask_paths": mask_paths,
            "parameter_fingerprint": fingerprint, "source_sha256": watched,
            "input": str(source), "output": str(destination), "output_sha256": _sha(destination),
            "sigma_path": str(sigma_path), "n_eff_map_path": str(coils_path) if coils_path else None,
            "sigma_grid_proof": sigma_proof, "coils_grid_proof": coils_proof,
            "output_grid_proof": output_proof, "gradients_preserved": True,
            "sources_unchanged": True, "denoising_performed": False,
            "sigma_estimated_or_converted": sigma_convention == "magnitude",
            "noise_profile_convention": sigma_convention, "calibration_snapshots": snapshots,
            "inferred_sigma_component_path": str(inferred_path) if inferred_path else None, "effective_coils_inferred": False,
            "positive_fitting_floor_applied": False, "threads": nthreads}
        (qc/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
