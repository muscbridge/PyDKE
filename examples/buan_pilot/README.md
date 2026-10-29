# BUAN pilot visualizations

This folder regenerates two figures and a browser overview from the saved pilot results. It does not require the original diffusion images or an internet connection. Supply the directory containing `buan_pilot_20261029` and `buan_odf_investigation_20261029` with `--data-root`. The data are not stored in this repository.

Install Python 3 with NumPy, Matplotlib and NiBabel. Then, from this folder, run:

```sh
python3 visualize.py --data-root /path/to/copied/pilot-parent
```

Open `figures/index.html` in a browser. For another destination use `python3 visualize.py --out /path/to/output`. If the pilot folders are elsewhere, add `--data-root /path/to/parent`. Existing generated figures at the output location are overwritten. The script reads saved tractograms, atlas files, labels and disk summaries; it does not generate tractography, refit ODFs, or run BUAN statistical tests.

The pilot covers three PD participants and two tracts only. The FA-derived mask affects sampling; atlas-based recognition is not independent anatomical validation. Profiles are unweighted exploratory disk means, not group comparisons. Sub-60 forceps major has as few as two valid streamlines per disk. See the pilot data folder’s `buan_odf_investigation_20261029/README.md` for processing details and limitations. A Python environment may be used instead of system Python; dependencies can be installed with `python3 -m pip install numpy matplotlib nibabel` in an appropriate environment.

To rerun the exploratory three-subject pilot, first install the [pyDKE BUAN feature branch](https://github.com/muscbridge/PyDKE/tree/feature/pydke-mrtrix-buan), MRtrix3 and DIPY, and ensure `python3` and `pydke-buan` use compatible environments. Then run:

```sh
bash run_pilot.sh /path/to/BIDS_Archive /path/to/copied/pilot-parent /path/to/new-output
```

The new output path must not already exist, and its parent must exist. The script reads the existing sub-59, sub-60 and sub-61 diffusion derivatives and atlas, creates FA masks and reruns the experimental pipeline. Those three inputs have a recorded export provenance and gradient convention; do not extend the script to other subjects without checking theirs. It never alters the input images. Results are stochastic and may vary. Inspect outputs, bundle coverage, registrations and native-space anatomy before interpreting them. FA mask selection can bias FA measurements, and MK image validity still requires review.

Before each subject, `verify_provenance.py` checks the ODF, fitting b-vectors and MRtrix gradient-bearing DWI against hashes from the original pilot manifest. The run stops if they do not match. This check does not establish quality of all scalar maps, anatomical alignment or results.
