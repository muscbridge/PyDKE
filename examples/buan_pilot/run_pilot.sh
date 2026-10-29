#!/bin/bash
# Experimental pilot; only the three subjects whose export provenance was checked.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  printf 'Usage: %s /path/to/BIDS_Archive /path/to/copied/pilot-parent /path/to/new-output\n' "$0" >&2
  exit 2
fi
bids="$1"; pilot="$2"; dest="$3"
script_dir="$(cd "$(dirname "$0")" && pwd)"
command -v pydke-buan >/dev/null || { echo 'Install the pyDKE feature branch and put pydke-buan on PATH.' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'Python 3 is required.' >&2; exit 1; }
[ ! -e "$dest" ] || { echo "Output already exists: $dest" >&2; exit 1; }
mkdir "$dest"
atlas="$pilot/buan_pilot_20261029/atlas/Atlas_30_Bundles"
for subject in sub-59 sub-60 sub-61; do
  dki="$bids/derivatives/dki/$subject"
  dwi="$bids/derivatives/dki_act_20261027_expanded/$subject/dwi/dwi_preprocessed.mif"
  mask="$dest/${subject}_FA02.nii.gz"
  python3 "$script_dir/verify_provenance.py" "$subject" "$bids" "$pilot"
  python3 "$script_dir/create_fa_mask.py" "$dki/brain_mask.nii" "$dki/metrics/dti_fa.nii" "$mask"
  pydke-buan --odf "$dki/metrics/dki_odf.nii" --mask "$dki/brain_mask.nii" \
    --reference "$dki/B0.nii" --bvec "$dki/dwi_preprocessed.bvec" --gradient-mif "$dwi" \
    --metric "$dki/metrics/dti_fa.nii" --metric "$dki/metrics/dki_mk.nii" \
    --atlas-wholebrain "$atlas/whole_brain/whole_brain_MNI.trk" \
    --atlas-bundle "$atlas/bundles/CST_L.trk" --atlas-bundle "$atlas/bundles/CC_ForcepsMajor.trk" \
    --out "$dest/$subject" --cutoff 0.05 --count 5000 --threads 2 \
    --basis pydke-tournier --coordinate-provenance-verified --exclude-extreme-voxels \
    --tracking-mask "$mask" --algorithm SD_STREAM --step-mm 1.25 --angle-deg 45 \
    --seed-limit 150000 --slr-x0 rigid --slr-min-cluster 3
done
printf 'Completed experimental pilot; inspect the manifests and QC before interpretation.\n'
