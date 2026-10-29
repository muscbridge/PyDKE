#!/usr/bin/env python3
"""Check pilot SH, gradients and input masks against saved provenance hashes."""
import hashlib,json,sys
from pathlib import Path
subject,bids,pilot=sys.argv[1:]; root=Path(bids)/'derivatives'; manifest=Path(pilot)/'buan_pilot_20261029'/('pydke_world_'+subject)/'manifest.json'
record=json.loads(manifest.read_text())['inputs']
inputs=[(root/'dki'/subject/'metrics/dki_odf.nii',f'/dki/{subject}/metrics/dki_odf.nii'),(root/'dki'/subject/'dwi_preprocessed.bvec',f'/dki/{subject}/dwi_preprocessed.bvec'),(root/'dki_act_20261027_expanded'/subject/'dwi/dwi_preprocessed.mif',f'/dki_act_20261027_expanded/{subject}/dwi/dwi_preprocessed.mif')]
for path,suffix in inputs:
 keys=[k for k in record if k.endswith(suffix)]
 if len(keys)!=1:raise ValueError(f'Cannot identify recorded provenance for {suffix}')
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if h.hexdigest()!=record[keys[0]]:raise ValueError(f'Input differs from recorded provenance: {path}')
print(f'{subject}: source SH and gradient hashes match saved pilot provenance')
