#!/usr/bin/env python3
"""Create a provisional FA seed/stopping mask on the native diffusion grid."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('brain_mask',type=Path);p.add_argument('fa',type=Path);p.add_argument('output',type=Path)
p.add_argument('--threshold',type=float,default=0.2)
a=p.parse_args()
if not 0 < a.threshold < 1: p.error('threshold must be between 0 and 1')
if a.output.exists(): raise FileExistsError(f'Output already exists: {a.output}')
mask=nib.load(a.brain_mask);fa=nib.load(a.fa)
if len(mask.shape)!=3 or mask.shape!=fa.shape or not np.allclose(mask.affine,fa.affine,atol=1e-4,rtol=0): raise ValueError('FA and brain-mask grids differ')
f=np.asarray(fa.dataobj);m=np.asarray(mask.dataobj)
if not np.isfinite(m).all(): raise ValueError('Brain mask is non-finite')
good=(m>0)&np.isfinite(f)&(f>=a.threshold)&(f<=1)
if not np.any(good):raise ValueError('No voxels meet mask criteria')
a.output.parent.mkdir(parents=True,exist_ok=True)
nib.save(nib.Nifti1Image(good.astype('uint8'),mask.affine),a.output)
print(f'Wrote {a.output}; voxels: {good.sum()}')
