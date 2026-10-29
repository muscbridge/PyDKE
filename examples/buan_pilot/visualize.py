#!/usr/bin/env python3
"""Render saved exploratory BUAN pilot figures without source patient images."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from nibabel.streamlines import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent.parent,
                   help='Folder containing both dated BUAN pilot directories')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'figures')
    args=p.parse_args(); root=args.data_root.resolve(); out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    prior=root/'buan_pilot_20261029'; research=root/'buan_odf_investigation_20261029'
    atlas=prior/'atlas/Atlas_30_Bundles/bundles'; subjects=['sub-59','sub-60','sub-61']; bundles=['CST_L','CC_ForcepsMajor']
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,b in enumerate(bundles):
        model=load(str(atlas/(b+'.trk'))).streamlines
        indices=np.linspace(0,len(model)-1,min(400,len(model)),dtype=int)
        for col,s in enumerate(subjects):
            folder=research/s/'dki_deterministic'; labels=[p for p in (folder/'rec').glob('*'+b+'*labels.npy') if not p.name.startswith('._')]
            if len(labels)!=1:raise RuntimeError(f'Expected one label file for {s} {b}, found {len(labels)}')
            ids=np.load(labels[0]); moving=load(str(folder/'slr/moved.trk')).streamlines
            ax=axes[row,col]
            ax.add_collection(LineCollection([model[i][:,[0,2]] for i in indices],colors='#899ba7',alpha=.1,linewidths=.5))
            ax.add_collection(LineCollection([moving[int(i)][:,[0,2]] for i in ids],colors='#d95335',alpha=.3,linewidths=.6))
            ax.autoscale_view();ax.set_aspect('equal');ax.set_title(f'{s} / {b} / n={len(ids)}');ax.set_xlabel('x (mm)');ax.set_ylabel('z (mm)')
    fig.suptitle('Atlas (gray) and participant tracks (red), after global rigid alignment');fig.tight_layout();fig.savefig(out/'bundles.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for i,b in enumerate(bundles):
        for j,metric in enumerate(['dti_fa','dki_mk']):
            ax=axes[i,j]
            for s in subjects:
                path=research/s/'dki_deterministic/profiles'/b/(metric+'_disks.csv')
                with path.open(newline='') as f: rows=list(csv.DictReader(f))
                vals=[float(x['mean']) if x['mean'] else np.nan for x in rows]
                ax.plot([int(x['disk']) for x in rows],vals,label=s,linewidth=1)
            ax.set_title(f'{b} / {metric}');ax.set_ylabel(metric);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Model centroid disk')
    fig.suptitle('Exploratory BUAN disk means; not group statistics');fig.tight_layout();fig.savefig(out/'profiles.png',dpi=150);plt.close(fig)
    (out/'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><title>BUAN pilot</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}img{max-width:100%}</style></head><body><h1>Exploratory BUAN pilot</h1><p>Three PD participants, two bundles. Atlas-based selection and the FA-derived mask affect results; these figures are technical QC, not group statistics. See the accompanying pilot report for methods and limitations.</p><h2>Tractograms</h2><p>Atlas in gray; participant tracks in red after global rigid alignment and before local recognition alignment.</p><img src="bundles.png" alt="Tractogram comparisons"><h2>FA and MK profiles</h2><p>Unweighted disk means; per-disk support varies. Sub-60 forceps major has only two contributing streamlines at its sparsest disk.</p><img src="profiles.png" alt="FA and MK profiles"></body></html>''',encoding='utf8')
    print(f'Figures and overview: {out}')

if __name__=='__main__':main()
