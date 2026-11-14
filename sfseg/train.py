"""Optional retained-recipe training, disabled unless explicitly requested.

`--plan` and `--validate-data` never initialize a model or run an optimizer.
Normal reproduction uses the supplied models, not this expensive training path.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n','utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def training_plan():
    c=read(REPO/'configs/training-recipe.json')
    d=read(REPO/'configs/detector-training.json')
    assert len(c['records'])==707
    for field in ('file_name','date','sha256'):
        sets=[{r[field] for r in c['records'] if r['fold']==h} for h in range(3)]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
    for row in c['records']:
        g=row['fold']; assert g not in c['base_jobs'][f'outer-{g}']
        for h in range(3):
            if h!=g:
                k=next(k for k in range(3) if k not in (g,h))
                assert not set(c['base_jobs'][f'single-{k}']) & {g,h}
    role_counts=dict(Counter(r['role'] for r in d['records']))
    assert role_counts['fit']==451 and role_counts['selection']==57 and role_counts['calibration']==57
    for field in ('file_name','date','sha256'):
        sets=[{r[field] for r in d['records'] if r['role']==role} for role in ('fit','selection','calibration','validation')]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(4) for j in range(i))
    return dict(training_enabled=False,physical_images=707,
        base_jobs=c['base_jobs'],base_updates_each=c['base_training']['epochs']*c['base_training']['steps'],
        refiner_jobs=['0','1','2','full'],refiner_updates_each=c['refiner_training']['epochs']*c['refiner_training']['steps'],
        tree_heads=2,tree_settings=c['tree'],detector_roles=role_counts,
        yolo_epochs=40,gray_refiner_epochs=15,
        sequence=['verify official data and frozen routes','download/check generic initialization only',
                  '7 Base fits and excluded-fold probability maps','4 OOF residual refiner fits',
                  '3 excluded-fold quality candidate pools','fit 2 HistGBDT regressors',
                  'A451 YOLO fit with A57 selection','A451 grayscale crop-refiner fit with A57 selection',
                  'export new five-file model bundle; retain frozen inference/calibration settings'],
        nested_upstream_date_hash_exclusion=True,
        calibration_scope='Source/conditional calibration maps in configs/inference.json are frozen retained training-side artifacts, not refitted by this optional training path. Their generation source is included separately.',
        training_rerun_byte_identity_claimed=False)


def validate_training_data(data_root):
    """Read only official training annotations and images; no test labels."""
    data_root=Path(data_root).resolve()
    candidates=list(data_root.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    if len(candidates)!=1: raise ValueError('Provide the extracted official dataset root containing exactly one training annotation file.')
    annotation=candidates[0]; root=annotation.parent.parent
    doc=read(annotation); c=read(REPO/'configs/training-recipe.json'); d=read(REPO/'configs/detector-training.json')
    if sha(annotation)!=d['annotation_sha256']: raise ValueError('Official training annotation hash mismatch.')
    expected={r['file_name']:r for r in c['records']}
    assert len(doc['images'])==1154 and {x['file_name'] for x in doc['images']}==set(expected)
    for f,row in expected.items():
        if sha(root/'train/train_images'/f)!=row['sha256']: raise ValueError('Training image hash mismatch: '+f)
    plan=training_plan()
    return dict(passed=True,training_executed=False,optimizer_updates=0,physical_images=707,expert_sets=1154,
                annotation_sha256=sha(annotation),nested_routes_checked=plan['nested_upstream_date_hash_exclusion']),root,doc


def _download_initialization(output):
    for name,asset in read(REPO/'configs/training-initialization.json').items():
        dest=output/asset['relative_path']; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            tmp=dest.with_suffix(dest.suffix+'.download')
            urllib.request.urlretrieve(asset['url'],tmp)
            if sha(tmp)!=asset['sha256']: raise ValueError('Generic initialization hash mismatch: '+name)
            tmp.replace(dest)
        if sha(dest)!=asset['sha256']: raise ValueError('Generic initialization changed: '+name)
        if name=='base': write(dest.with_name('manifest.json'),asset)


def _fit_trees(base_root, config, target):
    import numpy as np
    import pickle
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    from .quality import spatial_features
    xs=[]; ys=[]
    for h in range(3):
        folder=base_root/'quality-data'/f'oof-{h}'
        for row in read(folder/'manifest.json')['records']:
            path=folder/(Path(row['file_name']).stem+'.npz')
            assert sha(path)==row['sha256']
            with np.load(path) as z:
                if not len(z['patches']): continue
                xs.append(spatial_features(z['patches'],z['features'],z['variants']))
                ys.append(z['targets'])
    if not xs: raise ValueError('No OOF candidates available for quality training.')
    x,y=np.concatenate(xs),np.concatenate(ys)
    assert x.shape[1]==192 and y.shape==(len(x),2) and np.isfinite(x).all() and np.isfinite(y).all()
    settings={k:v for k,v in config['tree'].items() if k not in ('features','candidate_rows','heads')}
    with threadpool_limits(limits=1):
        state=dict(arm='hist_boost',models=[HistGradientBoostingRegressor(**settings).fit(x,y[:,h]) for h in range(2)])
    target.write_bytes(pickle.dumps(state,protocol=5))
    write(target.with_suffix('.json'),dict(candidate_rows=len(x),reference_candidate_rows=17676,settings=settings,
          note='A new upstream fit can produce a different candidate count; no forced parity with historical training outputs.'))


def execute_training(data_root, output_dir, *, enabled=False):
    if not enabled:
        raise RuntimeError('Training is disabled. Explicitly enable it only when a new training run is intended.')
    # Refuse destinations in the release, data inputs or any existing experiment.
    output=Path(output_dir).resolve(); data_root=Path(data_root).resolve()
    if output==REPO or REPO in output.parents or output==data_root or data_root in output.parents:
        raise ValueError('Use a new external training output directory, separate from the release and official input dataset.')
    if output.exists(): raise FileExistsError('Use a new output directory; retained weights and existing runs are never overwritten.')
    audit,dataset_root,doc=validate_training_data(data_root)
    if shutil.disk_usage(output.parent).free<150*1024**3: raise OSError('At least 150 GiB free is required for nested OOF training artifacts.')
    output.mkdir(parents=True)
    write(output/'input-audit.json',audit)
    _download_initialization(output)
    # Imports with numerical libraries occur only after the explicit training gate.
    from .training import base_pipeline as b, detector_pipeline as d, detector_crops as crops
    import torch
    from ultralytics import YOLO
    from unittest.mock import patch
    c=read(REPO/'configs/training-recipe.json')
    spec=read(REPO/'configs/detector-training.json')
    c['validation_records']=[]
    b.ROOT=output/'base'; b.DATA_ROOT=data_root; b.PREPARED=output/'prepared'; b.INITIAL_ROOT=output
    b.ROOT.mkdir(); write(b.ROOT/'protocol.json',c)
    by_file={r['file_name']:r for r in c['records']}
    folds=[dict(by_file[im['file_name']],image_id=str(im['id']),split='train') for im in doc['images']]
    write(b.PREPARED/'folds.json',folds); write(b.PREPARED/'audit.json',dict(annotation_sha256=spec['annotation_sha256']))
    for name in c['base_jobs']:
        b.train_base(name,c)
        if name!='deployment': b.generate_probabilities(name,c)
        import gc
        gc.collect(); torch.cuda.empty_cache()
    for h in (0,1,2,'full'):
        b.build_data(h,c); b.train_refiner(h,c)
        gc.collect(); torch.cuda.empty_cache()
    for h in range(3): b.build_quality_data(h,c)
    bundle=output/'model-bundle'; bundle.mkdir()
    _fit_trees(b.ROOT,c,bundle/'histgbdt.pkl')
    d.ROOT=output; d.DATA_ROOT=data_root
    crops.IMG_DIR=dataset_root/'train/train_images'
    spec['data_root']=str(dataset_root)
    spec['pretrained']={'B1280':dict(path=str(output/'pretrained/yolo11m-seg.pt'))}
    write(output/'protocol.json',spec)
    folders=d.build_yolo_data(spec)
    d.set_seed(0)
    detector=YOLO(spec['pretrained']['B1280']['path'])
    # Original runner bypassed Ultralytics' separate AMP-test download; training AMP stays enabled.
    with patch('ultralytics.engine.trainer.check_amp',return_value=True):
        detector.train(**d.ultra_args(spec,'B1280',folders['segment']))
    del detector; gc.collect(); torch.cuda.empty_cache()
    d.train_torch(spec,'R')
    sources={'base.pt':b.ROOT/'backbones/deployment/inference.pt',
             'refiner.pt':b.ROOT/'refiners/full/inference.pt',
             'yolo11m.pt':output/'reference/models/B1280/weights/best.pt',
             'gray_refiner.pt':output/'reference/models/R/inference.pt'}
    for name,source in sources.items(): shutil.copyfile(source,bundle/name)
    manifest={name:dict(sha256=sha(bundle/name),bytes=(bundle/name).stat().st_size) for name in [*sources,'histgbdt.pkl']}
    write(bundle/'manifest.json',manifest)
    write(output/'training-complete.json',dict(training_executed=True,model_bundle=str(bundle),
          frozen_calibration_retained=True,reference_csv_parity_claimed=False))
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan',action='store_true')
    action.add_argument('--validate-data',action='store_true')
    action.add_argument('--execute-training',action='store_true')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    if args.plan: print(json.dumps(training_plan(),indent=2)); return
    if not args.data_root: p.error('--data-root is required')
    if args.validate_data: print(json.dumps(validate_training_data(args.data_root)[0],indent=2)); return
    if not args.output_dir: p.error('--output-dir is required for an actual training run')
    print(execute_training(args.data_root,args.output_dir,enabled=True))

if __name__=='__main__': main()
