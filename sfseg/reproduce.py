"""Image-to-CSV inference, followed by a separate reference comparison.

No annotations, cached probabilities, reference predictions or GT are read by
run(). verify() is a separate post-inference operation.
"""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
(ROOT / '.runtime/yolo').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('ULTRALYTICS_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime/yolo'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch
import pycocotools.mask as mu
from ultralytics import YOLO
from .base import StrongUnet, Refiner, raw_image, probability, core_map, original_proposals, refine_image, resolve
from .quality import proposal_features, spatial_features, predict
from .supplement import PretrainedRefiner, _yolo_style_candidates, refine_candidate, compose


def read(path): return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', 'utf-8')


def rles(labels):
    return [dict(source='Base', index=i, rle=mu.encode(np.asfortranarray((labels == cid).astype(np.uint8)))['counts'].decode())
            for i, cid in enumerate([int(x) for x in np.unique(labels) if x > 0], 1)]


def verify_models(model_dir=None):
    model_dir = ROOT/'models' if model_dir is None else Path(model_dir).resolve()
    manifest = read(model_dir/'manifest.json')
    assert set(manifest) == {'base.pt','refiner.pt','histgbdt.pkl','yolo11m.pt','gray_refiner.pt'}
    for name, row in manifest.items():
        if sha(model_dir/name) != row['sha256']: raise ValueError('Weight hash mismatch: '+name)
    return manifest


def run(test_dir, output_dir, limit=None, *, model_dir=None):
    test_dir, output_dir = Path(test_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir/'submission.csv'
    if target.exists(): raise FileExistsError('Use a new output directory; existing output is never silently reused.')
    model_dir = ROOT/'models' if model_dir is None else Path(model_dir).resolve()
    weights = verify_models(model_dir)
    rows = read(ROOT/'configs/test-images.json')
    if limit is not None: rows=rows[:limit]
    for row in rows:
        if sha(test_dir/row['file_name']) != row['sha256']: raise ValueError('Input image mismatch: '+row['file_name'])
    if not torch.cuda.is_available(): raise RuntimeError('The validated BF16 inference path requires an NVIDIA CUDA GPU.')
    random.seed(13102026); np.random.seed(13102026); torch.manual_seed(13102026); torch.cuda.manual_seed_all(13102026)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic()
    base=StrongUnet('base_pretrained').cuda().eval()
    ck=torch.load(model_dir/'base.pt',map_location='cpu',weights_only=False)
    assert ck['weights']=='EMA' and ck['step']==9216
    base.load_state_dict(ck['model']); del ck
    refiner=Refiner().cuda().eval()
    refiner.load_state_dict(torch.load(model_dir/'refiner.pt',map_location='cpu',weights_only=False)['model'])
    # Pickle is accepted only after verifying the supplied trusted SHA256.
    state=pickle.loads((model_dir/'histgbdt.pkl').read_bytes())
    detector=YOLO(str(model_dir/'yolo11m.pt'))
    ck=torch.load(model_dir/'gray_refiner.pt',map_location='cpu',weights_only=False)
    gray=PretrainedRefiner(ck['encoder'],ck.get('in_channels',1),ck.get('out_channels',1),False).cuda().eval()
    gray.load_state_dict(ck['model']); gray.crop_size=ck.get('crop_size',256); del ck
    config=read(ROOT/'configs/inference.json')
    per_image=[]
    with target.open('w',newline='',encoding='utf-8') as f, torch.inference_mode():
        writer=csv.writer(f); writer.writerow(['filament_id','segmentation_rle'])
        for n,row in enumerate(rows,1):
            path=test_dir/row['file_name']; raw=raw_image(path)
            # Always regenerate the Base map and both feature variants.
            p=probability(base,raw); core=core_map(p)
            original=original_proposals(core,p); refined=refine_image(refiner,raw,p,core)
            ox,of,oi=proposal_features(raw,p,original); rx,rf,ri=proposal_features(raw,p,refined)
            patches=np.concatenate([ox,rx]); features=np.concatenate([of,rf])
            variants=np.r_[np.zeros(len(oi),np.uint8),np.ones(len(ri),np.uint8)]
            x=spatial_features(patches,features,variants) if len(patches) else np.zeros((0,192),np.float32)
            scores=np.prod(predict(state,x),axis=1)
            accepted={int(cid) for cid,sc in zip(ri,scores[len(oi):]) if sc>=.1}
            base_rows=rles(resolve(refined,raw.shape,accepted))
            # Fresh detector and four-view grayscale crop refinement, no candidate cache.
            candidates=[]
            for score,coarse in _yolo_style_candidates(detector,str(path),1280,.05):
                mask=refine_candidate(gray,'cuda',raw,coarse)
                if mask.any(): candidates.append(dict(score=float(score),rle=mu.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()))
            final=compose(base_rows,candidates,config)
            for i,r in enumerate(final,1): writer.writerow([f'{path.stem}_{i}',r['rle']])
            per_image.append(dict(file_name=path.name,image_sha256=row['sha256'],base_instances=len(base_rows),
                                  raw_supplement_candidates=len(candidates),supplement_instances=len(final)-len(base_rows),instances=len(final)))
            if n%10==0 or n==len(rows): print(json.dumps({'images':n,'total':len(rows),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    report=dict(state='inference_complete',images=per_image,submission_sha256=sha(target),
                instances=sum(r['instances'] for r in per_image),base_instances=sum(r['base_instances'] for r in per_image),
                supplement_instances=sum(r['supplement_instances'] for r in per_image),
                elapsed_seconds=time.monotonic()-started,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                gpu=torch.cuda.get_device_name(0),python=sys.version,
                versions={k:metadata.version(k) for k in ['torch','torchvision','segmentation-models-pytorch','timm','ultralytics','numpy','scipy','scikit-image','scikit-learn','pillow','pycocotools','opencv-python']},
                model_hashes={k:v['sha256'] for k,v in weights.items()},reads_test_labels=False,uses_prediction_caches=False)
    write(output_dir/'inference-manifest.json',report)
    return report


def groups(path):
    result={}
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['filament_id','segmentation_rle']: raise ValueError('Unexpected CSV schema')
        for row in reader:
            stem,n=row['filament_id'].rsplit('_',1); bucket=result.setdefault(stem,[])
            if int(n)!=len(bucket)+1: raise ValueError('Nonconsecutive or duplicate instance ID')
            bucket.append(row['segmentation_rle'])
    return result


def verify(output_dir):
    output_dir=Path(output_dir); manifest=read(output_dir/'inference-manifest.json')
    actual=groups(output_dir/'submission.csv'); expected=groups(ROOT/'verification/reference-submission.csv')
    stems=[Path(r['file_name']).stem for r in manifest['images']]
    if set(actual)-set(stems): raise ValueError('Unexpected image IDs')
    count=0
    for stem in stems:
        masks=actual.get(stem,[])
        if masks!=expected.get(stem,[]): raise AssertionError('Reference RLE mismatch: '+stem)
        occupied=np.zeros((2048,2048),bool)
        for rle in masks:
            mask=mu.decode(dict(size=[2048,2048],counts=rle.encode())).astype(bool)
            if not mask.any() or (mask&occupied).any(): raise AssertionError('Empty or overlapping mask')
            assert mu.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()==rle
            occupied|=mask; count+=1
    full=len(stems)==180
    digest=sha(output_dir/'submission.csv')
    if full: assert digest==read(ROOT/'verification/reference.json')['submission_sha256']
    report=dict(passed=True,full_test_set=full,images=len(stems),instances=count,exact_instance_RLE=True,
                nonempty_nonoverlapping=True,byte_identical_to_reference=full,submission_sha256=digest,
                reference_used_only_after_inference=True)
    write(output_dir/'verification.json',report); print(json.dumps(report,indent=2)); return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if not args.verify_only:
        if args.test_dir is None: parser.error('--test-dir is required for inference')
        run(args.test_dir,args.output_dir,args.limit)
    verify(args.output_dir)
