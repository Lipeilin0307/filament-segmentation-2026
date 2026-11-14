"""Retained Base/OOF/refiner training kernels. Only runtime paths and imports are adapted."""
import copy, gc, hashlib, json, os, random, re, time
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy import ndimage as ndi
from safetensors.torch import load_file
from .data import TrainingCrops as OriginalTrainingCrops, collate, sha256, write_json, load_dataset, image_path, annotation_mask
from .losses import training_loss, expert_loss
from ..base import StrongUnet, Refiner, raw_image, probability, core_map, original_proposals, input_patch, tiles, refine_image
from ..quality import proposal_features
ROOT = DATA_ROOT = PREPARED = INITIAL_ROOT = None
ARMS = {'base_pretrained':'base'}
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)
def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.writing.json'); write_json(tmp,value); os.replace(tmp,path)
def status(stage,**kw): print(json.dumps(dict(stage=stage,**kw)),flush=True)
class TrainingCrops(OriginalTrainingCrops):
    def __init__(self,**kwargs): super().__init__(data=DATA_ROOT,prepared=PREPARED,**kwargs)
def dataset():
    root,_,doc=load_dataset(DATA_ROOT)
    images,anns=defaultdict(list),defaultdict(list)
    for im in doc['images']: images[im['file_name']].append(im)
    for a in doc['annotations']: anns[str(a['image_id'])].append(a)
    return root,images,anns
def grayscale_stem(weight,bias,mean=MEAN,std=STD):
    """Exact RGB-replication/ImageNet normalization for a no-padding stem.

    Existing input x=(gray-.5)/.25. W*(gray-mean)/std+b becomes
    Wgray*x+bgray. Bias absorbs the per-channel normalization offsets.
    """
    if weight.ndim!=4 or weight.shape[1]!=3:raise ValueError('Expected RGB convolution weight')
    means=weight.new_tensor(mean)[None,:,None,None]
    stds=weight.new_tensor(std)[None,:,None,None]
    return .25*(weight/stds).sum(1,keepdim=True),bias+(weight*((.5-means)/stds)).sum((1,2,3))


def initialize(model,arm):
    if arm!='base_pretrained': raise ValueError('Only the retained ImageNet Base recipe is packaged.')
    size=ARMS[arm]; path=INITIAL_ROOT/'pretrained'/size/'model.safetensors'
    manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
    assert sha256(path)==manifest['sha256']
    source=load_file(str(path),device='cpu')
    target=model.net.encoder.model; converted={};used=[]
    for key,value in target.state_dict().items():
        original=re.sub(r'^(stem|stages)_(\d+)\.',r'\1.\2.',key)
        if original not in source:raise KeyError(f'Pretrained parameter missing: {original}')
        converted[key]=source[original].clone();used.append(original)
    stem=target.stem_0
    assert stem.padding==(0,0) and stem.kernel_size==(4,4) and stem.stride==(4,4)
    weight,bias=grayscale_stem(converted['stem_0.weight'],converted['stem_0.bias'])
    converted['stem_0.weight']=weight;converted['stem_0.bias']=bias
    target.load_state_dict(converted,strict=True)
    ignored=sorted(set(source)-set(used))
    assert all(k.startswith('head.') for k in ignored),ignored
    return {'kind':'imagenet1k_supervised_encoder_random_decoder','manifest':manifest,
            'encoder_tensors_loaded':len(converted),'ignored_classifier_keys':ignored,
            'gray_input_adaptation':'Exact replicated-RGB ImageNet-normalized stem; no image resizing.'}


def make_ema(model):
    result=copy.deepcopy(model).eval()
    result.requires_grad_(False)
    return result


@torch.no_grad()
def update_ema(ema,model,step,decay=.999):
    coefficient=min(decay,(1+step)/(10+step))
    source=model.state_dict()
    for key,destination in ema.state_dict().items():
        if destination.is_floating_point():destination.lerp_(source[key],1-coefficient)
        else:destination.copy_(source[key])
    return coefficient

def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("These experiments require CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def save_checkpoint(state, path):
    temporary = path.with_suffix(".writing.pt")
    torch.save(state, temporary)
    temporary.replace(path)

def batches(ds, batch_size):
    """Bounded CPU read/crop prefetch; deterministic index order, no global RNG."""
    def load_batch(start): return collate([ds[i] for i in range(start, start+batch_size)])
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = deque(); starts = iter(range(0, len(ds), batch_size))
        for _ in range(8):
            index = next(starts, None)
            if index is not None: pending.append(executor.submit(load_batch, index))
        while pending:
            result = pending.popleft().result()
            index = next(starts, None)
            if index is not None: pending.append(executor.submit(load_batch, index))
            yield result


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state_all(state['cuda'])

def probability_source(record, refiner_fold):
    """Inner OOF bases exclude both refiner holdout and the image's fold."""
    g=record['fold']
    if refiner_fold=='full':return f'outer-{g}'
    h=int(refiner_fold)
    if g==h:raise ValueError('Held image cannot be a refiner training example')
    return f'single-{next(k for k in range(3) if k not in (h,g))}'

def truths_for(experts, anns, shape):
    return [[annotation_mask(a['segmentation'], shape) for a in anns[str(e['id'])]] for e in experts]


def proposal_target(proposal, truths):
    y, x = proposal['top'], proposal['left']; mask = proposal['mask']; area = int(mask.sum())
    region = np.s_[y:y+mask.shape[0], x:x+mask.shape[1]]
    best = []
    for expert in truths:
        values = []
        for gt in expert:
            intersection = int((gt[region] & mask).sum())
            values.append(intersection / max(1, int(gt.sum()) + area - intersection))
        best.append(max(values, default=0.))
    return [float(np.mean(best)), float(np.mean(np.asarray(best) > .5))], best


def save_proposals(path, proposals):
    arrays = {'coords':np.asarray([[c['id'],c['top'],c['left']] for c in proposals],np.int32).reshape(-1,3)}
    for i,c in enumerate(proposals): arrays['p'+str(i)] = c['probability']; arrays['m'+str(i)] = c['mask']
    np.savez_compressed(path, **arrays)

def training_dataset(spec, name):
    config=spec['base_training']; allowed=set(spec['base_jobs'][name])
    records=[r for r in spec['records'] if r['fold'] in allowed]; files={r['file_name'] for r in records}
    ds=TrainingCrops(size=config['size'],length=config['batch']*config['steps'],seed=config['seed'])
    ds.files=sorted(files);ds.by_file={f:ds.by_file[f] for f in ds.files}
    ids={str(im['id']) for ims in ds.by_file.values() for im in ims}
    ds.annotations=defaultdict(list,{k:v for k,v in ds.annotations.items() if k in ids})
    assert set(ds.by_file)==files and all(r['fold'] in allowed for r in records)
    return ds, records


def train_base(name,spec):
    out=ROOT/'backbones'/name;out.mkdir(parents=True,exist_ok=True);ph=sha256(ROOT/'protocol.json')
    if (out/'summary.json').exists():
        summary=read(out/'summary.json');assert summary['protocol_sha256']==ph and sha256(out/'inference.pt')==summary['checkpoint_sha256'];return
    status('train_oof_backbone',name=name);c=spec['base_training'];setup(c['seed'])
    ds,records=training_dataset(spec,name)
    model=StrongUnet('base_pretrained');initial=initialize(model,'base_pretrained');model.cuda();ema=make_ema(model)
    encoder=[p for n,p in model.named_parameters() if n.startswith('net.encoder.')]
    decoder=[p for n,p in model.named_parameters() if not n.startswith('net.encoder.')]
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':5e-5},{'params':decoder,'lr':2e-4}],weight_decay=c['weight_decay'])
    epoch=0;step=0;history=[]
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert saved['protocol_sha256']==ph and saved['name']==name
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema_model']);optimizer.load_state_dict(saved['optimizer'])
        epoch=saved['epoch'];step=saved['step'];history=saved['history'];restore_rng(saved['rng']);del saved
    write_json(out/'training-files.json',{'records':records,'protocol_sha256':ph})
    for current in range(epoch,c['epochs']):
        ds.epoch=current+c['sample_epoch_offset'];model.train();losses=[];began=time.perf_counter();digest=hashlib.sha256()
        for batch in batches(ds,c['batch']):
            digest.update(batch['pixels'].numpy().tobytes())
            for mask in batch['masks']:digest.update(np.asarray(mask.shape,np.int64).tobytes());digest.update(mask.numpy().tobytes())
            lrs=c['lr_history'][current]['learning_rates']
            if step<c['warmup_updates']:lrs=[5e-5*(step+1)/c['warmup_updates'],2e-4*(step+1)/c['warmup_updates']]
            for group,lr in zip(optimizer.param_groups,lrs):group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=training_loss(model,'convnext_unet',batch,'cuda')
            if not torch.isfinite(loss):raise FloatingPointError('Base adaptation training nonfinite')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();step+=1
            update_ema(ema,model,step,c['ema_decay']);losses.append(float(loss.detach()))
        epoch=current+1;row={'epoch':epoch,'step':step,'loss':float(np.mean(losses)),'seconds':time.perf_counter()-began,'data_sha256':digest.hexdigest()};history.append(row)
        if epoch%4==0 or epoch==c['epochs']:
            save_checkpoint({'name':name,'protocol_sha256':ph,'model':model.state_dict(),'ema_model':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'epoch':epoch,'step':step,'history':history,'rng':rng_state()},out/'last.pt')
        atomic_json(out/'progress.json',{'state':'training','name':name,**row,'max_updates':c['epochs']*c['steps']})
        print(json.dumps({'stage':'oof_backbone','name':name,**row}),flush=True)
    save_checkpoint({'model':ema.state_dict(),'name':name,'weights':'EMA','protocol_sha256':ph,'step':step,'training_files':ds.files},out/'inference.pt')
    summary={'name':name,'protocol_sha256':ph,'checkpoint_sha256':sha256(out/'inference.pt'),'updates':step,'images':len(ds.files),'initialization':initial,'history':history}
    atomic_json(out/'summary.json',summary);atomic_json(out/'progress.json',{'state':'complete','name':name,'updates':step})


def generate_probabilities(name,spec):
    status('oof_probabilities',name=name)
    trained=set(spec['base_jobs'][name]); records=[r for r in spec['records'] if r['fold'] not in trained]
    checkpoint=ROOT/'backbones'/name/'inference.pt';ch=sha256(checkpoint)
    assert ch==read(checkpoint.parent/'summary.json')['checkpoint_sha256']
    setup(spec['seed']);model=StrongUnet('base_pretrained').cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    root,images,_=dataset();out=ROOT/'probabilities'/name;out.mkdir(parents=True,exist_ok=True);manifest=[]
    for number,record in enumerate(records,1):
        filename=record['file_name'];dest=out/(Path(filename).stem+'.npy');side=dest.with_suffix('.json')
        identity={'checkpoint_sha256':ch,'image_sha256':record['sha256'],'protocol_sha256':sha256(ROOT/'protocol.json')}
        if dest.exists() and side.exists():
            meta=read(side);assert all(meta[k]==v for k,v in identity.items()) and sha256(dest)==meta['sha256']
        else:
            source=image_path(root,images[filename][0]);assert sha256(source)==record['sha256']
            p=probability(model,raw_image(source));temporary=dest.with_suffix('.writing.npy');np.save(temporary,p);temporary.replace(dest)
            meta={**record,**identity,'sha256':sha256(dest)};write_json(side,meta)
        manifest.append(meta)
        if number%30==0:print(json.dumps({'stage':'oof_probabilities','name':name,'images':number,'total':len(records)}),flush=True)
    write_json(out/'manifest.json',manifest)
    del model;gc.collect();torch.cuda.empty_cache()

def load_probability(name,filename,spec):
    path=ROOT/'probabilities'/name/(Path(filename).stem+'.npy');meta=read(path.with_suffix('.json'))
    assert meta['protocol_sha256']==sha256(ROOT/'protocol.json') and sha256(path)==meta['sha256']
    assert meta['checkpoint_sha256']==read(ROOT/'backbones'/name/'summary.json')['checkpoint_sha256']
    record=next(r for r in spec['records'] if r['file_name']==filename)
    assert record['fold'] not in spec['base_jobs'][name]
    return np.load(path),meta


def build_data(fold,spec):
    status('build_refiner_data',fold=str(fold));out=ROOT/'refiners'/str(fold);out.mkdir(parents=True,exist_ok=True)
    dest=out/'data.json'
    if dest.exists():
        info=read(dest)
        assert info['protocol_sha256']==sha256(ROOT/'protocol.json')
        for p,h in info['hashes'].items():assert sha256(out/p)==h
        return
    root,images,anns=dataset();records=[r for r in spec['records'] if fold=='full' or r['fold']!=int(fold)];shards=[]
    for number,record in enumerate(records,1):
        filename=record['file_name'];name=probability_source(record,fold);source=ROOT/'probabilities'/name/(Path(filename).stem+'.npy')
        target=out/'shards'/(Path(filename).stem+'.npz');side=target.with_suffix('.json')
        identity={'probability_sha256':sha256(source),'image_sha256':record['sha256'],'protocol_sha256':sha256(ROOT/'protocol.json'),'source_base':name}
        if target.exists() and side.exists():
            meta=read(side);assert all(meta[k]==v for k,v in identity.items()) and sha256(target)==meta['sha256']
        else:
            raw=raw_image(image_path(root,images[filename][0]));p,_=load_probability(name,filename,spec);labels=core_map(p)
            truths=truths_for(images[filename],anns,raw.shape);assert 1<=len(truths)<=3
            xs=[];ys=[];counts=[]
            for cid,box in enumerate(ndi.find_objects(labels),1):
                if box is None:continue
                mask=labels[box]==cid;area=int(mask.sum());chosen=[]
                for expert in truths:
                    overlaps=[int((g[box]&mask).sum()) for g in expert]
                    chosen.append(expert[int(np.argmax(overlaps))] if overlaps and max(overlaps)>=.1*area else None)
                for y,x in tiles(box,raw.shape,limit=4):
                    patch=input_patch(raw,p,labels,cid,y,x)
                    if not patch[2].any():continue
                    target_masks=np.zeros((3,256,256),np.uint8)
                    for ei,truth in enumerate(chosen):
                        if truth is not None:target_masks[ei]=truth[y:y+256,x:x+256]
                    xs.append(patch);ys.append(target_masks);counts.append(len(truths))
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.writing.npz')
            np.savez_compressed(temporary,x=np.asarray(xs,np.uint8).reshape(-1,3,256,256),y=np.asarray(ys,np.uint8).reshape(-1,3,256,256),counts=np.asarray(counts,np.int64))
            temporary.replace(target);meta={**record,**identity,'patches':len(xs),'sha256':sha256(target)};write_json(side,meta)
        shards.append(meta)
        if number%30==0:print(json.dumps({'stage':'refiner_data','fold':str(fold),'images':number,'total':len(records)}),flush=True)
    n=sum(r['patches'] for r in shards);assert n>=8
    arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype=np.uint8,shape=(n,3,256,256)) for k in ['x','y']}
    counts=[];offset=0
    for r in shards:
        with np.load(out/'shards'/(Path(r['file_name']).stem+'.npz')) as d:
            count=len(d['x'])
            for k in arrays:arrays[k][offset:offset+count]=d[k]
            counts.extend(d['counts'].tolist());offset+=count
    for a in arrays.values():a.flush()
    arrays.clear();np.save(out/'counts.npy',np.asarray(counts,np.int64))
    write_json(dest,{'protocol_sha256':sha256(ROOT/'protocol.json'),'images':len(records),'patches':n,'records':shards,
                     'hashes':{p:sha256(out/p) for p in ['x.npy','y.npy','counts.npy']}})


def train_refiner(fold,spec):
    out=ROOT/'refiners'/str(fold);ph=sha256(ROOT/'protocol.json')
    if (out/'summary.json').exists():
        s=read(out/'summary.json');assert s['protocol_sha256']==ph and sha256(out/'inference.pt')==s['checkpoint_sha256'];return
    status('train_refiner',fold=str(fold));c=spec['refiner_training'];setup(c['seed'])
    x=np.load(out/'x.npy',mmap_mode='r');y=np.load(out/'y.npy',mmap_mode='r');counts=np.load(out/'counts.npy')
    model=Refiner().cuda();opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']);epoch=0;history=[]
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['protocol_sha256']==ph
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);epoch=state['epoch'];history=state['history']
        rng.bit_generator.state=state['sample_rng'];restore_rng(state['rng']);del state
    for current in range(epoch,c['epochs']):
        model.train();losses=[];began=time.perf_counter()
        for _ in range(c['steps']):
            ids=rng.choice(len(x),c['batch'],replace=False);k=int(rng.integers(4))
            bx=np.rot90(np.array(x[ids]),k,axes=(-2,-1));by=np.rot90(np.array(y[ids]),k,axes=(-2,-1))
            if rng.random()<.5:bx=bx[:,:,:,::-1];by=by[:,:,:,::-1]
            bx=torch.from_numpy(bx.copy()).cuda().float()/255;by=torch.from_numpy(by.copy()).cuda().float();bc=torch.from_numpy(counts[ids]).cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=expert_loss(model(bx).float(),by,bc)
            if not torch.isfinite(loss):raise FloatingPointError('Adapted refiner nonfinite')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        epoch=current+1;row={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.perf_counter()-began};history.append(row)
        save_checkpoint({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,
                         'rng':rng_state(),'sample_rng':rng.bit_generator.state,'protocol_sha256':ph},out/'last.pt')
        atomic_json(out/'progress.json',{'state':'training','fold':str(fold),**row,'max_epochs':c['epochs']})
        print(json.dumps({'stage':'refiner_training','fold':str(fold),**row}),flush=True)
    save_checkpoint({'model':model.state_dict(),'epochs':epoch,'protocol_sha256':ph,'training_records':read(out/'data.json')['records']},out/'inference.pt')
    atomic_json(out/'summary.json',{'checkpoint_sha256':sha256(out/'inference.pt'),'protocol_sha256':ph,'history':history})
    atomic_json(out/'progress.json',{'state':'complete','fold':str(fold),'epochs':epoch})

def build_quality_data(fold,spec):
    if fold not in (0,1,2): raise ValueError('Only training-side excluded-fold quality pools are exposed.')
    outer=fold=='validation';name='validation' if outer else f'oof-{fold}';status('quality_data',cohort=name)
    out=ROOT/'quality-data'/name;out.mkdir(parents=True,exist_ok=True)
    ph=sha256(ROOT/'protocol.json');root,images,anns=dataset();setup(spec['seed'])
    if outer:
        files=sorted({r['file_name'] for r in read('data/processed/folds.json') if r['split']=='val'})
        dates={r['file_name']:r['date'] for r in read('data/processed/folds.json')}
        records=[{'file_name':f,'date':dates[f],'fold':-1} for f in files]
        checkpoint=Path(spec['base_checkpoint']);base=StrongUnet('base_pretrained').cuda().eval()
        base.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);refiner_fold='full'
    else:
        records=[r for r in spec['records'] if r['fold']==int(fold)];refiner_fold=str(fold)
        checkpoint=ROOT/'backbones'/f'outer-{fold}'/'inference.pt';base=None
    refiner_checkpoint=ROOT/'refiners'/refiner_fold/'inference.pt'
    refiner=Refiner().cuda().eval();refiner.load_state_dict(torch.load(refiner_checkpoint,map_location='cpu',weights_only=False)['model'])
    identity={'protocol_sha256':ph,'base_sha256':sha256(checkpoint),'refiner_sha256':sha256(refiner_checkpoint)}
    manifest=[]
    for number,record in enumerate(records,1):
        filename=record['file_name'];stem=Path(filename).stem;dest=out/(stem+'.npz');side=dest.with_suffix('.json')
        if dest.exists() and side.exists():
            meta=read(side);assert all(meta[k]==v for k,v in identity.items()) and sha256(dest)==meta['sha256']
            if outer:
                for variant in ('original','refined'):assert sha256(out/(stem+f'.{variant}.npz'))==meta[variant+'_sha256']
        else:
            raw=raw_image(image_path(root,images[filename][0]))
            p=probability(base,raw) if outer else load_probability(f'outer-{fold}',filename,spec)[0]
            core=core_map(p);original=original_proposals(core,p);refined=refine_image(refiner,raw,p,core)
            truths=truths_for(images[filename],anns,raw.shape)
            patches=[];features=[];targets=[];ids=[];variants=[]
            for variant,proposals in enumerate((original,refined)):
                bx,bf,bi=proposal_features(raw,p,proposals);lookup={c['id']:c for c in proposals}
                patches.append(bx);features.append(bf);ids.append(bi);variants.extend([variant]*len(bi))
                targets.append(np.asarray([proposal_target(lookup[int(cid)],truths)[0] for cid in bi],np.float32).reshape(-1,2))
            temporary=dest.with_suffix('.writing.npz')
            np.savez_compressed(temporary,patches=np.concatenate(patches),features=np.concatenate(features),targets=np.concatenate(targets),
                ids=np.concatenate(ids),variants=np.asarray(variants,np.uint8));temporary.replace(dest)
            meta={**record,**identity,'candidates':sum(map(len,ids)),'sha256':sha256(dest)}
            if outer:
                for variant,proposals in [('original',original),('refined',refined)]:
                    path=out/(stem+f'.{variant}.npz');save_proposals(path,proposals);meta[variant+'_sha256']=sha256(path)
            write_json(side,meta)
        manifest.append(meta)
        if number%25==0:print(json.dumps({'stage':'quality_data','cohort':name,'images':number,'total':len(records)}),flush=True)
    write_json(out/'manifest.json',{'records':manifest,**identity})
    del refiner,base;gc.collect();torch.cuda.empty_cache()
