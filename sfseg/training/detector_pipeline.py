"""Retained A451 YOLO/R34 training definitions; paths are isolated by the public runner."""
import gc, hashlib, json, os, random, shutil, sys, time
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from . import detector_crops
from .data import load_dataset
from ..supplement import PretrainedRefiner
ROOT=DATA_ROOT=None
def environment():
    torch.set_num_threads(4)
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'settings')
    os.environ['ULTRALYTICS_AUTOINSTALL']='false'
    return torch
def index(): return load_dataset(DATA_ROOT)
def _load_coco():
    _,_,doc=index(); anns=defaultdict(list)
    for a in doc['annotations']: anns[a['image_id']].append(a)
    return doc,anns
def modules():
    return SimpleNamespace(_load_coco=_load_coco),None,detector_crops,sys.modules[__name__],None
def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers monitor append-only logs, not this file, while it is being written.
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    os.replace(tmp, path)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def status(stage, **kw):
    row = dict(stage=stage, updated_local=now(), pid=os.getpid(), **kw)
    write(ROOT / 'runtime.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def set_seed(seed=0):
    import numpy as np
    torch = environment()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gray_refiner(initialize=True):
    """Same SMP ImageNet encoder as upstream, loaded explicitly and offline.

    SMP's legacy ImageNet URL is333f7ec4, distinct from the newer torchvision
    b627a593 weights used in a previous adaptation experiment.
    """
    torch=environment();_,_,_,ref,_=modules()
    model=ref.PretrainedRefiner('resnet34',1,2,False)
    if initialize:
        path=ROOT/'pretrained/resnet34-333f7ec4.pth'
        assert sha(path)=='333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'
        # Official, hash-verified legacy PyTorch tar format predates weights_only.
        source=torch.load(path,map_location='cpu',weights_only=False)
        source={k:v for k,v in source.items() if not k.startswith('fc.')}
        source['conv1.weight']=source['conv1.weight'].sum(1,keepdim=True)
        model.net.encoder.load_state_dict(source,strict=True)
    return model


def state_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    import numpy as np
    import torch
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state_all(s['cuda'])


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing.pt')
    torch.save(value, tmp)
    os.replace(tmp, path)


def entries_for(spec, role):
    _, _, doc = index()
    files = {r['file_name'] for r in spec['reference_records'] if (r['role'] == 'train') == (role == 'train')}
    return [e for e in doc['images'] if e['file_name'] in files]


def build_yolo_data(spec):
    """One sample per expert, exactly the source polygon/box label conversion."""
    _, _, doc = index()
    anns = defaultdict(list)
    for a in doc['annotations']:
        anns[a['image_id']].append(a)
    folders = {}
    for box_only in [False, True]:
        d = ROOT / 'reference/data' / ('boxes' if box_only else 'segments')
        for split, role in [('train', 'train'), ('val', 'val')]:
            image_dir, label_dir = d / 'images' / split, d / 'labels' / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for e in entries_for(spec, role):
                stem = str(e['id']).replace('/', '_').replace('\\', '_')
                src = Path(spec['data_root']) / 'train/train_images' / e['file_name']
                dest = image_dir / (stem + '.jpeg')
                if not dest.exists():
                    shutil.copy2(src, dest)  # upstream also copies; never rewrite official originals
                assert sha(dest) == next(r['sha256'] for r in spec['reference_records'] if r['file_name'] == e['file_name'])
                lines = []
                for a in anns[e['id']]:
                    for poly in a['segmentation']:
                        if len(poly) < 6:
                            continue
                        if box_only:
                            xs, ys = poly[0::2], poly[1::2]
                            vals = [(min(xs)+max(xs))/4096, (min(ys)+max(ys))/4096,
                                    (max(xs)-min(xs))/2048, (max(ys)-min(ys))/2048]
                        else:
                            vals = [v/2048 for v in poly]
                        lines.append('0 ' + ' '.join(f'{v:.6f}' for v in vals))
                (label_dir / (stem + '.txt')).write_text('\n'.join(lines), 'utf-8')
        yaml = d / 'data.yaml'
        yaml.write_text('path: ' + json.dumps(str(d.resolve())) + '\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n', 'utf-8')
        folders['C' if box_only else 'segment'] = str(yaml)
    write(ROOT / 'reference/data-manifest.json', dict(protocol_sha256=sha(ROOT / 'protocol.json'),
        expert_sample_counts=spec['reference_counts'], datasets=folders,
        label_hashes={str(p): sha(p) for p in sorted((ROOT / 'reference/data').rglob('*.txt'))}))
    return folders


def ultra_args(spec, component, data, smoke=False):
    r = spec['recipes'][component]
    return dict(model=spec['pretrained'][component]['path'], data=data,
                epochs=r['epochs'], batch=r['batch'], imgsz=r['imgsz'], cls=r['cls'],
                patience=r['patience'], seed=0, deterministic=True, amp=True,
                workers=0 if smoke else 2, device=0, plots=False, save=True,
                project=str(ROOT / ('smoke/ultralytics' if smoke else 'reference/models')),
                name=component, exist_ok=False, verbose=False)


def train_torch(spec, component):
    if component!='R': raise ValueError('Only the retained grayscale R34 refiner is exposed.')
    torch = environment(); set_seed()
    ds, tr, crop, ref, _ = modules()
    _, anns = ds._load_coco()
    out = ROOT / 'reference/models' / component
    done = out / 'summary.json'
    if done.exists():
        s = read(done); assert sha(s['selected']) == s['selected_sha256']; return
    out.mkdir(parents=True, exist_ok=True)
    r = spec['recipes'][component]
    datasets = []
    for role in ['train', 'val']:
        cache = ROOT / 'reference/crops'
        prefix = cache / role
        ready = prefix.with_suffix('.json')
        if not ready.exists():
            paths = crop.build_spine_crop_cache(entries_for(spec, role), anns, cache, role)
            write(ready, dict(paths=[str(p) for p in paths[:3]], count=paths[3],
                              protocol_sha256=sha(ROOT/'protocol.json'),
                              files=sorted({e['file_name'] for e in entries_for(spec, role)}),
                              hashes={str(p):sha(p) for p in paths[:3]}))
        c = read(ready)
        assert c['protocol_sha256'] == sha(ROOT/'protocol.json')
        datasets.append(crop.SpineCropDataset(*map(Path, c['paths']), augment=role=='train'))
    loader = torch.utils.data.DataLoader(datasets[0], batch_size=16, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(datasets[1], batch_size=16, shuffle=False, num_workers=0)
    model = gray_refiner(True).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    criterion = ref.MaskSpineLoss(); scaler = None
    history, start, best = [], 0, 1e99
    last = out/'last.pt'
    if last.exists():
        s = torch.load(last, map_location='cpu', weights_only=False)
        assert s['protocol_sha256'] == sha(ROOT/'protocol.json')
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        if scaler is not None: scaler.load_state_dict(s['scaler'])
        restore_rng(s['rng']); history=s['history']; start=s['epoch']+1; best=s['best']
    for epoch in range(start, r['epochs']):
        model.train(); started=time.perf_counter(); losses=[]
        status('training_reference', state='running', component=component, epoch=epoch+1)
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            img, target=batch
            loss=criterion(model(img.cuda()),target.cuda()); assert torch.isfinite(loss)
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        value=ref.evaluate(model,val_loader,criterion,'cuda') if val_loader else None
        improved=component=='A' or value<best
        if improved: best=value if value is not None else best
        history.append(dict(epoch=epoch+1, loss=sum(losses)/len(losses), selection_loss=value, seconds=time.perf_counter()-started))
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict() if scaler else None,rng=state_rng(),epoch=epoch,history=history,best=best,
                     protocol_sha256=sha(ROOT/'protocol.json'),min_size=2048,max_size=2048,
                     in_channels=1,out_channels=2,encoder='resnet34' if component=='R' else None,crop_size=256)
        save_torch(last,payload)
        if improved:save_torch(out/'inference.pt',dict(payload,optimizer=None,scheduler=None,rng=None))
        print(json.dumps(history[-1]),flush=True)
    selected=out/'inference.pt'
    write(done,dict(component=component,protocol_sha256=sha(ROOT/'protocol.json'),history=history,
        selected=str(selected),selected_sha256=sha(selected)))
    del model,opt;gc.collect();torch.cuda.empty_cache()

class DiceBCELoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce_w, self.dice_w = bce_w, dice_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum((2, 3))
        union = probs.sum((2, 3)) + targets.sum((2, 3))
        dice = 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return self.bce_w * bce + self.dice_w * dice


class MaskSpineLoss(nn.Module):
    """logits: (B,2,H,W) -- channel 0 predicts the mask (DiceBCE, the real
    target), channel 1 predicts the GT spine centerline (plain BCE, purely
    auxiliary -- ignored at inference). The spine term only has to nudge
    the shared encoder/decoder features toward the true centerline; it
    never has to stand on its own the way the mask does."""

    def __init__(self, spine_w=0.3):
        super().__init__()
        self.mask_loss = DiceBCELoss()
        self.spine_bce = nn.BCEWithLogitsLoss()
        self.spine_w = spine_w

    def forward(self, logits, targets):
        mask_logits, spine_logits = logits[:, 0:1], logits[:, 1:2]
        mask_t, spine_t = targets[:, 0:1], targets[:, 1:2]
        return (1 - self.spine_w) * self.mask_loss(mask_logits, mask_t) + \
            self.spine_w * self.spine_bce(spine_logits, spine_t)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        losses.append(criterion(model(img), mask).item())
    return sum(losses) / len(losses)
