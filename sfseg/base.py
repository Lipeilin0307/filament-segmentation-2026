import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from PIL import Image
import segmentation_models_pytorch as smp
ARMS={"base_pretrained":"base"}
SIDE=256
MARGIN=64

class StrongUnet(nn.Module):
    def __init__(self,arm):
        super().__init__()
        size=ARMS[arm]
        self.net=smp.Unet('tu-convnext_'+size,encoder_weights=None,in_channels=1,classes=1,
                          decoder_channels=(256,128,64,32,16))
        # Common training-stability setting: longer running-stat averaging.
        for module in self.net.decoder.modules():
            if isinstance(module,nn.BatchNorm2d):module.momentum=.01

    def forward(self,x):return self.net(x)
def blocks(a, b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0=blocks(3,16); self.e1=blocks(16,32); self.e2=blocks(32,64); self.e3=blocks(64,128)
        self.d2=blocks(192,64); self.d1=blocks(96,32); self.d0=blocks(48,16)
        self.head=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self,x):
        a=self.e0(x); b=self.e1(F.avg_pool2d(a,2)); c=self.e2(F.avg_pool2d(b,2)); d=self.e3(F.avg_pool2d(c,2))
        d=self.d2(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        d=self.d1(torch.cat([F.interpolate(d,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        d=self.d0(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.head(d)+(x[:,2:3]*2-1)*2.1972245773362196

def tiles(box,shape,limit=None):
    starts=[]
    for s,dim in zip(box,shape):
        lo=max(0,s.start-MARGIN); hi=min(dim,s.stop+MARGIN)
        if hi-lo<=SIDE:
            axis=[max(0,min(dim-SIDE,(lo+hi-SIDE)//2))]
        else:
            axis=list(range(lo,hi-SIDE+1,SIDE//2))
            if axis[-1]!=hi-SIDE: axis.append(hi-SIDE)
        starts.append(axis)
    result=[(y,x) for y in starts[0] for x in starts[1]]
    if limit and len(result)>limit:
        result=[result[i] for i in np.linspace(0,len(result)-1,limit).round().astype(int)]
    return result

def input_patch(raw,probability,labels,identity,y,x):
    region=np.s_[y:y+SIDE,x:x+SIDE]
    p=(probability[region]*255).clip(0,255).astype(np.uint8)
    return np.stack([raw[region],p,(labels[region]==identity).astype(np.uint8)*255])

def attached_mask(probability,seed):
    """Keep predicted components touching original seed and within search radius."""
    candidate=(probability>=.5)&(ndi.distance_transform_edt(~seed)<=MARGIN)
    labels,_=ndi.label(candidate,np.ones((3,3)))
    ids=np.unique(labels[seed]); ids=ids[ids>0]
    return np.isin(labels,ids) if len(ids) else np.zeros_like(seed)

def resolve(candidates,shape,accepted=None):
    """Pixel-exclusive instances. Later scoring can reclaim rejected overlaps."""
    output=np.zeros(shape,np.int32); strength=np.zeros(shape,np.float32)
    for c in candidates:
        if accepted is not None and c['id'] not in accepted: continue
        y,x=c['top'],c['left']; p=c['probability']; mask=c['mask']
        region=np.s_[y:y+p.shape[0],x:x+p.shape[1]]
        take=mask&(p>strength[region]); output[region][take]=c['id']; strength[region][take]=p[take]
    return output

@torch.inference_mode()
def refine_image(model,raw,probability,labels):
    candidates=[]
    feather=np.maximum(.05,np.outer(np.hanning(SIDE),np.hanning(SIDE))).astype(np.float32)
    for identity,box in enumerate(ndi.find_objects(labels),1):
        if box is None: continue
        locations=tiles(box,labels.shape)
        y0=min(y for y,x in locations); x0=min(x for y,x in locations)
        y1=max(y for y,x in locations)+SIDE; x1=max(x for y,x in locations)+SIDE
        sums=np.zeros((y1-y0,x1-x0),np.float32); weights=np.zeros_like(sums)
        for start in range(0,len(locations),16):
            batch=locations[start:start+16]
            patches=np.stack([input_patch(raw,probability,labels,identity,y,x) for y,x in batch])
            inputs=torch.from_numpy(patches).cuda().float()/255
            with torch.autocast('cuda',dtype=torch.bfloat16): pred=model(inputs).float().sigmoid()[:,0].cpu().numpy()
            for (y,x),p in zip(batch,pred):
                region=np.s_[y-y0:y-y0+SIDE,x-x0:x-x0+SIDE]
                sums[region]+=p*feather; weights[region]+=feather
        predicted=sums/np.maximum(weights,1e-8)
        seed=labels[y0:y1,x0:x1]==identity
        mask=attached_mask(predicted,seed)
        candidates.append({'id':identity,'top':y0,'left':x0,'probability':predicted,'mask':mask})
    return candidates
def raw_image(path):
    with Image.open(path) as im:
        raw = np.asarray(im.convert('L'))
    assert raw.shape == (2048, 2048)
    return raw

@torch.inference_mode()
def probability(model, raw):
    x = torch.from_numpy((raw.astype(np.float32)[None, None] / 255 - .5) / .25).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x).float().sigmoid()[0, 0].cpu().numpy()
def core_map(p):
    labels, _ = ndi.label(p >= .35, np.ones((3, 3)))
    areas = np.bincount(labels.ravel())
    peak = ndi.maximum(p, labels, np.arange(len(areas)))
    ids = np.flatnonzero((areas >= 50) & (peak >= .75)); ids = ids[ids != 0]
    lookup = np.zeros(len(areas), np.int32); lookup[ids] = np.arange(1, len(ids) + 1)
    return lookup[labels]
def original_proposals(labels,p):
    return [{'id':cid,'top':box[0].start,'left':box[1].start,'probability':p[box].copy(),'mask':labels[box]==cid}
            for cid,box in enumerate(ndi.find_objects(labels),1) if box is not None]
