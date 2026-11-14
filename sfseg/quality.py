import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from PIL import Image
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize
FEATURES=['log_area','mean_p','q10_p','q90_p','max_p','mean_gray','std_gray','mean_darkness','median_darkness',
          'elongation','bbox_fill','perimeter_per_sqrt_area','log_skeleton_length','branch_cluster_fraction','log_aspect','radial_fraction','width_mean']
def tubed_skeleton(mask, radius=2):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or radius<0:
        raise ValueError('Expected 2D mask and nonnegative radius')
    result=skeletonize(np.ascontiguousarray(mask))
    if radius:
        result=ndi.binary_dilation(result,structure=ndi.generate_binary_structure(2,1),iterations=radius)
    return result & mask
def patch_features(raw,probability,darkness,labels,identity,box):
    y0,y1=box[0].start,box[0].stop; x0,x1=box[1].start,box[1].stop
    mask=labels[box]==identity; p=probability[box][mask]; gray=raw[box][mask].astype(float); dark=darkness[box][mask]
    yy,xx=np.where(mask); area=len(yy); sk=tubed_skeleton(mask,0)
    d=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
    branches=ndi.label(sk&(d>=3),np.ones((3,3)))[1]
    eig=np.linalg.eigvalsh(np.cov(np.stack([yy,xx]))) if area>1 else np.ones(2)
    elongation=np.sqrt((eig[-1]+1)/(eig[0]+1)); perimeter=(mask&~ndi.binary_erosion(mask)).sum()
    features=np.array([np.log1p(area),p.mean(),np.quantile(p,.1),np.quantile(p,.9),p.max(),gray.mean()/255,gray.std()/64,
       dark.mean()/32,np.median(dark)/32,min(elongation,100),area/mask.size,perimeter/np.sqrt(area),np.log1p(sk.sum()),
       branches/max(1,int(sk.sum())),np.log(max(y1-y0,x1-x0)/max(1,min(y1-y0,x1-x0))),
       np.hypot(yy.mean()+y0-1024,xx.mean()+x0-1024)/1024,(2*ndi.distance_transform_edt(mask)[sk]).mean() if sk.any() else 0],np.float32)
    size=min(raw.shape[0],max(64,max(y1-y0,x1-x0)+64)); left=max(0,min(raw.shape[1]-size,(x0+x1-size)//2)); top=max(0,min(raw.shape[0]-size,(y0+y1-size)//2))
    region=(slice(top,top+size),slice(left,left+size))
    channels=[raw[region],(probability[region]*255).clip(0,255).astype(np.uint8),((labels[region]==identity)*255).astype(np.uint8)]
    patch=np.stack([np.asarray(Image.fromarray(c).resize((96,96),Image.Resampling.NEAREST if i==2 else Image.Resampling.BILINEAR)) for i,c in enumerate(channels)])
    return patch,features
def proposal_features(raw, p, proposals):
    darkness = ndi.uniform_filter(raw.astype(np.float32),65,mode='reflect') - raw.astype(np.float32)
    labels = np.zeros(raw.shape, np.uint8); patches, features, ids = [], [], []
    for c in proposals:
        mask = c['mask']
        if not mask.any(): continue
        y, x = c['top'], c['left']; yy, xx = np.where(mask)
        region = np.s_[y:y+mask.shape[0], x:x+mask.shape[1]]; labels[region] = mask
        box = (slice(y+int(yy.min()),y+int(yy.max())+1),slice(x+int(xx.min()),x+int(xx.max())+1))
        patch, feature = patch_features(raw,p,darkness,labels,1,box)
        patches.append(patch); features.append(feature); ids.append(c['id']); labels[region] = 0
    return (np.asarray(patches,np.uint8).reshape(-1,3,96,96),
            np.asarray(features,np.float32).reshape(-1,len(FEATURES)), np.asarray(ids,np.int32))
def spatial_features(patches, features, variants):
    """Original 17 + variant + spatial pyramids + mask/ring appearance summaries."""
    assert patches.ndim == 4 and patches.shape[1:] == (3, 96, 96)
    x = patches.astype(np.float32) / 255.
    blocks = [features.astype(np.float32), np.asarray(variants, np.float32)[:, None]]
    for size in (1, 2, 4):
        cells = x.reshape(len(x), 3, size, 96//size, size, 96//size)
        blocks += [cells.mean((3, 5)).reshape(len(x), -1), cells.std((3, 5)).reshape(len(x), -1)]
    regions = []
    for z in x:
        mask = z[2] > .5
        ring = ndi.binary_dilation(mask, iterations=8) & ~ndi.binary_dilation(mask, iterations=2)
        edge = mask & ~ndi.binary_erosion(mask)
        gradient = np.hypot(ndi.sobel(z[0], axis=0), ndi.sobel(z[0], axis=1))
        row = []
        for region in (mask, ring, edge):
            row.append(float(region.mean()))
            for channel in (z[0], z[1], gradient):
                values = channel[region]
                row.extend([float(values.mean()), float(values.std()), *np.quantile(values, [.1, .5, .9]).tolist()] if len(values) else [0.]*5)
        regions.append(row)
    blocks.append(np.asarray(regions, np.float32).reshape(len(x), 48))
    result = np.concatenate(blocks, axis=1)
    assert np.isfinite(result).all()
    return result
def predict(state, features):
    if not len(features):
        return np.zeros((0, 2), np.float32)
    y = np.column_stack([m.predict(features) for m in state['models']])
    assert np.isfinite(y).all()
    return np.clip(y, 0, 1).astype(np.float32)
