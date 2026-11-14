import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from PIL import Image
import segmentation_models_pytorch as smp
import pycocotools.mask as mu
H=W=2048
CONTEXT=1.8
MIN_CROP=96
REFINER_THRESHOLD=.5
TTA_AREA_RATIO_MIN=.60
TTA_AREA_RATIO_MAX=1.70
SOURCES=("B1280",)
class PretrainedRefiner(nn.Module):
    """smp U-Net with an ImageNet-pretrained encoder (refiner v10); same I/O as
    RefinerUNet: 1-channel [0,1] crop in, mask (+ spine) logits out."""

    def __init__(self, encoder="resnet34", in_channels=1, out_channels=1, pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.in_channels, self.out_channels, self.encoder = in_channels, out_channels, encoder
        self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                            in_channels=in_channels, classes=out_channels)

    def forward(self, x):
        return self.net((x - 0.45) / 0.225)  # ImageNet gray mean/std
def square_bounds(mask: np.ndarray, context: float = CONTEXT, minimum: int = MIN_CROP):
    """Square crop box around a mask's bounding box, padded and clamped to image bounds."""
    height, width = mask.shape
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height
    return x0, y0, x1, y1
def refine_with_tta(refiner, device, x: np.ndarray) -> np.ndarray:
    """x: (C,256,256) float32 in [0,1] -- gray crop, plus the detector's coarse
    mask as a hint channel for the 2-channel refiner. Returns a 256x256
    probability map."""
    views = {
        "identity": x,
        "hflip": x[:, :, ::-1],
        "vflip": x[:, ::-1, :],
        "hvflip": x[:, ::-1, ::-1],
    }
    probs = {}
    for name, view in views.items():
        x_t = torch.from_numpy(np.ascontiguousarray(view)).float().unsqueeze(0)
        p = torch.sigmoid(refiner(x_t.to(device)))[0, 0].cpu().numpy()
        if name == "hflip":
            p = np.fliplr(p)
        elif name == "vflip":
            p = np.flipud(p)
        elif name == "hvflip":
            p = np.flipud(np.fliplr(p))
        probs[name] = p

    identity_area = (probs["identity"] > REFINER_THRESHOLD).sum()
    avg_prob = np.mean(list(probs.values()), axis=0)
    if identity_area == 0:
        return avg_prob

    avg_area = (avg_prob > REFINER_THRESHOLD).sum()
    ratio = avg_area / identity_area
    if ratio < TTA_AREA_RATIO_MIN or ratio > TTA_AREA_RATIO_MAX:
        return probs["identity"]
    return avg_prob
@torch.no_grad()
def refine_candidate(refiner, device, gray, coarse):
    x0, y0, x1, y1 = square_bounds(coarse.astype(bool))
    crop = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
        (refiner.crop_size, refiner.crop_size), Image.BILINEAR)).astype(np.float32) / 255.0
    channels = [crop]
    if refiner.in_channels == 2:  # hint refiner (v8): the proposal itself, same frame
        channels.append((np.array(Image.fromarray(coarse[y0:y1, x0:x1].astype(np.uint8) * 255).resize(
            (refiner.crop_size, refiner.crop_size), Image.NEAREST)) > 127).astype(np.float32))
    prob = refine_with_tta(refiner, device, np.stack(channels))
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    refined_crop = (prob_full > 0.5).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = refined_crop
    return full_mask

def _yolo_style_candidates(model, img_path, imgsz, floor):
    src = img_path if isinstance(img_path, np.ndarray) else str(img_path)
    out = model.predict(source=src, imgsz=imgsz, conf=floor, verbose=False)[0]
    cands = []
    if out.boxes is not None and len(out.boxes) > 0:
        boxes = out.boxes.xyxy.cpu().numpy()
        scores = out.boxes.conf.cpu().numpy()
        for j in range(len(boxes)):
            x0, y0, x1, y1 = boxes[j]
            x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            coarse = np.zeros((H, W), dtype=np.uint8)
            coarse[y0:y1, x0:x1] = 1
            cands.append((float(scores[j]), coarse))
    return cands
def enc(s):
    return dict(size=[2048, 2048], counts=s.encode())

def select(cache, maps, sources):
    """Keep source order/ties, whole-mask NMS, then original panoptic paint."""
    pool = []
    for source in SOURCES:
        if source not in sources:
            continue
        cal = maps[source]
        for i, row in enumerate(cache[source]):
            score = float(np.interp(row['score'], cal['x'], cal['y']))
            if score >= .5:
                pool.append(dict(source=source, index=i, score=score, rle=row['rle']))
    pool.sort(key=lambda row: -row['score'])  # stable original source/candidate ties
    rs = [enc(p['rle']) for p in pool]
    iou = mu.iou(rs, rs, [0]*len(rs)) if rs else np.empty((0, 0))
    chosen = []
    for i, p in enumerate(pool):
        blockers = [j for j in chosen if iou[i, j] > .05]
        p['blockers'] = [[pool[j]['source'], pool[j]['index']] for j in blockers]
        p['nms_kept'] = not blockers
        p['paint_kept'] = False
        if not blockers:
            chosen.append(i)
    occupied = np.zeros((2048, 2048), bool)
    final = []
    for i in chosen:
        p = pool[i]
        mask = mu.decode(rs[i]).astype(bool) & ~occupied
        if int(mask.sum()) < 20:
            continue
        occupied |= mask
        p['paint_kept'] = True
        final.append(dict(source=p['source'], index=p['index'], score=p['score'],
                          rle=mu.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()))
    return final, pool
def add_protected(base, yolo):
    """Keep every Base mask. Append calibrated/NMS-selected YOLO if IoU<=.05."""
    bs = [enc(p['rle']) for p in base]
    ys = [enc(p['rle']) for p in yolo]
    overlap = mu.iou(ys, bs, [0]*len(bs)) if ys and bs else np.zeros((len(ys), len(bs)))
    occupied = np.zeros((2048, 2048), bool)
    for b in bs:
        occupied |= mu.decode(b).astype(bool)
    output = [dict(p) for p in base]; decisions = []
    for j, p in enumerate(yolo):
        blockers = np.flatnonzero(overlap[j] > .05).astype(int).tolist()
        row = dict(yolo_index=p['index'], blockers=[base[i]['index'] for i in blockers],
                   blocker_positions=blockers, original_rle=p['rle'])
        if blockers:
            row['status'] = 'base_nms_blocked'
        else:
            mask = mu.decode(ys[j]).astype(bool) & ~occupied
            if int(mask.sum()) < 20:
                row['status'] = 'remaining_area_lt20'
            else:
                row['status'] = 'added'
                row['painted_rle'] = mu.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()
                occupied |= mask
                output.append(dict(p, rle=row['painted_rle']))
        decisions.append(row)
    assert output[:len(base)] == base
    return output, decisions

def compose(base, raw, config):
    yolo, _ = select({'B1280': raw}, {'B1280': config['source_map']}, SOURCES)
    full, _ = add_protected(base, yolo)
    c = config['conditional_map']
    return list(base) + [p for p in full[len(base):]
        if float(np.interp(raw[p['index']]['score'], c['x'], c['y'])) >= .5]
