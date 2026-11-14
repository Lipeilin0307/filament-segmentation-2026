import torch
from torch.nn import functional as F
def erode_cross(values):
    vertical = -F.max_pool2d(-values, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-values, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def skeleton(values, iterations=10):
    current = values
    result = torch.zeros_like(values)
    for _ in range(iterations + 1):
        eroded = erode_cross(current)
        opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
        ridge = (current - opened).relu()
        result = result + (ridge - result * ridge).relu()
        current = eroded
    return result


def contour_dice_loss(probability, target):
    def contour(values):
        # Inner one-pixel contour, excluding artificial borders outside the crop.
        return values + F.max_pool2d(-values, 3, stride=1, padding=1)
    predicted, truth = contour(probability), contour(target)
    axes = (1, 2, 3)
    overlap = (predicted * truth).sum(axes)
    return (1 - (2 * overlap + 1) / (predicted.sum(axes) + truth.sum(axes) + 1)).mean()


def soft_cldice_loss(probability, target, iterations=10):
    axes = (1, 2, 3)
    nonempty = target.sum(axes) > 0
    # Empty crops are supervised by the base BCE/Dice and contour loss.
    predicted_skeleton = skeleton(probability, iterations)
    with torch.no_grad():
        target_skeleton = skeleton(target, iterations)
    precision = ((predicted_skeleton * target).sum(axes) + 1) / (predicted_skeleton.sum(axes) + 1)
    recall = ((target_skeleton * probability).sum(axes) + 1) / (target_skeleton.sum(axes) + 1)
    loss = 1 - 2 * precision * recall / (precision + recall).clamp_min(1e-7)
    return (loss * nonempty).sum() / nonempty.sum().clamp_min(1)


def foreground_loss(logits, target, settings=None):
    settings = settings or {}
    unknown = set(settings) - {'contour_weight', 'cldice_weight', 'skeleton_iterations'}
    if unknown:
        raise ValueError(f'Unknown loss settings: {unknown}')
    cw, sw = settings.get('contour_weight', 0.), settings.get('cldice_weight', 0.)
    iterations = settings.get('skeleton_iterations', 10)
    if cw < 0 or sw < 0 or not isinstance(iterations, int) or iterations < 0:
        raise ValueError('Loss weights and skeleton iterations must be nonnegative')
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(5.))
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice = 1 - ((2*intersection + 1) / (probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)).mean()
    loss = bce + dice
    if cw:
        loss = loss + cw * contour_dice_loss(probability, target)
    if sw:
        loss = loss + sw * soft_cldice_loss(probability, target, iterations)
    return loss

def training_loss(model, kind, batch, device, loss_settings=None):
    pixels = batch["pixels"].to(device)
    masks = [m.to(device) for m in batch["masks"]]
    if kind == "mask2former":
        result = model(pixel_values=pixels, mask_labels=masks,
                       class_labels=[torch.zeros(len(m), device=device, dtype=torch.long) for m in masks])
        return result.loss
    logits = model(pixels)
    target = torch.stack([m.amax(dim=0) if len(m) else torch.zeros(m.shape[1:], device=device) for m in masks])[:, None]
    return foreground_loss(logits, target, loss_settings)

def expert_loss(logits,targets,counts):
    """Expected per-expert Dice plus BCE; ambiguous pixels have weight 0.25."""
    valid=(torch.arange(targets.shape[1],device=targets.device)[None,:]<counts[:,None]).float()
    mean=(targets*valid[:,:,None,None]).sum(1,keepdim=True)/counts[:,None,None,None]
    agreement=((mean==0)|(mean==1)).float(); weight=.25+.75*agreement
    bce=F.binary_cross_entropy_with_logits(logits,mean,reduction='none',pos_weight=logits.new_tensor(2.))
    bce=(bce*weight).sum((1,2,3))/weight.sum((1,2,3))
    p=logits.sigmoid(); intersection=(p*targets*weight).sum((2,3))
    denominator=((p+targets)*weight).sum((2,3))
    dice=1-(2*intersection+1)/(denominator+1)
    return (bce+(dice*valid).sum(1)/counts).mean()
