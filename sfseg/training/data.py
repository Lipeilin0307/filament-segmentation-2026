"""Original official-data and deterministic crop definitions; paths supplied by the caller."""
import hashlib, json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
def annotation_mask(segmentation, shape):
    height, width = shape
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if rle.get("size") != [height, width]:
            raise ValueError("Annotation RLE size differs from image size")
        if isinstance(rle["counts"], list):
            rle = coco_mask.frPyObjects(rle, height, width)
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    elif isinstance(segmentation, list) and segmentation:
        polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
        rle = coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
    else:
        raise ValueError("Empty or unsupported segmentation")
    array = coco_mask.decode(rle).astype(bool)
    if array.shape != (height, width) or not array.any():
        raise ValueError("Empty mask or incorrect shape")
    return array

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_dataset(root):
    root = Path(root).resolve()
    candidates = sorted(root.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one training annotation JSON under {root}, found {len(candidates)}. Extract the competition ZIP first.")
    annotation_path = candidates[0]
    dataset_root = annotation_path.parent.parent
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    for key in ("images", "annotations"):
        if not isinstance(document.get(key), list) or not document[key]:
            raise ValueError(f"Missing/empty {key}")
    return dataset_root, annotation_path, document


def image_path(root, image):
    filename = Path(image["file_name"]).name
    folder = Path(root) / "train" / "train_images"
    exact = folder / filename
    if exact.is_file():
        return exact
    # Some COCO exports use .jpg while the distributed files use .jpeg.
    candidates = [p for p in folder.glob(Path(filename).stem + ".*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cannot uniquely resolve {filename}")
    return candidates[0]

def crop_instances(annotations, box, full_shape=(2048, 2048)):
    x, y, size = box
    result = []
    for annotation in annotations:
        bx, by, bw, bh = annotation["bbox"]
        if bx + bw <= x or by + bh <= y or bx >= x + size or by >= y + size:
            continue
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
            shifted = []
            for polygon in polygons:
                coords = np.asarray(polygon, dtype=float).reshape(-1, 2) - [x, y]
                shifted.append(coords.ravel().tolist())
            rle = coco_mask.merge(coco_mask.frPyObjects(shifted, size, size))
            mask = coco_mask.decode(rle).astype(bool)
        else:
            mask = annotation_mask(segmentation, full_shape)[y:y+size, x:x+size]
        if mask.sum() >= 4:
            result.append(mask)
    return np.stack(result) if result else np.zeros((0, size, size), dtype=bool)


class TrainingCrops(torch.utils.data.Dataset):
    def __init__(self, data="data/raw", prepared="data/processed", size=512, length=256, seed=2026, augment=True):
        self.root, path, doc = load_dataset(data)
        prepared = Path(prepared)
        audit = json.loads((prepared / "audit.json").read_text(encoding="utf-8"))
        if audit["annotation_sha256"] != sha256(path):
            raise ValueError("Training annotations changed since audit")
        self.annotation_hash = audit["annotation_sha256"]
        self.folds_hash = sha256(prepared / "folds.json")
        folds = json.loads((prepared / "folds.json").read_text(encoding="utf-8"))
        ids = {r["image_id"] for r in folds if r["split"] == "train"}
        self.by_file = defaultdict(list)
        for im in doc["images"]:
            if str(im["id"]) in ids:
                self.by_file[im["file_name"]].append(im)
        self.files = sorted(self.by_file)
        self.annotations = defaultdict(list)
        for ann in doc["annotations"]:
            if str(ann["image_id"]) in ids:
                self.annotations[str(ann["image_id"])].append(ann)
        self.size, self.length, self.seed, self.epoch, self.augment = size, length, seed, 0, augment

    def __len__(self):
        return self.length

    @lru_cache(maxsize=8)
    def read_image(self, filename):
        with Image.open(image_path(self.root, {"file_name": filename})) as im:
            return np.asarray(im.convert("L")).copy()

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        # Uniform physical-image sampling avoids overweighting multi-annotator images.
        filename = self.files[int(rng.integers(len(self.files)))]
        versions = self.by_file[filename]
        im = versions[int(rng.integers(len(versions)))]
        annotations = self.annotations[str(im["id"])]
        raw = self.read_image(filename)
        height, width = raw.shape
        size = self.size
        if size > min(height, width):
            raise ValueError("Crop cannot be larger than the original image")
        if annotations and rng.random() < 0.75:
            ann = annotations[int(rng.integers(len(annotations)))]
            x, y, w, h = ann["bbox"]
            cx, cy = x + w/2, y + h/2
            x = int(np.clip(cx - size/2 + rng.uniform(-size/4, size/4), 0, width-size))
            y = int(np.clip(cy - size/2 + rng.uniform(-size/4, size/4), 0, height-size))
        else:
            x, y = int(rng.integers(width-size+1)), int(rng.integers(height-size+1))
        pixels = raw[y:y+size, x:x+size].astype(np.float32) / 255.0
        instances = crop_instances(annotations, (x, y, size), raw.shape)
        if self.augment:
            turns = int(rng.integers(4))
            pixels, instances = np.rot90(pixels, turns), np.rot90(instances, turns, axes=(1, 2))
            if rng.random() < 0.5:
                pixels, instances = pixels[:, ::-1], instances[:, :, ::-1]
            pixels = np.clip((pixels - .5) * rng.uniform(.9, 1.1) + .5 + rng.uniform(-.03, .03), 0, 1)
        return {"pixels": torch.from_numpy(np.ascontiguousarray((pixels[None] - .5) / .25)),
                "masks": torch.from_numpy(np.ascontiguousarray(instances)).float(),
                "image_id": str(im["id"]), "crop": (x, y, size)}


def collate(samples):
    return {"pixels": torch.stack([s["pixels"] for s in samples]),
            "masks": [s["masks"] for s in samples], "image_ids": [s["image_id"] for s in samples]}
