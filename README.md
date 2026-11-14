# Solar Filament Segmentation — retained public-0.40 system

Peilin Li, Nankai University — 2510547@mail.nankai.edu.cn

This repository reproduces `submission_Base_Full707_HistGBDT_YOLO_Conditional_A_20261028.csv` from input images. It packages the selected system, not the project's historical experiments. The public score **displayed as 0.40** was reported by the participant; its unrounded value and private score are unknown.

## Contents

- `reproduce.ipynb`: complete image-to-CSV pipeline and post-inference verification.
- `sfseg/`: minimal inference code (no imports from the original project).
- `models/`: all five runtime weights, totaling about 494 MiB, plus SHA256 manifest.
- `configs/`: frozen source/conditional calibration, official test image hashes and training recipe.
- `verification/`: reference CSV, numerical evidence, source provenance and environment lock. The reference CSV is **never an inference input**.
- `licenses/`: upstream attribution and publication review notes.

## Validated environment

Python 3.14.7, Windows x86_64, NVIDIA RTX 5090, PyTorch 2.12.0+cu130, torchvision 0.27.0+cu130. Base/refiner inference uses CUDA BF16; the detector crop refiner uses FP32. See `requirements.txt` for exact packages. Other GPUs/OS versions have not been verified for byte identity.

Create an isolated Python 3.14 environment, then install the pinned requirements:

```text
python -m venv .venv
# Activate .venv using your shell's standard activation command.
python -m pip install -r requirements.txt
python -m ipykernel install --user --name filament-final --display-name "Filament final"
```

Open `reproduce.ipynb` from this repository in a Jupyter-compatible editor and select that kernel. Set `SFSEG_TEST_DIR` to the folder containing the official 180 test images, or edit `TEST_DIR` in its first code cell. Run all cells. No training annotations, probability caches, or additional model downloads are required. Test data are not redistributed here; obtain the competition-provided images through the normal competition access process.

Equivalent command-line reproduction:

```text
python -m sfseg.reproduce --test-dir PATH_TO_TEST_IMAGES --output-dir outputs/run-1
```

The output directory must be new. Inference regenerates every prediction; it does not silently resume from an old CSV. For a short two-image smoke check add `--limit 2`. A smoke check is not full validation.

## Expected complete result

180 images; 1,261 instances = 1,220 Base + 41 detector supplements. CSV SHA256:

```text
16a9b935222168d93cb106e7939a763732d259ee165fc2ebd870fda157f48957
```

The verifier checks exact per-image instance order/RLE, decode/re-encode, nonempty masks, nonoverlap, and complete file bytes. Empty prediction images are covered by the image manifest even though the CSV has no object row for them. Changing numerical precision, package versions, painting order, quantization or thresholds may break parity.

## Local verification completed

Both the standalone CLI and `reproduce.executed.ipynb` regenerated all 180 images from scratch and produced the exact target CSV bytes. CLI inference took **314.6 s**; Notebook inference took **672.3 s** in this local session. Both recorded **2.20 GiB peak PyTorch allocated memory**, which excludes driver allocations, caching and other applications. Runtime is environment-dependent.

The executed Notebook retains all five executed code cells, their outputs, and a prediction visualization. Reproduction records are stored under `verification/`. Run `python verify_local_package.py` for a CPU-only package integrity check. Validation used the existing pinned inference libraries plus isolated Notebook tooling; a clean internet-based environment installation and an anonymous public clone are not yet tested.

## Model and evaluation scope

The Base deployment model and its downstream training use the official 707-image population with nested OOF construction. The retained YOLO11m/ResNet34 branch was trained on **451 A-training-side images**, with separate 57-image selection and 57-image source-calibration sets; conditional calibration uses 190 training-side OOF images. This is not a full707 detector refit. The source snapshot is documented in `verification/source-provenance.json`.

Historical recipe-matched A/B local PQ is 0.4524085 / 0.4518270. Each uses its own 565-fit / 142-held weights. These repeatedly inspected local splits are not a new blind test, and are not independent local PQ measurements for the actual 707-trained weights. These historical estimates are exploratory and do not establish a new blind-test gain. Never run these 707 weights against training labels and call the result held-out PQ.

## Downloading the complete repository

Install Git LFS before cloning so the model files are downloaded, rather than only their small pointer files:

```text
git lfs install
git clone https://github.com/Lipeilin0307/filament-segmentation-2026.git
cd filament-segmentation-2026
git lfs pull
python verify_local_package.py
```

The five model files total about 494 MiB. Downloading a source ZIP alone may not include the actual LFS model objects. No training annotations or historical prediction caches are needed. The official test images are obtained separately through the competition's authorized access process.

## Attribution and release scope

This release contains inference code, model weights, configuration, dependency pins, a complete Notebook and numerical verification records. **The technical report, its LaTeX source and report figures are intentionally withheld pending author revision.** They are not included in this repository's history.

The community-derived detector/refiner routines are explicitly attributed in `licenses/THIRD-PARTY-NOTICES.txt` and `verification/source-provenance.json`. No upstream LICENSE was found in that pinned source or the upstream repository when checked on 2026-11-14. This repository does not claim that those portions have an open-source redistribution license, and does not grant a blanket MIT or other license over third-party material. See `licenses/PUBLICATION-REVIEW.txt` for unresolved licensing matters and dependency notices. Public availability is not a statement that all downstream uses are licensed.

The original incumbent files outside this package remain unchanged. No test labels, training datasets, credentials, optimizer resume checkpoints or other historical experiments are included.
