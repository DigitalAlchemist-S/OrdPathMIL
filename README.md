# OrdPath-MIL

OrdPath-MIL is a compact PyTorch implementation of the ordinal pathology
multiple-instance learning model used for weakly supervised whole-slide image
(WSI) grading.

This release intentionally focuses on the model architecture. It does not
include private datasets, preprocessing scripts, training loops, or experiment
configuration files. The model accepts a pre-computed WSI feature bag, such as
CONCH patch features, and returns case-level ordinal class probabilities.

This repository follows the lightweight release style of small paper-code
repositories: the goal is to make the architecture easy to inspect and run, not
to reproduce every private training detail.

## Model Overview

OrdPath-MIL combines three modules:

- **TGE: Target-Gated Evidence Aggregation** identifies target-relevant
  positive-grade patch evidence and summarizes high-evidence patches.
- **ODM: Ordinal Distribution Module** models ordered diagnostic grades as an
  ordinal distribution.
- **NEM: Negative Evidence Module** explicitly models the zero/non-positive
  class and the zero-positive diagnostic boundary.

The input is a tensor of shape `[num_patches, feature_dim]`. Class `0` is the
zero or non-positive class, and classes `1..K-1` are increasing positive
diagnostic grades.

The constructor exposes the main ablation switches:

- `use_negative_module=True`: enables the NEM zero/non-positive branch.
- `use_ordinal_module=True`: enables ODM high-grade statistics and ordinal
  distribution fusion.

For experiments that use the ordinal branch without the negative branch, create
the model with `use_negative_module=False, use_ordinal_module=True`.

## Files

- `model.py`: standalone OrdPath-MIL model.
- `demo_inference.py`: random or `.pt` feature-bag inference demo.
- `requirements.txt`: minimal dependency list.

## Quick Start

```bash
pip install -r requirements.txt
python demo_inference.py
```

Expected output:

```text
logits: tensor(...)
probabilities: tensor(...)
predicted_grade: ...
attention_shape: (512,)
```

To run inference on a saved PyTorch feature tensor:

```bash
python demo_inference.py --feature-path path/to/features.pt --num-classes 4
```

The feature file should contain either a tensor with shape
`[num_patches, feature_dim]` or a dictionary with one of the keys `features`,
`feats`, or `x`.

## Example Usage

```python
import torch
from model import OrdPathMIL

features = torch.randn(512, 512)  # [num_patches, feature_dim]
model = OrdPathMIL(input_dim=512, num_classes=4)
model.eval()

with torch.no_grad():
    output = model(features)

print(output.probabilities)
print(output.attention)
```

## Training Integration

OrdPath-MIL is a single-bag model. A training loop can combine the main
case-level classification loss with the auxiliary losses returned by the model:

```python
features = torch.randn(512, 512)
label = torch.tensor([2])

model = OrdPathMIL(input_dim=512, num_classes=4)
output = model(features, target=label)

main_loss = torch.nn.functional.cross_entropy(output.logits, label)
loss = main_loss + sum(output.aux_losses.values())
loss.backward()
```

`output.logits` has shape `[1, num_classes]`, `output.probabilities` contains
softmax-normalized class probabilities, and `output.attention` contains one
attention weight per patch.

## Data Format

This release assumes patch features have already been extracted from WSIs. A
typical minimal dataset can store one feature file per slide:

```text
DATASET/
  slide_001.pt
  slide_002.pt
  labels.csv
```

where each `.pt` file contains a `[num_patches, feature_dim]` tensor and
`labels.csv` maps slide IDs to ordinal labels. WSI tiling, patch feature
extraction, cross-validation splits, and dataset-specific preprocessing are
outside the scope of this lightweight release.

## Citation

If you use this code, please cite the corresponding MICCAI workshop paper once
the final bibliographic information is available.

```bibtex
@inproceedings{ordpathmil2026,
  title     = {OrdPath-MIL: Ordinal Pathology Multiple Instance Learning for Whole-Slide Image Grading},
  author    = {Anonymous},
  booktitle = {MICCAI Workshop},
  year      = {2026}
}
```

## Notes

- This repository is a lightweight model release, not a full reproduction
  package.
- The model runs on random tensors and can be adapted to pre-computed WSI patch
  features.
- Coordinates are not required by this release implementation.
- No pretrained checkpoint is included.
