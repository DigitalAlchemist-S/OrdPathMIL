"""Minimal inference demo for OrdPath-MIL.

By default this script runs on a random feature bag.  Pass ``--feature-path`` to
run the model on a saved PyTorch tensor with shape [num_patches, feature_dim].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import OrdPathMIL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OrdPath-MIL on one WSI feature bag.")
    parser.add_argument("--feature-path", type=Path, default=None, help="Optional .pt tensor feature bag.")
    parser.add_argument("--feature-dim", type=int, default=512, help="Feature dimension for random input.")
    parser.add_argument("--num-patches", type=int, default=512, help="Number of patches for random input.")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of ordinal classes.")
    parser.add_argument(
        "--disable-negative-module",
        action="store_true",
        help="Disable explicit zero/non-positive evidence modeling.",
    )
    return parser.parse_args()


def load_features(path: Path | None, *, num_patches: int, feature_dim: int) -> torch.Tensor:
    if path is None:
        return torch.randn(num_patches, feature_dim)
    features = torch.load(path, map_location="cpu")
    if isinstance(features, dict):
        for key in ("features", "feats", "x"):
            if key in features:
                features = features[key]
                break
    if not isinstance(features, torch.Tensor):
        raise TypeError("feature file must contain a tensor or a dict with features/feats/x.")
    return features


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)

    features = load_features(args.feature_path, num_patches=args.num_patches, feature_dim=args.feature_dim)
    if features.ndim != 2:
        raise ValueError(f"features must be a [num_patches, feature_dim] tensor, got {tuple(features.shape)}.")
    model = OrdPathMIL(
        input_dim=features.shape[1],
        num_classes=args.num_classes,
        use_negative_module=not args.disable_negative_module,
    )
    model.eval()

    with torch.no_grad():
        output = model(features)

    predicted_grade = int(output.probabilities.argmax(dim=1).item())
    print("logits:", output.logits)
    print("probabilities:", output.probabilities)
    print("predicted_grade:", predicted_grade)
    print("attention_shape:", tuple(output.attention.shape))


if __name__ == "__main__":
    main()
