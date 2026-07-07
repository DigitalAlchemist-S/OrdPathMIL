"""OrdPath-MIL model.

This file is a compact, standalone release version of the full OrdPath-MIL
architecture used in the paper.  It keeps the core modules:

- Target-Gated Evidence Aggregation (TGE)
- Ordinal Distribution Module (ODM)
- Negative Evidence Module (NEM)

The implementation only depends on PyTorch and accepts a single WSI bag as a
feature matrix of shape [num_patches, feature_dim].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class OrdPathOutput:
    """Output container returned by OrdPathMIL."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    embedding: torch.Tensor
    attention: torch.Tensor
    target_scores: torch.Tensor
    negative_scores: torch.Tensor | None
    aux_losses: dict[str, torch.Tensor]


@dataclass
class _EvidenceSummary:
    topk_mean: torch.Tensor
    topk_features: torch.Tensor
    stats: torch.Tensor


@dataclass
class _NegativeSummary:
    feature: torch.Tensor
    stats: torch.Tensor


class OrdPathMIL(nn.Module):
    """Ordinal pathology MIL with target evidence and ordinal fusion.

    Parameters
    ----------
    input_dim:
        Dimension of pre-computed patch features, e.g. CONCH features.
    num_classes:
        Number of ordinal diagnostic classes. Class 0 is treated as the
        zero/non-positive class and classes 1..K-1 as ordered positive grades.
    projection_dim:
        Hidden dimension used after feature projection.
    attention_dim:
        Hidden dimension for gated attention and prediction heads.
    top_k:
        Number of high-evidence patches summarized for each evidence branch.
    use_negative_module:
        Enable explicit zero/non-positive evidence modeling. Set this to
        ``False`` for the ordinal-only ablation used by some experiments.
    use_ordinal_module:
        Enable high-grade evidence statistics and ordinal distribution fusion.

    Notes
    -----
    The model operates on one bag at a time.  It does not use WSI coordinates.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        projection_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
        top_k: int = 32,
        topk_temperature: float = 0.5,
        evidence_temperature: float = 1.0,
        base_prediction_weight: float = 0.7,
        learnable_fusion_weight: bool = True,
        use_negative_module: bool = True,
        use_ordinal_module: bool = True,
        target_loss_weight: float = 0.2,
        grade_loss_weight: float = 0.2,
        ordinal_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if input_dim <= 0 or projection_dim <= 0 or attention_dim <= 0:
            raise ValueError("input_dim, projection_dim, and attention_dim must be positive.")
        if top_k <= 0 or topk_temperature <= 0 or evidence_temperature <= 0:
            raise ValueError("top_k and temperatures must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= base_prediction_weight <= 1.0:
            raise ValueError("base_prediction_weight must be in [0, 1].")
        for name, value in {
            "target_loss_weight": target_loss_weight,
            "grade_loss_weight": grade_loss_weight,
            "ordinal_loss_weight": ordinal_loss_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

        if not use_ordinal_module:
            learnable_fusion_weight = False
            base_prediction_weight = 1.0
            grade_loss_weight = 0.0
            ordinal_loss_weight = 0.0
        if not use_negative_module:
            target_loss_weight = 0.0

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.positive_classes = num_classes - 1
        self.top_k = top_k
        self.topk_temperature = topk_temperature
        self.evidence_temperature = evidence_temperature
        self.use_negative_module = use_negative_module
        self.use_ordinal_module = use_ordinal_module
        self.target_loss_weight = target_loss_weight
        self.grade_loss_weight = grade_loss_weight
        self.ordinal_loss_weight = ordinal_loss_weight
        self.learnable_fusion_weight = learnable_fusion_weight

        # Global ABMIL-style context, used alongside explicit evidence summaries.
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention_v = nn.Sequential(nn.Linear(projection_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(projection_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)

        # Patch-level target relevance and ordered positive-grade evidence.
        self.target_gate = nn.Linear(projection_dim, 1)
        self.positive_evidence_head = nn.Linear(projection_dim, self.positive_classes)
        if use_negative_module:
            # Explicit zero/non-positive evidence branch.
            self.negative_evidence_head = nn.Linear(projection_dim, 1)

        evidence_stat_dim = self.positive_classes * 6 + 2
        topk_feature_dim = self.positive_classes * projection_dim
        negative_dim = projection_dim + 6 if use_negative_module else 0
        high_grade_dim = 4 if use_ordinal_module else 0
        summary_dim = projection_dim + topk_feature_dim + evidence_stat_dim + negative_dim + high_grade_dim

        self.summary = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.target_head = self._head(projection_dim, attention_dim, 1, dropout)
        self.positive_grade_head = self._head(projection_dim, attention_dim, self.positive_classes, dropout)
        self.ordinal_head = self._head(projection_dim, attention_dim, self.positive_classes, dropout)
        if use_negative_module:
            self.zero_head = self._head(projection_dim, attention_dim, 1, dropout)

        if learnable_fusion_weight:
            clipped = min(max(base_prediction_weight, 1e-4), 1.0 - 1e-4)
            self.fusion_logit = nn.Parameter(torch.tensor(math.log(clipped / (1.0 - clipped))))
        else:
            self.register_buffer("fixed_fusion_weight", torch.tensor(float(base_prediction_weight)))

    @staticmethod
    def _head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor, target: torch.Tensor | None = None) -> OrdPathOutput:
        """Run inference for one WSI feature bag.

        Parameters
        ----------
        features:
            Tensor with shape [num_patches, input_dim].
        target:
            Optional scalar class label.  When provided, auxiliary losses are
            returned for training/debugging. Add these losses to your main
            classification loss during training.
        """

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"features must be a non-empty [N, D] tensor, got {tuple(features.shape)}.")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"expected feature dimension {self.input_dim}, got {features.shape[1]}.")

        hidden = self.projection(features.float())
        attention_logits = self.attention_w(self.attention_v(hidden) * self.attention_u(hidden)).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=0)
        bag_embedding = torch.sum(attention.unsqueeze(-1) * hidden, dim=0, keepdim=True)

        target_logits = self.target_gate(hidden).squeeze(-1)
        target_scores = torch.sigmoid(target_logits)
        positive_scores = self.positive_evidence_head(hidden)
        gated_scores = positive_scores + torch.log(target_scores.clamp_min(1e-6)).unsqueeze(-1)

        evidence = self._summarize_evidence(hidden, gated_scores, target_scores)
        summary_parts = [
            bag_embedding,
            evidence.topk_features.reshape(1, -1),
            evidence.stats.reshape(1, -1),
        ]

        negative_scores = None
        negative_logits = None
        if self.use_negative_module:
            negative_logits = self.negative_evidence_head(hidden).squeeze(-1)
            negative_scores = torch.sigmoid(negative_logits)
            negative = self._summarize_negative(hidden, negative_logits, target_scores)
            summary_parts.extend([negative.feature.reshape(1, -1), negative.stats.reshape(1, -1)])

        if self.use_ordinal_module:
            summary_parts.append(self._high_grade_stats(gated_scores, target_scores, evidence.topk_mean).reshape(1, -1))

        summary = self.summary(torch.cat(summary_parts, dim=1))
        target_bag_logit = self.target_head(summary).squeeze(1)
        positive_grade_logits = self.positive_grade_head(summary)
        ordinal_logits = self.ordinal_head(summary)
        zero_logit = self.zero_head(summary).squeeze(1) if self.use_negative_module else None

        logits, ordinal_probabilities = self._compose_fused_logits(
            target_bag_logit,
            positive_grade_logits,
            ordinal_logits,
            zero_logit=zero_logit,
        )
        aux_losses = self._auxiliary_losses(
            target=target,
            target_bag_logit=target_bag_logit,
            positive_grade_logits=positive_grade_logits,
            ordinal_probabilities=ordinal_probabilities,
            target_logits=target_logits,
        )
        return OrdPathOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=1),
            embedding=summary,
            attention=attention,
            target_scores=target_scores,
            negative_scores=negative_scores,
            aux_losses=aux_losses,
        )

    def _compose_fused_logits(
        self,
        target_bag_logit: torch.Tensor,
        positive_grade_logits: torch.Tensor,
        ordinal_logits: torch.Tensor,
        *,
        zero_logit: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if zero_logit is None:
            base_zero = F.logsigmoid(-target_bag_logit).reshape(1, 1)
            base_positive = F.logsigmoid(target_bag_logit).reshape(1, 1) + F.log_softmax(
                positive_grade_logits,
                dim=1,
            )
        else:
            zero_positive_gate = F.log_softmax(torch.stack((zero_logit, target_bag_logit), dim=1), dim=1)
            base_zero = zero_positive_gate[:, :1]
            base_positive = zero_positive_gate[:, 1:] + F.log_softmax(positive_grade_logits, dim=1)

        base_log_probs = torch.cat((base_zero, base_positive), dim=1)
        ordinal_probabilities = self._ordinal_probabilities(ordinal_logits)
        ordinal_log_probs = torch.log(ordinal_probabilities.clamp_min(1e-8))
        alpha = torch.sigmoid(self.fusion_logit) if self.learnable_fusion_weight else self.fixed_fusion_weight
        log_scores = alpha * base_log_probs + (1.0 - alpha) * ordinal_log_probs
        return F.log_softmax(log_scores, dim=1), ordinal_probabilities

    def _ordinal_probabilities(self, ordinal_logits: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumprod(torch.sigmoid(ordinal_logits), dim=1)
        zero_prob = 1.0 - cumulative[:, :1]
        if self.positive_classes == 1:
            return torch.cat((zero_prob, cumulative), dim=1)
        middle = cumulative[:, :-1] - cumulative[:, 1:]
        last = cumulative[:, -1:]
        return torch.cat((zero_prob, middle, last), dim=1).clamp_min(1e-8)

    def _summarize_evidence(
        self,
        hidden: torch.Tensor,
        gated_scores: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> _EvidenceSummary:
        topk_features = []
        topk_mean = []
        stats = []
        for class_index in range(self.positive_classes):
            scores = gated_scores[:, class_index]
            k = min(self.top_k, scores.shape[0])
            values, indices = torch.topk(scores, k=k, dim=0)
            weights = torch.softmax(values / self.topk_temperature, dim=0)
            topk_features.append(torch.sum(weights.unsqueeze(-1) * hidden[indices], dim=0))
            topk_mean.append(values.mean())
            stats.extend(
                [
                    scores.max(),
                    values.mean(),
                    self._top_fraction_mean(scores, fraction=0.05),
                    self._top_fraction_mean(scores, fraction=0.10),
                    torch.sigmoid(scores / self.evidence_temperature).mean(),
                    (torch.sigmoid(scores / self.evidence_temperature) * target_scores).mean(),
                ]
            )

        k_target = min(self.top_k, target_scores.shape[0])
        stats.extend([torch.topk(target_scores, k=k_target, dim=0).values.mean(), target_scores.mean()])
        return _EvidenceSummary(
            topk_mean=torch.stack(topk_mean, dim=0),
            topk_features=torch.stack(topk_features, dim=0),
            stats=torch.stack(stats, dim=0),
        )

    def _summarize_negative(
        self,
        hidden: torch.Tensor,
        negative_logits: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> _NegativeSummary:
        k = min(self.top_k, negative_logits.shape[0])
        values, indices = torch.topk(negative_logits, k=k, dim=0)
        weights = torch.softmax(values / self.topk_temperature, dim=0)
        feature = torch.sum(weights.unsqueeze(-1) * hidden[indices], dim=0)
        negative_prob = torch.sigmoid(negative_logits)
        stats = torch.stack(
            [
                negative_logits.max(),
                values.mean(),
                self._top_fraction_mean(negative_logits, fraction=0.05),
                self._top_fraction_mean(negative_logits, fraction=0.10),
                negative_prob.mean(),
                (negative_prob * (1.0 - target_scores)).mean(),
            ],
            dim=0,
        )
        return _NegativeSummary(feature=feature, stats=stats)

    def _high_grade_stats(
        self,
        gated_scores: torch.Tensor,
        target_scores: torch.Tensor,
        topk_mean: torch.Tensor,
    ) -> torch.Tensor:
        high_scores = gated_scores[:, -1]
        lower_reference = topk_mean[:-1].max() if self.positive_classes > 1 else torch.zeros_like(topk_mean[-1])
        return torch.stack(
            [
                self._top_fraction_mean(high_scores, fraction=0.05),
                torch.sigmoid(high_scores / self.evidence_temperature).mean(),
                (torch.sigmoid(high_scores / self.evidence_temperature) * target_scores).mean(),
                topk_mean[-1] - lower_reference,
            ],
            dim=0,
        )

    @staticmethod
    def _top_fraction_mean(scores: torch.Tensor, *, fraction: float) -> torch.Tensor:
        k = max(1, int(round(scores.shape[0] * fraction)))
        k = min(k, scores.shape[0])
        return torch.topk(scores, k=k, dim=0).values.mean()

    def _auxiliary_losses(
        self,
        *,
        target: torch.Tensor | None,
        target_bag_logit: torch.Tensor,
        positive_grade_logits: torch.Tensor,
        ordinal_probabilities: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if target is None:
            return {}

        label = target.reshape(-1)[0].long()
        label_int = int(label.item())
        aux_losses: dict[str, torch.Tensor] = {}

        if self.target_loss_weight > 0:
            positive_label = torch.tensor(
                [float(label_int > 0)],
                dtype=target_bag_logit.dtype,
                device=target_bag_logit.device,
            )
            k = min(self.top_k, target_logits.shape[0])
            patch_target_logit = torch.topk(target_logits, k=k, dim=0).values.mean().reshape(1)
            target_logit = 0.5 * (target_bag_logit.reshape(1) + patch_target_logit)
            aux_losses["target"] = self.target_loss_weight * F.binary_cross_entropy_with_logits(
                target_logit,
                positive_label,
            )

        if self.grade_loss_weight > 0 and label_int > 0:
            aux_losses["grade"] = self.grade_loss_weight * F.cross_entropy(
                positive_grade_logits,
                torch.tensor([label_int - 1], dtype=torch.long, device=positive_grade_logits.device),
            )

        if self.ordinal_loss_weight > 0:
            threshold_targets = torch.tensor(
                [[float(label_int >= threshold) for threshold in range(1, self.num_classes)]],
                dtype=ordinal_probabilities.dtype,
                device=ordinal_probabilities.device,
            )
            cumulative_probs = 1.0 - torch.cumsum(ordinal_probabilities[:, :-1], dim=1)
            cumulative_probs = cumulative_probs.clamp(min=1e-6, max=1.0 - 1e-6)
            ordinal_loss = -(
                threshold_targets * torch.log(cumulative_probs)
                + (1.0 - threshold_targets) * torch.log1p(-cumulative_probs)
            ).mean()
            aux_losses["ordinal"] = self.ordinal_loss_weight * ordinal_loss

        return aux_losses
