"""Encoder statistics caching for self-sufficient compressed context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
import einops

from taco.model.tabpfn_arch.model.encoders import (
    NanHandlingEncoderStep,
    InputNormalizationEncoderStep,
    VariableNumFeaturesEncoderStep,
    LinearInputEncoderStep,
)


# Constants from NanHandlingEncoderStep
NAN_INDICATOR = -2.0
INF_INDICATOR = 2.0
NEG_INF_INDICATOR = 4.0


@dataclass
class EncoderStats:
    """Minimal statistics needed to encode test data without raw training data."""

    # NanHandlingEncoderStep
    feature_means: Tensor              # (B*num_groups, g)

    # InputNormalizationEncoderStep
    norm_mean: Optional[Tensor]        # (B*num_groups, g)
    norm_std: Optional[Tensor]         # (B*num_groups, g)

    # VariableNumFeaturesEncoderStep for nan_indicators (first one)
    num_features_nan_ind: int          # Target size for nan_indicators

    # VariableNumFeaturesEncoderStep for main (second one)
    num_used_features_main: Tensor     # (B*num_groups, 1) - for scaling main
    num_features_main: int             # Target size for main

    # Metadata
    batch_size: int
    features_per_group: int

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'EncoderStats':
        def _move(t: Optional[Tensor]) -> Optional[Tensor]:
            if t is None:
                return None
            if dtype is not None:
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        return EncoderStats(
            feature_means=_move(self.feature_means),
            norm_mean=_move(self.norm_mean),
            norm_std=_move(self.norm_std),
            num_features_nan_ind=self.num_features_nan_ind,
            num_used_features_main=self.num_used_features_main.to(device=device),
            num_features_main=self.num_features_main,
            batch_size=self.batch_size,
            features_per_group=self.features_per_group,
        )

    def detach(self) -> 'EncoderStats':
        def _detach(t: Optional[Tensor]) -> Optional[Tensor]:
            return t.detach() if t is not None else None

        return EncoderStats(
            feature_means=_detach(self.feature_means),
            norm_mean=_detach(self.norm_mean),
            norm_std=_detach(self.norm_std),
            num_features_nan_ind=self.num_features_nan_ind,
            num_used_features_main=self.num_used_features_main.detach(),
            num_features_main=self.num_features_main,
            batch_size=self.batch_size,
            features_per_group=self.features_per_group,
        )


def extract_encoder_stats(
    encoder: nn.Module,
    batch_size: int,
    features_per_group: int,
) -> EncoderStats:
    """Extract fitted statistics from a SequentialEncoder."""

    feature_means = None
    norm_mean = None
    norm_std = None

    var_steps = []

    for step in encoder:
        if isinstance(step, NanHandlingEncoderStep):
            if hasattr(step, 'feature_means_') and step.feature_means_ is not None:
                feature_means = step.feature_means_.clone()

        elif isinstance(step, InputNormalizationEncoderStep):
            if hasattr(step, 'mean_for_normalization') and step.mean_for_normalization is not None:
                norm_mean = step.mean_for_normalization.clone()
            if hasattr(step, 'std_for_normalization') and step.std_for_normalization is not None:
                norm_std = step.std_for_normalization.clone()

        elif isinstance(step, VariableNumFeaturesEncoderStep):
            var_steps.append({
                'in_keys': step.in_keys,
                'num_features': step.num_features,
                'normalize': step.normalize_by_used_features,
                'num_used_features': step.number_of_used_features_.clone() if hasattr(step, 'number_of_used_features_') and step.number_of_used_features_ is not None else None,
            })

    num_features_nan_ind = None
    num_features_main = None
    num_used_features_main = None

    for vs in var_steps:
        if 'nan_indicators' in vs['in_keys']:
            num_features_nan_ind = vs['num_features']
        elif 'main' in vs['in_keys']:
            num_features_main = vs['num_features']
            num_used_features_main = vs['num_used_features']

    return EncoderStats(
        feature_means=feature_means,
        norm_mean=norm_mean,
        norm_std=norm_std,
        num_features_nan_ind=num_features_nan_ind,
        num_used_features_main=num_used_features_main,
        num_features_main=num_features_main,
        batch_size=batch_size,
        features_per_group=features_per_group,
    )


def encode_test_data_with_stats(
    test_x: Tensor,                    # (T, B, F_raw) - original format
    encoder_stats: EncoderStats,
    linear_layer: nn.Linear,
) -> Tensor:
    """
    Encode test data using cached statistics.

    Pipeline:
    0. Reshape to grouped format
    1. NanHandlingEncoderStep: create nan_indicators, replace NaNs in main
    2. VariableNumFeaturesEncoderStep on nan_indicators: pad only (no scaling)
    3. InputNormalizationEncoderStep on main: normalize only
    4. VariableNumFeaturesEncoderStep on main: scale and pad
    5. LinearInputEncoderStep: cat(main, nan_indicators), apply linear
    """
    T, B, F_raw = test_x.shape
    g = encoder_stats.features_per_group

    num_groups = (F_raw + g - 1) // g
    target_F = num_groups * g

    if F_raw < target_F:
        padding = test_x.new_zeros(T, B, target_F - F_raw)
        x = torch.cat([test_x, padding], dim=-1)
    else:
        x = test_x.clone()

    main = einops.rearrange(x, "t b (f g) -> t (b f) g", g=g)

    stats = encoder_stats.to(main.device, main.dtype)
    BF = main.shape[1]

    # Step 1: NanHandlingEncoderStep
    nan_mask = torch.isnan(main)
    inf_pos_mask = torch.isinf(main) & (torch.sign(main) == 1)
    inf_neg_mask = torch.isinf(main) & (torch.sign(main) == -1)

    nan_indicators = (
        nan_mask.float() * NAN_INDICATOR +
        inf_pos_mask.float() * INF_INDICATOR +
        inf_neg_mask.float() * NEG_INF_INDICATOR
    ).to(main.dtype)

    bad_mask = nan_mask | torch.isinf(main)
    if bad_mask.any() and stats.feature_means is not None:
        means_expanded = stats.feature_means.unsqueeze(0).expand_as(main)
        main = torch.where(bad_mask, means_expanded, main)

    # Step 2: VariableNumFeaturesEncoderStep on nan_indicators (pad only)
    if stats.num_features_nan_ind is not None:
        current_g = nan_indicators.shape[-1]
        if current_g < stats.num_features_nan_ind:
            pad_size = stats.num_features_nan_ind - current_g
            nan_indicators = torch.cat([nan_indicators, nan_indicators.new_zeros(T, BF, pad_size)], dim=-1)
        elif current_g > stats.num_features_nan_ind:
            nan_indicators = nan_indicators[..., :stats.num_features_nan_ind]

    # Step 3: InputNormalizationEncoderStep on main
    if stats.norm_mean is not None and stats.norm_std is not None:
        mean = stats.norm_mean.unsqueeze(0)
        std = stats.norm_std.unsqueeze(0)
        main = (main - mean) / (std + 1e-16)
        main = torch.clamp(main, -100, 100)

    # Step 4: VariableNumFeaturesEncoderStep on main (scale and pad)
    if stats.num_used_features_main is not None and stats.num_features_main is not None:
        scale = torch.sqrt(
            stats.num_features_main / stats.num_used_features_main.clamp(min=1).float()
        )
        main = main * scale.unsqueeze(0)

        current_g = main.shape[-1]
        if current_g < stats.num_features_main:
            pad_size = stats.num_features_main - current_g
            main = torch.cat([main, main.new_zeros(T, BF, pad_size)], dim=-1)
        elif current_g > stats.num_features_main:
            main = main[..., :stats.num_features_main]

    # Step 5: LinearInputEncoderStep
    x = torch.cat([main, nan_indicators], dim=-1)
    x = linear_layer(x)

    return x