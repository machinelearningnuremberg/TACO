from __future__ import annotations
from collections import OrderedDict
from torch import nn, Tensor
from typing import Tuple, Dict, Any
import torch

from taco.model.taco_model import TACO as TACOWrapper

class TACOShim(nn.Module):
    def __init__(
        self,
        *,
        use_compressor: bool = True,
        row_compression_percentage: float = 10.0,
        checkpoint_path: str = "",
        rcp_sampling = None,
        new_tabpfn_config: Dict[str, Any] | None = None,
        max_chunk_size: int = 10000,
        **wrapper_kwargs,
    ):
        super().__init__()
        self._cache_slots = 8

        ckpt = None
        sd = None
        if checkpoint_path == "original":
            raise ValueError(
                "Loading original TabPFN weights is disabled. "
                "Use a TACO/POT checkpoint instead."
            )
        if checkpoint_path and checkpoint_path != "original":
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            sd = _strip_wrapper_tokens(sd)

            checkpoint_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            if isinstance(checkpoint_config, dict):
                if "use_compressor" in checkpoint_config:
                    use_compressor = bool(checkpoint_config["use_compressor"])

                checkpoint_tabpfn_config = (
                    checkpoint_config.get("new_tabpfn_config")
                    or checkpoint_config.get("tabpfn_config")
                )
                if new_tabpfn_config is None and checkpoint_tabpfn_config is not None:
                    new_tabpfn_config = checkpoint_tabpfn_config

        self.core = TACOWrapper(
            use_compressor=use_compressor,
            row_compression_percentage=row_compression_percentage,
            rcp_sampling=rcp_sampling,
            new_tabpfn_config=new_tabpfn_config,
            max_chunk_size=max_chunk_size,
            **wrapper_kwargs,
        )
        self.use_compressor = use_compressor

        if sd is not None:
            missing, unexpected = self.core.load_state_dict(sd, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"Strict load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
                )

        self.ninp = getattr(self.predictor, "ninp")
        self.features_per_group = getattr(self.predictor, "features_per_group", 1)
        self.n_out = getattr(self.predictor, "n_out", 1)

        if hasattr(self.predictor, "transformer_encoder"):
            self.transformer_encoder = self.predictor.transformer_encoder
        if hasattr(self.predictor, "transformer_decoder"):
            self.transformer_decoder = self.predictor.transformer_decoder

    @property
    def predictor(self) -> nn.Module:
        return self.core.predictor

    @property
    def compressor(self) -> nn.Module | None:
        return getattr(self.core, "compressor", None)

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        if hasattr(self.predictor, "reset_save_peak_mem_factor"):
            self.predictor.reset_save_peak_mem_factor(factor)
        if self.use_compressor and self.compressor and hasattr(self.compressor, "reset_save_peak_mem_factor"):
            self.compressor.reset_save_peak_mem_factor(factor)

    def empty_trainset_representation_cache(self) -> None:
        if hasattr(self.predictor, "empty_trainset_representation_cache"):
            self.predictor.empty_trainset_representation_cache()
        if self.use_compressor and self.compressor and hasattr(self.compressor, "empty_trainset_representation_cache"):
            self.compressor.empty_trainset_representation_cache()
        if self.use_compressor and hasattr(self.core, "clear_eval_cache"):
            self.core.clear_eval_cache()

    @property
    def cache_trainset_representation(self) -> bool:
        return getattr(self.predictor, "cache_trainset_representation", False)

    @cache_trainset_representation.setter
    def cache_trainset_representation(self, v: bool) -> None:
        if hasattr(self.predictor, "cache_trainset_representation"):
            self.predictor.cache_trainset_representation = v
        if self.use_compressor and self.compressor and hasattr(self.compressor, "cache_trainset_representation"):
            self.compressor.cache_trainset_representation = v

    # --- Forward --------------------------------------------------------------
    def forward(self, style, X_full: Tensor, y_train: Tensor | None, **kwargs) -> Tensor:
        # Without the compressor, TACO should behave like the native TabPFN
        # predictor so the standard fit_with_cache engine can prime and reuse KV.
        if not self.use_compressor:
            return self.predictor(
                style,
                X_full,
                y_train,
                only_return_standard_out=kwargs.get("only_return_standard_out", True),
                categorical_inds=kwargs.get("categorical_inds"),
                single_eval_pos=kwargs.get("single_eval_pos"),
            )

        if y_train is None:
            raise ValueError(
                "Compressed TACO inference requires training labels. "
                "Use fit_with_compressor_cache for cached compressed inference."
            )

        try:
            slot = int(style) % self._cache_slots
        except Exception:
            slot = 0
        setattr(self.core, "_current_cache_slot", slot)

        X = X_full.transpose(0, 1)
        B = X.shape[0]

        if y_train.ndim == 1:
            y_wrapped = y_train.unsqueeze(0)              # (1,N)
        elif y_train.ndim == 2:
            if y_train.shape[1] == B:                     # (N,B) -> (B,N)
                y_wrapped = y_train.transpose(0, 1)
            elif y_train.shape[0] == B:                   # (B,N)
                y_wrapped = y_train
            else:
                raise ValueError(
                    f"Ambiguous y_train shape {tuple(y_train.shape)} for batch B={B}; expected (N,), (N,B) or (B,N)."
                )
        else:
            raise TypeError(f"Unsupported y_train shape {tuple(y_train.shape)}")

        logits_BMC = self.core(X, y_wrapped)

        return logits_BMC.permute(1, 0, 2)



def _strip_wrapper_tokens(sd: dict) -> dict:
    """Remove tokens introduced by wrappers like DDP ('module') and torch.compile ('_orig_mod')."""
    def strip_tokens(k: str) -> str:
        parts = k.split(".")
        parts = [p for p in parts if p not in {"module", "_orig_mod"}]
        return ".".join(parts)

    out = OrderedDict()
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[strip_tokens(k)] = v
    return out
