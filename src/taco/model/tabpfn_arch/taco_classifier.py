from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from taco.model.tabpfn_arch.base import determine_precision
from taco.model.tabpfn_arch.classifier import TabPFNClassifier as BaseTabPFNClassifier
from taco.model.tabpfn_arch.config import ModelInterfaceConfig
from taco.model.tabpfn_arch.utils import infer_device_and_type, infer_random_state
from .shim_model import TACOShim


DEFAULT_CHECKPOINT_REPO_ID = "zabergjg/TabPFN-TACO"
TACO_CHECKPOINT_FILENAME = "TabPFN-TACO-classifier.ckpt"
POT_CHECKPOINT_FILENAME = "TabPFN-POT-classifier.ckpt"


class TACOClassifier(BaseTabPFNClassifier):
    def __init__(
        self,
        *,
        use_compressor: bool = True,
        row_compression_percentage: float = 10.0,
        checkpoint_path: str | Path | None = "auto",
        checkpoint_repo_id: str | None = None,
        checkpoint_filename: str | None = None,
        checkpoint_revision: str | None = None,
        local_files_only: bool = False,
        wrapper_kwargs: dict[str, Any] | None = None,
        rcp_sampling = None,
        new_tabpfn_config: dict[str, Any] | None = None,
        max_chunk_size: int = 10000000,
        device: str = "cuda",
        **kwargs: Any,
    ):
        super().__init__(device=device, **kwargs)
        self._use_compressor = use_compressor
        self._row_compression_percentage = row_compression_percentage
        self._checkpoint_path = checkpoint_path
        self._checkpoint_repo_id = checkpoint_repo_id
        self._checkpoint_filename = checkpoint_filename
        self._checkpoint_revision = checkpoint_revision
        self._local_files_only = local_files_only
        self._wrapper_kwargs = wrapper_kwargs or {}
        self._rcp_sampling = rcp_sampling
        self._new_tabpfn_config = new_tabpfn_config
        self._max_chunk_size = max_chunk_size

    def _resolve_checkpoint_path(self) -> str:
        if self._checkpoint_path not in (None, "", "auto"):
            return str(self._checkpoint_path)

        repo_id = (
            self._checkpoint_repo_id
            or os.environ.get("TACO_CHECKPOINT_REPO_ID")
            or DEFAULT_CHECKPOINT_REPO_ID
        )
        filename = self._checkpoint_filename or (
            TACO_CHECKPOINT_FILENAME if self._use_compressor else POT_CHECKPOINT_FILENAME
        )
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=self._checkpoint_revision,
            local_files_only=self._local_files_only,
        )

    def _initialize_model_variables(self):
        _, rng = infer_random_state(self.random_state)
        self.device_ = infer_device_and_type(self.device)
        (
            self.use_autocast_,
            self.forced_inference_dtype_,
            byte_size,
        ) = determine_precision(self.inference_precision, self.device_)
        self.interface_config_ = ModelInterfaceConfig.from_user_input(
            inference_config=self.inference_config,
        )
        checkpoint_path = self._resolve_checkpoint_path()

        self.model_ = TACOShim(
            use_compressor=self._use_compressor,
            row_compression_percentage=self._row_compression_percentage,
            checkpoint_path=checkpoint_path,
            rcp_sampling=self._rcp_sampling,
            new_tabpfn_config=self._new_tabpfn_config,
            max_chunk_size=self._max_chunk_size,
            **self._wrapper_kwargs,
        ).to(self.device_)
        self.config_ = self.model_.core.tabpfn_config
        self.model_.cache_trainset_representation = (self.fit_mode == "fit_with_cache")
        self.model_.eval()

        if getattr(self, "forced_inference_dtype_", None) is not None:
            self.model_ = self.model_.to(dtype=self.forced_inference_dtype_)

        return byte_size, rng

    def predict_proba(self, X):
        with torch.no_grad():
            return super().predict_proba(X)

    def predict(self, X):
        with torch.no_grad():
            return super().predict(X)

    def predict_logits(self, X):
        with torch.no_grad():
            return super().predict_logits(X)

    @property
    def K_values_(self) -> list[int] | None:
        # executor_ exists after fit
        eng = getattr(self, "executor_", None)
        if eng is None:
            return None
        return getattr(eng, "K_values", None)  # your engine stores this

    @property
    def K_total_(self) -> int | None:
        Ks = self.K_values_
        return int(sum(Ks)) if Ks is not None else None
