from __future__ import annotations

import random
from typing import Literal, Optional, Iterable, Tuple, List
import math
import torch
from einops import einops
from torch import nn, Tensor

from .tabpfn_arch.model.loading import (
    load_default_taco_model_config,
    load_model_criterion_from_config,
)
from taco.model.encoder_stats import extract_encoder_stats, EncoderStats


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class TACO(nn.Module):

    def __init__(
        self,
        *,
        use_compressor: bool = True,
        row_compression_percentage: float = 10.0,
        new_tabpfn_config: dict | None = None,
        rcp_sampling: Literal["none", "uniform"] = "none",
        rcp_choices: Optional[Iterable[float]] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        rcp_k_min: int = 1,
        max_chunk_size: int = 1000000,
        stats_sample_size: Optional[int] = 3000,
    ):
        super().__init__()

        self.use_compressor = use_compressor
        self.row_compression_percentage = float(row_compression_percentage)
        self.rcp_sampling = rcp_sampling
        self.rcp_choices = tuple(rcp_choices) if rcp_choices is not None else None
        self.rcp_k_min = int(rcp_k_min)

        self.max_chunk_size = max_chunk_size
        self.stats_sample_size = stats_sample_size
        self._eval_cache: dict[tuple[object, ...], dict[str, object]] = {}
        self._eval_cache_cap = 8

        if self.use_compressor:
            self.compressor, _, _ = self._build_tabpfn_backbone(
                new_config=new_tabpfn_config,
            )
            comp_ninp = getattr(self.compressor, "ninp")

            self.predictor, _, pred_config = self._build_tabpfn_backbone(
                new_config=new_tabpfn_config,
            )
            self.tabpfn_config = pred_config
            pred_ninp = getattr(self.predictor, "ninp")

            if comp_ninp == pred_ninp:
                self.z_to_pred = ResidualMLP(comp_ninp, hidden_mult=2)
            else:
                self.z_to_pred = nn.Linear(comp_ninp, pred_ninp, bias=False)

        else:
            self.predictor, _, pred_config = self._build_tabpfn_backbone(
                new_config=new_tabpfn_config,
            )
            self.tabpfn_config = pred_config

    @staticmethod
    def _build_tabpfn_backbone(
        *,
        new_config: dict | None,
    ):
        config = new_config if new_config is not None else load_default_taco_model_config()
        return load_model_criterion_from_config(
            config,
            cache_trainset_representation=False,
        )

    def _keep_count(self, train_size: int) -> int:
        return max(1, math.ceil(train_size * self.row_compression_percentage / 100.0))

    def _sample_keep_count(self, train_size: int) -> int:
        if self.rcp_sampling == "uniform" and self.rcp_choices:
            p = float(random.choice(self.rcp_choices))
        else:
            p = self.row_compression_percentage
        K = math.ceil(train_size * p / 100.0)
        return max(self.rcp_k_min, min(K, train_size))

    def _no_grad_ctx(self):
        return torch.inference_mode() if not self.training else torch.enable_grad()

    def _compress_latents(self, X: Tensor, y_train: Tensor, train_size: int):
        """
        Returns:
          compressed_ctx: (B, K, F + 1, Ec)
          K: int
        """
        assert self.use_compressor, "Compressor disabled."
        assert train_size <= X.shape[1]
        B, T, F = X.shape
        K = self._sample_keep_count(train_size)

        self.compressor = self.compressor.to(X.device)
        x_train = X[:, :train_size, :].transpose(0, 1)
        x_queries = X[:, train_size - K:train_size, :].transpose(0, 1)
        x_for_comp = torch.cat([x_train, x_queries], dim=0)
        y_for_comp = y_train[:, :train_size].to(X.dtype).transpose(0, 1)
        compressed_ctx = self.compressor.emit_context(
            x=x_for_comp,
            y=y_for_comp,
            single_eval_pos=train_size,
            source="test",
        )


        return compressed_ctx, K


    def _compress_single_chunk(
        self,
        X_chunk: Tensor,  # (B, N_chunk, F)
        y_chunk: Tensor,  # (B, N_chunk)
    ) -> Tuple[Tensor, int]:
        """
        Compress a single chunk of data.

        Args:
            X_chunk: Features for this chunk (B, N_chunk, F)
            y_chunk: Labels for this chunk (B, N_chunk)

        Returns:
            compressed_ctx: (B, K, F+1, E) compressed context
            K: number of compressed rows
        """
        B, N_chunk, F = X_chunk.shape
        K = self._sample_keep_count(N_chunk)

        self.compressor = self.compressor.to(X_chunk.device, dtype=X_chunk.dtype)

        x_train = X_chunk.transpose(0, 1)  # (N_chunk, B, F)
        x_queries = X_chunk[:, -K:, :].transpose(0, 1)  # (K, B, F)
        x_for_comp = torch.cat([x_train, x_queries], dim=0)  # (N_chunk + K, B, F)
        y_for_comp = y_chunk.to(X_chunk.dtype).transpose(0, 1)  # (N_chunk, B)

        compressed_ctx = self.compressor.emit_context(
            x=x_for_comp,
            y=y_for_comp,
            single_eval_pos=N_chunk,
            source="test",
        )

        return compressed_ctx, K

    def _compress_chunked(
        self,
        X: Tensor,
        y_train: Tensor,
    ) -> Tuple[Tensor, int, List[int]]:
        """
        Compress large dataset by splitting into chunks.

        Args:
            X: Full feature tensor (B, N_total, F)
            y_train: Full label tensor (B, N_total)

        Returns:
            compressed_ctx: (B, K_total, F+1, E) concatenated compressed contexts
            K_total: total number of compressed rows
            chunk_Ks: list of K values per chunk
        """
        B, N_total, F = X.shape

        if N_total <= self.max_chunk_size:
            ctx, K = self._compress_latents(X, y_train, N_total)
            return ctx, K, [K]

        num_chunks = math.ceil(N_total / self.max_chunk_size)
        chunk_size = math.ceil(N_total / num_chunks)

        compressed_chunks = []
        chunk_Ks = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N_total)

            X_chunk = X[:, start_idx:end_idx, :]
            y_chunk = y_train[:, start_idx:end_idx]

            with self._no_grad_ctx():
                ctx_chunk, K_chunk = self._compress_single_chunk(X_chunk, y_chunk)

            compressed_chunks.append(ctx_chunk)
            chunk_Ks.append(K_chunk)

        compressed_ctx = torch.cat(compressed_chunks, dim=1)
        K_total = sum(chunk_Ks)

        return compressed_ctx, K_total, chunk_Ks

    def _needs_chunking(self, train_size: int) -> bool:
        """Check if dataset needs chunked compression."""
        return train_size > self.max_chunk_size

    def _predict_with_tabpfn_hybrid(self, X: Tensor, y_train: Tensor) -> Tensor:
        B, T, F = X.shape
        N = y_train.shape[1]
        assert N <= T
        M = T - N
        if M == 0:
            C = getattr(self.predictor, "n_out", 1)
            return X.new_zeros((B, 0, C))

        self.predictor = self.predictor.to(X.device)
        self.z_to_pred = self.z_to_pred.to(X.device)

        needs_chunking = self._needs_chunking(N) and not self.training
        use_eval_cache = (not self.training) and self.rcp_sampling in (None, "none")
        cache_key = None
        cached = None

        if use_eval_cache:
            slot = int(getattr(self, "_current_cache_slot", 0))
            cache_key = (
                "slot",
                slot,
                int(N),
                int(F),
                float(self.row_compression_percentage),
                str(self.rcp_sampling),
                int(self.max_chunk_size),
                int(self.stats_sample_size) if self.stats_sample_size is not None else None,
                bool(needs_chunking),
            )
            cached = self._eval_cache.get(cache_key)

        if cached is not None:
            ctx_cached = cached["ctx"]
            enc_stats_cached = cached["enc_stats"]
            compressed_ctx = ctx_cached.to(
                device=X.device,
                dtype=ctx_cached.dtype,
                non_blocking=True,
            )
            encoder_stats = enc_stats_cached.to(
                device=X.device,
                dtype=ctx_cached.dtype,
            )
        else:
            if needs_chunking:
                with self._no_grad_ctx():
                    compressed_ctx_Ec, _, _ = self._compress_chunked(X[:, :N, :], y_train)
                compressed_ctx = self.z_to_pred(compressed_ctx_Ec)
            else:
                with self._no_grad_ctx():
                    compressed_ctx_Ec, _ = self._compress_latents(X, y_train, N)
                compressed_ctx = self.z_to_pred(compressed_ctx_Ec)

            if (
                needs_chunking
                and self.stats_sample_size is not None
                and N > self.stats_sample_size
            ):
                sample_idx = torch.randperm(N, device=X.device)[:self.stats_sample_size]
                stats_x = X[:, sample_idx, :].transpose(0, 1)
            else:
                stats_x = X[:, :N, :].transpose(0, 1)

            num_groups = compressed_ctx.shape[2] - 1
            encoder_stats = self._compute_encoder_stats_from_x(
                stats_x=stats_x,
                num_groups=num_groups,
            )

            if use_eval_cache and cache_key is not None:
                if cache_key not in self._eval_cache and len(self._eval_cache) >= self._eval_cache_cap:
                    self._eval_cache.clear()
                self._eval_cache[cache_key] = {
                    "ctx": compressed_ctx.detach().to(device=X.device),
                    "enc_stats": encoder_stats.detach().to(device=X.device),
                }

        test_x = X[:, N:, :].transpose(0, 1)
        del X
        del y_train
        with self._no_grad_ctx():
            out = self.predictor.predict_with_preembedded_context(
                context_block=compressed_ctx,
                test_x=test_x,
                encoder_stats=encoder_stats,
            )

        return out

    def _predict_without_compressor(self, X: Tensor, y_train: Tensor) -> Tensor:
        B, T, F = X.shape
        N = y_train.shape[1]

        assert N <= T
        M = T - N
        if M == 0:
            C = getattr(self.predictor, "n_out", 1)
            return X.new_zeros((B, 0, C))

        self.predictor = self.predictor.to(X.device, dtype=X.dtype)

        train_x = X[:, :N, :].transpose(0, 1)
        test_x = X[:, N:, :].transpose(0, 1)
        y_train = y_train.transpose(0, 1).to(dtype=X.dtype)


        with self._no_grad_ctx():
            logits = self.predictor(
                train_x=train_x,
                train_y=y_train,
                test_x=test_x,
                only_return_standard_out=True,
            )
        return logits.permute(1, 0, 2)

    def clear_eval_cache(self) -> None:
        """Clear cached compressed contexts used for repeated evaluation."""
        self._eval_cache.clear()

    def _train_forward_compress(self, X: Tensor, y_train: Tensor) -> Tensor:
        return self._predict_with_tabpfn_hybrid(X=X, y_train=y_train)

    def _train_forward(self, X: Tensor, y_train: Tensor) -> Tensor:
        return self._predict_without_compressor(X, y_train)

    def forward(self, X: Tensor, y_train: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        if not self.use_compressor:
            logits = self._train_forward(X, y_train)
        else:
            logits = self._train_forward_compress(X, y_train)
        return logits

    def _compute_encoder_stats_from_x(
        self,
        *,
        stats_x: torch.Tensor,
        num_groups: int,
    ) -> EncoderStats:
        """Compute encoder statistics from raw features."""
        B = stats_x.shape[1]
        g = int(getattr(self.predictor, "features_per_group", 1))
        needed = num_groups * g

        x = stats_x
        if x.shape[2] < needed:
            x = torch.cat([x, x.new_zeros(x.shape[0], x.shape[1], needed - x.shape[2])], dim=-1)
        elif x.shape[2] > needed:
            x = x[:, :, :needed]

        x_grouped = einops.rearrange(x, "s b (f n) -> s (b f) n", f=num_groups, n=g)

        with torch.no_grad():
            _ = self.predictor.encoder(
                {"main": x_grouped},
                single_eval_pos=x_grouped.shape[0],
                cache_trainset_representation=False,
            )

            enc_stats = extract_encoder_stats(
                encoder=self.predictor.encoder,
                batch_size=B,
                features_per_group=g,
            ).detach()

        return enc_stats
