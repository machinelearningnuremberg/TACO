"""Module that defines different ways to run inference with TabPFN."""

#  Copyright (c) Prior Labs GmbH 2025.
#  Modified by the TACO contributors in 2026. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override
import math
import joblib
import numpy as np
import torch

from taco.model.tabpfn_arch.model.memory import MemoryUsageEstimator
from taco.model.tabpfn_arch.preprocessing import fit_preprocessing
from taco.model.tabpfn_arch.utils import get_autocast_context

if TYPE_CHECKING:
    from taco.model.tabpfn_arch.model.preprocessing import SequentialFeatureTransformer
    from taco.model.tabpfn_arch.model.transformer import PerFeatureTransformer
    from taco.model.tabpfn_arch.preprocessing import EnsembleConfig


logger = logging.getLogger(__name__)


@dataclass
class InferenceEngine(ABC):
    """These define how tabpfn inference can be run.

    As there are many things that can be cached, with multiple ways to parallelize,
    `Executor` defines three primary things:

    Most will define a method `prepare()` which is specific to that inference engine.
    These do not share a common interface.

    1. What to cache:

        As we can prepare a lot of the transformers context, there is a tradeoff in
        terms of how much memory to be spent in caching. This memory is used when
        `prepare()` is called, usually in `fit()`.

    2. Using the cached data for inference:

        Based on what has been prepared for the transformer context,
        `iter_outputs()` will use this cached information to make predictions.

    3. Controlling parallelism:

        As we have trivially parallel parts for inference, we can parallelize them.
        However as the GPU is typically a bottle-neck in most systems, we can define,
        where and how we would like to parallelize the inference.

    The InferenceEngineBatchedNoPreprocessing
    InferenceEngineCachePreprocessing engines also support toggling
    `torch.use_torch_inference_mode` via `use_torch_inference_mode`
    to enable/disable gradient tracking during prediction.
    """

    save_peak_mem: bool | Literal["auto"] | float | int
    dtype_byte_size: int
    ensemble_configs: Sequence[EnsembleConfig]

    @abstractmethod
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor, EnsembleConfig]]:
        """Iterate over the outputs of the model.

        One for each ensemble configuration that was used to initialize the executor.

        Args:
            X: The input data to make predictions on.
            device: The device to run the model on.
            autocast: Whether to use torch.autocast during inference.
        """
        ...

    def use_torch_inference_mode(self, *, use_inference: bool):
        """Enable/Disable `torch.inference_mode`.

        Disabling allows backpropagation (gradients) but is slower and uses more
        memory during prediction. Enabling is faster for pure inference.

        Only `InferenceEngineBatchedNoPreprocessing` and
        `InferenceEngineCachePreprocessing` currently support this method. Other
        engines will raise `NotImplementedError`.

        Called internally by methods like
        `TabPFNClassifier.predict_proba_from_preprocessed` (for batched engine) and
        `TabPFNRegressor.forward` (for batched & fit_preprocessors engines)
        when gradients might be needed (e.g., for fine-tuning) or when pure
        inference speed is desired.

        """
        raise NotImplementedError(
            "This inference engine does not support torch.inference_mode changes."
        )

    def save_state_expect_model_weights(self, path: str | Path) -> None:
        """Persist the executor state to ``path`` without the model weights.

        The state is first moved to CPU so the resulting file can be loaded
        on machines without a GPU. The large model weights are explicitly
        excluded to keep the file small and efficient.
        """
        state_copy = deepcopy(self)

        # Decouple the large model weights before serialization
        if hasattr(state_copy, "model"):
            state_copy.model = None
        if hasattr(state_copy, "models"):
            state_copy.models = None  # For KV cache engine

        joblib.dump(state_copy, path)

    @staticmethod
    def load_state(path: str | Path) -> InferenceEngine:
        """Load an executor saved with :meth:`save_state`."""
        return joblib.load(Path(path))


@dataclass
class InferenceEngineOnDemand(InferenceEngine):
    """Inference engine that does not cache anything, computes everything as needed.

    This is one of the slowest ways to run inference, as computation that could be
    cached is recomputed on every call. However the memory demand is lowest and
    can be more trivially parallelized across GPUs with some work.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    ensemble_configs: Sequence[EnsembleConfig]
    cat_ix: list[int]
    static_seed: int
    n_workers: int
    model: PerFeatureTransformer
    force_inference_dtype: torch.dtype | None

    @classmethod
    def prepare(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
        model: PerFeatureTransformer,
        ensemble_configs: Sequence[EnsembleConfig],
        rng: np.random.Generator,
        n_workers: int,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        shuffle_rows: bool = False,
    ) -> InferenceEngineOnDemand:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            rng: The random number generator.
            n_workers: The number of workers to use.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            shuffle_rows: Whether to shuffle the training data rows.
        """
        # Shuffle training data if enabled
        if shuffle_rows:
            shuffle_idx = rng.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

        # We save it as a static seed to be reproducible across predicts
        static_seed = rng.integers(0, int(np.iinfo(np.int32).max))
        return cls(
            X_train=X_train,
            y_train=y_train,
            ensemble_configs=ensemble_configs,
            cat_ix=cat_ix,
            model=model,
            static_seed=static_seed,
            n_workers=n_workers,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        rng = np.random.default_rng(self.static_seed)
        itr = fit_preprocessing(
            configs=self.ensemble_configs,
            X_train=self.X_train,
            y_train=self.y_train,
            random_state=rng,
            cat_ix=self.cat_ix,
            n_workers=self.n_workers,
            parallel_mode="as-ready",
        )

        self.model = self.model.to(device)
        if self.force_inference_dtype is not None:
            self.model = self.model.type(self.force_inference_dtype)

        for config, preprocessor, X_train, y_train, cat_ix in itr:
            X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)  # noqa: PLW2901

            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)

            X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
            batched_cat_ix = [cat_ix]
            y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)  # type: ignore  # noqa: PLW2901

            MemoryUsageEstimator.reset_peak_memory_if_required(
                save_peak_mem=self.save_peak_mem,
                model=self.model,
                X=X_full,
                cache_kv=False,
                dtype_byte_size=self.dtype_byte_size,
                device=device,
                safety_factor=1.2,  # TODO(Arjun): make customizable
            )

            if self.force_inference_dtype is not None:
                X_full = X_full.type(self.force_inference_dtype)
                y_train = y_train.type(self.force_inference_dtype)  # type: ignore  # noqa: PLW2901

            style = None
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(),
            ):
                output = self.model(
                    *(style, X_full, y_train),
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_cat_ix,
                    single_eval_pos=len(y_train),
                )

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, config

        self.model = self.model.cpu()


@dataclass
class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
    """Inference engine that uses preprocessed inputs, and allows batched predictions
    on several datasets at once.

    Args:
            X_trains: The training data.
            y_trains    : The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to enable torch inference mode.
    """

    X_trains: list[torch.Tensor]
    y_trains: list[torch.Tensor]
    cat_ix: list[list[list[int]]]
    model: PerFeatureTransformer
    ensemble_configs: Sequence[EnsembleConfig]
    force_inference_dtype: torch.dtype | None
    inference_mode: bool

    @classmethod
    def prepare(
        cls,
        X_trains: list[torch.Tensor],
        y_trains: list[torch.Tensor],
        *,
        cat_ix: list[list[list[int]]],
        model: PerFeatureTransformer,
        ensemble_configs: Sequence[EnsembleConfig],
        force_inference_dtype: torch.dtype | None,
        inference_mode: bool,
        dtype_byte_size: int,
        save_peak_mem: bool | Literal["auto"] | float | int,
        shuffle_rows: bool = False,
    ) -> InferenceEngineBatchedNoPreprocessing:
        """Prepare the inference engine.

        Args:
            X_trains: The training data.
            y_trains: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            inference_mode: Whether to use torch inference mode.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            shuffle_rows: Whether to shuffle the training data rows.
        """
        # Shuffle training data if enabled
        if shuffle_rows:
            shuffled_X_trains = []
            shuffled_y_trains = []
            for i, (X_train, y_train) in enumerate(zip(X_trains, y_trains)):
                shuffle_idx = torch.randperm(X_train.shape[0], device=X_train.device)
                shuffled_X_trains.append(X_train[shuffle_idx])
                shuffled_y_trains.append(y_train[shuffle_idx])
            X_trains = shuffled_X_trains
            y_trains = shuffled_y_trains

        # We save it as a static seed to be reproducible across predicts
        return cls(
            X_trains=X_trains,
            y_trains=y_trains,
            cat_ix=cat_ix,
            model=model,
            ensemble_configs=ensemble_configs,
            force_inference_dtype=force_inference_dtype,
            inference_mode=inference_mode,
            dtype_byte_size=dtype_byte_size,
            save_peak_mem=save_peak_mem,
        )

    @override
    def iter_outputs(
        self,
        X: list[torch.Tensor],
        *,
        device: torch.device,
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        self.model = self.model.to(device)
        ensemble_size = len(self.X_trains)
        for i in range(ensemble_size):
            single_eval_pos = self.X_trains[i].size(-2)  # End of train data
            train_x_full = torch.cat([self.X_trains[i], X[i]], dim=-2)
            train_y_batch = self.y_trains[i]
            train_x_full = train_x_full.to(device)
            train_y_batch = train_y_batch.to(device)
            if self.force_inference_dtype is not None:
                train_x_full = train_x_full.type(self.force_inference_dtype)
                train_y_batch = train_y_batch.type(self.force_inference_dtype)  # type: ignore

            style = None
            with (
                torch.autocast(device.type, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                output = self.model(
                    *(
                        style,
                        train_x_full.transpose(0, 1),
                        train_y_batch.transpose(0, 1),
                    ),
                    only_return_standard_out=True,
                    categorical_inds=list([cat_item[i] for cat_item in self.cat_ix]),  # noqa: C411
                    single_eval_pos=single_eval_pos,
                )

            yield output, self.ensemble_configs[i]
        if self.inference_mode:  ## if inference
            self.model = self.model.cpu()

    @override
    def use_torch_inference_mode(self, use_inference: bool):
        self.inference_mode = use_inference


@dataclass
class InferenceEngineCachePreprocessing(InferenceEngine):
    """Inference engine that caches the preprocessing for feeding as model context on
    predict.

    This will fit the preprocessors on the training data, as well as cache the
    transformed training data on RAM (not GPU RAM).

    This saves some time on each predict call, at the cost of increasing the amount
    of memory in RAM. The main functionality performed at `predict()` time is to
    forward pass through the model which is currently done sequentially.
    """

    X_trains: Sequence[np.ndarray | torch.Tensor]
    y_trains: Sequence[np.ndarray | torch.Tensor]
    cat_ixs: Sequence[list[int]]
    ensemble_configs: Sequence[EnsembleConfig]
    preprocessors: Sequence[SequentialFeatureTransformer]
    model: PerFeatureTransformer
    force_inference_dtype: torch.dtype | None
    inference_mode: bool
    no_preprocessing: bool = False

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        cat_ix: list[int],
        model: PerFeatureTransformer,
        ensemble_configs: Sequence[EnsembleConfig],
        n_workers: int,
        rng: np.random.Generator,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        inference_mode: bool,
        no_preprocessing: bool = False,
        shuffle_rows: bool = False,
    ) -> InferenceEngineCachePreprocessing:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            n_workers: The number of workers to use.
            rng: The random number generator.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to use torch.inference mode
                (this is quicker but disables backpropagation)
            no_preprocessing: If turned of, the preprocessing on the test
                tensors is tuned off. Used for differentiablity.
            shuffle_rows: Whether to shuffle the training data rows.

        Returns:
            The prepared inference engine.
        """
        # Shuffle training data if enabled
        if shuffle_rows:
            if isinstance(X_train, np.ndarray):
                shuffle_idx = rng.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
            elif isinstance(X_train, torch.Tensor):
                shuffle_idx = torch.randperm(X_train.shape[0], device=X_train.device)
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]

        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_workers=n_workers,
            parallel_mode="block",
        )
        configs, preprocessors, X_trains, y_trains, cat_ixs = list(zip(*itr))
        return InferenceEngineCachePreprocessing(
            X_trains=X_trains,
            y_trains=y_trains,
            model=model,
            cat_ixs=cat_ixs,
            ensemble_configs=configs,
            preprocessors=preprocessors,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
            inference_mode=inference_mode,
            no_preprocessing=no_preprocessing,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray | torch.tensor,
        *,
        device: torch.device,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        self.model = self.model.to(device)
        if self.force_inference_dtype is not None:
            self.model = self.model.type(self.force_inference_dtype)
        for est_idx, (preprocessor, X_train, y_train, config, cat_ix) in enumerate(zip(
            self.preprocessors,
            self.X_trains,
            self.y_trains,
            self.ensemble_configs,
            self.cat_ixs,
        )):
            if not isinstance(X_train, torch.Tensor):
                X_train = torch.as_tensor(X_train, dtype=torch.float32)  # noqa: PLW2901
            X_train = X_train.to(device)  # noqa: PLW2901
            X_test = preprocessor.transform(X).X if not self.no_preprocessing else X
            if not isinstance(X_test, torch.Tensor):
                X_test = torch.as_tensor(X_test, dtype=torch.float32)
            X_test = X_test.to(device)
            X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
            if not isinstance(y_train, torch.Tensor):
                y_train = torch.as_tensor(y_train, dtype=torch.float32)  # noqa: PLW2901
            y_train = y_train.to(device)  # noqa: PLW2901

            batched_cat_ix = [cat_ix]

            # Handle type casting
            with contextlib.suppress(Exception):  # Avoid overflow error
                X_full = X_full.float()
            if self.force_inference_dtype is not None:
                X_full = X_full.type(self.force_inference_dtype)
                y_train = y_train.type(self.force_inference_dtype)  # type: ignore # noqa: PLW2901

            if self.inference_mode:
                MemoryUsageEstimator.reset_peak_memory_if_required(
                   save_peak_mem=self.save_peak_mem,
                   model=self.model,
                   X=X_full,
                   cache_kv=False,
                   device=device,
                   dtype_byte_size=self.dtype_byte_size,
                   safety_factor=1.2,  # TODO(Arjun): make customizable
                )
            else:
                pass

            style = est_idx  # per-estimator id: 0..n_estimators-1
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                output = self.model(
                    *(style, X_full, y_train),
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_cat_ix,
                    single_eval_pos=len(y_train),
                )

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, config
        if self.inference_mode:  ## if inference
            self.model = self.model.cpu()

    @override
    def use_torch_inference_mode(self, use_inference: bool):
        self.inference_mode = use_inference


@dataclass
class InferenceEngineCacheKV(InferenceEngine):
    """Inference engine that caches the actual KV cache calculated from the context
    of the processed training data.

    This is by far the most memory intensive inference engine, as for each ensemble
    member we store the full KV cache of that model. For now this is held in CPU RAM
    (TODO(eddiebergman): verify)
    """

    preprocessors: list[SequentialFeatureTransformer]
    ensemble_configs: list[EnsembleConfig]
    cat_ixs: Sequence[list[int]]
    models: list[PerFeatureTransformer]
    n_train_samples: list[int]
    force_inference_dtype: torch.dtype | None

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        cat_ix: list[int],
        ensemble_configs: Sequence[EnsembleConfig],
        n_workers: int,
        model: PerFeatureTransformer,
        device: torch.device,
        rng: np.random.Generator,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        autocast: bool,
        only_return_standard_out: bool = True,
        shuffle_rows: bool = False,
    ) -> InferenceEngineCacheKV:
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            ensemble_configs: The ensemble configurations to use.
            n_workers: The number of workers to use.
            model: The model to use.
            device: The device to run the model on.
            rng: The random number generator.
            dtype_byte_size: Size of the dtype in bytes.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            autocast: Whether to use torch.autocast during inference.
            only_return_standard_out: Whether to only return the standard output
            shuffle_rows: Whether to shuffle the training data rows.
        """
        # Shuffle training data if enabled
        if shuffle_rows:
            shuffle_idx = rng.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_workers=n_workers,
            parallel_mode="as-ready",
        )
        models: list[PerFeatureTransformer] = []
        preprocessors: list[SequentialFeatureTransformer] = []
        correct_order_configs: list[EnsembleConfig] = []
        cat_ixs: Sequence[list[int]] = []
        n_train_samples: list[int] = []

        for config, preprocessor, X, y, preprocessor_cat_ix in itr:
            cat_ixs.append(preprocessor_cat_ix)
            preprocessors.append(preprocessor)
            correct_order_configs.append(config)
            n_train_samples.append(len(y))

            ens_model = deepcopy(model)
            ens_model = ens_model.to(device)
            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X, dtype=torch.float32, device=device)  # noqa: PLW2901
            X = X.unsqueeze(1)  # noqa: PLW2901
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.float32, device=device)  # noqa: PLW2901

            batched_preprocessor_cat_ix = [preprocessor_cat_ix]

            # We do not reset the peak memory for cache_kv mode
            # because the entire data has to be passed through the model
            # at once to generate the KV cache
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(),
            ):
                ens_model.forward(
                    *(None, X, y),
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_preprocessor_cat_ix,
                    single_eval_pos=len(X),
                )

            if device.type != "cpu":
                ens_model = ens_model.cpu()

            models.append(ens_model)

        return InferenceEngineCacheKV(
            preprocessors=preprocessors,
            ensemble_configs=correct_order_configs,
            cat_ixs=cat_ixs,
            n_train_samples=n_train_samples,
            models=models,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        for preprocessor, model, config, cat_ix, X_train_len in zip(
            self.preprocessors,
            self.models,
            self.ensemble_configs,
            self.cat_ixs,
            self.n_train_samples,
        ):
            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            X_test = X_test.unsqueeze(1)
            batched_cat_ix = [cat_ix]

            MemoryUsageEstimator.reset_peak_memory_if_required(
                save_peak_mem=self.save_peak_mem,
                model=model,
                X=X_test,
                cache_kv=True,
                device=device,
                dtype_byte_size=self.dtype_byte_size,
                safety_factor=1.2,  # TODO(Arjun): make customizable
                n_train_samples=X_train_len,
            )

            model = model.to(device)  # noqa: PLW2901
            style = None

            if self.force_inference_dtype is not None:
                model = model.type(self.force_inference_dtype)  # noqa: PLW2901
                X_test = X_test.type(self.force_inference_dtype)
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(),
            ):
                output = model(
                    *(style, X_test, None),
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_cat_ix,
                    single_eval_pos=None,
                )

            # TODO(eddiebergman): This is not really what we want.
            # We'd rather just say unload from GPU, we already have it available on CPU.
            model = model.cpu()  # noqa: PLW2901

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, config

@dataclass
class InferenceEngineCacheCompressorKV(InferenceEngine):
    """Inference engine that caches the compressor output and predictor KV cache
    when using the compressor.

    This engine:
    1. Caches the compressed context output from the compressor
    2. Caches the KV cache from the predictor model after processing the compressed context
    3. Allows saving/loading these caches for later inference on the same dataset

    This is memory intensive but allows for very fast repeated inference on the same
    training data with different test sets.
    """

    preprocessors: list[SequentialFeatureTransformer]
    ensemble_configs: list[EnsembleConfig]
    cat_ixs: Sequence[list[int]]
    models: list[PerFeatureTransformer]
    n_train_samples: list[int]
    force_inference_dtype: torch.dtype | None
    compressed_contexts: list[torch.Tensor]  # Cached compressor outputs
    encoder_stats_list: list[Any]  # Cached encoder statistics
    K_values: list[int]  # Number of compressed rows per ensemble member
    X_trains: Sequence[np.ndarray | torch.Tensor] | None = None  # Preprocessed training data for recomputation
    y_trains: Sequence[np.ndarray | torch.Tensor] | None = None  # Preprocessed training targets for recomputation
    recompute_compressed_context: bool = False  # If True, recompute compressed context instead of using cache

    @classmethod
    def prepare(  # noqa: PLR0913
            cls,
            X_train: np.ndarray,
            y_train: np.ndarray,
            *,
            cat_ix: list[int],
            ensemble_configs: Sequence[EnsembleConfig],
            n_workers: int,
            model: PerFeatureTransformer,
            device: torch.device,
            rng: np.random.Generator,
            dtype_byte_size: int,
            force_inference_dtype: torch.dtype | None,
            save_peak_mem: bool | Literal["auto"] | float | int,
            autocast: bool,
            only_return_standard_out: bool = True,
            recompute_compressed_context: bool = False,
            shuffle_rows: bool = False,
    ) -> "InferenceEngineCacheCompressorKV":
        """Prepare the inference engine with compressor caching.

        Args:
            recompute_compressed_context: If True, store preprocessed training data
                to allow recomputing compressed context during inference instead of
                using cached values. Useful for ablation studies.
            shuffle_rows: Whether to shuffle the training data rows.
        """
        # Shuffle training data if enabled
        if shuffle_rows:
            shuffle_idx = rng.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

        # Check if model has compressor
        has_compressor = (
                hasattr(model, "core")
                and hasattr(model.core, "use_compressor")
                and model.core.use_compressor
        )
        if not has_compressor:
            raise ValueError(
                "InferenceEngineCacheCompressorKV requires a model with compressor enabled. "
                "Use TabPFNClassifierWithCompressor with use_compressor=True."
            )

        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_workers=n_workers,
            parallel_mode="as-ready",
        )

        items = list(itr)

        models: list[PerFeatureTransformer] = []
        preprocessors: list[SequentialFeatureTransformer] = []
        correct_order_configs: list[EnsembleConfig] = []
        cat_ixs: Sequence[list[int]] = []
        n_train_samples: list[int] = []
        compressed_contexts: list[torch.Tensor] = []
        encoder_stats_list: list[Any] = []
        K_values: list[int] = []
        # Store preprocessed training data if recomputation is needed
        X_trains_list: list[np.ndarray | torch.Tensor] = []
        y_trains_list: list[np.ndarray | torch.Tensor] = []

        dummy_test_x_list: list[torch.Tensor] = []

        # PASS 1: Run compressor (+ z_to_pred + encoder_stats) for each member
        for config, preprocessor, X, y, preprocessor_cat_ix in items:
            cat_ixs.append(preprocessor_cat_ix)
            preprocessors.append(preprocessor)
            correct_order_configs.append(config)
            n_train_samples.append(len(y))

            if recompute_compressed_context:
                X_trains_list.append(X.copy() if isinstance(X, np.ndarray) else X.clone())
                y_trains_list.append(y.copy() if isinstance(y, np.ndarray) else y.clone())

            ens_model = deepcopy(model).to(device)

            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X, dtype=torch.float32, device=device)  # noqa: PLW2901
            X = X.unsqueeze(1)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.float32, device=device)  # noqa: PLW2901

            N = len(y)

            X_for_compressor = X.transpose(0, 1)  # (1, N, F)

            if y.ndim == 1:
                y_train_tensor = y.unsqueeze(0)  # (1, N)
            else:
                y_train_tensor = y
                if y_train_tensor.shape[0] != 1 and y_train_tensor.shape[1] == 1:
                    y_train_tensor = y_train_tensor.transpose(0, 1)

            ens_model.core.eval()
            def _layer_factor(m):
                try:
                    return m.transformer_encoder.layers[0].save_peak_mem_factor
                except Exception as e:
                    return f"<no factor attr: {e}>"

            ens_model.core.compressor.reset_save_peak_mem_factor(MemoryUsageEstimator.SAVE_PEAK_MEM_FACTOR)
            with torch.inference_mode():
                # Check if chunking is needed
                needs_chunking = (
                        hasattr(ens_model.core, "_needs_chunking")
                        and ens_model.core._needs_chunking(N)
                )
                with get_autocast_context(device, enabled=autocast):
                    if needs_chunking:
                        compressed_ctx_Ec, K, _ = ens_model.core._compress_chunked(
                            X_for_compressor, y_train_tensor
                        )
                    else:
                        compressed_ctx_Ec, K = ens_model.core._compress_latents(
                            X_for_compressor, y_train_tensor, N
                        )

                    compressed_ctx = ens_model.core.z_to_pred(compressed_ctx_Ec)

                if needs_chunking and hasattr(ens_model.core, "stats_sample_size"):
                    stats_sample_size = ens_model.core.stats_sample_size
                    if stats_sample_size is not None and N > stats_sample_size:
                        sample_idx = torch.randperm(N, device=X_for_compressor.device)[:stats_sample_size]
                        stats_x = X_for_compressor[:, sample_idx, :].transpose(0, 1)
                    else:
                        stats_x = X_for_compressor.transpose(0, 1)
                else:
                    compression_source = getattr(ens_model.core, "compression_source", "test")
                    if compression_source == "test":
                        S_stats = N - K
                    else:
                        S_stats = N
                    stats_x = X_for_compressor[:, :S_stats, :].transpose(0, 1)

                num_groups = compressed_ctx.shape[2] - 1
                encoder_stats = ens_model.core._compute_encoder_stats_from_x(
                    stats_x=stats_x,
                    num_groups=num_groups,
                )

                dummy_test_x = X_for_compressor[:, :1, :].transpose(0, 1)

                # Move to CPU for storage
                if device.type != "cpu":
                    ens_model = ens_model.cpu()
                    compressed_ctx = compressed_ctx.to("cpu")
                    encoder_stats = encoder_stats.to("cpu")
                    dummy_test_x = dummy_test_x.to("cpu")

            models.append(ens_model)
            compressed_contexts.append(compressed_ctx)
            encoder_stats_list.append(encoder_stats)
            dummy_test_x_list.append(dummy_test_x)
            K_values.append(int(K))

        # PASS 2: Run predictor once per member to build & cache KV
        for i, ens_model in enumerate(models):
            model_i = ens_model.to(device)  # noqa: PLW2901
            model_i.core.eval()

            predictor = model_i.core.predictor.to(device)

            compressed_ctx = compressed_contexts[i].to(device)
            encoder_stats = encoder_stats_list[i].to(device)
            dummy_test_x = dummy_test_x_list[i].to(device)

            if hasattr(predictor, "cache_trainset_representation"):
                predictor.cache_trainset_representation = True

            if force_inference_dtype is not None:
                predictor = predictor.type(force_inference_dtype)  # noqa: PLW2901
                compressed_ctx = compressed_ctx.type(force_inference_dtype)
            with torch.inference_mode():
                with get_autocast_context(device, enabled=autocast):
                    _ = predictor.predict_with_preembedded_context(
                        context_block=compressed_ctx,
                        test_x=dummy_test_x,
                        encoder_stats=encoder_stats,
                    )

            if device.type != "cpu":
                models[i] = model_i.cpu()
                compressed_contexts[i] = compressed_contexts[i].to("cpu")
                encoder_stats_list[i] = encoder_stats_list[i].to("cpu")
                dummy_test_x_list[i] = dummy_test_x_list[i].to("cpu")

            del model_i, predictor, compressed_ctx, encoder_stats, dummy_test_x
            torch.cuda.empty_cache()

        return InferenceEngineCacheCompressorKV(
            preprocessors=preprocessors,
            ensemble_configs=correct_order_configs,
            cat_ixs=cat_ixs,
            n_train_samples=n_train_samples,
            models=models,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
            compressed_contexts=compressed_contexts,
            encoder_stats_list=encoder_stats_list,
            K_values=K_values,
            X_trains=X_trains_list if recompute_compressed_context else None,
            y_trains=y_trains_list if recompute_compressed_context else None,
            recompute_compressed_context=recompute_compressed_context,
        )

    @override
    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        """Iterate over outputs using cached compressor output and KV cache."""
        if self.recompute_compressed_context:
            if self.X_trains is None or self.y_trains is None:
                raise ValueError(
                    "recompute_compressed_context is True but X_trains/y_trains are not stored. "
                    "Reinitialize the engine with recompute_compressed_context=True in prepare()."
                )
            iterable = zip(
                self.preprocessors,
                self.models,
                self.ensemble_configs,
                self.cat_ixs,
                self.n_train_samples,
                self.X_trains,
                self.y_trains,
            )
        else:
            iterable = zip(
                self.preprocessors,
                self.models,
                self.ensemble_configs,
                self.cat_ixs,
                self.n_train_samples,
                self.compressed_contexts,
                self.encoder_stats_list,
                self.K_values,
            )

        for item in iterable:
            if self.recompute_compressed_context:
                preprocessor, model, config, cat_ix, X_train_len, X_train, y_train = item
            else:
                preprocessor, model, config, cat_ix, X_train_len, compressed_ctx, encoder_stats, K = item

            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            X_test = X_test.unsqueeze(1)

            model = model.to(device)  # noqa: PLW2901

            if self.recompute_compressed_context:
                if not isinstance(X_train, torch.Tensor):
                    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device)
                else:
                    X_train_tensor = X_train.to(device)
                X_train_tensor = X_train_tensor.unsqueeze(1)

                if not isinstance(y_train, torch.Tensor):
                    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)
                else:
                    y_train_tensor = y_train.to(device)

                N = len(y_train_tensor)

                X_for_compressor = X_train_tensor.transpose(0, 1)

                if y_train_tensor.ndim == 1:
                    y_train_tensor = y_train_tensor.unsqueeze(0)
                else:
                    if y_train_tensor.shape[0] != 1 and y_train_tensor.shape[1] == 1:
                        y_train_tensor = y_train_tensor.transpose(0, 1)

                model.core.eval()

                needs_chunking = (
                    hasattr(model.core, "_needs_chunking")
                    and model.core._needs_chunking(N)
                )

                with get_autocast_context(device, enabled=autocast):
                    if needs_chunking:
                        compressed_ctx_Ec, K, _ = model.core._compress_chunked(
                            X_for_compressor, y_train_tensor
                        )
                    else:
                        compressed_ctx_Ec, K = model.core._compress_latents(
                            X_for_compressor, y_train_tensor, N
                        )

                    compressed_ctx = model.core.z_to_pred(compressed_ctx_Ec)

                if needs_chunking and hasattr(model.core, "stats_sample_size"):
                    stats_sample_size = model.core.stats_sample_size
                    if stats_sample_size is not None and N > stats_sample_size:
                        sample_idx = torch.randperm(N, device=X_for_compressor.device)[:stats_sample_size]
                        stats_x = X_for_compressor[:, sample_idx, :].transpose(0, 1)
                    else:
                        stats_x = X_for_compressor.transpose(0, 1)
                else:
                    compression_source = getattr(model.core, "compression_source", "test")
                    if compression_source == "test":
                        S_stats = N - K
                    else:
                        S_stats = N
                    stats_x = X_for_compressor[:, :S_stats, :].transpose(0, 1)

                num_groups = compressed_ctx.shape[2] - 1
                encoder_stats = model.core._compute_encoder_stats_from_x(
                    stats_x=stats_x,
                    num_groups=num_groups,
                )
            else:
                compressed_ctx = compressed_ctx.to(device)
                encoder_stats = encoder_stats.to(device)

            predictor = model.core.predictor

            if hasattr(predictor, "cache_trainset_representation"):
                predictor.cache_trainset_representation = True

            if self.force_inference_dtype is not None:
                predictor = predictor.type(self.force_inference_dtype)  # noqa: PLW2901
                compressed_ctx = compressed_ctx.type(self.force_inference_dtype)
                X_test = X_test.type(self.force_inference_dtype)

            M = X_test.shape[0]
            max_chunk_size = getattr(model.core, "max_chunk_size", 10000)
            max_test_batch = max_chunk_size - K

            if M <= max_test_batch:
                with (
                    get_autocast_context(device, enabled=autocast),
                    torch.inference_mode(),
                ):
                    output = predictor.predict_with_preembedded_context(
                        context_block=compressed_ctx,
                        test_x=X_test,
                        encoder_stats=encoder_stats,
                        use_cached_embeddings=True,
                    )
            else:
                num_batches = (M + max_test_batch - 1) // max_test_batch
                outputs = []
                for i in range(num_batches):
                    start_idx = i * max_test_batch
                    end_idx = min((i + 1) * max_test_batch, M)
                    test_batch = X_test[start_idx:end_idx]

                    with (
                        get_autocast_context(device, enabled=autocast),
                        torch.inference_mode(),
                    ):
                        batch_out = predictor.predict_with_preembedded_context(
                            context_block=compressed_ctx,
                            test_x=test_batch,
                            encoder_stats=encoder_stats,
                            use_cached_embeddings=True,
                        )
                    outputs.append(batch_out)

                output = torch.cat(outputs, dim=1)  # (B, M, C)

            output = output.permute(1, 0, 2)  # (M, B, C)
            output = output if isinstance(output, dict) else output.squeeze(1)  # (M, C) when B=1

            yield output, config


    @override
    def use_torch_inference_mode(self, use_inference: bool):
        self.inference_mode = use_inference

    @override
    def save_state_expect_model_weights(self, path: str | Path) -> None:
        """Persist the executor state without model weights, but with cached data.
        
        This saves the compressed contexts and encoder statistics, which are the
        expensive parts to compute. The KV cache is stored in the model's buffers
        and will be regenerated when the model is loaded and used for inference.
        To save the KV cache as well, you would need to save the models (which
        includes the model weights and is much larger).
        """
        state_copy = deepcopy(self)

        # Decouple the large model weights before serialization
        if hasattr(state_copy, "model"):
            state_copy.model = None
        if hasattr(state_copy, "models"):
            state_copy.models = None

        # Keep compressed contexts and encoder stats (they're already on CPU)
        # Note: KV cache is stored in model buffers and will be regenerated on load
        joblib.dump(state_copy, path)




@dataclass
class InferenceEngineWithCompression(InferenceEngine):
    """Inference engine that compresses training data during prediction.

    This engine performs compression of the training data during prediction (not fitting)
    to reduce the context size. Compression can be done via:
    1. Random sampling
    2. K nearest neighbors for each test point (K determined by context window size)

    This engine should only be used when the compressor is NOT already used.
    """

    X_trains: Sequence[np.ndarray | torch.Tensor]
    y_trains: Sequence[np.ndarray | torch.Tensor]
    cat_ixs: Sequence[list[int]]
    ensemble_configs: Sequence[EnsembleConfig]
    preprocessors: Sequence[SequentialFeatureTransformer]
    model: PerFeatureTransformer
    force_inference_dtype: torch.dtype | None
    inference_mode: bool
    compression_method: Literal["random", "knn"] = "knn"
    compression_rate_percentage: int | None = None
    rng: np.random.Generator | None = None

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        cat_ix: list[int],
        model: PerFeatureTransformer,
        ensemble_configs: Sequence[EnsembleConfig],
        n_workers: int,
        rng: np.random.Generator,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        inference_mode: bool,
        compression_method: Literal["random", "knn"] = "knn",
        compression_rate_percentage: int | None = None,
        shuffle_rows: bool = False,
    ) -> InferenceEngineWithCompression:
        """Prepare the inference engine with compression support.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            n_workers: The number of workers to use.
            rng: The random number generator.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to use torch.inference mode.
            compression_method: Method for compression ("random" or "knn").
            context_window_size: Target context window size. If None, uses a default
                based on model config or training data size.
            shuffle_rows: Whether to shuffle the training data rows.
        """
        # Check that compressor is not already used
        has_compressor = (
            hasattr(model, "core")
            and hasattr(model.core, "use_compressor")
            and model.core.use_compressor
        )
        if has_compressor:
            raise ValueError(
                "InferenceEngineWithCompression should not be used when the model "
                "already has a compressor. Use InferenceEngineCacheCompressorKV instead."
            )

        # Shuffle training data if enabled
        if shuffle_rows:
            if isinstance(X_train, np.ndarray):
                shuffle_idx = rng.permutation(len(X_train))
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]
            elif isinstance(X_train, torch.Tensor):
                shuffle_idx = torch.randperm(X_train.shape[0], device=X_train.device)
                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]

        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_workers=n_workers,
            parallel_mode="block",
        )
        configs, preprocessors, X_trains, y_trains, cat_ixs = list(zip(*itr))

        # Determine default context window size if not provided
        if compression_rate_percentage is None:
            raise ValueError("compression_rate_percentage must be provided.")
        return InferenceEngineWithCompression(
            X_trains=X_trains,
            y_trains=y_trains,
            model=model,
            cat_ixs=cat_ixs,
            ensemble_configs=configs,
            preprocessors=preprocessors,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
            inference_mode=inference_mode,
            compression_method=compression_method,
            compression_rate_percentage=compression_rate_percentage,
            rng=rng,
        )

    def _compress_training_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        method: Literal["random", "knn"],
        compression_rate_percentage: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress training data based on the specified method.

        Args:
            X_train: Training features tensor of shape (N, F).
            y_train: Training targets tensor of shape (N,).
            X_test: Test features tensor of shape (M, F).
            method: Compression method ("random" or "knn").
            compression_rate_percentage: Target compression rate as a percentage.
            device: Device to perform computation on.

        Returns:
            Compressed X_train and y_train tensors.
        """
        N = X_train.shape[0]
        M = X_test.shape[0]
        context_window_size = math.ceil(int(compression_rate_percentage * N) / 100)

        # If training data is already small enough, no compression needed
        if N <= context_window_size:
            return X_train, y_train

        if method == "random":
            # Random sampling
            num_samples = min(context_window_size, N)
            # Use numpy RNG if available for reproducibility, otherwise use torch
            if self.rng is not None:
                indices_np = self.rng.choice(N, size=num_samples, replace=False)
                indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
            else:
                indices = torch.randperm(N, device=device)[:num_samples]
            return X_train[indices], y_train[indices]

        elif method == "knn":
            context_window_size = math.ceil(int(compression_rate_percentage * N) / 100)
            K_per_test = max(1, context_window_size // M) if M > 0 else context_window_size
            K_per_test = min(K_per_test, N)
            distances = torch.cdist(X_test, X_train, p=2)  # (M, N)
            _, knn_indices = torch.topk(distances, k=K_per_test, dim=1, largest=False)  # (M, K_per_test)

            unique_indices = torch.unique(knn_indices.flatten())

            target = min(context_window_size, N)

            if unique_indices.numel() < target:
                k = min(max(K_per_test + 1, 1), N)
                while unique_indices.numel() < target and k < N:
                    k = min(max(k * 2, k + 1), N)

                    _, knn_indices = torch.topk(distances, k=k, dim=1, largest=False)
                    unique_indices = torch.unique(knn_indices.flatten())
            if len(unique_indices) > context_window_size:
                num_samples = context_window_size
                if self.rng is not None:
                    indices_np = self.rng.choice(len(unique_indices), size=num_samples, replace=False)
                    perm_indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
                else:
                    perm_indices = torch.randperm(len(unique_indices), device=device)[:num_samples]
                selected_indices = unique_indices[perm_indices]
            else:
                selected_indices = unique_indices
            return X_train[selected_indices], y_train[selected_indices]

        else:
            raise ValueError(f"Unknown compression method: {method}")

    @override
    def iter_outputs(
        self,
        X: np.ndarray | torch.tensor,
        *,
        device: torch.device,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        """Iterate over outputs with compressed training data."""
        self.model = self.model.to(device)
        if self.force_inference_dtype is not None:
            self.model = self.model.type(self.force_inference_dtype)

        for est_idx, (preprocessor, X_train, y_train, config, cat_ix) in enumerate(zip(
            self.preprocessors,
            self.X_trains,
            self.y_trains,
            self.ensemble_configs,
            self.cat_ixs,
        )):
            if not isinstance(X_train, torch.Tensor):
                X_train = torch.as_tensor(X_train, dtype=torch.float32)  # noqa: PLW2901
            X_train = X_train.to(device)  # noqa: PLW2901

            X_test = preprocessor.transform(X).X if not hasattr(self, "no_preprocessing") or not self.no_preprocessing else X
            if not isinstance(X_test, torch.Tensor):
                X_test = torch.as_tensor(X_test, dtype=torch.float32)
            X_test = X_test.to(device)

            if not isinstance(y_train, torch.Tensor):
                y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)
            else:
                y_train_tensor = y_train.to(device)

            X_train_compressed, y_train_compressed = self._compress_training_data(
                X_train=X_train,
                y_train=y_train_tensor,
                X_test=X_test,
                method=self.compression_method,
                compression_rate_percentage=self.compression_rate_percentage,
                device=device,
            )

            if not isinstance(y_train_compressed, torch.Tensor):
                y_train_compressed = torch.as_tensor(y_train_compressed, dtype=torch.float32, device=device)

            X_full = torch.cat([X_train_compressed, X_test], dim=0).unsqueeze(1)
            batched_cat_ix = [cat_ix]

            with contextlib.suppress(Exception):  # Avoid overflow error
                X_full = X_full.float()
            if self.force_inference_dtype is not None:
                X_full = X_full.type(self.force_inference_dtype)
                y_train_compressed = y_train_compressed.type(self.force_inference_dtype)  # type: ignore

            if self.inference_mode:
                MemoryUsageEstimator.reset_peak_memory_if_required(
                    save_peak_mem=self.save_peak_mem,
                    model=self.model,
                    X=X_full,
                    cache_kv=False,
                    device=device,
                    dtype_byte_size=self.dtype_byte_size,
                    safety_factor=1.2,  # TODO(Arjun): make customizable
                )
            else:
                pass

            style = est_idx  # per-estimator id: 0..n_estimators-1
            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                output = self.model(
                    *(style, X_full, y_train_compressed),
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=batched_cat_ix,
                    single_eval_pos=len(y_train_compressed),
                )

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, config
        if self.inference_mode:  ## if inference
            self.model = self.model.cpu()

    @override
    def use_torch_inference_mode(self, use_inference: bool):
        self.inference_mode = use_inference



def _to_cpu_float32(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(x, device="cpu", dtype=torch.float32)


def _y_to_cpu_float32_1d(y: np.ndarray | torch.Tensor) -> torch.Tensor:
    y_t = _to_cpu_float32(y)
    return y_t.view(-1)


@dataclass
class InferenceEngineChunkedCompressor:
    """
    Chunked compressor inference engine.

    - prepare(): fits preprocessing per ensemble member, then builds compressed context by streaming train
      chunks to GPU via core._compress_single_chunk, caches (compressed_context, encoder_stats, K) on CPU.
    - iter_outputs(): preprocesses test, moves cached ctx/stats to device, runs predictor in batches.

    NOTE: This keeps the same general policy as other engines:
      - caches live on CPU
      - model is moved to GPU during iter_outputs and moved back to CPU afterward
      - no attempt to keep everything resident on GPU between calls
    """

    save_peak_mem: bool | Literal["auto"] | float | int
    dtype_byte_size: int
    ensemble_configs: Sequence["EnsembleConfig"]

    preprocessors: Sequence["SequentialFeatureTransformer"]
    cat_ixs: Sequence[list[int]]
    X_trains: Sequence[np.ndarray | torch.Tensor]   # stored CPU side
    y_trains: Sequence[np.ndarray | torch.Tensor]   # stored CPU side

    model: "PerFeatureTransformer"
    force_inference_dtype: torch.dtype | None
    inference_mode: bool

    compressed_contexts: list[torch.Tensor]         # each: (1, K_total, F+1, Epred) on CPU
    encoder_stats_list: list["EncoderStats"]        # each on CPU
    K_values: list[int]

    train_chunk_size: int | None = None
    max_test_batch: int | None = None
    stats_sample_size_override: int | None = None

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        cat_ix: list[int],
        model: "PerFeatureTransformer",
        ensemble_configs: Sequence["EnsembleConfig"],
        n_workers: int,
        rng: np.random.Generator,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        device: torch.device,
        autocast: bool,
        inference_mode: bool,
        shuffle_rows: bool = False,
        train_chunk_size: int | None = None,
        max_test_batch: int | None = None,
        stats_sample_size_override: int | None = None,
    ) -> "InferenceEngineChunkedCompressor":
        if shuffle_rows:
            if isinstance(X_train, np.ndarray):
                idx = rng.permutation(len(X_train))
                X_train = X_train[idx]
                y_train = y_train[idx]
            elif isinstance(X_train, torch.Tensor):
                idx = torch.randperm(X_train.shape[0], device=X_train.device)
                X_train = X_train[idx]
                y_train = y_train[idx]

        # Must be compressor-enabled
        has_compressor = (
            hasattr(model, "core")
            and hasattr(model.core, "use_compressor")
            and bool(model.core.use_compressor)
        )
        if not has_compressor:
            raise ValueError(
                "InferenceEngineChunkedCompressor requires a compressor-enabled model "
                "(model.core.use_compressor=True)."
            )

        logger.debug("ChunkedCompressor.prepare: fitting preprocessors")
        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_workers=n_workers,
            parallel_mode="block",
        )
        items = list(itr)
        logger.debug("ChunkedCompressor.prepare: got %d ensemble members", len(items))

        preprocessors: list["SequentialFeatureTransformer"] = []
        configs_out: list["EnsembleConfig"] = []
        cat_ixs: list[list[int]] = []
        X_trains: list[np.ndarray | torch.Tensor] = []
        y_trains: list[np.ndarray | torch.Tensor] = []

        compressed_contexts: list[torch.Tensor] = []
        encoder_stats_list: list["EncoderStats"] = []
        K_values: list[int] = []

        for est_idx, (config, preprocessor, Xp, yp, cat_ix_p) in enumerate(items):
            logger.debug(
                "ChunkedCompressor.prepare: ensemble member %d/%d",
                est_idx + 1,
                len(items),
            )

            preprocessors.append(preprocessor)
            configs_out.append(config)
            cat_ixs.append(cat_ix_p)
            X_trains.append(Xp)
            y_trains.append(yp)

            X_cpu = _to_cpu_float32(Xp)          # (N,F)
            y_cpu = _y_to_cpu_float32_1d(yp)     # (N,)
            N = int(y_cpu.numel())
            F = int(X_cpu.shape[1])

            logger.debug("ChunkedCompressor.prepare: train shape N=%d F=%d", N, F)

            # Put model on device for compression
            model_i = model.to(device)
            if force_inference_dtype is not None:
                model_i = model_i.type(force_inference_dtype)
            core = model_i.core
            core.eval()

            core_max_chunk = int(getattr(core, "max_chunk_size", 10000))
            chunk_size = int(train_chunk_size or core_max_chunk)
            n_chunks = int(math.ceil(N / chunk_size))

            logger.debug(
                "ChunkedCompressor.prepare: train_chunk_size=%d core.max_chunk_size=%d chunks=%d autocast=%s dtype=%s",
                chunk_size,
                core_max_chunk,
                n_chunks,
                autocast,
                force_inference_dtype or torch.float32,
            )

            if inference_mode:
                try:
                    MemoryUsageEstimator.reset_peak_memory_if_required(
                        save_peak_mem=save_peak_mem,
                        model=model_i,
                        X=(Xp if isinstance(Xp, torch.Tensor) else torch.empty((1, 1, 1))),  # best-effort
                        cache_kv=False,
                        device=device,
                        dtype_byte_size=dtype_byte_size,
                        safety_factor=1.2,
                    )
                except Exception:
                    pass

            ctx_Ec_chunks_cpu: list[torch.Tensor] = []
            K_total = 0

            with torch.inference_mode(inference_mode):
                for ci, s in enumerate(range(0, N, chunk_size)):
                    e = min(s + chunk_size, N)
                    logger.debug(
                        "ChunkedCompressor.prepare: compressing chunk %d/%d rows %d:%d (n=%d)",
                        ci + 1,
                        n_chunks,
                        s,
                        e,
                        e - s,
                    )

                    Xc = X_cpu[s:e].to(device, non_blocking=True).unsqueeze(0)
                    yc = y_cpu[s:e].to(device, non_blocking=True).unsqueeze(0)

                    if force_inference_dtype is not None:
                        Xc = Xc.to(dtype=force_inference_dtype)
                        yc = yc.to(dtype=force_inference_dtype)

                    with get_autocast_context(device, enabled=autocast):
                        ctx_Ec, Kc = core._compress_single_chunk(Xc, yc)

                    ctx_Ec_chunks_cpu.append(ctx_Ec.detach().cpu())
                    K_total += int(Kc)

                    logger.debug(
                        "ChunkedCompressor.prepare: chunk done kept Kc=%d running K_total=%d",
                        int(Kc),
                        K_total,
                    )

                    del Xc, yc, ctx_Ec
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                logger.debug(
                    "ChunkedCompressor.prepare: compression done, concatenating %d context chunks",
                    len(ctx_Ec_chunks_cpu),
                )
                ctx_Ec_cpu = torch.cat(ctx_Ec_chunks_cpu, dim=1)  # (1,Ktot,F+1,Ec)
                logger.debug(
                    "ChunkedCompressor.prepare: ctx_Ec_cpu shape=%s K_total=%d",
                    tuple(ctx_Ec_cpu.shape),
                    K_total,
                )

                logger.debug("ChunkedCompressor.prepare: projecting context")
                ctx_Ec_gpu = ctx_Ec_cpu.to(device, non_blocking=True)
                if force_inference_dtype is not None:
                    ctx_Ec_gpu = ctx_Ec_gpu.to(dtype=force_inference_dtype)

                with get_autocast_context(device, enabled=autocast):
                    ctx_pred_gpu = core.z_to_pred(ctx_Ec_gpu)

                ctx_pred_cpu = ctx_pred_gpu.detach().cpu()
                logger.debug(
                    "ChunkedCompressor.prepare: ctx_pred_cpu shape=%s",
                    tuple(ctx_pred_cpu.shape),
                )

                del ctx_Ec_gpu, ctx_pred_gpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                logger.debug("ChunkedCompressor.prepare: computing encoder stats")
                stats_sample_size = stats_sample_size_override
                if stats_sample_size is None:
                    stats_sample_size = getattr(core, "stats_sample_size", 10000)

                if stats_sample_size is not None and N > int(stats_sample_size):
                    idx = torch.randperm(N)[: int(stats_sample_size)]
                    stats_x_cpu = X_cpu[idx]  # (S,F)
                    logger.debug(
                        "ChunkedCompressor.prepare: stats sampling S=%d of N=%d",
                        int(stats_sample_size),
                        N,
                    )
                else:
                    stats_x_cpu = X_cpu
                    logger.debug(
                        "ChunkedCompressor.prepare: stats using full train S=%d",
                        stats_x_cpu.shape[0],
                    )

                stats_x = stats_x_cpu.to(device, non_blocking=True).unsqueeze(1)  # (S,1,F)
                if force_inference_dtype is not None:
                    stats_x = stats_x.to(dtype=force_inference_dtype)

                num_groups = ctx_pred_cpu.shape[2] - 1
                enc_stats = core._compute_encoder_stats_from_x(
                    stats_x=stats_x,
                    num_groups=num_groups,
                )
                enc_stats_cpu = enc_stats.detach().to("cpu")
                logger.debug("ChunkedCompressor.prepare: encoder stats done")

                del stats_x, enc_stats
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            compressed_contexts.append(ctx_pred_cpu)
            encoder_stats_list.append(enc_stats_cpu)
            K_values.append(int(K_total))

            if device.type != "cpu":
                model_i = model_i.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.debug("ChunkedCompressor.prepare: all members prepared")
        return cls(
            preprocessors=preprocessors,
            ensemble_configs=configs_out,
            cat_ixs=cat_ixs,
            X_trains=X_trains,
            y_trains=y_trains,
            model=model.cpu() if device.type != "cpu" else model,
            force_inference_dtype=force_inference_dtype,
            inference_mode=inference_mode,
            dtype_byte_size=dtype_byte_size,
            save_peak_mem=save_peak_mem,
            compressed_contexts=compressed_contexts,
            encoder_stats_list=encoder_stats_list,
            K_values=K_values,
            train_chunk_size=train_chunk_size,
            max_test_batch=max_test_batch,
            stats_sample_size_override=stats_sample_size_override,
        )

    def iter_outputs(
        self,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
    ) -> Iterator[tuple[torch.Tensor | dict, "EnsembleConfig"]]:
        logger.debug("ChunkedCompressor.iter_outputs: moving model to device")
        self.model = self.model.to(device)
        if self.force_inference_dtype is not None:
            self.model = self.model.type(self.force_inference_dtype)

        for est_idx, (preprocessor, config, cat_ix, ctx_cpu, enc_cpu, K) in enumerate(
            zip(
                self.preprocessors,
                self.ensemble_configs,
                self.cat_ixs,
                self.compressed_contexts,
                self.encoder_stats_list,
                self.K_values,
            )
        ):
            logger.debug(
                "ChunkedCompressor.iter_outputs: ensemble member %d/%d",
                est_idx + 1,
                len(self.ensemble_configs),
            )

            logger.debug("ChunkedCompressor.iter_outputs: preprocessing test")
            X_test_np = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test_np, dtype=torch.float32, device=device).unsqueeze(1)  # (M,1,F)
            if self.force_inference_dtype is not None:
                X_test = X_test.to(dtype=self.force_inference_dtype)
            M = int(X_test.shape[0])
            logger.debug("ChunkedCompressor.iter_outputs: test shape M=%d", M)

            core = self.model.core
            predictor = core.predictor

            logger.debug("ChunkedCompressor.iter_outputs: moving cached context/stats to device")
            ctx = ctx_cpu.to(device, non_blocking=True)
            enc = enc_cpu.to(device=device, dtype=(self.force_inference_dtype or None))

            if self.force_inference_dtype is not None:
                ctx = ctx.to(dtype=self.force_inference_dtype)

            L_max = 10000

            max_test = L_max - int(K)
            if self.max_test_batch is not None:
                max_test = min(max_test, int(self.max_test_batch))

            if max_test <= 0:
                raise ValueError(
                    f"Context too long: K={int(K)} leaves no room for test tokens under L_max={L_max}. "
                    "Reduce compression (K) or increase model limit."
                )

            n_batches = (M + max_test - 1) // max_test  # ceil division

            logger.debug(
                "ChunkedCompressor.iter_outputs: predicting core.max_chunk_size=%d K=%d max_test_batch=%d batches=%d",
                L_max,
                int(K),
                max_test,
                n_batches,
            )

            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                if M <= max_test:
                    logger.debug("ChunkedCompressor.iter_outputs: predicting single batch")
                    out = predictor.predict_with_preembedded_context(
                        context_block=ctx,
                        test_x=X_test,
                        encoder_stats=enc,
                    )
                else:
                    outs = []
                    for bi, s in enumerate(range(0, M, max_test)):
                        e = min(s + max_test, M)
                        logger.debug(
                            "ChunkedCompressor.iter_outputs: predicting batch %d/%d rows %d:%d (n=%d)",
                            bi + 1,
                            n_batches,
                            s,
                            e,
                            e - s,
                        )
                        batch = X_test[s:e]
                        batch_out = predictor.predict_with_preembedded_context(
                            context_block=ctx,
                            test_x=batch,
                            encoder_stats=enc,
                        )
                        outs.append(batch_out)
                    out = torch.cat(outs, dim=1)  # (B,M,C)

            out = out.permute(1, 0, 2).squeeze(1)

            logger.debug("ChunkedCompressor.iter_outputs: done member")
            yield out, config

        if self.inference_mode:
            logger.debug("ChunkedCompressor.iter_outputs: moving model back to CPU")
            self.model = self.model.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def use_torch_inference_mode(self, *, use_inference: bool):
        self.inference_mode = use_inference
