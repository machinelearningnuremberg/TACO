from __future__ import annotations

import argparse
import logging
import math
import time

import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score

from taco.model.tabpfn_arch.taco_classifier import TACOClassifier


PERCENTAGE_SCALE = 10_000


def estimate_chunking(
    *,
    n_train_rows: int,
    max_chunk_size: int,
    row_compression_percentage: float,
) -> tuple[int, int, int]:
    n_chunks = 0
    max_rows_per_chunk = 0
    compressed_context_rows = 0

    for start in range(0, n_train_rows, max_chunk_size):
        chunk_rows = min(max_chunk_size, n_train_rows - start)
        kept_rows = max(1, math.ceil(chunk_rows * row_compression_percentage / 100.0))
        n_chunks += 1
        max_rows_per_chunk = max(max_rows_per_chunk, chunk_rows)
        compressed_context_rows += kept_rows

    return n_chunks, max_rows_per_chunk, compressed_context_rows


def calculate_row_compression_percentage(
    *,
    n_train_rows: int,
    max_chunk_size: int,
    compressed_context_budget: int,
) -> float:
    n_chunks, _, minimum_context_rows = estimate_chunking(
        n_train_rows=n_train_rows,
        max_chunk_size=max_chunk_size,
        row_compression_percentage=0.0,
    )
    if minimum_context_rows > compressed_context_budget:
        raise ValueError(
            "Cannot satisfy the compressed context budget: keeping one row per "
            f"chunk already needs {minimum_context_rows:,} rows across "
            f"{n_chunks:,} chunks, but the budget is {compressed_context_budget:,}."
        )

    low = 0
    high = 100 * PERCENTAGE_SCALE
    while low < high:
        mid = (low + high + 1) // 2
        percentage = mid / PERCENTAGE_SCALE
        _, _, compressed_context_rows = estimate_chunking(
            n_train_rows=n_train_rows,
            max_chunk_size=max_chunk_size,
            row_compression_percentage=percentage,
        )
        if compressed_context_rows <= compressed_context_budget:
            low = mid
        else:
            high = mid - 1

    return low / PERCENTAGE_SCALE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chunked TabPFN-TACO inference on synthetic data.")
    parser.add_argument(
        "--checkpoint-path",
        default="auto",
        help="Path to a TabPFN-TACO checkpoint, or 'auto' to download from Hugging Face.",
    )
    parser.add_argument(
        "--checkpoint-repo-id",
        default=None,
        help="Optional Hugging Face repo id for --checkpoint-path auto.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use an already cached Hugging Face checkpoint.",
    )
    parser.add_argument("--total-rows", default=50000, type=int)
    parser.add_argument("--train-rows", default=40000, type=int)
    parser.add_argument("--test-rows", default=10000, type=int)
    parser.add_argument("--n-features", default=10, type=int)
    parser.add_argument("--max-context-rows", default=10000, type=int)
    parser.add_argument("--max-chunk-size", default=9500, type=int)
    parser.add_argument(
        "--min-test-batch-rows",
        default=500,
        type=int,
        help="Reserve at least this many test rows per prediction batch.",
    )
    parser.add_argument(
        "--row-compression-percentage",
        default=None,
        type=float,
        help="Optional override. If omitted, it is calculated from the context budget.",
    )
    parser.add_argument("--n-estimators", default=1, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-precision", default="auto")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed checkpoint, preprocessing, chunk compression, and prediction progress logs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("taco.model.tabpfn_arch.inference").setLevel(logging.DEBUG)
        logging.getLogger("taco.model.tabpfn_arch.base").setLevel(logging.INFO)

    if args.n_features < 4:
        raise ValueError("--n-features must be at least 4.")
    if args.train_rows + args.test_rows > args.total_rows:
        raise ValueError("--train-rows + --test-rows must be <= --total-rows.")

    compressed_context_budget = min(
        args.max_context_rows,
        args.max_chunk_size - args.min_test_batch_rows,
    )
    if compressed_context_budget <= 0:
        raise ValueError(
            "--max-chunk-size must be larger than --min-test-batch-rows."
        )

    row_compression_percentage = args.row_compression_percentage
    if row_compression_percentage is None:
        row_compression_percentage = calculate_row_compression_percentage(
            n_train_rows=args.train_rows,
            max_chunk_size=args.max_chunk_size,
            compressed_context_budget=compressed_context_budget,
        )

    n_chunks, max_rows_per_chunk, compressed_context_rows = estimate_chunking(
        n_train_rows=args.train_rows,
        max_chunk_size=args.max_chunk_size,
        row_compression_percentage=row_compression_percentage,
    )
    test_batch_rows = args.max_chunk_size - compressed_context_rows

    if max_rows_per_chunk >= args.max_context_rows:
        raise ValueError(
            f"Raw train chunks must be < {args.max_context_rows:,} rows; "
            f"got {max_rows_per_chunk:,}."
        )
    if compressed_context_rows > args.max_context_rows:
        raise ValueError(
            f"Compressed context must be <= {args.max_context_rows:,} rows; "
            f"got {compressed_context_rows:,}."
        )
    if test_batch_rows < args.min_test_batch_rows:
        raise ValueError(
            f"Need at least {args.min_test_batch_rows:,} test rows per prediction "
            f"batch, but only {test_batch_rows:,} remain."
        )

    print("Chunking plan", flush=True)
    print(f"  total rows: {args.total_rows:,}")
    print(f"  train rows: {args.train_rows:,}")
    print(f"  test rows: {args.test_rows:,}")
    print(f"  row compression percentage: {row_compression_percentage:.4f}")
    print(f"  compressed context budget: {compressed_context_budget:,}")
    print(f"  max raw rows per train chunk: {max_rows_per_chunk:,}")
    print(f"  train chunks: {n_chunks:,}")
    print(f"  compressed context rows: {compressed_context_rows:,}")
    print(f"  test rows per prediction batch: {test_batch_rows:,}")

    print("Generating synthetic data...", flush=True)
    X, y = make_classification(
        n_samples=args.total_rows,
        n_features=args.n_features,
        n_informative=max(2, args.n_features // 2),
        n_redundant=max(0, args.n_features // 5),
        n_classes=2,
        random_state=args.seed,
    )
    X = X.astype("float32", copy=False)

    X_train = X[: args.train_rows]
    y_train = y[: args.train_rows]
    X_test = X[args.train_rows : args.train_rows + args.test_rows]
    y_test = y[args.train_rows : args.train_rows + args.test_rows]

    print(
        f"Train shape: {X_train.shape}, test shape: {X_test.shape}, "
        f"full array size: {X.nbytes / 1024**2:.1f} MiB"
    )
    print(
        f"Running TabPFN-TACO with checkpoint={args.checkpoint_path}, "
        f"device={args.device}, n_estimators={args.n_estimators}"
    )

    if torch.cuda.is_available():
        print(f"CUDA detected: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not detected. Large chunking runs can be slow on CPU.")

    clf_taco = TACOClassifier(
        use_compressor=True,
        row_compression_percentage=row_compression_percentage,
        checkpoint_path=args.checkpoint_path,
        checkpoint_repo_id=args.checkpoint_repo_id,
        local_files_only=args.local_files_only,
        fit_mode="fit_with_chunking",
        max_chunk_size=args.max_chunk_size,
        n_estimators=args.n_estimators,
        device=args.device,
        inference_precision=args.inference_precision,
        ignore_pretraining_limits=True,
        random_state=args.seed,
        shuffle_rows=True,
    )

    fit_start = time.perf_counter()
    print(
        "Fitting TabPFN-TACO: resolving/loading checkpoint, fitting preprocessors, "
        f"and compressing {args.train_rows:,} training rows across {n_chunks:,} chunks...",
        flush=True,
    )
    clf_taco.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    print(
        f"Predicting {args.test_rows:,} test rows in batches of up to {test_batch_rows:,}...",
        flush=True,
    )
    prediction_probabilities = clf_taco.predict_proba(X_test)
    predict_seconds = time.perf_counter() - predict_start

    predictions = prediction_probabilities.argmax(axis=1)

    print(f"Fit time: {fit_seconds:.2f}s")
    print(f"Predict time: {predict_seconds:.2f}s")
    print("TabPFN-TACO ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
    print("TabPFN-TACO Accuracy:", accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()
