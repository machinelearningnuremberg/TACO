from taco.prior import dataset
from transformers import default_data_collator
import time
import json
from pathlib import Path
from typing import Optional, Union
from scipy.stats import loguniform
import torch
from torch.utils.data import IterableDataset
import json
from pathlib import Path
from torch.nn import functional as F
import numpy as np
import pandas as pd
import torch.distributed as dist
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings("ignore", message="Bins whose width are too small")



class MixedDataset(IterableDataset):
    def __init__(self, real_dataset, prior_dataset, total_steps, 
                 start_alpha=0.0, end_alpha=1.0):
        """
        Args:
            real_dataset: IterableDataset for real data
            prior_dataset: IterableDataset for prior/synthetic data
            total_steps: number of training steps for full curriculum
            start_alpha: initial probability of sampling real data (e.g. 0.0)
            end_alpha: final probability of sampling real data (e.g. 1.0)
        """
        self.real_dataset = real_dataset
        self.prior_dataset = prior_dataset
        self.total_steps = total_steps
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha

        self.step = 0  # will be incremented during training

        self.real_iter = iter(real_dataset) if real_dataset is not None else None
        self.prior_iter = iter(prior_dataset) if prior_dataset is not None else None

    def set_step(self, step):
        """Called by training loop to update current step."""
        self.step = step
        if self.real_dataset is not None:
            self.real_dataset.current_idx = step

    def get_alpha(self):
        """Linearly schedule alpha from start_alpha to end_alpha."""
        progress = min(1.0, self.step / self.total_steps)
        return self.start_alpha + progress * (self.end_alpha - self.start_alpha)

    def __iter__(self):
        while True:
            if self.real_dataset is None:
                if self.prior_dataset is None:
                    raise ValueError("Both real_dataset and prior_dataset are None.")
                yield next(self.prior_iter)
                continue

            alpha = self.get_alpha()
            choice = torch.rand(1).item()
            if self.real_dataset.ddp_world_size > 1 and dist.is_initialized():
                choice = _broadcast_choice(choice, device='cpu')
            if choice < alpha:
                try:
                    yield next(self.real_iter)
                except StopIteration:
                    print("Restarting real dataset iterator.", flush=True)
                    self.real_iter = iter(self.real_dataset)
                    yield next(self.real_iter)
            else:
                try:
                    yield next(self.prior_iter)
                except StopIteration:
                    self.prior_iter = iter(self.prior_dataset)
                    yield next(self.prior_iter)


def _broadcast_choice(choice, device):
    """Helper to broadcast a random choice from rank 0 to all ranks."""
    tensor = torch.tensor([choice], device=device)
    if dist.get_backend() == 'nccl':
        tensor = tensor.cuda()
    dist.broadcast(tensor, src=0)
    return tensor.to(device=device).item()

def _broadcast_array(arr, device):
    """Helper to broadcast a numpy array from rank 0 to all ranks."""
    tensor = torch.from_numpy(arr).to(device=device)
    if dist.get_backend() == 'nccl':
        tensor = tensor.cuda()
    dist.broadcast(tensor, src=0)
    return tensor.to(device=device).numpy()

def _broadcast_json(payload: Optional[dict] = None, device="cpu", src=0):
    """Broadcast a JSON-serializable dictionary from rank `src` to all other ranks."""
    if dist.get_backend() == 'nccl':
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    if dist.get_rank() == src:
        encoded = json.dumps(payload).encode("utf-8")
        length = torch.tensor([len(encoded)], device=device)
        buffer = torch.ByteTensor(list(encoded)).to(device)
    else:
        length = torch.tensor([0], device=device)
        buffer = None

    # Broadcast length
    dist.broadcast(length, src=src)

    # Allocate buffer on receiving ranks
    if dist.get_rank() != src:
        buffer = torch.empty(length.item(), dtype=torch.uint8, device=device)

    # Broadcast actual payload
    dist.broadcast(buffer, src=src)
    decoded = bytes(buffer.tolist()).decode("utf-8")
    return json.loads(decoded)

class LoadRealDatasets(IterableDataset):
    """
    DDP-compatible streaming loader for real tabular datasets.
    
    Splits files and rows among ranks to prevent duplication in distributed training.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        start_from: int = 0,
        max_batches: Optional[int] = None,
        timeout: int = 60,
        device: str = "cpu",
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        eval_mode: bool = False,
        augment_X: bool = False,
        randomize_y: bool = False,
        max_classes: int = 10,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.current_idx = start_from
        self.max_batches = max_batches
        self.timeout = timeout
        self.device = device
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.eval_mode = eval_mode
        self.augment_X = augment_X
        self.randomize_y = randomize_y
        self.max_classes = max_classes

        # Metadata
        metadata_file = self.data_dir / "metadata.json"
        self.metadata = None
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata.json: {e}")

        # List all dataset files
        self.dataset_files = sorted(
            [f for f in self.data_dir.iterdir() if f.suffix in [".csv", ".parquet"]]
        )
        if not self.dataset_files:
            raise ValueError(f"No dataset files found in {self.data_dir}")
        
    def __iter__(self):
        return self

    def _load_file(self, file_path: Path):
        # Load file using pandas instead of HuggingFace datasets
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        feature_keys = [k for k in df.columns if k != "target"]

        # --- randomize_y ---
        if self.randomize_y and not self.eval_mode:
            eligible_indices = [i for i, k in enumerate(feature_keys) if df[k].nunique() > 1]
            if not eligible_indices:
                raise RuntimeError(f"No eligible features for randomize_y in {file_path}")

            if self.ddp_world_size > 1 and dist.is_initialized():
                if self.ddp_rank == 0:
                    new_target_index = np.random.choice(eligible_indices)
                    new_target = feature_keys[new_target_index]
                    y_full = df[new_target].values.astype(np.float32)
                    num_bins = min(np.unique(y_full).size, self.max_classes)
                    bin_edges = np.quantile(y_full, np.linspace(0, 1, num_bins + 1))
                    payload = {
                        "target_index": int(new_target_index),
                        "bin_edges": [float(x) for x in bin_edges.tolist()],
                    }
                else:
                    payload = None
                payload = _broadcast_json(payload, device=self.device, src=0)
                new_target_index = payload["target_index"]
                bin_edges = np.array(payload["bin_edges"], dtype=np.float32)
            else:
                new_target_index = np.random.choice(eligible_indices)
                new_target = feature_keys[new_target_index]
                y_full = df[new_target].values.astype(np.float32)
                num_bins = min(np.unique(y_full).size, self.max_classes)
                bin_edges = np.quantile(y_full, np.linspace(0, 1, num_bins + 1))

            new_target = feature_keys[new_target_index]
            y_full = df[new_target].values.astype(np.float32)
            y_binned = np.digitize(y_full, bin_edges[:-1], right=True) - 1
            y_binned = np.clip(y_binned, 0, len(bin_edges) - 2)

            df = df.drop(columns=["target", new_target], errors="ignore")
            df["target"] = y_binned.astype(np.float32)

        # --- shard after target assignment ---
        if self.ddp_world_size > 1 and not self.eval_mode:
            df = df.iloc[self.ddp_rank::self.ddp_world_size]

        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def sample_seq_len(min_seq_len, max_seq_len, log=False):
        if min_seq_len is None:
            return max_seq_len
        return int(loguniform.rvs(min_seq_len, max_seq_len)) if log else np.random.randint(min_seq_len, max_seq_len)
        
    def __next__(self):
        if self.max_batches is not None and self.current_idx >= self.max_batches:
            raise StopIteration

        file_idx = self.current_idx % len(self.dataset_files)
        file_path = self.dataset_files[file_idx]

        # Wait for file
        wait_time = 0
        while not file_path.exists():
            if wait_time >= self.timeout:
                raise RuntimeError(f"Timeout waiting for file {file_path}")
            time.sleep(1)
            wait_time += 1

        df = self._load_file(file_path)
        N = len(df)
        print("dataset loaded:", file_path, "with", N, "rows", " rank:", self.ddp_rank, " index:", self.current_idx, flush=True)

        # --- sequence length and batch size ---
        seq_len = min(self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len), N)
        B = N // seq_len
        if B == 0:
            raise StopIteration

        # --- feature extraction ---
        feature_keys = [k for k in df.columns if k != "target"]
        F = len(feature_keys)

        X_np = df[feature_keys].iloc[:B * seq_len].to_numpy(dtype=np.float32).reshape(B, seq_len, F)
        y_np = df["target"].iloc[:B * seq_len].to_numpy(dtype=np.float32).reshape(B, seq_len)

        X = torch.from_numpy(X_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        # --- DDP-safe augmentation ---
        if self.augment_X and not self.eval_mode:
            if self.ddp_world_size > 1 and dist.is_initialized():
                if self.ddp_rank == 0:
                    num_to_drop = torch.randint(0, F // 3 + 1, (1,))
                    drop_indices = torch.randperm(F)[:num_to_drop]
                    mask = torch.ones(F, dtype=torch.bool, device=self.device)
                    mask[drop_indices] = False
                else:
                    mask = torch.empty(F, dtype=torch.bool, device=self.device)
                if dist.get_backend() == 'nccl':
                    mask = mask.cuda()
                dist.broadcast(mask, src=0)
                mask = mask.to(self.device)
            else:
                num_to_drop = torch.randint(0, F // 3 + 1, (1,)).item()
                mask = torch.ones(F, dtype=torch.bool, device=self.device)
                if num_to_drop > 0:
                    drop_indices = torch.randperm(F)[:num_to_drop]
                    mask[drop_indices] = False

            X = X[:, :, mask]
            F = X.shape[-1]

        self.current_idx += 1

        # --- metadata ---
        d = torch.full((B,), F, device=self.device)
        seq_lens = torch.full((B,), seq_len, device=self.device)
        train_sizes = torch.full((B,), int(0.8 * seq_len), device=self.device)

        return X, y, d, seq_lens, train_sizes

class LoadRealDatasetsHuddled(IterableDataset):
    """
    LoadRealDatasetsHuddled
    DDP-compatible streaming loader for real tabular datasets.
    
    Splits files and rows among ranks to prevent duplication in distributed training.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        max_batches: Optional[int] = None,
        device: str = "cpu",
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        batch_size: int = 32,
        max_classes: int = 10,
        max_features: int = 100,
        max_train_size: int = 0.9,
        min_train_size: int = 0.1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.max_batches = max_batches
        self.device = device
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.max_classes = max_classes
        self.max_features = max_features
        self.batch_size = batch_size
        self.eval_mode = False
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size

        # Metadata
        metadata_file = self.data_dir / "metadata.json"
        self.metadata = None
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata.json: {e}")

        # List all dataset files
        self.dataset_files = sorted(
            [f for f in self.data_dir.iterdir() if f.suffix in [".csv", ".parquet"]]
        )
        if not self.dataset_files:
            raise ValueError(f"No dataset files found in {self.data_dir}")

        self.MAX_ROWS_FOR_CACHE = np.inf  # max rows to cache in memory
        self.df_cache = {}
        self._init_feature_distribution()

    def _init_feature_distribution(self):
        """Scan dataset files and compute distribution of available feature counts."""
        print("Scanning dataset files to determine feature histogram...", flush=True)
        feature_counter = Counter()
        self.feature_size_distribution = {}
        for file_path in self.dataset_files:
            try:
                df = self._load_file(file_path)
                num_features = df.shape[1] 
                self.feature_size_distribution[file_path] = num_features
                if num_features > 0:
                    feature_counter[num_features] += 1
                if len(df) <= self.MAX_ROWS_FOR_CACHE:
                    self.df_cache[file_path] = df    
            except Exception as e:
                print(f"Warning: Could not inspect {file_path}: {e}")

        if not feature_counter:
            raise ValueError("Could not determine feature histogram from dataset files.")

        values = np.array(list(feature_counter.keys()))
        counts = np.array(list(feature_counter.values()), dtype=np.float32)
        probs = counts / counts.sum()
        print("done. Feature counts and probabilities:", flush=True)

        # Clip to self.max_features
        mask = values <= self.max_features
        self.feature_values = values[mask]
        self.feature_probs = probs[mask]
        self.feature_probs /= self.feature_probs.sum()  # renormalize

    def _eligible_files_for_F(self, F):
        return [f for f, num_feat in self.feature_size_distribution.items() if num_feat  >= F + 2]  # +2 for target column and random target

    def __iter__(self):
        return self
    def sample_train_size(self, seq_len) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types
        """
        min_train_size = self.min_train_size
        max_train_size = self.max_train_size
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    def _sample_spec(self):
        """Sample (seq_len, F) once globally and share with all nodes."""
        if self.ddp_world_size > 1 and dist.is_initialized():
            if self.ddp_rank == 0:
                payload = {
                    "seq_len": self.sample_seq_len(
                        self.min_seq_len, self.max_seq_len, log=self.log_seq_len
                    ),
                    "F": int(np.random.choice(self.feature_values, p=self.feature_probs))
,
                }
            else:
                payload = None
            payload = _broadcast_json(payload, device=self.device, src=0)
            return payload["seq_len"], payload["F"]
        else:
            return (
                self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len
                ),
                int(np.random.choice(self.feature_values, p=self.feature_probs))
,
            )

    def _load_file(self, file_path: Path):
        # Load file using pandas instead of HuggingFace datasets
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        df = df.sample(frac=1).reset_index(drop=True)
        return df

    @staticmethod
    def sample_seq_len(min_seq_len, max_seq_len, log=False):
        if min_seq_len is None:
            return max_seq_len
        return int(loguniform.rvs(min_seq_len, max_seq_len)) if log else np.random.randint(min_seq_len, max_seq_len)

    
    def __next__(self):
        seq_len, F = self._sample_spec()
        batch_X = np.empty((self.batch_size, seq_len, F), dtype=np.float32)
        batch_y = np.empty((self.batch_size, seq_len), dtype=np.float32)
        d = torch.full((self.batch_size,), F, device=self.device)
        seq_lens = torch.full((self.batch_size,), seq_len, device=self.device)
        train_sizes = np.empty((self.batch_size,), dtype=np.int32)
        eligible_files = self._eligible_files_for_F(F)
        for b in range(self.batch_size):
            train_sizes[b] = self.sample_train_size(seq_len)
            file_path = np.random.choice(eligible_files)
            if file_path in self.df_cache:
                df = self.df_cache[file_path]
                df = df.sample(frac=1).reset_index(drop=True)
            else:
                df = self._load_file(file_path)
                print("file not in cache:", file_path,"num of rows:", len(df), flush=True)
            N = len(df)

            # Select target column randomly
            feature_keys = [k for k in df.columns if k != "target"]
            eligible_features = [k for k in feature_keys if df[k].nunique() > 1]
            new_target = np.random.choice(eligible_features)
            feature_keys.remove(new_target)
            y_full = df[new_target].values.astype(np.float32)

            # Choose F features randomly 
            chosen_features = np.random.choice(feature_keys, size=F, replace=False)
            X_np = df[chosen_features].to_numpy(dtype=np.float32)

            # If not enough rows, tile data
            if N < seq_len:
                reps = int(np.ceil(seq_len / N))
                X_np = np.tile(X_np, (reps, 1))[:seq_len]
                y_full = np.tile(y_full, reps)[:seq_len]
                N = len(X_np)

            # Randomly select seq_len rows
            random_indices = np.random.choice(N, size=seq_len, replace=False)
            X_np = X_np[random_indices]
            y_full = y_full[random_indices]

            # If too many classes, bin the labels
            if np.unique(y_full).size > self.max_classes:
                y_full = y_full * 1e3  # spread out values to avoid collisions
                y_full_binned = safe_binning(
                    y_full,
                    max_bins=self.max_classes,
                    min_bins_required=2,
                    max_retries=5,
                    apply_log=False,
                    use_rank=False,
                    fallback_to_dividing_over_median=True,
                    verbose=False,
                )
                y_full = y_full_binned.astype(np.float32)

            # Randomly remap class IDs
            unique_classes = np.random.permutation(np.unique(y_full))
            mapping = {old: new for new, old in enumerate(unique_classes)}
            y_full = np.vectorize(mapping.get)(y_full).astype(np.float32)



            #change order of features and normalize
            X_np = X_np[:, np.random.permutation(F)]
            X_np = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-6)
            y_np = y_full

            batch_X[b] = X_np
            batch_y[b] = y_np

        batch_X = torch.from_numpy(batch_X).to(self.device)
        batch_y = torch.from_numpy(batch_y).to(self.device)
        return batch_X, batch_y, d, seq_lens, train_sizes

def safe_binning(
    y_values,
    max_bins=10,
    min_bins_required=2,
    max_retries=5,
    apply_log=False,
    use_rank=False,
    fallback_to_dividing_over_median=True,
    verbose=True,
):
    """
    Safely bins continuous values into discrete classes using qcut or cut,
    with fallbacks to log/rank transform and label encoding.

    Parameters:
        y_values (array-like): 1D array of target values to bin
        max_bins (int): Maximum number of bins to attempt
        min_bins_required (int): Minimum number of distinct bins for success
        max_retries (int): Number of retries with different settings
        apply_log (bool): Whether to apply log1p() to y_values before binning
        use_rank (bool): Whether to bin ranks instead of raw values
        fallback_to_label_encoding (bool): Fallback to label encoding if binning fails
        verbose (bool): Print warnings and debug info

    Returns:
        np.ndarray of integer bin labels
    """
    y = np.asarray(y_values).flatten().astype(np.float32)

    if apply_log:
        y = np.log1p(y)

    if use_rank:
        y = pd.Series(y).rank(method="average").values

    unique_vals = np.unique(y)
    if unique_vals.size < 2:
        raise ValueError("Cannot bin: target has fewer than 2 unique values.")

    for attempt in range(max_retries):
        n_bins = np.random.randint(2, min(max_bins + 1, unique_vals.size + 1))
        strategy = np.random.choice(["quantile", "uniform", "kmeans"])

        try:
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy=strategy
            )
            y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten()
            bin_count = np.unique(y_binned).size
            if bin_count >= min_bins_required:
                if verbose and bin_count < n_bins:
                    print(f"Warning: Only {bin_count} < {n_bins} bins used. Strategy: {strategy}")
                return y_binned.astype(np.int32)

            if verbose:
                print(f"[Retry {attempt+1}] Only {bin_count} bins with {strategy}, trying again...")

        except Exception as e:
            if verbose:
                print(f"[Retry {attempt+1}] Binning failed: {e}")

    if fallback_to_dividing_over_median:
        if verbose:
            print("Falling back to dividing over median.")
        median = np.median(y)
        y_encoded = (y > median).astype(np.int32)
        return y_encoded.astype(np.int32)

    raise ValueError("Binning failed after retries and no fallback allowed.")
