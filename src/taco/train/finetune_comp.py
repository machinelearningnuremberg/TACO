from __future__ import annotations

import os
import timeit
import warnings
import functools
from dataclasses import asdict, is_dataclass
from contextlib import nullcontext

import math
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from tqdm import tqdm
import wandb

from taco.model.taco_model import TACO
from taco.prior.dataset import PriorDataset
from taco.prior.genload import LoadPriorDataset
from taco.train.optim import get_scheduler
from taco.train.train_config import build_parser
from taco.prior.real_dataset import LoadRealDatasets, MixedDataset, LoadRealDatasetsHuddled
from taco.model.tabpfn_arch.model.config import ModelConfig
from datetime import timedelta
warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def _serializable_config(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _serializable_config(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_serializable_config(item) for item in value)
    if isinstance(value, list):
        return [_serializable_config(item) for item in value]
    return value


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper

class TrainerCompFinetuner:
    """    Trainer for fine-tuning the TACO model with the compression module.
    """

    def __init__(self, config):
        self.config = config
        print("trainer config:", self.config)
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        self.configure_evaluator()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        if self.master_process:
            print("Trainer initialized with config:")
            print(self.config)
        self.failed_batches = 0
        if self.config.freeze_predictor:
            # Whether to freeze the predictor
            self._freeze_predictor()

    def _freeze_predictor(self):
        """Freeze the predictor to prevent its parameters from being updated during training."""
        self.raw_model.predictor.eval()
        for param in self.raw_model.predictor.parameters():
            param.requires_grad = False

        if self.master_process:
            num_trainable = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
            print(f"After freezing predictor: {num_trainable} trainable parameters.")

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]), timeout=timedelta(seconds=7200))
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
                settings=wandb.Settings(init_timeout=300)
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None


    def build_model(self):
        """Create the TACO model and optionally load a local checkpoint."""
        ckpt_arg = getattr(self.config, "pretrained_ckpt", None)

        wrapper_cfg = dict(
            use_compressor=getattr(self.config, "use_compressor", True),
            row_compression_percentage=getattr(self.config, "row_compression_percentage", 10.0),
            rcp_sampling=getattr(self.config, "rcp_sampling", "uniform"),
            rcp_choices=getattr(self.config, "rcp_choices", (1, 2, 4, 8, 16, 32)),
            rcp_k_min=getattr(self.config, "rcp_k_min", 1),
        )
        tab_pfn_cfg = None
        if getattr(self.config, "overwrite_tabpfn_config", False):
            tab_pfn_cfg =  ModelConfig(
                emsize=self.config.emsize,
                features_per_group=self.config.features_per_group,
                max_num_classes=self.config.max_num_classes,
                nhead=self.config.nhead,
                remove_duplicate_features=self.config.remove_duplicate_features,
                num_buckets=self.config.num_buckets,
                max_num_features=self.config.max_num_features,
                two_sets_of_queries=self.config.two_sets_of_queries,
                dropout=self.config.dropout,  # from your existing shared-architecture arg
                encoder_use_bias=self.config.encoder_use_bias,
                feature_positional_embedding=self.config.feature_positional_embedding,
                multiquery_item_attention=self.config.multiquery_item_attention,
                nan_handling_enabled=self.config.nan_handling_enabled,
                nan_handling_y_encoder=self.config.nan_handling_y_encoder,
                nhid_factor=self.config.nhid_factor,
                nlayers=self.config.nlayers,
                normalize_by_used_features=self.config.normalize_by_used_features,
                normalize_on_train_only=self.config.normalize_on_train_only,
                normalize_to_ranking=self.config.normalize_to_ranking,
                normalize_x=self.config.normalize_x,
                recompute_attn=self.config.recompute_attn,
                recompute_layer=self.config.recompute_layer,
                remove_empty_features=self.config.remove_empty_features,
                remove_outliers=self.config.remove_outliers,
                use_separate_decoder=self.config.use_separate_decoder,
                use_flash_attention=self.config.use_flash_attention,
                multiquery_item_attention_for_test_set=self.config.multiquery_item_attention_for_test_set,
                attention_init_gain=self.config.attention_init_gain,
                dag_pos_enc_dim=self.config.dag_pos_enc_dim,
                item_attention_type=self.config.item_attention_type,
                feature_attention_type=self.config.feature_attention_type,
                seed=self.config.seed,
        )



        if ckpt_arg:
            ckpt_path = Path(ckpt_arg)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            checkpoint_tabpfn_config = None
            if isinstance(checkpoint_config, dict):
                checkpoint_tabpfn_config = (
                    checkpoint_config.get("new_tabpfn_config")
                    or checkpoint_config.get("tabpfn_config")
                )

            self.model_config = dict(wrapper_cfg)  # store what we actually build now
            if tab_pfn_cfg is not None:
                self.model_config["new_tabpfn_config"] = tab_pfn_cfg
            elif checkpoint_tabpfn_config is not None:
                self.model_config["new_tabpfn_config"] = checkpoint_tabpfn_config
            model = TACO(**self.model_config).to(self.config.device)

            missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
            if self.master_process:
                print(f"Loaded checkpoint (strict=False) from {ckpt_path}")
                print(f"  Missing keys: {len(missing)}  Unexpected: {len(unexpected)}")

        else:
            # start from scratch
            self.model_config = dict(wrapper_cfg)
            if tab_pfn_cfg is not None:
                self.model_config["new_tabpfn_config"] = tab_pfn_cfg
            model = TACO(**self.model_config).to(self.config.device)

        if self.master_process:
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_trainable} trainable parameters.")

        # Compile if requested
        if getattr(self.config, "model_compile", False):
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        # DDP wrap
        if self.ddp:
            if self.config.use_compressor:
                self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False,
                                 find_unused_parameters=True)
            else:
                self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

        if self.master_process:
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            num_total = sum(p.numel() for p in model.parameters())
            print(f"Model has {num_trainable} trainable parameters out of {num_total} total.")
            if self.wandb_run is not None:
                wandb.log({"trainable_params": num_trainable})

    def configure_prior(self):
        """
        Sets up data sources for training:
        - Real dataset (if available).
        - Prior dataset (generated or loaded).
        Creates a mixed dataloader that yields batches from both
        sources according to a configurable mixing ratio.
        """

        if self.config.prior_dir is None:
            prior_dataset = PriorDataset(
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                device=self.config.prior_device,
                n_jobs=1,  # Avoid nested parallelism in DDP
            )
        else:
            prior_dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        real_dataset = None
        if getattr(self.config, "real_data_dir", None) is not None:
            if getattr(self.config, "huddle_real_data", True):
                real_dataset = LoadRealDatasetsHuddled(
                    data_dir=self.config.real_data_dir,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    device=self.config.prior_device,
                    min_seq_len=self.config.min_seq_len,
                    max_seq_len=self.config.max_seq_len,
                    max_classes=self.config.max_classes,
                    max_features=self.config.max_features,
                    batch_size=self.config.batch_size,
                    min_train_size=self.config.min_train_size,
                    max_train_size=self.config.max_train_size,
                )
            else:
                real_dataset = LoadRealDatasets(
                    data_dir=self.config.real_data_dir,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    device=self.config.prior_device,
                    min_seq_len=self.config.min_seq_len,
                    max_seq_len=self.config.max_seq_len,
                    augment_X=self.config.augment_real_data,
                    randomize_y=self.config.augment_real_data,
                    max_classes=self.config.max_classes,
                )

        if real_dataset is not None:
            # Wrap both into a combined dataset
            dataset = MixedDataset(
                real_dataset=real_dataset,
                prior_dataset=prior_dataset,
                total_steps=self.config.max_steps,
            )
        else:
            if self.master_process:
                print("No real dataset provided; training only on prior data.")
            dataset = MixedDataset(
                real_dataset=None,
                prior_dataset=prior_dataset,
                total_steps=self.config.max_steps,
                start_alpha=getattr(self.config, "mixed_dataloader_start_alpha", 0.0),
                end_alpha=getattr(self.config, "mixed_dataloader_end_alpha", 1.0),
            )


        # -----------------------------
        # Dataloader
        # -----------------------------
        num_workers = int(getattr(self.config, "num_workers", 0))
        pin_memory = self.config.prior_device == "cpu" and "cuda" in self.config.device
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=None,  # internal batching handled in datasets
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 4
        if pin_memory:
            dataloader_kwargs["pin_memory_device"] = self.config.device
        self.dataloader = DataLoader(**dataloader_kwargs)

    def configure_evaluator(self):
        """Set up evaluation datasets and dataloaders."""
        if not self.master_process:
            self.eval_dataloader = None
            return
        if self.config.eval_data_dir is None:
            self.eval_dataloader = None
            return
        eval_dataloader = LoadRealDatasets(
            data_dir=self.config.eval_data_dir,
            ddp_world_size=1,
            ddp_rank=0,
            device=self.config.prior_device,
            min_seq_len=self.config.min_seq_len,
            max_seq_len=self.config.max_seq_len,
            eval_mode=True,
            augment_X=False,
            randomize_y=False,
        )
        self.eval_dataloader = MixedDataset(
            real_dataset=eval_dataloader,
            prior_dataset=None,
            total_steps=1,
            start_alpha=1.0,
            end_alpha=1.0,
        )

    def configure_optimizer(self):
        base_lr = self.config.lr
        pred_lr = base_lr * self.config.predictor_lr_mult

        predictor_params = list(self.raw_model.predictor.parameters())

        # Everything except predictor
        predictor_param_ids = {id(p) for p in predictor_params}
        base_params = [p for p in self.raw_model.parameters()
                    if p.requires_grad and id(p) not in predictor_param_ids]

        param_groups = [
            {"params": base_params, "lr": base_lr},
            {"params": predictor_params, "lr": pred_lr},
        ]
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                params=param_groups, lr=base_lr, weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)


    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""
        self.amp = self.config.amp and "cuda" in self.config.device

        if not self.amp:
            self.scaler = torch.GradScaler("cuda", enabled=False)
            self.amp_ctx = nullcontext()
            return

        dtype_str = str(self.config.dtype).lower()
        if dtype_str in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
        elif dtype_str in ("fp16", "float16", "half"):
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32

        self.scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

        if self.master_process:
            print(f"Automatic Mixed Precision is enabled (autocast dtype={amp_dtype}).")

        self.amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def _apply_lr_override(self, new_lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        if hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [new_lr for _ in self.scheduler.base_lrs]

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch or from the pretrained model.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        self.raw_model.load_state_dict(checkpoint["state_dict"])

        # Optionally load optimizer and scheduler state
        if self.config.only_load_model:
            self.curr_step = checkpoint["curr_step"]
            print(f"Only loading model weights. Resuming training at step {self.curr_step}")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")

        if getattr(self.config, "override_lr", None) is not None:
            self._apply_lr_override(self.config.override_lr)
            print(f"Overriding LR on resume to {self.config.override_lr}")

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": _serializable_config(self.model_config),
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        """
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            print("No evaluation dataset provided.")
            return

        self.model.eval()
        self.raw_model.eval()
        total_loss = 0
        accuracies = []
        print("Starting evaluation...", flush=True)
        print(f"Eval dataset has {len(self.eval_dataloader.real_dataset.dataset_files)} batches.", flush=True)
        N = len(self.eval_dataloader.real_dataset.dataset_files)
        iterator = iter(self.eval_dataloader)
        number_of_failed_batches = 0
        with torch.no_grad():
            for _ in range(N):
                batch = next(iterator)
                results = self.run_batch(batch, train=False)
                del batch
                if results is not None:
                    total_loss += results['ce']
                    accuracies.append(results['accuracy'])
                else:
                    number_of_failed_batches += 1
                    print(f"rank {self.ddp_rank} skipping eval batch due to failure at step {self.curr_step}. total failed eval batches: {number_of_failed_batches}", flush=True)


                

        avg_loss = total_loss / N
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        print(f"Evaluation loss: {avg_loss}", flush=True)
        print(f"Evaluation accuracy at step {self.curr_step}: {avg_accuracy}", flush=True)
        self.model.train()
        self.raw_model.train()
        # clean CUDA cache after evaluation to free memory
        torch.cuda.empty_cache()

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)
        dataloader = iter(self.dataloader)
        for step in step_progress:
            # Get the next batch
            self.dataloader.dataset.set_step(self.curr_step)

            # Evaluate at configured intervals
            needs_eval = (self.curr_step % self.config.eval_every == 0)
            if needs_eval and self.master_process:
                self.evaluate()
            if self.ddp and needs_eval:
                dist.barrier()


            with Timer() as prior_timer:
                batch = next(dataloader)
            prior_time = prior_timer.elapsed
            
            # Train the model on the batch
            with Timer() as train_timer:
                results = self.run_batch(batch)
                if results is None:
                    # All micro-batches failed; skip this step entirely
                    self.failed_batches += 1
                    print(f"rank {self.ddp_rank} skipping step {self.curr_step} due to failed batch. total failed batches: {self.failed_batches}", flush=True)
                    torch.cuda.empty_cache()
                    self.curr_step += 1
                    continue
            train_time = train_timer.elapsed
            self.curr_step = step + 1
            if self.master_process:
                # Add timing information to results
                results.update({"prior_time": prior_time, "train_time": train_time})

                # Update progress bar with rounded values for cleaner display
                step_progress.set_postfix(**{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()})

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                # Add learning rate to results
                results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)
    

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """
        Validate consistent sequence length and train size within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length and
        train/test split position, required for efficient batch processing during
        gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor (micro_batch_size,)
            Sequence lengths for each dataset.

        micro_train_size : Tensor (micro_batch_size,)
            Training sizes (split positions) for each dataset.

        Returns
        -------
        tuple (int, int)
            The common (seq_len, train_size) for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths or train sizes are inconsistent.
        """
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")

        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")

        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()

        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        """
        Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (`micro_d.max()`) respectively. This optimizes
        memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor (B, T, H)
            Input features per dataset.

        micro_y : Tensor (B, T)
            Target labels per dataset.

        micro_d : Tensor (B,)
            Number of active features per dataset.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        tuple (Tensor, Tensor)
            Truncated (micro_X, micro_y) tensors with shapes
            (B, seq_len, micro_d.max()) and (B, seq_len).
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]

        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches, training_mode: bool = True):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size) tensors for the micro batch

        micro_batch_idx : int
            Index of the current micro batch

        num_micro_batches : int
            Total number of micro batches

        Returns
        -------
        dict
            Result dictionary
        """
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)
        # Move to device
        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]
        micro_y = micro_y.detach()
        

        # # Set DDP gradient sync for last micro batch only
        # if self.ddp and training_mode:
        #     self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        model = self.model if training_mode else self.raw_model
        with self.amp_ctx:
            pred = model(micro_X, y_train)
            micro_X = micro_X.detach()
            y_train = y_train.detach()
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            loss = F.cross_entropy(pred, true)
            y_test = y_test.detach()
            true = true.detach()
            pred = pred.detach()
            

        # Scale loss for gradient accumulation and backpropagate
        scaled_loss = loss / num_micro_batches
        if training_mode and self.model.training:
            self.scaler.scale(scaled_loss).backward()
        with torch.no_grad():
            micro_results = {}
            micro_results["ce"] = loss.detach().float().item() / num_micro_batches
            accuracy = (pred.argmax(dim=1) == true).sum() / len(true)
            micro_results["accuracy"] = accuracy.item() / num_micro_batches
        del pred, true, loss, scaled_loss
        return micro_results


    def run_batch(self, batch, train: bool = True):
        if train:
            self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Pad & split (your existing code)
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        batch_size = batch[0].size(0)
        num_micro_batches = math.ceil(batch_size / self.config.micro_batch_size)
        micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
        micro_batches = list(zip(*micro_batches))

        # Initialize results dict with all possible keys
        results = {"ce": 0.0, "accuracy": 0.0}

        failed_batches = 0
        acceptable_failure_rate = 0.1
        for idx, micro_batch in enumerate(micro_batches):
            is_last = (idx == len(micro_batches) - 1)
            context = self.model.no_sync() if self.ddp and not is_last else nullcontext()
    
            try:
                with context:
                    micro_results = self.run_micro_batch(micro_batch, idx, num_micro_batches)
                for k, v in micro_results.items():
                    results[k] += v
            except torch.cuda.OutOfMemoryError as e:
                print("CUDA OOM encountered during micro-batch processing.", flush=True)
                X, _, _, _, _ = micro_batch
                Xshape = X.shape
                seq_len_train = micro_batch[3]
                train_size_train = micro_batch[4]
                dims = torch.unique(micro_batch[2])
                print(f"rank {self.ddp_rank} caught OOM on micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}: {e}. X shape: {Xshape}, seq_len_train: {seq_len_train}, train_size_train: {train_size_train}‚ dims: {dims}", flush=True)
                failed_batches += 1
                torch.cuda.empty_cache()
                continue
        failure_ratio = failed_batches / num_micro_batches
        local_ok = (failure_ratio < acceptable_failure_rate)  # Allow up to 10% micro-batch failures
        if not local_ok:
            print(f"rank {self.ddp_rank} had {failure_ratio * 100:.2f}% failed micro-batches at step {self.curr_step}.", flush=True)
            raise RuntimeError(f"Too many micro-batch failures; failure rate exceeded acceptable limit at step {self.curr_step} for rank {self.ddp_rank}. failure ratio: {failure_ratio:.2f}, acceptable limit: {acceptable_failure_rate:.2f}")
        elif failure_ratio > 0:
            print(f"rank {self.ddp_rank} had {failure_ratio * 100:.2f}% failed micro-batches but within acceptable limit at step {self.curr_step}.", flush=True)

        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)

        # No extra manual all-reduce needed; DDP will sync on first comm op after no_sync()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results


    
    def _global_all_ok(self, local_ok: bool) -> bool:
        if not self.ddp:
            return local_ok
        t = torch.tensor(1 if local_ok else 0, device=self.config.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)  # 1 only if everyone is 1
        return bool(t.item())


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = TrainerCompFinetuner(config)
    trainer.train()
