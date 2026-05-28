"""Define argument parser for TACO training."""

import argparse


def str2bool(value):
    return value.lower() == "true"


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def build_parser():
    """Build parser with all TACO training arguments."""
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument("--wandb_log", default=False, type=str2bool, help="Log results using wandb")
    parser.add_argument("--wandb_project", type=str, default="TabPFN-TACO", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb logging directory")
    parser.add_argument(
        "--wandb_mode", default="offline", type=str, help="Wandb logging mode: online, offline, or disabled"
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument("--device", default="cuda", type=str, help="Device for training: cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="Data type (supported for float16, float32) used for training"
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--max_steps", type=int, default=60000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=8, help="Size of micro-batches for gradient accumulation"
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="cosine_warmup", help="Learning rate scheduler: see optim.py for options."
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay / L2 regularization penalty")
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=1.0,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument("--predictor_lr_mult", type=float, default=1.0, help="Learning rate multiplier for the predictor")
    parser.add_argument("--cosine_lr_end", type=float, default=0, help="Final learning rate for cosine_with_restarts")
    parser.add_argument(
        "--poly_decay_lr_end", type=float, default=1e-7, help="Final learning rate for polynomial decay scheduler"
    )
    parser.add_argument(
        "--poly_decay_power", type=float, default=1.0, help="Power factor for polynomial decay scheduler"
    )

    # Prior Dataset Config
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )

    parser.add_argument("--real_data_dir", type=str, default=None, help="Directory for real datasets")
    parser.add_argument("--eval_data_dir", type=str, default=None, help="Directory for evaluation datasets")
    parser.add_argument("--eval_every", type=int, default=1000, help="Steps between evaluations")
    parser.add_argument(
        "--mixed_dataloader_start_alpha",
        type=float,
        default=0.1,
        help="Initial probability of sampling real data in mixed dataloader",
    )
    parser.add_argument(
        "--mixed_dataloader_end_alpha",
        type=float,
        default=1.0,
        help="Final probability of sampling real data in mixed dataloader",
    )
    parser.add_argument( "--huddle_real_data", default=False, type=str2bool, help="Whether to use huddle sampling for real data")
    parser.add_argument("--augment_real_data", default=False, type=str2bool, help="Whether to augment real data")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=5, help="The minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="The maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="The maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum samples per dataset")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum samples per dataset")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Starting position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Ending position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type", default="mix_scm", type=str, help="Prior type: dummy, mlp_scm, tree_scm, mix_scm"
    )
    parser.add_argument("--prior_device", default="cpu", type=str, help="Device for prior data generation")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers")

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )

    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")

    # Compression Config
    parser.add_argument("--use_compressor", action="store_true", help="Whether to use compressor")
    parser.add_argument("--row_compression_percentage", type=float, default=0.5, help="Percentage of rows to compress")
    parser.add_argument("--rcp_sampling", type=str, default="none", help="RCP sampling strategy: none or uniform")

    parser.add_argument(
        "--pretrained_ckpt", type=str, default=None,
        help=(
            "Absolute or relative path to a local .ckpt file that was trained without\n"
            "the compressor. If omitted (default) training starts from random\n"
            "initialisation."
        )
    )
    parser.add_argument("--freeze_predictor", default=False, type=str2bool, help="Whether to freeze the predictor")



    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory for checkpoint saving and loading")
    parser.add_argument("--save_temp_every", default=50, type=int, help="Steps between temporary checkpoints")
    parser.add_argument("--save_perm_every", default=5000, type=int, help="Steps between permanent checkpoints")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of temporary checkpoints to keep. Permanent checkpoints are not counted.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to specific checkpoint file to load")
    parser.add_argument("--only_load_model", default=False, type=str2bool, help="Whether to only load model weights")
    parser.add_argument("--override_lr", default=None, type=float, help="Whether to override learning rate from checkpoint")

    ###########################################################################
    ###### Global ModelConfig #################################################
    ###########################################################################

    parser.add_argument("--emsize", type=int, default=192, help="ModelConfig: embedding size")
    parser.add_argument(
        "--features_per_group",
        type=int,
        default=2,
        help="ModelConfig: number of features per group",
    )
    parser.add_argument(
        "--max_num_classes",
        type=int,
        default=10,
        help="ModelConfig: maximum number of classes",
    )
    parser.add_argument("--nhead", type=int, default=6, help="ModelConfig: number of attention heads")
    parser.add_argument(
        "--remove_duplicate_features",
        default=False,
        type=str2bool,
        help="ModelConfig: whether to remove duplicate features",
    )
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=1000,
        help="ModelConfig: number of buckets (e.g. for embeddings / hashing)",
    )
    parser.add_argument(
        "--max_num_features",
        type=int,
        default=85,
        help="ModelConfig: maximum number of features",
    )
    parser.add_argument(
        "--two_sets_of_queries",
        default=False,
        type=str2bool,
        help="ModelConfig: whether to use two sets of queries",
    )
    parser.add_argument(
        "--encoder_use_bias",
        default=False,
        type=str2bool,
        help="ModelConfig: whether encoder attention uses bias",
    )
    parser.add_argument(
        "--feature_positional_embedding",
        type=str,
        default="subspace",
        help="ModelConfig: feature positional embedding type",
    )
    parser.add_argument(
        "--multiquery_item_attention",
        default=False,
        type=str2bool,
        help="ModelConfig: enable multi-query item attention",
    )
    parser.add_argument(
        "--nan_handling_enabled",
        default=True,
        type=str2bool,
        help="ModelConfig: enable special handling for NaNs",
    )
    parser.add_argument(
        "--nan_handling_y_encoder",
        default=True,
        type=str2bool,
        help="ModelConfig: enable NaN handling in y encoder",
    )
    parser.add_argument(
        "--nhid_factor",
        type=int,
        default=4,
        help="ModelConfig: hidden dimension factor (multiplicative)",
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        default=12,
        help="ModelConfig: number of transformer layers",
    )
    parser.add_argument(
        "--normalize_by_used_features",
        default=True,
        type=str2bool,
        help="ModelConfig: normalize by number of used features",
    )
    parser.add_argument(
        "--normalize_on_train_only",
        default=True,
        type=str2bool,
        help="ModelConfig: apply normalization on train data only",
    )
    parser.add_argument(
        "--normalize_to_ranking",
        default=False,
        type=str2bool,
        help="ModelConfig: normalize labels/targets to ranking",
    )
    parser.add_argument(
        "--normalize_x",
        default=True,
        type=str2bool,
        help="ModelConfig: normalize input features",
    )
    parser.add_argument(
        "--recompute_attn",
        default=False,
        type=str2bool,
        help="ModelConfig: recompute attention to save memory",
    )
    parser.add_argument(
        "--recompute_layer",
        default=True,
        type=str2bool,
        help="ModelConfig: recompute whole layers (checkpointing)",
    )
    parser.add_argument(
        "--remove_empty_features",
        default=True,
        type=str2bool,
        help="ModelConfig: remove features that are entirely empty/NaN",
    )
    parser.add_argument(
        "--remove_outliers",
        default=False,
        type=str2bool,
        help="ModelConfig: remove outlier feature values",
    )
    parser.add_argument(
        "--use_separate_decoder",
        default=False,
        type=str2bool,
        help="ModelConfig: use a separate decoder module",
    )
    parser.add_argument(
        "--use_flash_attention",
        default=True,
        type=str2bool,
        help="ModelConfig: use flash attention implementation",
    )
    parser.add_argument(
        "--multiquery_item_attention_for_test_set",
        default=True,
        type=str2bool,
        help="ModelConfig: use multi-query item attention at test time",
    )
    parser.add_argument(
        "--attention_init_gain",
        type=float,
        default=1.0,
        help="ModelConfig: initialization gain for attention layers",
    )
    parser.add_argument(
        "--dag_pos_enc_dim",
        type=int,
        default=None,
        help="ModelConfig: positional encoding dimension for DAG (if used)",
    )
    parser.add_argument(
        "--item_attention_type",
        type=str,
        default="full",
        help="ModelConfig: item attention type (e.g., full, causal, etc.)",
    )
    parser.add_argument(
        "--feature_attention_type",
        type=str,
        default="full",
        help="ModelConfig: feature attention type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="ModelConfig: random seed used inside the model",
    )
    parser.add_argument(
        "--overwrite_tabpfn_config",
        default=True,
        type=str2bool,
        help="ModelConfig: whether to overwrite TabPFN config settings",
        )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw"],
        help="Optimizer to use.",
    )

    return parser
