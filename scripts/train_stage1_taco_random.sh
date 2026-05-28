MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 apptainer exec --nv \
  --bind necessary directories \
    torchrun \
      --nnodes="$SLURM_NNODES" \
      --nproc_per_node=4 \
      --node_rank="$SLURM_NODEID" \
      --rdzv_id="$SLURM_JOB_ID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        /taco/src/taco/train/finetune_comp.py  \
            --wandb_log True \
            --wandb_project TabPFN-TACO \
            --wandb_name Stage1_TabPFN-TACO_random_sampling \
            --wandb_dir /wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype bfloat16 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 80000 \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --num_workers 1 \
            --batch_size_per_gp 16 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --checkpoint_dir /checkpoint/dir \
            --save_temp_every 50 \
            --save_perm_every 1000 \
            --use_compressor \
            --row_compression_percentage 20 \
            --rcp_sampling "uniform" \
            --amp True
