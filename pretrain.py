#!/usr/bin/env python3
"""
GenoBERT Pretraining Script

Distributed training script for GenoBERT masked language modeling.
Supports multi-GPU and multi-node training via PyTorch DDP.

Usage:
    # Single GPU
    python pretrain.py --configFile configs/example.yaml

    # Multi-GPU (SLURM)
    srun python pretrain.py --configFile configs/example.yaml
"""

# Set CUDA device before importing torch (for SLURM)
import os
if 'SLURM_LOCALID' in os.environ:
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

from functools import partial
from datetime import datetime
import re
from config.modelconfig import ModelConfig
from data.dataset import SNPsDataset_HDF5, MultiGeneDataset_HDF5
from data.utils import (
    mask_random_positions,
    mask_random_positions_bias,
    find_latest_checkpoint,
    save_checkpoint,
    cleanup_memory,
    FocalLoss,
    GatedFocalLoss,
    F1Loss,
    get_pretrain_dataset_paths,
)
from dataclasses import asdict
from model.genobert import GenoBERTMLM, print_model_summary
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import List, Tuple, Optional
import argparse
import glob
import h5py
import math
import numpy as np
import random
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

# Optional: wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def print_exp_summary(use_gpu, world_size, rank, backend, config, train_hdf5_files, val_hdf5_files, latest_checkpoint_file):
    """Print experiment configuration summary."""
    print("============= TRAINING CONFIGURATION =============")
    print("Is CUDA available:", use_gpu)
    if use_gpu:
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No CUDA device")
        print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Training on CPU nodes")

    print(f"World size: {world_size}, Rank: {rank}, Backend: {backend}")

    print("\n============= MODEL CONFIGURATION =============")
    for field in vars(config):
        value = getattr(config, field)
        if isinstance(value, list) and len(value) > 10:
            print(f"{field}: [{value[0]}, {value[1]}, ..., {value[-1]}] (length {len(value)})")
        else:
            print(f"{field}: {value}")

    total_train_samples = get_num_samples(train_hdf5_files)
    num_batches_per_epoch = total_train_samples // config.batchSize
    total_steps = num_batches_per_epoch * config.totalEpochs
    batches_per_device_per_epoch = num_batches_per_epoch // max(1, world_size)

    print(f"\n============= TRAINING STATISTICS =============")
    print(f"Total samples: {total_train_samples:,}")
    print(f"Batches per device in one epoch: {batches_per_device_per_epoch:,.1f}")
    print(f"Batch size: {config.batchSize}")
    print(f"Batches per epoch: {num_batches_per_epoch:,}")
    print(f"Total epochs: {config.totalEpochs}")
    print(f"Total training steps: {total_steps:,}")
    print(f"Save checkpoint frequency: Every {config.saveCheckpointFreq} epoch(s)")
    print(f"Checkpoint directory: {config.modelDir}")

    if latest_checkpoint_file:
        print(f"\nLoading checkpoint from '{latest_checkpoint_file}'")
    else:
        print("\nStarting fresh training run without checkpoint")

    print("=================================================")


def get_optimal_num_workers(config=None):
    """Get optimal number of workers for data loading."""
    if config is not None and hasattr(config, 'numWorkers') and config.numWorkers is not None:
        return config.numWorkers

    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    return max(1, cpus_per_task - 1)


def get_optimal_bucket_size(num_gpus):
    """Get optimal gradient bucket size for DDP."""
    if num_gpus <= 8:
        return 100
    else:
        return 200


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GenoBERT Pretraining")
    parser.add_argument(
        '--configFile',
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    return parser.parse_args()


def lr_lambda(config, current_epoch):
    """Learning rate schedule with warmup and cooldown."""
    warmup_epochs = config.warmupEpochs
    total_epochs = config.totalEpochs
    cooldown_epochs = config.cooldownEpochs
    scheduler_type = config.scheduler

    # Warmup phase
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(max(1, warmup_epochs))

    # Cooldown phase
    elif current_epoch > total_epochs - cooldown_epochs:
        cooldown_epoch = current_epoch - (total_epochs - cooldown_epochs)
        return float(cooldown_epochs - cooldown_epoch) / float(max(1, cooldown_epochs))

    # Main training phase
    else:
        if scheduler_type == "cosineAnn":
            cosine_epoch = current_epoch - warmup_epochs
            cosine_total = max(1, total_epochs - warmup_epochs - cooldown_epochs)
            return 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_total))

        elif scheduler_type == "stepLR":
            step_size = max(1, config.schedulerStepSize)
            step_factor = (current_epoch - warmup_epochs) // step_size
            return config.schedulerGamma ** step_factor

        return 1.0


def aggregate_scalar(value, device):
    """Aggregate values across all distributed processes."""
    if isinstance(value, torch.Tensor):
        tensor = value.clone().detach()
    else:
        tensor = torch.tensor(value, device=device)

    tensor = tensor.to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def train_ddp(use_gpu, rank, world_size, config, train_hdf5_files, val_hdf5_files, checkpoint_file):
    """Main distributed training function."""
    # Set random seeds
    seed = config.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if use_gpu:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'

    if rank == 0:
        print_exp_summary(use_gpu, world_size, rank, backend, config, train_hdf5_files, val_hdf5_files, checkpoint_file)

    # Initialize distributed process group
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Initialize model
    model = GenoBERTMLM(config).to(device)

    if rank == 0:
        print_model_summary(model)

    # Load checkpoint
    checkpoint = None
    current_epoch = 1
    if checkpoint_file and os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Wrap model in DDP
    if device.type == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[0],
            output_device=0,
            bucket_cap_mb=get_optimal_bucket_size(world_size)
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    num_workers = get_optimal_num_workers(config)

    # Initialize wandb
    if rank == 0 and config.useWandB and WANDB_AVAILABLE:
        if config.WandBKey:
            os.environ["WANDB_API_KEY"] = config.WandBKey
        wandb.init(
            settings=wandb.Settings(init_timeout=600),
            project=config.WandBProjName,
            name=config.run,
            config=asdict(config),
            resume='allow',
            id=config.runId
        )
        wandb.watch(model, log_freq=100)

    # Mixed precision
    scaler = None
    if config.mixedPrecisionTraining and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')

    # Loss function
    criterion = {
        "crossEntropy": nn.CrossEntropyLoss(),
        "focalLoss": FocalLoss(config=config),
        "gatedFocalLoss": GatedFocalLoss(config=config),
        "f1Loss": F1Loss(config=config)
    }[config.loss].to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learningRate,
        betas=(config.adamwBeta1, config.adamwBeta2),
        eps=config.adamwEps,
        weight_decay=config.adamwWeightDecay
    )

    # Load optimizer state
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group["lr"]
    else:
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]

    # Learning rate scheduler
    last_epoch_val = -1 if current_epoch == 1 else current_epoch - 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(lr_lambda, config),
        last_epoch=last_epoch_val
    )

    if checkpoint is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Load datasets
    cache_dir = getattr(config, 'cacheDir', './cache')
    train_cache_file = os.path.join(cache_dir, f"{config.dataset}_chr{config.chromosome}_{config.population}_train_merged.hdf5")
    val_cache_file = os.path.join(cache_dir, f"{config.dataset}_chr{config.chromosome}_{config.population}_val_merged.hdf5")

    if rank == 0:
        print(f"\n=== Loading Training Data ===")
        print(f"  Train files: {len(train_hdf5_files)}")
        print(f"  Val files: {len(val_hdf5_files)}")

    # Check for merged file
    use_merged = (len(train_hdf5_files) == 1 and '_all.hdf5' in train_hdf5_files[0])

    if use_merged:
        if rank == 0:
            print(f"  Mode: Merged file (preprocessed)")
            print(f"  Train: {train_hdf5_files[0]}")
        train_dataset = SNPsDataset_HDF5(train_hdf5_files[0], preload=True)
        val_dataset = SNPsDataset_HDF5(val_hdf5_files[0], preload=True) if val_hdf5_files else None
    else:
        if rank == 0:
            print(f"  Mode: Per-gene files (will deduplicate and cache)")
            print(f"  Cache dir: {cache_dir}")
        train_dataset = MultiGeneDataset_HDF5(
            train_hdf5_files,
            preload=True,
            cache_file=train_cache_file
        )
        val_dataset = MultiGeneDataset_HDF5(
            val_hdf5_files,
            preload=True,
            cache_file=val_cache_file
        ) if val_hdf5_files else None

    if rank == 0:
        print(f"  Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"  Val samples: {len(val_dataset)}")
        print(f"=== Data Loading Complete ===\n")

    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    ) if val_dataset else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batchSize,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batchSize,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=False,
        persistent_workers=num_workers > 0
    ) if val_dataset else None

    # Upsampling configuration
    if config.sampling == "upsampling":
        required_count = int(config.maskProb * config.upsamplingRatio * config.batchSize * config.segLen)
        required_count = torch.tensor(required_count, device=device)

    # Training loop
    for epoch in range(current_epoch, config.totalEpochs + 1):
        train_sampler.set_epoch(epoch)

        epoch_train_loss = torch.tensor(0.0, device=device)
        epoch_train_correct = torch.tensor(0, device=device)
        epoch_train_total = torch.tensor(0, device=device)

        model.train()

        if rank == 0:
            pbar_epoch = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch}/{config.totalEpochs} - Training",
                dynamic_ncols=True,
                unit="batch"
            )

        for batch_data in train_loader:
            # Unpack batch
            if len(batch_data) == 2:
                batch_snps, batch_snpsIndex = batch_data
            else:
                batch_snps, batch_snpsIndex, _ = batch_data

            batch_snps = batch_snps.to(device, non_blocking=True)

            # Skip batches with insufficient variants (upsampling mode)
            if config.sampling == "upsampling":
                variant_count = ((batch_snps == 2) | (batch_snps == 3) | (batch_snps == 4)).sum()
                if variant_count < required_count:
                    continue

            batch_snpsIndex = batch_snpsIndex.to(device, non_blocking=True)

            # Compute genomic position bias
            if config.enableBias:
                batch_bias = torch.zeros_like(batch_snpsIndex[:, :, 1], dtype=torch.float32)
                for b in range(batch_snpsIndex.shape[0]):
                    valid_pos_mask = batch_snpsIndex[b, :, 1] > 0
                    if valid_pos_mask.any():
                        genomic_positions = batch_snpsIndex[b, valid_pos_mask, 1].float()
                        min_pos = genomic_positions.min()
                        max_pos = genomic_positions.max()
                        cls_pos_mask = batch_snpsIndex[b, :, 1] == -1
                        sep_pos_mask = batch_snpsIndex[b, :, 1] == -2
                        pad_pos_mask = batch_snpsIndex[b, :, 1] == -3
                        if cls_pos_mask.any():
                            batch_bias[b, cls_pos_mask] = min_pos - 1
                        if sep_pos_mask.any():
                            batch_bias[b, sep_pos_mask] = max_pos + 1
                        batch_bias[b, valid_pos_mask] = batch_snpsIndex[b, valid_pos_mask, 1].float()
                        if max_pos > min_pos:
                            norm_min = min_pos - 1
                            norm_max = max_pos + 1
                            non_pad_mask = ~pad_pos_mask
                            batch_bias[b, non_pad_mask] = (batch_bias[b, non_pad_mask] - norm_min) / (norm_max - norm_min)
            else:
                batch_bias = None

            batch_padding_mask = (batch_snps != config.padId).to(device)

            # Apply masking
            if config.sampling == "upsampling":
                masked_input, missing_mask = mask_random_positions_bias(config, batch_snps)
            else:
                masked_input, missing_mask = mask_random_positions(config, batch_snps)

            labels = batch_snps.flatten()
            valid_positions = batch_padding_mask.flatten()
            if not valid_positions.any():
                continue

            optimizer.zero_grad()

            # Forward pass
            if device.type == 'cuda' and config.mixedPrecisionTraining:
                with torch.amp.autocast('cuda'):
                    logits = model(masked_input, batch_bias, batch_padding_mask)
                    predicted_genotypes = logits.argmax(dim=2)
                    predicted_genotypes[~batch_padding_mask] = config.padId
                    predicted_genotypes = predicted_genotypes.flatten()

                    if config.benchmarkAll:
                        logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                        labels_valid = labels[valid_positions]
                        if not labels_valid.any():
                            continue
                        loss = criterion(logits_valid, labels_valid)
                        batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                        batch_total = valid_positions.sum()
                    else:
                        masked_positions = missing_mask.flatten() & valid_positions
                        logits_valid = logits.reshape(-1, config.vocabSize)[masked_positions]
                        labels_valid = labels[masked_positions]
                        if not labels_valid.any():
                            continue
                        loss = criterion(logits_valid, labels_valid)
                        batch_correct = (predicted_genotypes[masked_positions] == labels_valid).sum()
                        batch_total = masked_positions.sum()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(masked_input, batch_bias, batch_padding_mask)
                predicted_genotypes = logits.argmax(dim=2)
                predicted_genotypes[~batch_padding_mask] = config.padId
                predicted_genotypes = predicted_genotypes.flatten()

                if config.benchmarkAll:
                    logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                    labels_valid = labels[valid_positions]
                    if not labels_valid.any():
                        continue
                    loss = criterion(logits_valid, labels_valid)
                    batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                    batch_total = valid_positions.sum()
                else:
                    masked_positions = missing_mask.flatten() & valid_positions
                    logits_valid = logits.reshape(-1, config.vocabSize)[masked_positions]
                    labels_valid = labels[masked_positions]
                    if not labels_valid.any():
                        continue
                    loss = criterion(logits_valid, labels_valid)
                    batch_correct = (predicted_genotypes[masked_positions] == labels_valid).sum()
                    batch_total = masked_positions.sum()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if torch.isnan(loss):
                if rank == 0:
                    print(f"WARNING: NaN loss detected at epoch {epoch}")
                continue

            epoch_train_loss += loss.detach()
            epoch_train_correct += batch_correct.detach()
            epoch_train_total += batch_total.detach()

            if rank == 0:
                pbar_epoch.update(1)
                pbar_epoch.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(batch_correct.float() / batch_total.float()).item():.4f}'
                })

        if rank == 0:
            pbar_epoch.close()

        # Aggregate training metrics
        epoch_train_loss = aggregate_scalar(epoch_train_loss, device)
        epoch_train_correct = aggregate_scalar(epoch_train_correct, device)
        epoch_train_total = aggregate_scalar(epoch_train_total, device)

        # Validation phase
        epoch_val_loss = torch.tensor(0.0, device=device)
        epoch_val_correct = torch.tensor(0, device=device)
        epoch_val_total = torch.tensor(0, device=device)

        model.eval()

        if val_loader:
            if rank == 0:
                pbar_val = tqdm(
                    total=len(val_loader),
                    desc=f"Epoch {epoch}/{config.totalEpochs} - Validation",
                    dynamic_ncols=True,
                    unit="batch"
                )

            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 2:
                        batch_snps, batch_snpsIndex = batch_data
                    else:
                        batch_snps, batch_snpsIndex, _ = batch_data

                    batch_snps = batch_snps.to(device, non_blocking=True)
                    batch_snpsIndex = batch_snpsIndex.to(device, non_blocking=True)

                    if config.enableBias:
                        batch_bias = torch.zeros_like(batch_snpsIndex[:, :, 1], dtype=torch.float32)
                        for b in range(batch_snpsIndex.shape[0]):
                            valid_pos_mask = batch_snpsIndex[b, :, 1] > 0
                            if valid_pos_mask.any():
                                genomic_positions = batch_snpsIndex[b, valid_pos_mask, 1].float()
                                min_pos = genomic_positions.min()
                                max_pos = genomic_positions.max()
                                cls_pos_mask = batch_snpsIndex[b, :, 1] == -1
                                sep_pos_mask = batch_snpsIndex[b, :, 1] == -2
                                pad_pos_mask = batch_snpsIndex[b, :, 1] == -3
                                if cls_pos_mask.any():
                                    batch_bias[b, cls_pos_mask] = min_pos - 1
                                if sep_pos_mask.any():
                                    batch_bias[b, sep_pos_mask] = max_pos + 1
                                batch_bias[b, valid_pos_mask] = batch_snpsIndex[b, valid_pos_mask, 1].float()
                                if max_pos > min_pos:
                                    norm_min = min_pos - 1
                                    norm_max = max_pos + 1
                                    non_pad_mask = ~pad_pos_mask
                                    batch_bias[b, non_pad_mask] = (batch_bias[b, non_pad_mask] - norm_min) / (norm_max - norm_min)
                    else:
                        batch_bias = None

                    batch_padding_mask = (batch_snps != config.padId).to(device)
                    labels = batch_snps.flatten()
                    masked_input, missing_mask = mask_random_positions(config, batch_snps)
                    valid_positions = batch_padding_mask.flatten()

                    if device.type == 'cuda' and config.mixedPrecisionTraining:
                        with torch.amp.autocast('cuda'):
                            logits = model(masked_input, batch_bias, batch_padding_mask)
                            predicted_genotypes = logits.argmax(dim=2)
                            predicted_genotypes[~batch_padding_mask] = config.padId
                            predicted_genotypes = predicted_genotypes.flatten()

                            if config.benchmarkAll:
                                logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                                labels_valid = labels[valid_positions]
                                if not labels_valid.any():
                                    continue
                                loss = criterion(logits_valid, labels_valid)
                                batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                                batch_total = valid_positions.sum()
                            else:
                                masked_positions = missing_mask.flatten() & valid_positions
                                logits_valid = logits.reshape(-1, config.vocabSize)[masked_positions]
                                labels_valid = labels[masked_positions]
                                if not labels_valid.any():
                                    continue
                                loss = criterion(logits_valid, labels_valid)
                                batch_correct = (predicted_genotypes[masked_positions] == labels_valid).sum()
                                batch_total = masked_positions.sum()
                    else:
                        logits = model(masked_input, batch_bias, batch_padding_mask)
                        predicted_genotypes = logits.argmax(dim=2)
                        predicted_genotypes[~batch_padding_mask] = config.padId
                        predicted_genotypes = predicted_genotypes.flatten()

                        if config.benchmarkAll:
                            logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                            labels_valid = labels[valid_positions]
                            if not labels_valid.any():
                                continue
                            loss = criterion(logits_valid, labels_valid)
                            batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                            batch_total = valid_positions.sum()
                        else:
                            masked_positions = missing_mask.flatten() & valid_positions
                            logits_valid = logits.reshape(-1, config.vocabSize)[masked_positions]
                            labels_valid = labels[masked_positions]
                            if not labels_valid.any():
                                continue
                            loss = criterion(logits_valid, labels_valid)
                            batch_correct = (predicted_genotypes[masked_positions] == labels_valid).sum()
                            batch_total = masked_positions.sum()

                    epoch_val_loss += loss.detach()
                    epoch_val_correct += batch_correct.detach()
                    epoch_val_total += batch_total.detach()

                    if rank == 0:
                        pbar_val.update(1)
                        pbar_val.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{(batch_correct.float() / batch_total.float()).item():.4f}'
                        })

            if rank == 0:
                pbar_val.close()

        # Aggregate validation metrics
        epoch_val_loss = aggregate_scalar(epoch_val_loss, device)
        epoch_val_correct = aggregate_scalar(epoch_val_correct, device)
        epoch_val_total = aggregate_scalar(epoch_val_total, device)

        # Calculate final metrics
        if epoch_train_total > 0:
            epoch_train_loss = epoch_train_loss / epoch_train_total
            epoch_train_accuracy = epoch_train_correct / epoch_train_total
        else:
            epoch_train_loss = float('inf')
            epoch_train_accuracy = 0.0

        if epoch_val_total > 0:
            epoch_val_loss = epoch_val_loss / epoch_val_total
            epoch_val_accuracy = epoch_val_correct / epoch_val_total
        else:
            epoch_val_loss = float('inf')
            epoch_val_accuracy = 0.0

        scheduler.step()

        # Save checkpoint and log
        if rank == 0:
            if epoch % config.saveCheckpointFreq == 0 or epoch == config.totalEpochs:
                model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, epoch, config, filename_prefix="pt")

            if config.useWandB and WANDB_AVAILABLE:
                wandb.log({
                    "train_loss": epoch_train_loss,
                    "train_accuracy": epoch_train_accuracy,
                    "val_loss": epoch_val_loss,
                    "val_accuracy": epoch_val_accuracy,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

    # Cleanup
    if hasattr(train_dataset, 'close'):
        train_dataset.close()
    if val_dataset and hasattr(val_dataset, 'close'):
        val_dataset.close()
    cleanup_memory(force=True)
    dist.destroy_process_group()

    if rank == 0 and config.useWandB and WANDB_AVAILABLE:
        wandb.finish()


def get_num_samples(hdf5_files):
    """Count total samples across HDF5 files."""
    total_samples = 0
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, 'r') as f:
            total_samples += f['snps'].shape[0]
    return total_samples


def main():
    args = parse_args()
    config = ModelConfig.from_yaml(args.configFile)

    # Set run name
    config.run = f"{config.runId}_{config.dataset}_chr{config.chromosome}_{config.population}_seg{config.segLen}_overlap{config.overlap}"

    # Get distributed training info
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    # Auto-generate checkpoint directory
    hla_suffix = ""
    if '_HLA' in getattr(config, 'resPtDir', '') or '_hla' in getattr(config, 'resPtDir', '').lower():
        hla_suffix = "_HLA"
    config.modelDir = f"checkpoints_pt/{config.dataset}_{config.population}_chr{config.chromosome}{hla_suffix}"

    if rank == 0:
        os.makedirs(config.modelDir, exist_ok=True)

    # Get dataset paths
    train_hdf5_files, val_hdf5_files = get_pretrain_dataset_paths(config)

    if rank == 0:
        print(f"Checkpoint dir: {config.modelDir}")
        print(f"Looking for train files in: {config.resPtDir}/train/")
        print(f"Found {len(train_hdf5_files)} train files, {len(val_hdf5_files)} val files")
        if train_hdf5_files:
            print(f"  Train: {train_hdf5_files[0]}")
        if val_hdf5_files:
            print(f"  Val: {val_hdf5_files[0]}")

    latest_checkpoint_file = find_latest_checkpoint(config, filename_prefix='pt')

    use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    train_ddp(
        use_gpu,
        rank=rank,
        world_size=world_size,
        config=config,
        train_hdf5_files=train_hdf5_files,
        val_hdf5_files=val_hdf5_files,
        checkpoint_file=latest_checkpoint_file
    )


if __name__ == "__main__":
    main()
