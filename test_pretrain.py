#!/usr/bin/env python3
"""
Test/Inference script for GenoBERT pretrained models.

Evaluates model performance on test data with various masking probabilities.
Supports single-GPU, multi-GPU, and multi-node testing via DDP.

Usage:
    # Single GPU
    python test_pretrain.py --configFile configs/config.yaml --checkpoint checkpoints/model.pth

    # Multi-GPU (via SLURM)
    srun python test_pretrain.py --configFile configs/config.yaml --checkpoint checkpoints/model.pth

    # Test with multiple mask probabilities
    python test_pretrain.py --configFile configs/config.yaml --checkpoint checkpoints/model.pth --maskProb 0.05 0.15 0.5
"""

import argparse
import glob
import json
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config.modelconfig import ModelConfig
from data.dataset import SNPsDataset_HDF5
from data.utils import (
    mask_random_positions,
    mask_random_positions_bias,
    FocalLoss,
    GatedFocalLoss,
    F1Loss,
)
from model.genobert import GenoBERTMLM


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_criterion(config, device):
    """Get loss function based on config."""
    loss_type = config.lossType.lower() if hasattr(config, 'lossType') else 'crossentropy'

    if loss_type == 'focalloss':
        return FocalLoss(config).to(device)
    elif loss_type == 'gatedfocalloss':
        return GatedFocalLoss(config).to(device)
    elif loss_type == 'f1loss':
        return F1Loss(config).to(device)
    else:
        return nn.CrossEntropyLoss().to(device)


def collate_fn(batch):
    """Collate function for DataLoader."""
    snps_batch = []
    index_batch = []

    for item in batch:
        snps, index = item[:2]  # Handle both 2-tuple and 3-tuple returns
        snps_batch.append(snps)
        index_batch.append(index)

    snps_tensor = torch.stack(snps_batch)
    index_tensor = torch.stack(index_batch)

    return snps_tensor, index_tensor


def get_test_files(config):
    """
    Get test HDF5 files based on config.

    Looks for files in:
        1. {resPtDir}/{dataset}/test/{prefix}_test_*.hdf5
        2. {resPtDir}/test/{prefix}_test_*.hdf5 (fallback)
    """
    base = getattr(config, 'resPtDir', './res_pt')
    genotype_ds = getattr(config, 'genotypeDataset', None) or config.dataset.split('_')[0]
    prefix = f"{genotype_ds}_chr{config.chromosome}_{config.population}_seg{config.segLen}_overlap{config.overlap}"

    # Try dataset subdirectory first
    base_paths = [
        f"{base}/{genotype_ds}",  # e.g., ./res_pt/1KGP
        base,                      # e.g., ./res_pt (fallback)
    ]

    test_files = []
    found_path = None
    for b in base_paths:
        # Check for merged test file
        merged_file = f"{b}/test/{prefix}_test_all.hdf5"
        if os.path.exists(merged_file):
            return [merged_file], merged_file

        # Check for per-gene test files
        pattern = f"{b}/test/{prefix}_test_*.hdf5"
        test_files = sorted(glob.glob(pattern))
        test_files = [f for f in test_files if '_all.hdf5' not in f]

        if test_files:
            found_path = f"{b}/test/"
            break

    return test_files, found_path


def compute_genomic_bias(snps_index, config, device):
    """Compute normalized genomic position bias for attention."""
    batch_size = snps_index.shape[0]
    batch_bias = torch.zeros(batch_size, snps_index.shape[1], dtype=torch.float32, device=device)

    for b in range(batch_size):
        # Get valid genomic positions (positive values)
        valid_pos_mask = snps_index[b, :, 1] > 0
        if valid_pos_mask.any():
            genomic_positions = snps_index[b, valid_pos_mask, 1].float()
            min_pos = genomic_positions.min()
            max_pos = genomic_positions.max()

            # Special token masks
            cls_pos_mask = snps_index[b, :, 1] == -1
            sep_pos_mask = snps_index[b, :, 1] == -2
            pad_pos_mask = snps_index[b, :, 1] == -3

            # Normalize to [0.1, 0.9] range
            if max_pos > min_pos:
                normalized = 0.1 + 0.8 * (genomic_positions - min_pos) / (max_pos - min_pos)
                batch_bias[b, valid_pos_mask] = normalized
            else:
                batch_bias[b, valid_pos_mask] = 0.5

            # Set special token positions
            if cls_pos_mask.any():
                batch_bias[b, cls_pos_mask] = 0.0
            if sep_pos_mask.any():
                batch_bias[b, sep_pos_mask] = 1.0

    return batch_bias


def evaluate(model, test_files, config, mask_prob, device, rank, world_size, seed):
    """
    Evaluate model on test data.

    Returns dict with loss, accuracy metrics.
    """
    model.eval()
    set_seed(seed)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    masked_correct = 0
    masked_samples = 0

    # Per-class metrics
    class_correct = torch.zeros(config.vocabSize, device=device)
    class_total = torch.zeros(config.vocabSize, device=device)

    criterion = get_criterion(config, device)

    # Progress bar for files (only on rank 0)
    file_iter = tqdm(test_files, disable=(rank != 0), desc=f"mask={mask_prob:.0%}", unit="file")

    with torch.no_grad():
        for test_file in file_iter:
            dataset = SNPsDataset_HDF5(test_file, preload=True)
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            loader = DataLoader(
                dataset,
                batch_size=config.batchSize,
                sampler=sampler,
                num_workers=config.numWorkers,
                pin_memory=True,
                collate_fn=collate_fn
            )

            for snps, snps_index in loader:
                snps = snps.to(device)
                snps_index = snps_index.to(device)
                batch_padding_mask = (snps != config.padId)

                # Compute genomic bias if enabled
                if config.enableBias:
                    batch_bias = compute_genomic_bias(snps_index, config, device)
                else:
                    batch_bias = None

                # Apply masking
                if config.sampling == 'upsampling':
                    masked_input, missing_mask = mask_random_positions_bias(
                        config, snps.clone()
                    )
                else:
                    masked_input, missing_mask = mask_random_positions(
                        config, snps.clone()
                    )

                # Override mask probability for testing
                if mask_prob != config.maskProb:
                    missing_mask = torch.rand_like(snps, dtype=torch.float) < mask_prob
                    missing_mask = missing_mask & batch_padding_mask
                    masked_input = snps.clone()
                    masked_input[missing_mask] = config.maskId

                # Forward pass
                logits = model(masked_input, batch_bias, batch_padding_mask)

                # Evaluate based on benchmarkAll setting
                if config.benchmarkAll:
                    valid_mask = batch_padding_mask
                else:
                    valid_mask = missing_mask & batch_padding_mask

                if valid_mask.sum() == 0:
                    continue

                logits_valid = logits[valid_mask]
                labels_valid = snps[valid_mask]

                loss = criterion(logits_valid, labels_valid)
                predictions = torch.argmax(logits_valid, dim=-1)

                total_loss += loss.item() * valid_mask.sum().item()
                total_correct += (predictions == labels_valid).sum().item()
                total_samples += valid_mask.sum().item()

                # Track per-class accuracy
                for c in range(config.vocabSize):
                    class_mask = labels_valid == c
                    class_total[c] += class_mask.sum()
                    class_correct[c] += ((predictions == labels_valid) & class_mask).sum()

                # Track masked-only accuracy
                masked_only = missing_mask & batch_padding_mask
                if masked_only.sum() > 0:
                    logits_masked = logits[masked_only]
                    labels_masked = snps[masked_only]
                    predictions_masked = torch.argmax(logits_masked, dim=-1)
                    masked_correct += (predictions_masked == labels_masked).sum().item()
                    masked_samples += masked_only.sum().item()

            dataset.close()

    # Aggregate across ranks
    metrics = torch.tensor(
        [total_loss, total_correct, total_samples, masked_correct, masked_samples],
        device=device
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(class_total, op=dist.ReduceOp.SUM)

    avg_loss = metrics[0].item() / max(metrics[2].item(), 1)
    accuracy = metrics[1].item() / max(metrics[2].item(), 1)
    masked_accuracy = metrics[3].item() / max(metrics[4].item(), 1) if metrics[4].item() > 0 else 0.0

    # Per-class accuracy
    per_class_acc = {}
    token_names = ['MASK', '0|0', '0|1', '1|0', '1|1', 'CLS', 'SEP', 'PAD']
    for c in range(config.vocabSize):
        if class_total[c].item() > 0:
            per_class_acc[token_names[c]] = class_correct[c].item() / class_total[c].item()

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'masked_accuracy': masked_accuracy,
        'total_samples': int(metrics[2].item()),
        'masked_samples': int(metrics[4].item()),
        'mask_prob': mask_prob,
        'seed': seed,
        'benchmark_all': config.benchmarkAll,
        'per_class_accuracy': per_class_acc
    }


def main():
    parser = argparse.ArgumentParser(description="Test GenoBERT pretrained model")
    parser.add_argument('--configFile', required=True, help="Path to config YAML file")
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint")
    parser.add_argument('--maskProb', type=float, nargs='+', default=[0.05, 0.15, 0.5],
                        help="Mask probabilities to test (default: 0.05 0.15 0.5)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output', type=str, default=None,
                        help="Output JSON file (default: test_results_{runId}.json)")
    args = parser.parse_args()

    # Initialize distributed training
    if 'SLURM_PROCID' in os.environ:
        # SLURM environment - use SLURM variables
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])

        # Calculate local rank: tasks per node
        tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', world_size))
        local_rank = rank % tasks_per_node

        # Ensure local_rank doesn't exceed available GPUs
        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            local_rank = rank % num_gpus

        # Set environment variables for torch distributed
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # Initialize process group
        dist.init_process_group(backend='nccl', init_method='env://')
    elif 'RANK' in os.environ:
        # torchrun or other launcher
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
    else:
        # Single GPU mode - no distributed
        rank = 0
        world_size = 1
        local_rank = 0

        # Initialize a single-process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', rank=0, world_size=1)

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"Distributed init: rank={rank}, world_size={world_size}, local_rank={local_rank}, device={device}")

    # Load config
    config = ModelConfig.from_yaml(args.configFile)

    # Get test files
    test_files, found_path = get_test_files(config)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"GenoBERT Testing")
        print(f"{'='*60}")
        print(f"Config: {args.configFile}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Dataset: {config.dataset}, Chr: {config.chromosome}, Pop: {config.population}")
        if found_path:
            print(f"Test files: {len(test_files)} files from {found_path}")
        else:
            print(f"Test files: {len(test_files)}")
        print(f"World size: {world_size} GPUs")
        print(f"Mask probs: {args.maskProb}")
        print(f"Seed: {args.seed}")
        print(f"Benchmark all: {config.benchmarkAll}")
        print(f"{'='*60}\n")

    if not test_files:
        if rank == 0:
            print("ERROR: No test files found!")
            print(f"  Looking for: {config.resPtDir}/{{dataset}}/test/...")
            print("  Run data preparation with --split test first.")
        dist.destroy_process_group()
        return

    # Load model
    model = GenoBERTMLM(config).to(device)

    # Load checkpoint
    if rank == 0:
        print(f"Loading checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if rank == 0 and 'epoch' in checkpoint:
            print(f"  Checkpoint from epoch {checkpoint['epoch']}")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    # Wrap with DDP
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

    # Test with different mask probabilities
    all_results = {}
    for mask_prob in args.maskProb:
        if rank == 0:
            print(f"\nTesting with mask_prob={mask_prob:.1%}...")

        results = evaluate(model, test_files, config, mask_prob, device, rank, world_size, args.seed)

        if rank == 0:
            print(f"  Loss: {results['loss']:.4f}")
            print(f"  All-position accuracy: {results['accuracy']:.4f}")
            print(f"  Masked-only accuracy: {results['masked_accuracy']:.4f}")
            print(f"  Samples evaluated: {results['total_samples']:,}")

        all_results[f'mask_{int(mask_prob*100)}pct'] = results

    # Save results
    if rank == 0:
        output_file = args.output or f"test_results_{config.runId}.json"

        output_data = {
            'config_file': args.configFile,
            'checkpoint': args.checkpoint,
            'run_id': config.runId,
            'dataset': config.dataset,
            'chromosome': config.chromosome,
            'population': config.population,
            'num_test_files': len(test_files),
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'Mask %':<10} {'Loss':<10} {'All Acc':<12} {'Masked Acc':<12}")
        print("-" * 46)
        for key, res in all_results.items():
            print(f"{res['mask_prob']*100:<10.0f} {res['loss']:<10.4f} "
                  f"{res['accuracy']:<12.4f} {res['masked_accuracy']:<12.4f}")

        print(f"\nPer-class accuracy (last mask prob):")
        last_result = list(all_results.values())[-1]
        for token, acc in last_result['per_class_accuracy'].items():
            print(f"  {token}: {acc:.4f}")

        print(f"\nResults saved to: {output_file}")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
