"""
Utility functions for GenoBERT pretraining.

Includes masking strategies, loss functions, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import os
import re
import gc
import glob
from collections import Counter
import h5py
from typing import Tuple, List, Optional


def mask_random_positions(config, batch_snps):
    """
    Apply random masking to input sequences.

    Args:
        config: Configuration object with maskProb and maskId
        batch_snps: Input tensor [batch_size, seq_len]

    Returns:
        masked_input: Tensor with masked positions
        missing_mask: Boolean mask indicating masked positions
    """
    device = batch_snps.device
    batch_size, seq_len = batch_snps.shape

    mask_prob = config.maskProb
    missing_mask = torch.rand(batch_size, seq_len, device=device) < mask_prob

    masked_input = batch_snps.clone()
    masked_input[missing_mask] = config.maskId

    return masked_input, missing_mask


def mask_random_positions_bias(config, data_batch, bert_strategy=False):
    """
    Mask tokens with bias toward minority alleles.

    Prioritizes masking heterozygous (2, 3) and homozygous alternate (4) tokens,
    falling back to homozygous reference (1) tokens when needed.

    Args:
        config: Configuration with maskProb, upsamplingRatio, and maskId
        data_batch: Input tensor with tokens
        bert_strategy: If True, applies BERT-style 80/10/10 masking

    Returns:
        masked_data: Masked input tensor
        mask_tensor: Boolean mask indicating masked positions
    """
    data_batch = data_batch.to(torch.int)
    maskable = (data_batch == 1) | (data_batch == 2) | (data_batch == 3) | (data_batch == 4)
    maskable_234 = (data_batch == 2) | (data_batch == 3) | (data_batch == 4)
    maskable_1 = data_batch == 1

    maskable_sum = maskable.sum()
    num_to_mask = (config.maskProb * maskable_sum.float()).int()

    if num_to_mask <= 0:
        return data_batch, torch.zeros_like(data_batch, dtype=torch.bool, device=data_batch.device)

    num_to_mask_234 = (num_to_mask.float() * config.upsamplingRatio).int()

    maskable_234_flat = maskable_234.flatten()
    maskable_234_indices = torch.where(maskable_234_flat)[0]

    chosen_indices = torch.empty(0, dtype=torch.long, device=data_batch.device)
    mask_234_count = maskable_234_indices.size(0)

    if mask_234_count >= num_to_mask_234:
        perm = torch.randperm(mask_234_count, device=data_batch.device)
        chosen_indices = maskable_234_indices[perm[:num_to_mask_234]]
    else:
        chosen_indices = maskable_234_indices
        remaining_mask = num_to_mask - mask_234_count

        if remaining_mask > 0:
            maskable_1_flat = maskable_1.flatten()
            maskable_1_indices = torch.where(maskable_1_flat)[0]
            mask_1_count = maskable_1_indices.size(0)

            if mask_1_count > 0:
                to_pick = torch.min(remaining_mask, torch.tensor(mask_1_count, device=data_batch.device))
                perm = torch.randperm(mask_1_count, device=data_batch.device)
                extra_indices = maskable_1_indices[perm[:to_pick]]
                chosen_indices = torch.cat((chosen_indices, extra_indices))

    mask_tensor = torch.zeros_like(data_batch, dtype=torch.bool, device=data_batch.device)
    mask_flat = mask_tensor.flatten()
    mask_flat[chosen_indices] = True
    mask_tensor = mask_flat.view_as(data_batch)

    masked_data = data_batch.clone()

    if bert_strategy:
        chosen_count = chosen_indices.size(0)
        num_mask_tokens = (0.8 * chosen_count).int()
        num_random_tokens = (0.1 * chosen_count).int()

        perm = torch.randperm(chosen_count, device=data_batch.device)
        mask_indices = chosen_indices[perm[:num_mask_tokens]]

        masked_data_flat = masked_data.flatten()
        masked_data_flat[mask_indices] = config.maskId

        if num_random_tokens > 0:
            random_indices = chosen_indices[perm[num_mask_tokens:num_mask_tokens + num_random_tokens]]
            random_tokens = torch.randint(1, 5, size=(num_random_tokens,), device=data_batch.device)
            masked_data_flat[random_indices] = random_tokens

        masked_data = masked_data_flat.view_as(data_batch)
    else:
        masked_data_flat = masked_data.flatten()
        masked_data_flat[chosen_indices] = config.maskId
        masked_data = masked_data_flat.view_as(data_batch)

    return masked_data, mask_tensor


def find_latest_checkpoint(config, filename_prefix="checkpoint"):
    """
    Find the latest checkpoint file by epoch number.

    Args:
        config: Configuration with modelDir and runId
        filename_prefix: Prefix for checkpoint files

    Returns:
        Path to latest checkpoint or None
    """
    pattern = os.path.join(config.modelDir, f"{filename_prefix}_{config.runId}_epoch_*.pth")
    list_of_files = glob.glob(pattern)

    if not list_of_files:
        return None

    latest_file = None
    highest_epoch = -1

    for file in list_of_files:
        match = re.search(r"epoch_(\d+)\.pth", file)
        if match:
            epoch_number = int(match.group(1))
            if epoch_number > highest_epoch:
                highest_epoch = epoch_number
                latest_file = file

    return latest_file


def save_checkpoint(state, epoch, config, filename_prefix='checkpoint'):
    """
    Save model checkpoint.

    Args:
        state: State dict with model, optimizer, scheduler states
        epoch: Current epoch number
        config: Configuration with modelDir and runId
        filename_prefix: Prefix for checkpoint file
    """
    checkpointName = os.path.join(
        config.modelDir,
        f'{filename_prefix}_{config.runId}_epoch_{epoch}.pth'
    )
    torch.save(state, checkpointName)


def get_pretrain_dataset_paths(config, gene_ids=None):
    """
    Get pretrain dataset paths from res_pt/{dataset}/{split}/ or res_pt/{split}/.

    File naming patterns:
        Per-gene:    {prefix}_{split}_{gene_id}.hdf5
        Merged all:  {prefix}_{split}_all.hdf5
        Gene list:   {prefix}_{split}_{genes_list_name}.hdf5

    Priority order:
        1. Gene list merged file (if genesListFile is set)
        2. All merged file (_all.hdf5)
        3. Per-gene files

    Args:
        config: ModelConfig instance
        gene_ids: Optional list of gene IDs to filter

    Returns:
        Tuple of (train_files, val_files)
    """
    from pathlib import Path

    base = getattr(config, 'resPtDir', None) or './res_pt'

    genotype_ds = getattr(config, 'genotypeDataset', None) or config.dataset.split('_')[0]

    prefix = f"{genotype_ds}_chr{config.chromosome}_{config.population}_seg{config.segLen}_overlap{config.overlap}"

    print(f"  Looking for data with prefix: {prefix}")
    print(f"  Base directory: {base}")
    print(f"  Genotype dataset: {genotype_ds}, Population: {config.population}")

    train_files = []
    val_files = []

    # Try multiple base paths (prefer dataset subdirectory first)
    base_paths = [
        f"{base}/{genotype_ds}",        # e.g., ./res_pt/1KGP (preferred)
        base,                           # e.g., ./res_pt (fallback)
    ]

    # Check for gene list merged file first (highest priority)
    genes_list_file = getattr(config, 'genesListFile', None)
    if genes_list_file:
        genes_list_name = Path(genes_list_file).stem

        for b in base_paths:
            train_genes_merged = f"{b}/train/{prefix}_train_{genes_list_name}.hdf5"
            val_genes_merged = f"{b}/val/{prefix}_val_{genes_list_name}.hdf5"

            print(f"  Checking gene list file: {train_genes_merged}")
            if os.path.exists(train_genes_merged):
                print(f"  Found gene list merged file: {train_genes_merged}")
                train_files = [train_genes_merged]
                val_files = [val_genes_merged] if os.path.exists(val_genes_merged) else []
                return train_files, val_files

        print(f"  Warning: genesListFile set but merged file not found")
        print(f"  Run merge_genes.py with --genes_list_file {genes_list_file} first")

    # Check for merged file (_all.hdf5)
    for b in base_paths:
        train_merged = f"{b}/train/{prefix}_train_all.hdf5"
        val_merged = f"{b}/val/{prefix}_val_all.hdf5"

        print(f"  Checking merged file: {train_merged}")
        if os.path.exists(train_merged):
            print(f"  Found merged file: {train_merged}")
            train_files = [train_merged]
            val_files = [val_merged] if os.path.exists(val_merged) else []
            return train_files, val_files

    # Look for per-gene files (try dataset subdirectory first)
    for b in base_paths:
        train_pattern = f"{b}/train/{prefix}_train_*.hdf5"
        val_pattern = f"{b}/val/{prefix}_val_*.hdf5"

        print(f"  Checking per-gene pattern: {train_pattern}")
        train_files = sorted(glob.glob(train_pattern))
        val_files = sorted(glob.glob(val_pattern))

        # Exclude merged/chunk files
        train_files = [f for f in train_files if '_all.hdf5' not in f and '_chunk' not in f]
        val_files = [f for f in val_files if '_all.hdf5' not in f and '_chunk' not in f]

        if train_files:
            print(f"  Found {len(train_files)} per-gene files")
            break

    # Check old location for backward compatibility
    if not train_files:
        print(f"  No files found in standard locations, checking legacy paths...")
        old_base = getattr(config, 'resDir', './res')
        old_prefix = f"{config.dataset}_chr{config.chromosome}_{config.population}_seg{config.segLen}_overlap{config.overlap}"
        train_chunk_pattern = f"{old_base}/{old_prefix}_train_chunk*.hdf5"
        val_chunk_pattern = f"{old_base}/{old_prefix}_val_chunk*.hdf5"
        train_files = sorted(glob.glob(train_chunk_pattern))
        val_files = sorted(glob.glob(val_chunk_pattern))

    # Filter by gene IDs if specified
    if gene_ids is None:
        gene_ids = config.get_gene_list() if hasattr(config, 'get_gene_list') else []

    if gene_ids and gene_ids != ["complete"]:
        if isinstance(gene_ids, str):
            gene_ids = [gene_ids]

        if train_files and not any(x in train_files[0] for x in ['_all.hdf5', '_chunk']):
            train_files = [f for f in train_files if any(gene in f for gene in gene_ids)]
        if val_files and not any(x in val_files[0] for x in ['_all.hdf5', '_chunk']):
            val_files = [f for f in val_files if any(gene in f for gene in gene_ids)]

    return train_files, val_files


def cleanup_memory(force=False):
    """
    Optimized memory cleanup.

    Args:
        force: Force cleanup regardless of memory state
    """
    if not torch.cuda.is_available():
        gc.collect()
        return

    device = torch.cuda.current_device()
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
    fragmentation = mem_reserved - mem_allocated

    if force or fragmentation > 5:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Memory cleanup: freed {fragmentation:.2f}GB of fragmented memory")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reduces loss for well-classified examples, focusing training on hard examples.

    Args:
        config: Configuration with focalGamma and useAlphaWeighting
        ignore_index: Index to ignore in loss calculation
    """
    def __init__(self, config, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = config.focalGamma
        self.use_alpha = getattr(config, 'useAlphaWeighting', True)

        if self.use_alpha:
            alpha_weights = torch.tensor([
                1.0,   # MASK (0)
                0.25,  # 0|0 (1) - common homozygous
                0.5,   # 0|1 (2) - heterozygous
                0.5,   # 1|0 (3) - heterozygous
                0.75,  # 1|1 (4) - rare homozygous
                0.1,   # CLS (5)
                0.1,   # SEP (6)
                0.1,   # PAD (7)
            ], dtype=torch.float32)
            self.register_buffer('alpha', alpha_weights)

    def forward(self, inputs, targets):
        if inputs.dim() == 3:
            inputs = inputs.view(-1, inputs.shape[-1])
        if targets.dim() > 1:
            targets = targets.view(-1)

        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if torch.isnan(inputs).any():
            inputs = torch.nan_to_num(inputs, nan=0.0)

        inputs = torch.clamp(inputs, min=-100, max=100)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        ce_loss = torch.clamp(ce_loss, max=100)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.use_alpha:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if torch.isnan(focal_loss).any():
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0)

        return focal_loss.mean()


class GatedFocalLoss(nn.Module):
    """
    Focal Loss with learnable gating between uniform and class-weighted alpha.

    Args:
        config: Configuration with focalGamma and useAlphaWeighting
        ignore_index: Index to ignore in loss calculation
    """
    def __init__(self, config, ignore_index=-100):
        super(GatedFocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = config.focalGamma
        self.use_alpha = getattr(config, 'useAlphaWeighting', False)

        if self.use_alpha:
            alpha_weights = getattr(config, 'alphaWeights',
                                   [1.0, 0.25, 0.5, 0.5, 0.75, 0.1, 0.1, 0.1])
            self.register_buffer('alpha', torch.tensor(alpha_weights, dtype=torch.float32))
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs, targets):
        if inputs.dim() == 3:
            inputs = inputs.view(-1, inputs.shape[-1])
        if targets.dim() > 1:
            targets = targets.view(-1)

        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.use_alpha:
            alpha_t = self.alpha[targets]
            effective_alpha = self.gate * alpha_t + (1 - self.gate) * 1.0
            focal_loss = effective_alpha * focal_loss

        return focal_loss.mean()


class F1Loss(nn.Module):
    """
    Differentiable F1 Loss for multi-class classification.

    Args:
        config: Configuration with vocabSize
        ignore_index: Index to ignore in loss calculation
        epsilon: Small value for numerical stability
    """
    def __init__(self, config, ignore_index=-100, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon
        self.num_classes = config.vocabSize

    def forward(self, inputs, targets):
        if inputs.dim() == 3:
            inputs = inputs.view(-1, inputs.shape[-1])
        if targets.dim() > 1:
            targets = targets.view(-1)

        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        probs = F.softmax(inputs, dim=-1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)

        return 1 - f1.mean()
