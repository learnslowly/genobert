#!/usr/bin/env python3
"""
Pretrain Data Preparation Script with Global SNP-Index Segmentation

This script prepares genotype data for GenoBERT pretraining by:
1. Loading all chromosome SNPs from a VCF file
2. Computing global segment boundaries by SNP index
3. Processing each gene region to extract overlapping segments
4. Saving per-gene HDF5 files with deduplication

GLOBAL SEGMENTATION STRATEGY:
- Segments are cut by SNP INDEX (count), not genomic position
- All genes share the same global segment boundaries
- Each gene file contains segments that overlap its flanking region
- Supports parallel processing via SLURM array jobs

Usage:
    # Process genes (SLURM array job)
    python pretrain_data_prep.py \\
        --genotype_ds 1KGP \\
        --gene_ds GEUVADIS \\
        --chr 22 \\
        --race EUR \\
        --split train \\
        --node_id $SLURM_ARRAY_TASK_ID \\
        --total_nodes $SLURM_ARRAY_TASK_COUNT \\
        --pretrain_vcf /path/to/chr22.vcf.gz \\
        --gene_exp_path /path/to/gene_regions \\
        --output_dir ./res_pt

    # Merge per-gene files after array job completes
    python pretrain_data_prep.py \\
        --genotype_ds 1KGP \\
        --chr 22 \\
        --race EUR \\
        --create_chunks_only \\
        --apply_dedup \\
        --output_dir ./res_pt
"""

import gzip
import pandas as pd
import os
import numpy as np
import re
import time
import h5py
import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Constants ===
GENOTYPE_ENCODING = {
    '0|0': '1', '0/0': '1',
    '0|1': '2', '0/1': '2',
    '1|0': '3', '1/0': '3',
    '1|1': '4', '1/1': '4',
    '.|.': '0', './.': '0'
}
CLS, SEP, PAD = 5, 6, 7
CLS_POS, SEP_POS, PAD_POS = -1, -2, -3

# Global variables to be set from command line arguments
model_input_width = None
token_span = None
overlap_size = None
stride = None
flank_size = None

# Global data structures (shared across gene processing)
global_snp_positions = None
global_genotype_matrix = None
global_sample_ids = None
global_segments_by_index = None


def count_non_padding_snps(segment):
    """Count non-padding SNPs in a segment (excluding CLS, SEP, PAD)"""
    special_tokens_mask = (segment == CLS) | (segment == SEP) | (segment == PAD)
    return np.sum(~special_tokens_mask)


def load_dataset_config(config_path):
    """Load dataset configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded dataset config: {config['dataset_name']}")
    return config


def load_sample_id_mapping(mapping_file, expr_col, geno_col, geno_prefix=""):
    """
    Load sample ID mapping for datasets where expression and genotype use different IDs.

    Args:
        mapping_file: Path to CSV mapping file
        expr_col: Column name for expression IDs
        geno_col: Column name for genotype IDs
        geno_prefix: Optional prefix to add to genotype IDs

    Returns:
        dict: {expression_id: genotype_id_with_prefix}
    """
    print(f"Loading sample ID mapping from {mapping_file}")
    mapping_df = pd.read_csv(mapping_file)

    mapping = {}
    for _, row in mapping_df.iterrows():
        expr_id = str(row[expr_col])
        geno_id = f"{geno_prefix}{row[geno_col]}"
        mapping[expr_id] = geno_id

    print(f"  Loaded {len(mapping)} sample ID mappings")
    return mapping


def load_gene_regions(gene_path):
    """Load gene regions from gene expression file."""
    print(f"Loading gene regions from {gene_path}")
    gene_df = pd.read_csv(gene_path, sep='\t')

    gene_regions = []
    for _, row in gene_df.iterrows():
        gene_id = row['TargetID']
        region_start = max(0, int(row['GeneStart']) - flank_size)
        region_end = int(row['GeneEnd']) + flank_size
        gene_regions.append((gene_id, region_start, region_end))

    print(f"Loaded {len(gene_regions)} gene regions")
    return gene_regions


def load_all_chromosome_snps(pretrain_vcf_path, sample_id_mapping=None):
    """
    Load ALL SNPs for the entire chromosome using all VCF samples.

    Args:
        pretrain_vcf_path: Path to VCF file
        sample_id_mapping: Optional dict mapping expression IDs to genotype IDs

    Returns:
        snp_positions: np.array of genomic positions [num_snps]
        genotype_matrix: np.array of encoded genotypes [num_snps, num_samples]
        sample_ids: np.array of sample IDs [num_samples]
    """
    print(f"Loading ALL chromosome SNPs from {pretrain_vcf_path}")
    start_time = time.time()

    rows, columns = [], []
    with gzip.open(pretrain_vcf_path, 'rt') as f:
        for l in f:
            if l.startswith('##'):
                continue
            elif l.startswith('#CHROM'):
                columns = l.lstrip('#').split('\t')
            else:
                rows.append(l.strip().split('\t'))

    genotype_df = pd.DataFrame(rows, columns=columns)
    pretrain_sample_ids = genotype_df.columns[9:].tolist()
    print(f"Found {len(pretrain_sample_ids)} samples in VCF")

    print(f"Using all {len(pretrain_sample_ids)} samples from VCF")
    genotype_df['POS'] = genotype_df['POS'].astype(int)
    genotype_df = genotype_df.set_index('POS').loc[:, pretrain_sample_ids]
    samples_to_encode = pretrain_sample_ids

    # Convert sample IDs to integers
    new_columns = []
    for sid in samples_to_encode:
        digits = re.sub(r'\D', '', sid)
        if digits:
            new_columns.append(int(digits))
        else:
            new_columns.append(hash(sid) % 1000000)
    genotype_df.columns = new_columns

    # Encode genotypes
    genotype_encoded = genotype_df.replace(GENOTYPE_ENCODING).astype(np.int8)

    snp_positions = genotype_encoded.index.to_numpy()
    genotype_matrix = genotype_encoded.to_numpy()
    sample_ids = genotype_encoded.columns.to_numpy()

    print(f"Loaded {len(snp_positions):,} SNPs for {len(sample_ids)} samples in {time.time() - start_time:.1f}s")

    return snp_positions, genotype_matrix, sample_ids


def compute_global_segments(num_snps, token_span, overlap_size):
    """
    Compute global segment boundaries by SNP INDEX (not position).

    Args:
        num_snps: Total number of SNPs in chromosome
        token_span: Number of SNPs per segment (model_input_width - 2)
        overlap_size: Number of overlapping SNPs between consecutive segments

    Returns:
        List of (start_idx, end_idx) tuples
    """
    stride = token_span - overlap_size
    segments = []

    start_idx = 0
    while start_idx < num_snps:
        end_idx = min(start_idx + token_span, num_snps)
        segments.append((start_idx, end_idx))
        start_idx += stride

    print(f"Computed {len(segments):,} global segments (token_span={token_span}, overlap={overlap_size}, stride={stride})")
    return segments


def process_gene_block_pretrain_global(args):
    """
    Process a single gene using global segmentation.

    Uses global variables:
        global_snp_positions, global_genotype_matrix, global_sample_ids, global_segments_by_index
    """
    (gene_id, region_start, region_end, min_snps_threshold, overlap_threshold) = args

    gene_region_start = region_start - flank_size
    gene_region_end = region_end + flank_size

    gene_snp_mask = (global_snp_positions >= gene_region_start) & (global_snp_positions <= gene_region_end)
    gene_snp_indices = np.where(gene_snp_mask)[0]

    if len(gene_snp_indices) == 0:
        return {'gene_id': gene_id, 'status': 'no_snps', 'reason': 'no SNPs in gene region'}

    segments_info = []
    total_segments_checked = 0
    segments_excluded_overlap = 0
    segments_excluded_minsnps = 0

    for seg_start_idx, seg_end_idx in global_segments_by_index:
        total_segments_checked += 1

        segment_snp_indices = np.arange(seg_start_idx, seg_end_idx)
        snps_in_gene_region = np.intersect1d(segment_snp_indices, gene_snp_indices)

        if len(snps_in_gene_region) == 0:
            continue

        overlap_ratio = len(snps_in_gene_region) / (seg_end_idx - seg_start_idx)
        if overlap_ratio < overlap_threshold:
            segments_excluded_overlap += 1
            continue

        segment_genotypes = global_genotype_matrix[seg_start_idx:seg_end_idx, :]
        segment_positions = global_snp_positions[seg_start_idx:seg_end_idx]

        num_snps_in_seg = segment_genotypes.shape[0]
        num_samples = segment_genotypes.shape[1]

        segment = segment_genotypes.T

        cls_col = np.full((num_samples, 1), CLS)
        sep_col = np.full((num_samples, 1), SEP)
        segment = np.hstack([cls_col, segment, sep_col])

        if segment.shape[1] < model_input_width:
            pad = np.full((num_samples, model_input_width - segment.shape[1]), PAD)
            segment = np.hstack([segment, pad])

        pos_seq = np.concatenate([[CLS_POS], segment_positions, [SEP_POS]])
        if len(pos_seq) < model_input_width:
            pos_seq = np.concatenate([pos_seq, [PAD_POS] * (model_input_width - len(pos_seq))])

        rows = np.repeat(global_sample_ids[:, np.newaxis], len(pos_seq), axis=1)
        cols = np.tile(pos_seq, (len(global_sample_ids), 1))
        snps_index = np.stack((rows, cols), axis=-1)

        non_padding_count = count_non_padding_snps(segment[0])
        if non_padding_count < min_snps_threshold:
            segments_excluded_minsnps += 1
            continue

        segments_info.append({
            'segment': segment,
            'snps_index': snps_index,
            'non_padding_count': non_padding_count,
            'global_seg_idx': (seg_start_idx, seg_end_idx)
        })

    if not segments_info:
        return {
            'gene_id': gene_id,
            'status': 'no_segments',
            'reason': f'no segments passed filters (checked {total_segments_checked}, excluded: {segments_excluded_overlap} overlap, {segments_excluded_minsnps} min_snps)'
        }

    all_snps_rows = [s['segment'] for s in segments_info]
    all_snpsIndex_rows = [s['snps_index'] for s in segments_info]

    snps = np.vstack(all_snps_rows)
    snpsIndex = np.vstack(all_snpsIndex_rows)

    return {
        'gene_id': gene_id,
        'status': 'success',
        'snps': snps,
        'snpsIndex': snpsIndex,
        'num_segments': len(segments_info),
        'total_rows': snps.shape[0],
        'num_snps_in_region': len(gene_snp_indices),
        'segments_checked': total_segments_checked,
        'segments_excluded_overlap': segments_excluded_overlap,
        'segments_excluded_minsnps': segments_excluded_minsnps
    }


def save_gene_result(result, args):
    """Save individual gene result to HDF5 file"""
    gene_id = result['gene_id']
    split_dir = f"{args.output_dir}/{args.split}"
    os.makedirs(split_dir, exist_ok=True)

    output_path = f"{split_dir}/{args.genotype_ds}_chr{args.chr}_{args.race}_seg{model_input_width}_overlap{overlap_size}_{args.split}_{gene_id}.hdf5"

    try:
        with h5py.File(output_path, "w") as f:
            f.create_dataset("snps", data=result['snps'], compression="gzip")
            f.create_dataset("snpsIndex", data=result['snpsIndex'], compression="gzip")

            f.attrs['gene_id'] = gene_id
            f.attrs['num_segments'] = result['num_segments']
            f.attrs['total_rows'] = result['total_rows']
            f.attrs['model_input_width'] = model_input_width
            f.attrs['overlap_size'] = overlap_size
            f.attrs['stride'] = stride
            f.attrs['flank_size'] = flank_size
            f.attrs['num_snps_in_region'] = result['num_snps_in_region']
            f.attrs['segmentation_type'] = 'global_snp_index'
            f.attrs['overlap_threshold'] = args.overlap_threshold
            f.attrs['min_snps_threshold'] = args.min_snps

        if args.verbose:
            print(f"Node {args.node_id}: Saved {gene_id}: {result['num_segments']} segments")
        return True
    except Exception as e:
        print(f"Node {args.node_id}: Failed to save gene {gene_id}: {e}")
        return False


def _process_and_save_gene(gene_args_with_save_args):
    """Process a single gene and save result. Used by parallel workers."""
    gene_args, save_args = gene_args_with_save_args
    gene_id = gene_args[0]

    try:
        result = process_gene_block_pretrain_global(gene_args)

        if result['status'] == 'success':
            split_dir = f"{save_args['output_dir']}/{save_args['split']}"
            os.makedirs(split_dir, exist_ok=True)

            output_path = f"{split_dir}/{save_args['genotype_ds']}_chr{save_args['chr']}_{save_args['race']}_seg{model_input_width}_overlap{overlap_size}_{save_args['split']}_{gene_id}.hdf5"

            with h5py.File(output_path, "w") as f:
                f.create_dataset("snps", data=result['snps'], compression="gzip")
                f.create_dataset("snpsIndex", data=result['snpsIndex'], compression="gzip")

                f.attrs['gene_id'] = gene_id
                f.attrs['num_segments'] = result['num_segments']
                f.attrs['total_rows'] = result['total_rows']
                f.attrs['model_input_width'] = model_input_width
                f.attrs['overlap_size'] = overlap_size
                f.attrs['stride'] = stride
                f.attrs['flank_size'] = flank_size
                f.attrs['num_snps_in_region'] = result['num_snps_in_region']
                f.attrs['segmentation_type'] = 'global_snp_index'
                f.attrs['overlap_threshold'] = save_args['overlap_threshold']
                f.attrs['min_snps_threshold'] = save_args['min_snps']

            return {
                'gene_id': gene_id,
                'status': 'saved',
                'num_segments': result['num_segments']
            }
        else:
            return {
                'gene_id': gene_id,
                'status': result['status'],
                'reason': result.get('reason', 'unknown')
            }
    except Exception as e:
        return {
            'gene_id': gene_id,
            'status': 'exception',
            'reason': str(e)
        }


def process_gene_subset_pretrain_global(gene_subset, args):
    """Process a subset of genes using global segmentation."""
    import multiprocessing as mp

    print(f"Node {args.node_id}/{args.total_nodes}: Processing {len(gene_subset)} genes")
    print(f"  Min SNPs per segment: {args.min_snps}")
    print(f"  Overlap threshold: {args.overlap_threshold:.2f}")
    print(f"  Parallel workers: {args.num_workers}")

    save_args = {
        'output_dir': args.output_dir,
        'split': args.split,
        'genotype_ds': args.genotype_ds,
        'chr': args.chr,
        'race': args.race,
        'node_id': args.node_id,
        'verbose': args.verbose,
        'overlap_threshold': args.overlap_threshold,
        'min_snps': args.min_snps
    }

    args_list = []
    for gene_id, region_start, region_end in gene_subset:
        gene_args = (gene_id, region_start, region_end, args.min_snps, args.overlap_threshold)
        args_list.append((gene_args, save_args))

    saved_count = 0
    failed_count = 0
    no_segments_count = 0
    filtered_genes = []

    if args.num_workers > 1:
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass

        print(f"Node {args.node_id}: Starting parallel processing with {args.num_workers} workers...")

        with mp.Pool(processes=args.num_workers) as pool:
            results_iter = pool.imap_unordered(_process_and_save_gene, args_list, chunksize=1)

            for i, result in enumerate(results_iter):
                gene_id = result['gene_id']

                if result['status'] == 'saved':
                    saved_count += 1
                elif result['status'] in ('no_segments', 'no_snps'):
                    no_segments_count += 1
                    filtered_genes.append({'gene_id': gene_id, 'reason': result.get('reason', result['status'])})
                else:
                    failed_count += 1
                    filtered_genes.append({'gene_id': gene_id, 'reason': result.get('reason', result['status'])})

                completed = i + 1
                if completed % 10 == 0 or completed == len(args_list):
                    print(f"Node {args.node_id}: Completed {completed}/{len(args_list)} genes")
    else:
        for i, (gene_args, _) in enumerate(args_list):
            gene_id = gene_args[0]
            try:
                result = process_gene_block_pretrain_global(gene_args)

                if result['status'] == 'success':
                    if save_gene_result(result, args):
                        saved_count += 1
                    else:
                        failed_count += 1
                        filtered_genes.append({'gene_id': gene_id, 'reason': 'failed to save'})
                elif result['status'] in ('no_segments', 'no_snps'):
                    no_segments_count += 1
                    filtered_genes.append({'gene_id': gene_id, 'reason': result['reason']})
                else:
                    failed_count += 1
                    filtered_genes.append({'gene_id': gene_id, 'reason': result.get('reason', result['status'])})

                completed = saved_count + failed_count + no_segments_count
                if completed % 10 == 0:
                    print(f"Node {args.node_id}: Completed {completed}/{len(args_list)} genes")

            except Exception as e:
                print(f"Node {args.node_id}: Gene {gene_id} exception: {e}")
                failed_count += 1
                filtered_genes.append({'gene_id': gene_id, 'reason': f'exception: {str(e)}'})

    print(f"\nNode {args.node_id}: Processing complete")
    print(f"  Total genes: {len(gene_subset)}")
    print(f"  Saved: {saved_count}")
    print(f"  No segments: {no_segments_count}")
    print(f"  Failed: {failed_count}")


def distribute_genes_to_nodes(gene_regions, total_nodes):
    """Distribute genes across nodes as evenly as possible"""
    genes_per_node = len(gene_regions) // total_nodes
    remainder = len(gene_regions) % total_nodes

    node_assignments = []
    start_idx = 0

    for node_id in range(total_nodes):
        node_size = genes_per_node + (1 if node_id < remainder else 0)
        end_idx = start_idx + node_size
        node_genes = gene_regions[start_idx:end_idx]
        node_assignments.append(node_genes)
        start_idx = end_idx

    return node_assignments


def rm_dup(snps, snpsIndex):
    """
    Remove duplicate segments based on SNP genotype values.

    Args:
        snps: Array of SNP genotypes [num_segments, seg_len]
        snpsIndex: Array of indices [num_segments, seg_len, 2]

    Returns:
        unique_snps, unique_snpsIndex: Deduplicated arrays
    """
    unique_ids = np.array([''.join(map(str, row)) for row in snps])
    _, unique_indices = np.unique(unique_ids, return_index=True)

    total_segments = len(snps)
    unique_segments = len(unique_indices)
    duplicate_segments = total_segments - unique_segments

    print(f"  Deduplication: {total_segments:,} -> {unique_segments:,} segments ({duplicate_segments:,} duplicates removed)")

    unique_indices = sorted(unique_indices)
    unique_snps = snps[unique_indices]
    unique_snpsIndex = snpsIndex[unique_indices]

    return unique_snps, unique_snpsIndex


def create_chunk_files(output_dir, split, genotype_ds, chr_num, race, seg_len, overlap,
                       chunk_size=None, apply_dedup=False, shuffle=True, seed=42):
    """
    Create merged HDF5 file from per-gene files with deduplication and shuffle.
    """
    import glob

    pattern = f"{output_dir}/{split}/{genotype_ds}_chr{chr_num}_{race}_seg{seg_len}_overlap{overlap}_{split}_*.hdf5"
    gene_files = sorted(glob.glob(pattern))

    # Exclude already-merged files
    gene_files = [f for f in gene_files if '_all.hdf5' not in f and '_chunk' not in f]

    if not gene_files:
        print(f"No gene files found for {split} split matching: {pattern}")
        return

    print(f"\nCreating chunk files for {split} from {len(gene_files)} gene files...")
    print(f"  Chunk size: {chunk_size if chunk_size else 'Single file'}")
    print(f"  Deduplication: {'Enabled' if apply_dedup else 'Disabled'}")

    start_time = time.time()

    all_snps = []
    all_snps_index = []
    total_rows = 0

    print("\nLoading per-gene files...")
    for gene_idx, gene_file in enumerate(gene_files):
        with h5py.File(gene_file, 'r') as f:
            snps = f['snps'][:]
            snps_index = f['snpsIndex'][:]

            all_snps.append(snps)
            all_snps_index.append(snps_index)
            total_rows += len(snps)

        if (gene_idx + 1) % 50 == 0:
            print(f"  Loaded {gene_idx + 1}/{len(gene_files)} files... Total segments: {total_rows:,}")

    all_snps = np.vstack(all_snps)
    all_snps_index = np.vstack(all_snps_index)

    print(f"\nTotal segments loaded: {len(all_snps):,}")

    if apply_dedup:
        print("\nApplying deduplication...")
        all_snps, all_snps_index = rm_dup(all_snps, all_snps_index)

    if shuffle:
        print(f"\nShuffling {len(all_snps):,} segments (seed={seed})...")
        rng = np.random.RandomState(seed)
        shuffle_idx = rng.permutation(len(all_snps))
        all_snps = all_snps[shuffle_idx]
        all_snps_index = all_snps_index[shuffle_idx]

    if chunk_size and chunk_size > 0:
        num_chunks = (len(all_snps) + chunk_size - 1) // chunk_size
        print(f"\nCreating {num_chunks} chunk files...")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_snps))

            chunk_snps = all_snps[start_idx:end_idx]
            chunk_snps_index = all_snps_index[start_idx:end_idx]

            output_file = f"{output_dir}/{split}/{genotype_ds}_chr{chr_num}_{race}_seg{seg_len}_overlap{overlap}_{split}_chunk{chunk_idx}.hdf5"

            with h5py.File(output_file, 'w') as f:
                f.create_dataset('snps', data=chunk_snps, dtype='i1', compression='gzip', chunks=True)
                f.create_dataset('snpsIndex', data=chunk_snps_index, dtype='i4', compression='gzip', chunks=True)

                f.attrs['chunk_index'] = chunk_idx
                f.attrs['total_chunks'] = num_chunks
                f.attrs['chunk_size'] = len(chunk_snps)
                f.attrs['deduplication_applied'] = apply_dedup
                f.attrs['shuffled'] = shuffle
                f.attrs['split'] = split

            print(f"  Chunk {chunk_idx}/{num_chunks-1}: {len(chunk_snps):,} segments")
    else:
        output_file = f"{output_dir}/{split}/{genotype_ds}_chr{chr_num}_{race}_seg{seg_len}_overlap{overlap}_{split}_all.hdf5"

        print(f"\nSaving {len(all_snps):,} segments to:")
        print(f"  {output_file}")

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('snps', data=all_snps, dtype='i1', compression='gzip', chunks=True)
            f.create_dataset('snpsIndex', data=all_snps_index, dtype='i4', compression='gzip', chunks=True)

            f.attrs['num_genes'] = len(gene_files)
            f.attrs['total_segments'] = len(all_snps)
            f.attrs['deduplication_applied'] = apply_dedup
            f.attrs['shuffled'] = shuffle
            f.attrs['shuffle_seed'] = seed if shuffle else -1
            f.attrs['split'] = split

    elapsed = time.time() - start_time
    print(f"\n{split.upper()} split completed in {elapsed:.1f}s")


def main():
    global model_input_width, token_span, overlap_size, stride, flank_size
    global global_snp_positions, global_genotype_matrix, global_sample_ids, global_segments_by_index

    parser = argparse.ArgumentParser(description='Pretrain Data Preparation with Global SNP-Index Segmentation')

    # Dataset configuration
    parser.add_argument('--dataset_config', type=str, default=None,
                       help='Path to dataset YAML config file')

    # Core arguments
    parser.add_argument('--genotype_ds', default='1KGP', help='Genotype dataset name')
    parser.add_argument('--gene_ds', default='GEUVADIS', help='Gene dataset name')
    parser.add_argument('--chr', type=int, default=22, help='Chromosome number')
    parser.add_argument('--race', default='ALL', help='Population code')
    parser.add_argument('--gene_pop', default=None, help='Population for gene regions lookup')
    parser.add_argument('--split', default='train', help='train/val/test')
    parser.add_argument('--node_id', type=int, help='Node ID (0-based)')
    parser.add_argument('--total_nodes', type=int, help='Total number of nodes')
    parser.add_argument('--gene_exp_path', default='<PATH_TO_GENE_REGIONS>',
                       help='Path to gene expression files for defining gene regions')
    parser.add_argument('--pretrain_vcf', help='Path to pretrain VCF file')
    parser.add_argument('--output_dir', default='./res_pt', help='Output directory')

    # Sample ID mapping options
    parser.add_argument('--sample_mapping_file', type=str, default=None)
    parser.add_argument('--expr_id_col', type=str, default='Subject_ID')
    parser.add_argument('--geno_id_col', type=str, default='LOS_ID')
    parser.add_argument('--geno_id_prefix', type=str, default='')
    parser.add_argument('--hla_suffix', type=str, default='')

    # Model parameters
    parser.add_argument('--model_input_width', type=int, default=258,
                       help='Model input width (default: 258)')
    parser.add_argument('--overlap_size', type=int, default=8,
                       help='Overlap size between segments (default: 8)')
    parser.add_argument('--flank_size', type=int, default=100000,
                       help='Flank size around genes in bp (default: 100000)')

    # Filtering options
    parser.add_argument('--min_snps', type=int, default=32,
                       help='Minimum non-padding SNPs per segment')
    parser.add_argument('--overlap_threshold', type=float, default=0.33,
                       help='Minimum ratio of SNPs in gene region')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers')

    # Chunk file creation mode
    parser.add_argument('--create_chunks_only', action='store_true',
                       help='Only create merged file from existing per-gene files')
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--apply_dedup', action='store_true')
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--shuffle_seed', type=int, default=42)

    args = parser.parse_args()

    # Validate required arguments
    if not args.create_chunks_only:
        if args.node_id is None:
            parser.error("--node_id is required for gene processing mode")
        if args.total_nodes is None:
            parser.error("--total_nodes is required for gene processing mode")
        if args.pretrain_vcf is None:
            parser.error("--pretrain_vcf is required for gene processing mode")

    # Set global parameters
    model_input_width = args.model_input_width
    token_span = model_input_width - 2
    overlap_size = args.overlap_size
    stride = token_span - overlap_size
    flank_size = args.flank_size

    # CHUNK CREATION MODE
    if args.create_chunks_only:
        print("\nCHUNK FILE CREATION MODE")
        print(f"  Dataset: {args.genotype_ds}")
        print(f"  Chromosome: {args.chr}")
        print(f"  Population: {args.race}")

        do_shuffle = not args.no_shuffle

        for split in ['train', 'val', 'test']:
            import glob
            split_pattern = f"{args.output_dir}/{split}/{args.genotype_ds}_chr{args.chr}_{args.race}_seg{token_span + 2}_overlap{overlap_size}_{split}_*.hdf5"
            split_files = [f for f in sorted(glob.glob(split_pattern))
                          if '_all.hdf5' not in f and '_chunk' not in f]

            if split_files:
                print(f"\nProcessing {split.upper()} split ({len(split_files)} gene files)")
                create_chunk_files(
                    output_dir=args.output_dir,
                    split=split,
                    genotype_ds=args.genotype_ds,
                    chr_num=args.chr,
                    race=args.race,
                    seg_len=token_span + 2,
                    overlap=overlap_size,
                    chunk_size=args.chunk_size,
                    apply_dedup=args.apply_dedup,
                    shuffle=do_shuffle,
                    seed=args.shuffle_seed
                )
            else:
                print(f"\nNo {split} files found")

        print("\nMerged file creation completed!")
        return

    # Load dataset config if provided
    dataset_config = None
    sample_id_mapping = None

    if args.dataset_config:
        dataset_config = load_dataset_config(args.dataset_config)
        args.genotype_ds = dataset_config.get('genotype', {}).get('original_vcf_pattern', '').split('_chr')[0] or args.genotype_ds
        args.race = dataset_config.get('population', args.race)

        if dataset_config.get('sample_mapping', {}).get('required', False):
            mapping_cfg = dataset_config['sample_mapping']
            sample_id_mapping = load_sample_id_mapping(
                mapping_cfg['mapping_file'],
                mapping_cfg['expression_id_column'],
                mapping_cfg['genotype_id_column'],
                mapping_cfg.get('genotype_id_prefix', '')
            )
    elif args.sample_mapping_file:
        sample_id_mapping = load_sample_id_mapping(
            args.sample_mapping_file,
            args.expr_id_col,
            args.geno_id_col,
            args.geno_id_prefix
        )

    # Print configuration
    print(f"\nGlobal SNP-Index Segmentation Configuration")
    print(f"  Model input width: {model_input_width}")
    print(f"  Token span: {token_span}")
    print(f"  Overlap size: {overlap_size}")
    print(f"  Stride: {stride}")
    print(f"  Flank size: {flank_size}")
    print(f"  Min SNPs per segment: {args.min_snps}")
    print(f"  Overlap threshold: {args.overlap_threshold}")

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Load gene regions
    # Path: {base}/{pop}/{ds}_chr{chr}_{pop}_{split}_gene_exp_peer_adjusted.txt
    gene_pop = args.gene_pop if args.gene_pop else args.race
    gene_path = f"{args.gene_exp_path}/{gene_pop}/{args.gene_ds}_chr{args.chr}_{gene_pop}_{args.split}_gene_exp_peer_adjusted{args.hla_suffix}.txt"
    print(f"Loading gene regions from: {gene_path}")
    gene_regions = load_gene_regions(gene_path)

    # Load all chromosome SNPs
    global_snp_positions, global_genotype_matrix, global_sample_ids = load_all_chromosome_snps(
        args.pretrain_vcf,
        sample_id_mapping=sample_id_mapping
    )

    # Compute global segment boundaries
    global_segments_by_index = compute_global_segments(
        num_snps=len(global_snp_positions),
        token_span=token_span,
        overlap_size=overlap_size
    )

    print(f"\nGlobal data loaded:")
    print(f"  SNPs: {len(global_snp_positions):,}")
    print(f"  Samples: {len(global_sample_ids)}")
    print(f"  Global segments: {len(global_segments_by_index):,}")
    print(f"  Memory usage: ~{(global_genotype_matrix.nbytes / 1024**2):.1f} MB\n")

    # Distribute genes across nodes
    node_assignments = distribute_genes_to_nodes(gene_regions, args.total_nodes)
    my_genes = node_assignments[args.node_id]

    print(f"Node {args.node_id}: Assigned {len(my_genes)} genes out of {len(gene_regions)} total")

    # Process genes
    process_gene_subset_pretrain_global(my_genes, args)

    print(f"\nNode {args.node_id}: Completed in {time.time() - start_time:.2f} seconds")
    print(f"\nNOTE: After ALL nodes complete, run with --create_chunks_only to create merged files.")


if __name__ == "__main__":
    main()
