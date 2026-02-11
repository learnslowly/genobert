#!/usr/bin/env python3
"""
Merge Per-Gene HDF5 Files with Optional Gene Filtering

This script merges per-gene HDF5 files created by pretrain_data_prep.py into
a single merged file. Optionally filters to include only genes from a gene list.

Workflow:
    1. Run pretrain_data_prep.py (via 01_data_prep_array.batch) to create per-gene HDF5 files
    2. Run this script (via 02_merge_genes.batch) to merge with optional gene filtering
    3. Run pretrain.py (via 03_pretrain.batch) with genesListFile config to use the merged file

Usage:
    # Merge ALL per-gene files (no filtering)
    python merge_genes.py \
        --input_dir ./res_pt/train \
        --output_dir ./res_pt/train \
        --prefix 1KGP_chr22_EUR_seg258_overlap8_train \
        --apply_dedup \
        --shuffle

    # Merge only genes from a gene list file
    python merge_genes.py \
        --input_dir ./res_pt/train \
        --output_dir ./res_pt/train \
        --prefix 1KGP_chr22_EUR_seg258_overlap8_train \
        --genes_list_file configs/gene_lists/bone_genes.txt \
        --apply_dedup \
        --shuffle

Gene List File Format (one gene ID per line):
    ENSG00000160801
    ENSG00000159692
    ENSG00000109320
    ...

Output:
    Without gene list: {output_dir}/{prefix}_all.hdf5
    With gene list:    {output_dir}/{prefix}_{genes_list_name}.hdf5
"""

import argparse
import glob
import h5py
import hashlib
import numpy as np
import os
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-gene HDF5 files with optional gene filtering"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help="Directory containing per-gene HDF5 files"
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="Output directory (default: same as input_dir)"
    )
    parser.add_argument(
        '--prefix', type=str, required=True,
        help="File prefix pattern (e.g., 1KGP_chr22_EUR_seg258_overlap8_train)"
    )
    parser.add_argument(
        '--genes_list_file', type=str, default=None,
        help="Path to gene list file (one gene ID per line). If not provided, merges all genes."
    )
    parser.add_argument(
        '--apply_dedup', action='store_true',
        help="Apply deduplication based on genotype signature"
    )
    parser.add_argument(
        '--shuffle', action='store_true', default=True,
        help="Shuffle segments after merging (default: True)"
    )
    parser.add_argument(
        '--no_shuffle', action='store_true',
        help="Disable shuffling"
    )
    parser.add_argument(
        '--shuffle_seed', type=int, default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Print verbose output"
    )
    return parser.parse_args()


def load_genes_list(genes_list_file):
    """Load gene IDs from a text file (one per line)."""
    genes = set()
    with open(genes_list_file, 'r') as f:
        for line in f:
            gene_id = line.strip()
            if gene_id and not gene_id.startswith('#'):
                base_gene_id = gene_id.split('.')[0]
                genes.add(base_gene_id)
    return genes


def find_per_gene_files(input_dir, prefix):
    """Find all per-gene HDF5 files matching the prefix."""
    pattern = os.path.join(input_dir, f"{prefix}_*.hdf5")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if '_all.hdf5' not in f and '_chunk' not in f]
    return files


def extract_gene_id_from_filename(filepath):
    """Extract gene ID from per-gene HDF5 filename."""
    filename = os.path.basename(filepath)
    name = filename.rsplit('.hdf5', 1)[0]
    parts = name.split('_')
    if parts:
        gene_id = parts[-1]
        if gene_id.startswith('ENSG'):
            return gene_id.split('.')[0]
    return None


def hash_row(row):
    """Create a hash of a row for deduplication (memory efficient)."""
    return hashlib.md5(row.tobytes()).hexdigest()


def merge_files_memory_efficient(gene_files, output_path, apply_dedup=False, shuffle=True,
                                  shuffle_seed=42, verbose=False):
    """
    Merge multiple per-gene HDF5 files with memory-efficient deduplication.

    Uses hash-based deduplication to avoid creating large string arrays.
    """
    print(f"\nPhase 1: Scanning {len(gene_files)} files...")

    # First pass: count segments and collect hashes for dedup
    seen_hashes = set()
    file_segment_info = []  # List of (file_path, local_indices_to_keep)
    total_unique = 0

    for i, gene_file in enumerate(gene_files):
        with h5py.File(gene_file, 'r') as f:
            snps = f['snps'][:]
            n_segments = len(snps)

            if apply_dedup:
                indices_to_keep = []
                for idx in range(n_segments):
                    h = hash_row(snps[idx])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        indices_to_keep.append(idx)
                file_segment_info.append((gene_file, indices_to_keep))
                total_unique += len(indices_to_keep)
            else:
                file_segment_info.append((gene_file, list(range(n_segments))))
                total_unique += n_segments

        if verbose and (i + 1) % 100 == 0:
            print(f"  Scanned {i + 1}/{len(gene_files)} files...")

    print(f"  Total unique segments: {total_unique:,}")

    if apply_dedup:
        original_count = sum(len(list(range(len(h5py.File(f, 'r')['snps']))))
                            for f, _ in file_segment_info[:10])  # Rough estimate
        # Clear the set to free memory
        del seen_hashes

    # Get data shape from first file
    with h5py.File(gene_files[0], 'r') as f:
        seg_len = f['snps'].shape[1]
        index_shape = f['snpsIndex'].shape[1:]

    # Phase 2: Write to output
    print(f"\nPhase 2: Writing {total_unique:,} segments...")

    with h5py.File(output_path, 'w') as out_f:
        # Create datasets
        snps_ds = out_f.create_dataset(
            'snps',
            shape=(total_unique, seg_len),
            dtype='i1',
            chunks=(min(10000, total_unique), seg_len),
            compression='gzip',
            compression_opts=4
        )
        snps_index_ds = out_f.create_dataset(
            'snpsIndex',
            shape=(total_unique,) + index_shape,
            dtype='i4',
            chunks=(min(10000, total_unique),) + index_shape,
            compression='gzip',
            compression_opts=4
        )

        write_idx = 0
        for i, (gene_file, indices) in enumerate(file_segment_info):
            if not indices:
                continue

            with h5py.File(gene_file, 'r') as f:
                snps = f['snps'][:]
                snps_index = f['snpsIndex'][:]

                # Select only the indices to keep
                snps_subset = snps[indices]
                snps_index_subset = snps_index[indices]

                n = len(snps_subset)
                snps_ds[write_idx:write_idx + n] = snps_subset
                snps_index_ds[write_idx:write_idx + n] = snps_index_subset
                write_idx += n

            if verbose and (i + 1) % 100 == 0:
                print(f"  Written {i + 1}/{len(file_segment_info)} files...")

        out_f.attrs['num_samples'] = total_unique
        out_f.attrs['deduplication_applied'] = apply_dedup
        out_f.attrs['num_source_files'] = len(gene_files)

    print(f"  Written {write_idx:,} segments")

    # Phase 3: Shuffle if needed (in-place using chunked approach)
    if shuffle:
        print(f"\nPhase 3: Shuffling {total_unique:,} segments (seed={shuffle_seed})...")

        rng = np.random.RandomState(shuffle_seed)
        perm = rng.permutation(total_unique)

        # Read, permute, write in chunks to manage memory
        chunk_size = 50000

        with h5py.File(output_path, 'r') as f:
            all_snps = f['snps'][:]
            all_snps_index = f['snpsIndex'][:]

        all_snps = all_snps[perm]
        all_snps_index = all_snps_index[perm]

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('snps', data=all_snps, dtype='i1',
                            compression='gzip', compression_opts=4)
            f.create_dataset('snpsIndex', data=all_snps_index, dtype='i4',
                            compression='gzip', compression_opts=4)
            f.attrs['num_samples'] = total_unique
            f.attrs['deduplication_applied'] = apply_dedup
            f.attrs['shuffled'] = True
            f.attrs['shuffle_seed'] = shuffle_seed
            f.attrs['num_source_files'] = len(gene_files)

        print(f"  Shuffle complete")
    else:
        with h5py.File(output_path, 'a') as f:
            f.attrs['shuffled'] = False

    return total_unique


def main():
    args = parse_args()

    start_time = time.time()

    # Set output directory
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    # Find per-gene files
    gene_files = find_per_gene_files(args.input_dir, args.prefix)

    if not gene_files:
        print(f"Error: No per-gene files found matching pattern: {args.prefix}_*.hdf5")
        print(f"  Input directory: {args.input_dir}")
        return 1

    print(f"Found {len(gene_files)} per-gene files")

    # Filter by gene list if provided
    genes_list_name = None
    if args.genes_list_file:
        if not os.path.exists(args.genes_list_file):
            print(f"Error: Gene list file not found: {args.genes_list_file}")
            return 1

        genes_list_name = Path(args.genes_list_file).stem
        genes_set = load_genes_list(args.genes_list_file)

        print(f"\nFiltering by gene list: {args.genes_list_file}")
        print(f"  Gene list name: {genes_list_name}")
        print(f"  Genes in list: {len(genes_set)}")

        filtered_files = []
        matched_genes = set()

        for filepath in gene_files:
            gene_id = extract_gene_id_from_filename(filepath)
            if gene_id and gene_id in genes_set:
                filtered_files.append(filepath)
                matched_genes.add(gene_id)

        missing_genes = genes_set - matched_genes
        if missing_genes:
            print(f"  Warning: {len(missing_genes)} genes from list not found in files:")
            for gene in sorted(missing_genes)[:10]:
                print(f"    - {gene}")
            if len(missing_genes) > 10:
                print(f"    ... and {len(missing_genes) - 10} more")

        gene_files = filtered_files
        print(f"  Matched {len(gene_files)} files ({len(matched_genes)} genes)")

        if not gene_files:
            print("Error: No files matched the gene list")
            return 1

    # Determine output filename
    if genes_list_name:
        output_filename = f"{args.prefix}_{genes_list_name}.hdf5"
    else:
        output_filename = f"{args.prefix}_all.hdf5"

    output_path = os.path.join(output_dir, output_filename)

    # Merge files
    do_shuffle = not args.no_shuffle and args.shuffle
    final_count = merge_files_memory_efficient(
        gene_files,
        output_path,
        apply_dedup=args.apply_dedup,
        shuffle=do_shuffle,
        shuffle_seed=args.shuffle_seed,
        verbose=args.verbose
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Final segments: {final_count:,}")

    return 0


if __name__ == "__main__":
    exit(main())
