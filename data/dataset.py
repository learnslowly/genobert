"""
Dataset classes for GenoBERT pretraining.

Provides HDF5-based datasets for efficient loading of genotype sequences.
"""

from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
import os
import pickle


class SNPsDataset_HDF5(Dataset):
    """
    Dataset for loading pretraining data from a single HDF5 file.

    HDF5 structure:
    - snps: [num_samples, seq_len] int - encoded genotype tokens
    - snpsIndex: [num_samples, seq_len, 2] int - (sample_id, genomic_position)

    Args:
        hdf5_filename: Path to HDF5 file
        preload: If True, load all data into memory. If False, load on-demand.
    """
    def __init__(self, hdf5_filename: str, preload=True):
        self.hdf5_filename = hdf5_filename
        self.preload = preload
        self.data_file = None

        if self.preload:
            with h5py.File(self.hdf5_filename, 'r') as f:
                self.snps = torch.from_numpy(f['snps'][:]).long()
                self.snpsIndex = torch.from_numpy(f['snpsIndex'][:]).long()
            self.length = len(self.snps)
        else:
            with h5py.File(self.hdf5_filename, 'r') as f:
                self.length = len(f['snps'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.preload:
            return self.snps[idx], self.snpsIndex[idx]

        if self.data_file is None:
            self.data_file = h5py.File(self.hdf5_filename, 'r')

        snps = torch.from_numpy(self.data_file['snps'][idx]).long()
        snpsIndex = torch.from_numpy(self.data_file['snpsIndex'][idx]).long()
        return snps, snpsIndex

    def __del__(self):
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()

    def close(self):
        """Explicitly close file handle and free memory."""
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()
            self.data_file = None
        if hasattr(self, 'snps'):
            del self.snps
        if hasattr(self, 'snpsIndex'):
            del self.snpsIndex


class MultiGeneDataset_HDF5(Dataset):
    """
    Dataset for multi-gene pretraining with deduplication.

    Loads multiple gene HDF5 files and deduplicates segments by genomic position
    signature. Supports caching for faster resume.

    Args:
        hdf5_files: List of HDF5 file paths
        preload: If True, load all data into memory
        chunk_size: Number of files to load at once (for chunked mode)
        cache_file: Path to save/load deduplicated data
    """
    def __init__(self, hdf5_files, preload=True, chunk_size=10, cache_file=None):
        self.hdf5_files = sorted(hdf5_files) if isinstance(hdf5_files, list) else [hdf5_files]
        self.preload = preload
        self.chunk_size = chunk_size
        self.cache_file = cache_file

        if not self.hdf5_files:
            raise ValueError("No HDF5 files provided")

        self.gene_metadata = []

        if self.preload:
            if cache_file and os.path.exists(cache_file):
                self._load_from_cache(cache_file)
            else:
                self._load_all_data()
                if cache_file:
                    self._save_to_cache(cache_file)
        else:
            self._load_metadata()
            self.current_chunk_files = []
            self.current_chunk_data = {}

    def _load_all_data(self):
        """Load all gene data with deduplication by genomic position signature."""
        unique_snps = []
        unique_snps_index = []
        seen_position_sigs = {}

        for gene_idx, hdf5_file in enumerate(self.hdf5_files):
            with h5py.File(hdf5_file, 'r') as f:
                snps_np = f['snps'][:]
                snps_index_np = f['snpsIndex'][:]

                snps = torch.from_numpy(snps_np).long()
                snps_index = torch.from_numpy(snps_index_np).long()

                gene_id = f.attrs.get('gene_id', f'gene_{gene_idx}')

                gene_position_groups = {}
                for row_idx in range(len(snps)):
                    full_index = snps_index[row_idx]
                    full_sig = tuple(full_index.flatten().tolist())

                    if full_sig not in gene_position_groups:
                        gene_position_groups[full_sig] = []
                    gene_position_groups[full_sig].append(row_idx)

                gene_rows_added = 0
                gene_rows_skipped = 0

                for full_sig, row_indices in gene_position_groups.items():
                    row_idx = row_indices[0]

                    if full_sig not in seen_position_sigs:
                        seen_position_sigs[full_sig] = gene_idx

                        unique_snps.append(snps[row_idx])
                        unique_snps_index.append(snps_index[row_idx])
                        gene_rows_added += 1
                    else:
                        gene_rows_skipped += 1

                self.gene_metadata.append({
                    'gene_id': gene_id,
                    'file_path': hdf5_file,
                    'num_rows_original': len(snps),
                    'num_rows_added': gene_rows_added,
                    'num_rows_skipped': gene_rows_skipped,
                    'num_unique_segments': len(gene_position_groups)
                })

        if unique_snps:
            self.snps = torch.stack(unique_snps, dim=0)
            self.snpsIndex = torch.stack(unique_snps_index, dim=0)
            self.length = len(self.snps)
        else:
            raise ValueError("No segments loaded after deduplication")

    def _save_to_cache(self, cache_file):
        """Save deduplicated data to HDF5 file."""
        print(f"Saving deduplicated data to cache: {cache_file}")
        cache_dir = os.path.dirname(cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('snps', data=self.snps.numpy(), compression='gzip', compression_opts=4)
            f.create_dataset('snpsIndex', data=self.snpsIndex.numpy(), compression='gzip', compression_opts=4)

            metadata = {
                'gene_metadata': self.gene_metadata
            }
            metadata_bytes = pickle.dumps(metadata)
            f.create_dataset('metadata', data=np.frombuffer(metadata_bytes, dtype=np.uint8))
            f.attrs['num_samples'] = self.length

        print(f"  Saved {self.length} unique samples to {cache_file}")

    def _load_from_cache(self, cache_file):
        """Load deduplicated data from cache file."""
        print(f"Loading deduplicated data from cache: {cache_file}")

        with h5py.File(cache_file, 'r') as f:
            self.snps = torch.from_numpy(f['snps'][:]).long()
            self.snpsIndex = torch.from_numpy(f['snpsIndex'][:]).long()
            self.length = len(self.snps)

            metadata_bytes = bytes(f['metadata'][:])
            metadata = pickle.loads(metadata_bytes)
            self.gene_metadata = metadata['gene_metadata']

        print(f"  Loaded {self.length} unique samples from cache")

    def _load_metadata(self):
        """Load only metadata for chunked loading mode."""
        for gene_idx, hdf5_file in enumerate(self.hdf5_files):
            with h5py.File(hdf5_file, 'r') as f:
                gene_id = f.attrs.get('gene_id', f'gene_{gene_idx}')
                num_segments = len(f['snps'])

                self.gene_metadata.append({
                    'gene_id': gene_id,
                    'file_path': hdf5_file,
                    'num_segments': num_segments
                })

        self.length = sum(meta['num_segments'] for meta in self.gene_metadata)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.preload:
            return self.snps[idx], self.snpsIndex[idx], None
        else:
            cumsum = 0
            for gene_idx, meta in enumerate(self.gene_metadata):
                if idx < cumsum + meta['num_segments']:
                    local_idx = idx - cumsum

                    if gene_idx not in self.current_chunk_data:
                        with h5py.File(meta['file_path'], 'r') as f:
                            self.current_chunk_data[gene_idx] = {
                                'snps': torch.from_numpy(f['snps'][:]).long(),
                                'snpsIndex': torch.from_numpy(f['snpsIndex'][:]).long()
                            }

                    gene_data = self.current_chunk_data[gene_idx]
                    return gene_data['snps'][local_idx], gene_data['snpsIndex'][local_idx], None

                cumsum += meta['num_segments']

        raise IndexError(f"Index {idx} out of range")

    def close(self):
        """Clean up resources."""
        if self.preload:
            if hasattr(self, 'snps'):
                del self.snps
            if hasattr(self, 'snpsIndex'):
                del self.snpsIndex
        else:
            self.current_chunk_data = {}
