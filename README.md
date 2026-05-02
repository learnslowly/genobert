[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://arxiv.org/abs/2604.00058) &nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

# GenoBERT

**GenoBERT: A Language Model for Accurate Genotype Imputation**

GenoBERT is a BERT-based deep learning model that treats genotype sequences as a language modeling problem. By leveraging masked language modeling (MLM) on genotype data, GenoBERT learns to capture linkage disequilibrium (LD) patterns and impute missing genotypes with high accuracy.

## Model Overview

GenoBERT adapts the ALBERT architecture for genomic sequences, treating each SNP genotype as a token in a sequence. The model learns bidirectional representations of genotype patterns, enabling accurate imputation of masked or missing genotypes.

### Key Innovations

- **Genomic Position Bias**: Incorporates physical genomic distances into attention scores to better model LD decay
- **Flexible Attention**: Supports switching between standard multi-head and sparse attention mechanisms
- **Genotype Segmentation**: Processes chromosomes using overlapping windows with consistent boundaries across samples
- **Class-Balanced Training**: Uses focal loss to handle the imbalance between common and rare alleles

### Architecture

| Component | Description |
|-----------|-------------|
| Encoder | ALBERT-based transformer with weight sharing |
| Position Encoding | Rotary Position Embeddings (RoPE) with optional Relative Genomic Position Bias (RGPB) |
| Attention | Multi-head attention with optional sparse attention |
| Feed-Forward | CNN-based bottleneck or GeGLU |
| Normalization | LayerNorm |

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/learnslowly/genobert.git
cd genobert

# Install dependencies
pip install -r requirements.txt
```

### SLURM Configuration

For running on HPC clusters, configure your SLURM settings:

```bash
# Copy the credentials template
cp job/credentials.sh.template job/credentials.sh

# Edit with your cluster settings
nano job/credentials.sh
```

The `credentials.sh` file contains:
- `SLURM_ACCOUNT`: Your SLURM account name
- `SLURM_PARTITION_CPU`: CPU partition for data preparation
- `SLURM_PARTITION_GPU`: GPU partition for training
- `SLURM_EMAIL`: Email for job notifications (optional)
- `CONDA_PATH`: Path to your miniconda/anaconda **root directory** (e.g., `/work/user/miniconda3`)
- `CONDA_ENV`: Name of conda environment to activate

This file is gitignored to keep your credentials private.

**Important**: All batch jobs should be submitted from the **project root directory**:
```bash
cd /path/to/GenoBERT
sbatch --partition=gpu2 --nodes=8 --ntasks=16 job/03_pretrain.batch configs/your_config.yaml
```

**Note on SLURM options**: Due to SLURM's parsing behavior, `#SBATCH` directives placed after shell commands in batch scripts are ignored. You have two options:

1. **Command-line approach** (recommended): Pass SLURM options when submitting:
   ```bash
   sbatch --partition=workq --mem=128G job/02_merge_genes.batch
   ```
   - Pros: Flexible, easy to adjust per-run
   - Cons: Longer command lines

2. **Hardcode in script**: Place all `#SBATCH` directives at the very top of the batch file, before any shell commands (including `source` for credentials):
   ```bash
   #!/bin/bash
   #SBATCH --partition=workq
   #SBATCH --mem=128G
   # ... other SBATCH directives ...

   # Shell commands start here
   source job/credentials.sh
   ```
   - Pros: Simpler submit commands
   - Cons: Can't use variables from credentials.sh in SBATCH directives, less flexible

## Genotype Encoding

GenoBERT uses integer encoding for diploid genotypes:

| Token | Meaning |
|-------|---------|
| 0 | MASK (masked/missing) |
| 1 | 0\|0 (homozygous reference) |
| 2 | 0\|1 (heterozygous) |
| 3 | 1\|0 (heterozygous) |
| 4 | 1\|1 (homozygous alternate) |
| 5 | CLS (sequence start) |
| 6 | SEP (sequence end) |
| 7 | PAD (padding) |

## Tutorial: Training GenoBERT

This tutorial demonstrates how to train GenoBERT on the 1000 Genomes Project (1KGP) dataset.

### Prerequisites

- 1KGP VCF files located at `../data/1KGP/`
- Gene expression data (for defining gene regions) located at `../data/GEUVADIS/` or similar

### Data Preparation Pipeline

The data preparation follows a two-step pipeline:

| Step | Batch Script | Python Script | Description |
|------|--------------|---------------|-------------|
| 1 | `01_data_prep_array.batch` | `pretrain_data_prep.py` | Creates per-gene HDF5 files from VCF |
| 2 | `02_merge_genes.batch` | `merge_genes.py` | Merges per-gene files with deduplication |
| 3 | `03_pretrain.batch` | `pretrain.py` | Model training |

### Step 1: Create Per-Gene Files

#### Option A: Single-Node Processing

```bash
# Prepare training data for chromosome 22, EUR population
python pretrain_data_prep.py \
    --genotype_ds 1KGP \
    --gene_ds GEUVADIS \
    --chr 22 \
    --race EUR \
    --split train \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf ../data/1KGP/split/1KGP_chr22_EUR_train.vcf.gz \
    --gene_exp_path ../data/GEUVADIS/split \
    --output_dir ./res_pt/1KGP \
    --model_input_width 258 \
    --overlap_size 8

# Repeat for validation and test split
python pretrain_data_prep.py \
    --genotype_ds 1KGP \
    --gene_ds GEUVADIS \
    --chr 22 \
    --race EUR \
    --split val \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf ../data/1KGP/split/1KGP_chr22_EUR_val.vcf.gz \
    --gene_exp_path ../data/GEUVADIS/split \
    --output_dir ./res_pt/1KGP

python pretrain_data_prep.py \
    --genotype_ds 1KGP \
    --gene_ds GEUVADIS \
    --chr 22 \
    --race EUR \
    --split test \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf ../data/1KGP/split/1KGP_chr22_EUR_val.vcf.gz \
    --gene_exp_path ../data/GEUVADIS/split \
    --output_dir ./res_pt/1KGP
```

#### Option B: SLURM Array Job (Recommended for Large Datasets)

You need to manually set the split within job/01\_data\_prep\_array.batch
```bash
# Edit job/01_data_prep_array.batch with your paths
# Then submit from the project root directory:
sbatch --array=0-199%50 --cpus-per-task=4 --mem=64G job/01_data_prep_array.batch
```

### Step 2: Merge Per-Gene Files

After per-gene files are created, merge them into a single file:

#### Merge ALL Genes

```bash
python merge_genes.py \
    --input_dir ./res_pt/1KGP/train \
    --prefix 1KGP_chr22_EUR_seg258_overlap8_train \
    --apply_dedup

python merge_genes.py \
    --input_dir ./res_pt/1KGP/val \
    --prefix 1KGP_chr22_EUR_seg258_overlap8_val \
    --apply_dedup
```

This creates `{prefix}_all.hdf5` files in the respective directories (e.g., `./res_pt/1KGP/train/`).

#### Merge with Gene List Filter (Optional)

To train on a subset of genes, create a gene list file and use it during merge:

```bash
# Create a gene list file (one gene ID per line)
cat > configs/gene_lists/my_genes.txt << EOF
ENSG00000160801
ENSG00000159692
ENSG00000109320
EOF

# Merge only genes in the list
python merge_genes.py \
    --input_dir ./res_pt/1KGP/train \
    --prefix 1KGP_chr22_EUR_seg258_overlap8_train \
    --genes_list_file configs/gene_lists/my_genes.txt \
    --apply_dedup
```

This creates `{prefix}_{genes_list_name}.hdf5` (e.g., `*_my_genes.hdf5`).

#### Using SLURM

```bash
# Edit job/02_merge_genes.batch with your settings
# Optionally set GENES_LIST_FILE for filtered merge
sbatch --partition=workq --mem=128G job/02_merge_genes.batch
```

### Step 3: Configure Training

Create a configuration file for your experiment:

```bash
cp configs/example_pretrain.yaml configs/1KGP_chr22_EUR.yaml
```

Edit key parameters:

```yaml
# Data - IMPORTANT: These must match your data prep parameters!
runId: 1KGP_EUR_chr22_v1
dataset: 1KGP           # Dataset name (also used for data path: ./res_pt/{dataset}/train/)
chromosome: 22
population: EUR         # Must match RACE used in data prep (01_data_prep_array.batch)
segLen: 258             # Must match --model_input_width in data prep
overlap: 8              # Must match --overlap_size in data prep
resPtDir: ./res_pt

# Optional: Use gene list filtered data
# genesListFile: configs/gene_lists/my_genes.txt

# Model
embDim: 512
numHeads: 4
numLayers: 6
enableBias: True

# Training
batchSize: 128
learningRate: 0.0008
totalEpochs: 100
maskProb: 0.5
lossType: focalLoss
```

**Data Path Resolution**: The training script looks for data files matching the pattern:
- `{resPtDir}/{dataset}/train/{dataset}_chr{chromosome}_{population}_seg{segLen}_overlap{overlap}_train_all.hdf5`
- Fallback: `{resPtDir}/train/...` (legacy path without dataset subdirectory)

### Step 4: Train the Model

#### Single GPU Training

```bash
python pretrain.py --configFile configs/1KGP_chr22_EUR.yaml
```

#### Multi-GPU Training (SLURM)

```bash
# Submit from the project root directory:
sbatch --partition=gpu2 --nodes=8 --ntasks=16 job/03_pretrain.batch configs/1KGP_chr22_EUR.yaml
```

#### Multi-GPU Training (torchrun)

```bash
torchrun --nproc_per_node=4 pretrain.py --configFile configs/1KGP_chr22_EUR.yaml
```

### Step 5: Monitor Training

Checkpoints are saved to `checkpoints_pt/{dataset}_{population}_chr{chr}/`.

Training metrics logged each epoch:
- `train_loss`: Training loss
- `train_accuracy`: Token prediction accuracy
- `val_loss`: Validation loss
- `val_accuracy`: Validation accuracy

Enable Weights & Biases logging:
```yaml
useWandB: True
wandbProject: GenoBERT
WandBKey: ""  # Optional - leave empty if using ~/.netrc for authentication
```

## Step 5: Test the Model

After training, evaluate the model on test data.

### Prepare Test Data

First, prepare test split data using the same pipeline as training:

```bash
# Prepare test data (same as Step 1, but with --split test)
python pretrain_data_prep.py \
    --genotype_ds 1KGP \
    --gene_ds GEUVADIS \
    --chr 22 \
    --race EUR \
    --gene_pop EUR \
    --split test \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf ../data/1KGP/split/1KGP_chr22_EUR_test.vcf.gz \
    --gene_exp_path ../data/GEUVADIS/split \
    --output_dir ./res_pt/1KGP

# Merge test files (same as Step 2, but for test split)
python merge_genes.py \
    --input_dir ./res_pt/1KGP/test \
    --prefix 1KGP_chr22_EUR_seg258_overlap8_test \
    --apply_dedup
```

### Run Testing

#### Single GPU

```bash
python test_pretrain.py \
    --configFile configs/1KGP_chr22_EUR.yaml \
    --checkpoint checkpoints_pt/1KGP_EUR_chr22/checkpoint_epoch_100.pth \
    --maskProb 0.05 0.15 0.5
```

#### Multi-GPU (SLURM)

```bash
sbatch --partition=gpu2 --nodes=1 --ntasks=2 \
    job/04_test_pretrain.batch \
    configs/1KGP_chr22_EUR.yaml \
    checkpoints_pt/1KGP_EUR_chr22/checkpoint_epoch_100.pth
```

#### Test Output

The test script outputs:
- Loss and accuracy at each mask probability
- Per-class accuracy (0|0, 0|1, 1|0, 1|1)
- Results saved to `test_results_{runId}.json`

```
============================================================
Summary:
Mask %     Loss       All Acc      Masked Acc
----------------------------------------------
5          0.1234     0.9512       0.8234
15         0.2345     0.9234       0.7856
50         0.4567     0.8756       0.6543

Per-class accuracy (last mask prob):
  0|0: 0.9234
  0|1: 0.7856
  1|0: 0.7912
  1|1: 0.8123
============================================================
```

## Project Structure

```
GenoBERT/
|-- model/
|   |-- genobert.py            # Model architecture
|-- data/
|   |-- dataset.py             # Dataset classes
|   |-- utils.py               # Utilities and loss functions
|-- config/
|   |-- modelconfig.py         # Configuration dataclass
|-- configs/
|   |-- example_pretrain.yaml
|   |-- gene_lists/            # Gene list files for filtered training
|-- job/
|   |-- 01_data_prep_array.batch  # Step 1: SLURM array job
|   |-- 02_merge_genes.batch      # Step 2: Merge job
|   |-- 03_pretrain.batch         # Step 3: Training job
|   |-- 04_test_pretrain.batch    # Step 5: Testing job
|   |-- credentials.sh.template
|-- res_pt/                    # Preprocessed data (created by data prep)
|   |-- 1KGP/                  # Per-dataset subdirectory
|   |   |-- train/             # Training split
|   |   |-- val/               # Validation split
|   |   |-- test/              # Test split
|-- pretrain_data_prep.py      # Step 1: Create per-gene HDF5 files
|-- merge_genes.py             # Step 2: Merge per-gene files
|-- pretrain.py                # Step 3: Training script
|-- test_pretrain.py           # Step 5: Testing script
|-- README.md
|-- requirements.txt
```

## Training on Custom Datasets

### LOS Dataset Example

For the Louisiana Osteoporosis Study (LOS) dataset:

```bash
# Step 1: Data preparation
python pretrain_data_prep.py \
    --genotype_ds LOS \
    --gene_ds LOS_mRNA \
    --chr 22 \
    --race AA \
    --split train \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf ../data/LOS/split/LOS_chr22_AA_train.vcf.gz \
    --gene_exp_path ../data/LOS_mRNA/split \
    --output_dir ./res_pt/LOS

# Step 2: Merge
python merge_genes.py \
    --input_dir ./res_pt/LOS/train \
    --prefix LOS_chr22_AA_seg258_overlap8_train \
    --apply_dedup
```

### Configuration Tips

- **Batch Size**: Start with 64-128, increase if GPU memory allows
- **Learning Rate**: 0.0008 works well for most cases
- **Mask Probability**: 0.15-0.5, higher values for denser LD regions
- **Enable Bias**: Set to `True` to incorporate genomic distances
- **Gene List**: Use `genesListFile` to train on specific gene regions

## Citation

```bibtex
@article{huang2026genobert,
  title        = {GenoBERT: A Language Model for Accurate Genotype Imputation},
  author       = {Huang, L. and others},
  year         = {2026},
  eprint       = {2604.00058},
  archivePrefix= {arXiv},
  primaryClass = {q-bio.GN},
  doi          = {10.48550/arXiv.2604.00058}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
