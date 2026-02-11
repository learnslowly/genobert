"""
Configuration class for GenoBERT pretraining.

Defines all hyperparameters for model architecture, training, and data loading.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import yaml
import os


@dataclass
class ModelConfig:
    """
    Configuration class for GenoBERT pretraining.

    Attributes are organized into sections:
    - Data Parameters: Dataset paths and preprocessing options
    - Vocabulary: Token IDs for genotype encoding
    - Model Architecture: Network structure hyperparameters
    - Training Parameters: Optimizer and schedule settings
    """

    # ==================== Data Parameters ====================
    runId: str  # Unique identifier for this run
    dataset: str  # Dataset name (e.g., '1KGP', 'GEUVADIS')
    chromosome: int  # Chromosome number
    population: str  # Population code (e.g., 'EUR', 'AFR', 'ALL')
    segLen: int  # Segment length (input sequence length)
    overlap: int  # Overlap between segments

    # Optional data parameters
    unique: bool = True  # Remove duplicate segments in preprocessing
    hlaOnly: bool = False  # HLA region only (chromosome 6)
    genotypeDataset: Optional[str] = None  # Genotype dataset name (if different from dataset)
    genesListFile: Optional[str] = None  # Path to gene list file for filtered pretraining

    # ==================== Vocabulary ====================
    vocabSize: int = 8
    padId: int = 7
    maskId: int = 0
    CLSId: int = 5
    SEPId: int = 6

    # ==================== Model Architecture ====================
    embDim: int = 512
    numHeads: int = 4
    numLayers: int = 6
    weightSharing: int = 0  # 0: None; 1: Attention; 2: FFN; 3: Both
    dropoutRate: float = 0.1

    # Positional bias
    enableBias: bool = False
    initBiasWeight: float = 0.0

    # Feed-forward network
    useCNNBottleneck: bool = True
    bottleneckShape: float = 4.0
    cnnKernelSize: int = 3
    cnnStride: int = 1

    # Sparse attention (BigBird-like)
    sparseAttention: bool = False  # Use sparse attention instead of full attention
    localWinSize: int = 64  # Local window size for sparse attention
    globalAttentionIds: List[int] = field(default_factory=lambda: [0])  # Global attention token indices (e.g., CLS)

    # ==================== Training Parameters ====================
    # Batch and data loading
    batchSize: int = 64
    numWorkers: int = 12
    persistentWorkers: bool = True
    seed: int = 0

    # Optimizer
    learningRate: float = 0.001
    weightDecay: float = 0.01
    adamwBeta1: float = 0.9
    adamwBeta2: float = 0.99
    adamwEps: float = 1e-8
    adamwWeightDecay: float = 0.001

    # Learning rate schedule
    scheduler: str = 'cosineAnn'  # 'cosineAnn' or 'stepLR'
    totalEpochs: int = 300
    warmupEpochs: int = 10
    cooldownEpochs: int = 10
    schedulerStepSize: int = 30  # For stepLR only
    schedulerGamma: float = 0.1  # For stepLR only

    # Masking and loss
    maskProb: float = 0.15
    sampling: str = 'normal'  # 'normal' or 'upsampling'
    upsamplingRatio: float = 0.8

    lossType: str = 'crossEntropy'
    loss: str = 'crossEntropy'
    focalGamma: float = 2.0
    focalAlpha: float = 0.25
    useAlphaWeighting: bool = True

    benchmarkAll: bool = True

    # Training options
    useMixedPrecision: bool = False
    mixedPrecisionTraining: bool = False

    # Directory paths
    resDir: str = "./res"
    resPtDir: str = "./res_pt"
    cacheDir: str = "./cache"

    # Checkpointing
    checkpointPrefix: str = ""
    checkpointDir: str = "./checkpoints"
    modelDir: str = "./checkpoints"
    saveCheckpointFreq: int = 1

    # Logging
    useWandB: bool = False
    wandbProject: str = 'GenoBERT'
    WandBProjName: str = 'GenoBERT'
    WandBKey: str = ""

    # Profiling
    enableProfiling: bool = False
    maxProfilingBatches: int = 10

    # ==================== Methods ====================
    def get_checkpoint_name_suffix(self) -> str:
        """Generate checkpoint filename suffix from config."""
        return f"{self.runId}_{self.dataset}_chr{self.chromosome}_{self.population}_seg{self.segLen}_overlap{self.overlap}"

    def get_gene_list(self) -> List[str]:
        """Get list of genes (for compatibility)."""
        return ["complete"]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """Create a ModelConfig instance from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Fill in default values for missing fields
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name not in config_dict:
                if field_def.default is not dataclasses.MISSING:
                    config_dict[field_name] = field_def.default
                elif field_def.default_factory is not dataclasses.MISSING:
                    config_dict[field_name] = field_def.default_factory()

        return cls(**config_dict)

    @classmethod
    def from_args(cls) -> 'ModelConfig':
        """Create a ModelConfig instance from command line arguments."""
        parser = argparse.ArgumentParser(description="GenoBERT Pretraining")

        # Required arguments
        parser.add_argument('--runId', type=str, required=True, help="Unique run identifier")
        parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
        parser.add_argument('--chromosome', type=int, required=True, help="Chromosome number")
        parser.add_argument('--population', type=str, required=True, help="Population group")
        parser.add_argument('--segLen', type=int, required=True, help="Segment length")
        parser.add_argument('--overlap', type=int, required=True, help="Segment overlap")

        # Model parameters
        parser.add_argument('--embDim', type=int, default=512)
        parser.add_argument('--numHeads', type=int, default=4)
        parser.add_argument('--numLayers', type=int, default=6)

        # Training parameters
        parser.add_argument('--batchSize', type=int, default=64)
        parser.add_argument('--learningRate', type=float, default=0.001)
        parser.add_argument('--totalEpochs', type=int, default=300)

        args = parser.parse_args()
        config_dict = vars(args)

        # Fill in defaults for unspecified fields
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name not in config_dict:
                if field_def.default is not dataclasses.MISSING:
                    config_dict[field_name] = field_def.default
                elif field_def.default_factory is not dataclasses.MISSING:
                    config_dict[field_name] = field_def.default_factory()

        return cls(**config_dict)

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Sync aliases
        if self.loss and not self.lossType:
            self.lossType = self.loss
        elif self.lossType and not self.loss:
            self.loss = self.lossType

        if self.mixedPrecisionTraining and not self.useMixedPrecision:
            self.useMixedPrecision = self.mixedPrecisionTraining
        elif self.useMixedPrecision and not self.mixedPrecisionTraining:
            self.mixedPrecisionTraining = self.useMixedPrecision

        if self.WandBProjName and not self.wandbProject:
            self.wandbProject = self.WandBProjName
        elif self.wandbProject and not self.WandBProjName:
            self.WandBProjName = self.wandbProject
