"""
GenoBERT: BERT-based model for SNP genotype prediction.

This module implements an ALBERT-based architecture for masked language modeling
on genotype sequences. Key components:
- AlbertEmbeddings: Token embeddings with LayerNorm
- MultiHeadAttentionVanilla: Standard multi-head attention with rotary position embeddings
- MultiHeadAttentionSparse: BigBird-like sparse attention with local window + global tokens
- CNNBottleneck: 1D CNN-based feed-forward network
- ALBERT: Main encoder with optional weight sharing
- GenoBERTMLM: Pretraining model with MLM head

Supports optional relative genomic position bias (RGPB) and switching between
standard multi-head and sparse attention mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
from rotary_embedding_torch import RotaryEmbedding
from typing import Optional


def print_model_summary(model):
    """Print summary of model parameters."""
    print("Model Summary:")
    total_params = 0

    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        print(f"{name:40} : Params: {param_count}")

    print(f"\nTotal Parameters: {total_params}\n")


def xavia_init(module):
    """
    Xavia initialization from "Path-norm optimization by reweighting" paper.
    Specifically designed for GeGLU networks.
    """
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.Linear):
        if hasattr(module, '_is_gate_proj'):
            nn.init.xavier_normal_(module.weight, gain=math.sqrt(1/3))
        else:
            nn.init.xavier_normal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class CNNBottleneck(nn.Module):
    """
    1D CNN-based feed-forward network for transformer layers.

    Uses convolution + pooling + upsampling instead of traditional FFN.
    """
    def __init__(self, emb_dim, config=None, kernel_size=7, stride=1, padding=3,
                 dropout_rate=0.1, bottleneck_factor=1.0):
        super(CNNBottleneck, self).__init__()

        if config is not None:
            kernel_size = getattr(config, 'cnnKernelSize', kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            stride = getattr(config, 'cnnStride', stride)
            padding = kernel_size // 2
            pool_kernel = getattr(config, 'poolKernelSize', 2)
            pool_stride = getattr(config, 'poolStride', 2)
            upsample_scale = getattr(config, 'upsampleScale', 2)
            dropout_rate = getattr(config, 'dropoutRate', dropout_rate)
            bottleneck_factor = getattr(config, 'bottleneckShape', bottleneck_factor)
        else:
            pool_kernel = 2
            pool_stride = 2
            upsample_scale = 2

        bottleneck_dim = int(emb_dim * bottleneck_factor)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=bottleneck_dim,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Dropout(dropout_rate)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=bottleneck_dim, out_channels=emb_dim,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=upsample_scale)
        )

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x


class GeGLU(nn.Module):
    """Gated GELU activation unit."""
    def __init__(self, input_dim, output_dim):
        super(GeGLU, self).__init__()
        self.gate_proj = nn.Linear(input_dim, output_dim)
        self.gate_proj._is_gate_proj = True
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.apply(xavia_init)

    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        value = self.value_proj(x)
        return gate * value


class WeightSharingStrategy(enum.Enum):
    """Weight sharing strategies for ALBERT."""
    NONE = 0
    EMBEDDING_ONLY = 1
    FFN_ONLY = 2
    BOTH = 3


class AlbertEmbeddings(nn.Module):
    """Token embeddings with LayerNorm and dropout."""
    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocabSize, config.embDim, padding_idx=config.padId)
        self.layerNorm = nn.LayerNorm(config.embDim)
        self.dropout = nn.Dropout(config.dropoutRate)
        self.apply(xavia_init)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttentionVanilla(nn.Module):
    """
    Multi-head attention with rotary position embeddings.

    Optionally adds genomic position bias to attention scores.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embDim
        self.numHeads = config.numHeads
        self.head_dim = config.embDim // config.numHeads
        assert self.head_dim * self.numHeads == self.embed_dim, "embed_dim must be divisible by numHeads"

        self.wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.wv = nn.Linear(self.embed_dim, self.embed_dim)

        if config.enableBias:
            self.pos_encoding_coeff = nn.Parameter(
                torch.full((self.numHeads, 1, 1), config.initBiasWeight, dtype=torch.float32),
                requires_grad=True
            )
        else:
            self.pos_encoding_coeff = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.rope_encoding = RotaryEmbedding(dim=self.head_dim)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.apply(xavia_init)

    def split_heads(self, x):
        """Split input into multiple heads."""
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.numHeads, self.head_dim).transpose(1, 2)

    def calculate_rpb(self, position_bias):
        """Calculate relative position bias matrix."""
        batch_size, seq_len = position_bias.shape
        pb1 = position_bias.unsqueeze(2).expand(-1, -1, seq_len)
        pb2 = position_bias.unsqueeze(1).expand(-1, seq_len, -1)
        return pb1 - pb2

    def forward(self, query, key, value, rel_pos_bias=None, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query, key, value: [batch_size, seq_len, embed_dim]
            rel_pos_bias: [batch_size, seq_len] - 1D position bias vector
            mask: [batch_size, seq_len] - padding mask
        """
        bsz, seq_len, _ = query.size()

        Q = self.split_heads(self.wq(query))
        K = self.split_heads(self.wk(key))
        V = self.split_heads(self.wv(value))

        Q = self.rope_encoding.rotate_queries_or_keys(Q)
        K = self.rope_encoding.rotate_queries_or_keys(K)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if rel_pos_bias is not None:
            rel_pos_matrix = self.calculate_rpb(rel_pos_bias)
            rel_pos_matrix = rel_pos_matrix.unsqueeze(1)
            attention_scores += self.pos_encoding_coeff * rel_pos_matrix

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        output = self.fc_out(attention_output)

        return output


class MultiHeadAttentionSparse(nn.Module):
    """
    BigBird-like sparse attention with local window and global tokens.

    Combines local sliding window attention with global attention for
    efficient processing of long sequences. Global tokens (e.g., CLS, SEP)
    attend to all positions while other tokens only attend within a local window.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embDim
        self.numHeads = config.numHeads
        self.head_dim = config.embDim // config.numHeads
        assert self.head_dim * self.numHeads == self.embed_dim, "embed_dim must be divisible by numHeads"

        self.wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.wv = nn.Linear(self.embed_dim, self.embed_dim)

        if config.enableBias:
            self.pos_encoding_coeff = nn.Parameter(
                torch.full((self.numHeads, 1, 1), config.initBiasWeight, dtype=torch.float32),
                requires_grad=True
            )
        else:
            self.pos_encoding_coeff = nn.Parameter(
                torch.zeros(self.numHeads, 1, 1, dtype=torch.float32),
                requires_grad=False
            )

        self.rope_encoding = RotaryEmbedding(dim=self.head_dim)

        # Sparse attention parameters
        self.local_window_size = getattr(config, 'localWinSize', 64)
        self.global_attention_indices = getattr(config, 'globalAttentionIds', [0])

        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.apply(xavia_init)

    def split_heads(self, x):
        """Split input into multiple heads."""
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.numHeads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, rel_pos_bias=None, mask=None):
        """
        Forward pass for sparse multi-head attention.

        Args:
            query, key, value: [batch_size, seq_len, embed_dim]
            rel_pos_bias: [batch_size, seq_len, seq_len] - relative position bias matrix
            mask: [batch_size, seq_len] - padding mask
        """
        bsz, seq_len, _ = query.size()
        device = query.device
        dtype = query.dtype

        Q = self.split_heads(self.wq(query))
        K = self.split_heads(self.wk(key))
        V = self.split_heads(self.wv(value))

        Q = self.rope_encoding.rotate_queries_or_keys(Q)
        K = self.rope_encoding.rotate_queries_or_keys(K)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add relative positional bias
        if rel_pos_bias is not None:
            rel_pos_bias = rel_pos_bias.to(dtype=dtype, device=device).unsqueeze(1)
            attention_scores += self.pos_encoding_coeff.unsqueeze(0) * rel_pos_bias

        # Build sparse attention mask: local window + global tokens
        row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
        col_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        distance = torch.abs(row_indices - col_indices)
        sparse_mask = distance <= self.local_window_size

        # Add global attention indices (e.g., CLS at position 0)
        if self.global_attention_indices is not None and len(self.global_attention_indices) > 0:
            global_idx = torch.tensor(self.global_attention_indices, device=device)
            global_idx = global_idx[(global_idx >= 0) & (global_idx < seq_len)]

            sparse_mask[:, global_idx] = True  # All tokens attend to global tokens
            sparse_mask[global_idx, :] = True  # Global tokens attend to all tokens

        # Expand for batch and heads
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)
        sparse_mask = sparse_mask.expand(bsz, -1, -1, -1)

        # Combine with padding mask
        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)
            sparse_mask = sparse_mask & padding_mask

        attention_scores = attention_scores.masked_fill(~sparse_mask, float('-inf'))

        # Numerical stability
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        output = self.fc_out(attention_output)

        return output


class ALBertLayer(nn.Module):
    """Single ALBERT encoder layer with attention and feed-forward."""
    def __init__(self, config, shared_attention=None, shared_ffn=None):
        super(ALBertLayer, self).__init__()

        if shared_attention is not None:
            self.attention = shared_attention
        else:
            use_sparse = getattr(config, 'sparseAttention', False)
            if use_sparse:
                self.attention = MultiHeadAttentionSparse(config)
            else:
                self.attention = MultiHeadAttentionVanilla(config)

        if shared_ffn is not None:
            self.feed_forward = shared_ffn
        else:
            use_cnn = getattr(config, 'useCNNBottleneck', True)
            bottleneck_factor = getattr(config, 'bottleneckShape', 4.0)

            if use_cnn:
                self.feed_forward = CNNBottleneck(
                    emb_dim=config.embDim,
                    config=config,
                    dropout_rate=config.dropoutRate,
                    bottleneck_factor=bottleneck_factor
                )
            else:
                ffn_dim = int(config.embDim * bottleneck_factor)
                self.feed_forward = nn.Sequential(
                    GeGLU(config.embDim, ffn_dim),
                    nn.Dropout(config.dropoutRate),
                    nn.Linear(ffn_dim, config.embDim)
                )
                self.feed_forward.apply(xavia_init)

        self.post_norm1 = nn.LayerNorm(config.embDim)
        self.post_norm2 = nn.LayerNorm(config.embDim)
        self.alpha = (2 * config.numLayers) ** 0.25

    def forward(self, x, batch_bias=None, batch_padding_mask=None):
        attention_output = self.attention(x, x, x, batch_bias, batch_padding_mask)
        x = self.post_norm1(self.alpha * x + attention_output)

        ff_output = self.feed_forward(x)
        x = self.post_norm2(self.alpha * x + ff_output)

        return x


class AlbertOutput:
    """Container for ALBERT output with optional hidden states."""
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class ALBERT(nn.Module):
    """
    ALBERT encoder with optional weight sharing.

    Weight sharing strategies:
    - 0 (NONE): No weight sharing
    - 1 (EMBEDDING_ONLY): Share attention weights across layers
    - 2 (FFN_ONLY): Share FFN weights across layers
    - 3 (BOTH): Share both attention and FFN weights
    """
    def __init__(self, config):
        super(ALBERT, self).__init__()
        self.embedding = AlbertEmbeddings(config)

        strategy = WeightSharingStrategy(config.weightSharing)
        shared_attention = None
        shared_ffn = None

        use_cnn = getattr(config, 'useCNNBottleneck', True)
        bottleneck_factor = getattr(config, 'bottleneckShape', 4.0)

        if strategy == WeightSharingStrategy.FFN_ONLY or strategy == WeightSharingStrategy.BOTH:
            if use_cnn:
                shared_ffn = CNNBottleneck(
                    emb_dim=config.embDim,
                    config=config,
                    dropout_rate=config.dropoutRate,
                    bottleneck_factor=bottleneck_factor
                )
            else:
                ffn_dim = int(config.embDim * bottleneck_factor)
                shared_ffn = nn.Sequential(
                    GeGLU(config.embDim, ffn_dim),
                    nn.Dropout(config.dropoutRate),
                    nn.Linear(ffn_dim, config.embDim)
                )
            shared_ffn.apply(xavia_init)

        if strategy == WeightSharingStrategy.EMBEDDING_ONLY or strategy == WeightSharingStrategy.BOTH:
            use_sparse = getattr(config, 'sparseAttention', False)
            if use_sparse:
                shared_attention = MultiHeadAttentionSparse(config)
            else:
                shared_attention = MultiHeadAttentionVanilla(config)

        self.encoder_layers = nn.ModuleList([
            ALBertLayer(config, shared_attention, shared_ffn) for _ in range(config.numLayers)
        ])

    def forward(self, input_ids, batch_bias=None, batch_padding_mask=None, output_hidden_states=False):
        embedding_output = self.embedding(input_ids)

        all_hidden_states = []
        if output_hidden_states:
            all_hidden_states.append(embedding_output)

        hidden_states = embedding_output

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, batch_bias, batch_padding_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        if output_hidden_states:
            return AlbertOutput(
                last_hidden_state=hidden_states,
                hidden_states=tuple(all_hidden_states)
            )
        else:
            return hidden_states


class GenoBERTMLM(nn.Module):
    """
    GenoBERT for Masked Language Modeling pretraining.

    Predicts masked genotype tokens in input sequences.

    Genotype vocabulary:
    - 0: MASK (masked token)
    - 1: 0|0 (homozygous reference)
    - 2: 0|1 (heterozygous)
    - 3: 1|0 (heterozygous)
    - 4: 1|1 (homozygous alternate)
    - 5: CLS (sequence start)
    - 6: SEP (sequence end)
    - 7: PAD (padding)
    """
    def __init__(self, config):
        super(GenoBERTMLM, self).__init__()
        self.albert = ALBERT(config)
        self.mlm_head = nn.Linear(config.embDim, config.vocabSize)
        self.apply(xavia_init)

    def forward(self, input_ids, batch_bias, batch_padding_mask):
        embeddings = self.albert(input_ids, batch_bias, batch_padding_mask)
        logits = self.mlm_head(embeddings)

        batch_size, seq_len, vocab_size = logits.size()

        # Never predict MASK token (index 0)
        logits[:, :, 0] = float('-inf')

        # Never predict PAD token (index 7)
        logits[:, :, 7] = float('-inf')

        # During inference, enforce special token positions
        if not self.training:
            # CLS only at first position
            positions_except_first = torch.ones(seq_len, dtype=torch.bool, device=logits.device)
            positions_except_first[0] = False
            logits[:, positions_except_first, 5] = float('-inf')

            # SEP only at last non-padding position
            if batch_padding_mask is not None:
                seq_lengths = batch_padding_mask.sum(dim=1).long() - 1
                for b in range(batch_size):
                    seq_len_b = seq_lengths[b].item()
                    sep_positions = torch.ones(seq_len, dtype=torch.bool, device=logits.device)
                    if 0 <= seq_len_b < seq_len:
                        sep_positions[seq_len_b] = False
                    logits[b, sep_positions, 6] = float('-inf')
            else:
                last_positions = torch.ones(batch_size, seq_len, dtype=torch.bool, device=logits.device)
                last_positions[:, -1] = False
                logits[last_positions.view(batch_size, seq_len, 1).expand(-1, -1, 1), 6] = float('-inf')

        return logits
