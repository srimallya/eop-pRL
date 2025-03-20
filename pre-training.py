## pretraining V2.1

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# ==========================================
# 1) Hyperparameters (modified for new dataset)
# ==========================================
hyperparams = {
    # Model Architecture
    'block_size': 1024,               # Sequence length for context
    'batch_size': 2,                  # Batch size
    'embed_dim': 1024,                # Transformer embedding dimension
    'n_heads': 16,                    # Number of attention heads
    'n_layers': 24,                   # Number of Transformer blocks
    'memory_n_layers': 8,             # Number of layers in the original MemoryModule
    'vocab_size': 256,                # Fixed vocabulary size for byte tokenization

    # Training Parameters
    'num_epochs': 100,                # Number of epochs
    'steps_per_epoch': 1000,          # Steps per epoch
    'eval_interval': 200,             # Steps between loss evaluations
    'eval_iters': 100,                # Iterations to average validation loss
    'accumulation_steps': 8,          # Number of steps to accumulate gradients over
    'validation_split': 0.1,          # Fraction of data to use for validation
    'sample_size': 1000000,           # Number of samples to use from dataset

    # Weighted Loss Parameters
    'use_weighted_loss': True,        # Whether to use weighted cross-entropy
    'alpha': 0.5,                     # Alpha parameter for (1/p_i)^alpha weighting
    'kl_lambda': 0.1,                 # Lambda for KL divergence regularization
    'min_freq': 1e-5,                 # Minimum frequency to avoid division by zero
    'max_weight': 10.0,               # Maximum weight cap to prevent instability

    # Generation Parameters
    'generate_num_tokens': 2048,      # Number of tokens to generate after each epoch
    'top_p': 0.8,                     # Top-p (nucleus) sampling parameter
    'start_prompt': "Explain why the statement 'I wore my lucky socks today, and I got an A on my test, so my socks must be lucky' is a logical fallacy.",

    # Special Tokens & Tags
    'thinking_tag': "<think>",        # Opening tag for thinking process
    'thinking_end_tag': "</think>",   # Closing tag for thinking process
    'answer_tag': "<answer>",         # Opening tag for final answer
    'answer_end_tag': "</answer>",    # Closing tag for final answer
    'bos_token': 254,                 # Beginning-of-sequence token (byte value)
    'eos_token': 255,                 # End-of-sequence token (byte value)

    # File Paths & Modes
    'checkpoint_path': "threshold_transformer_checkpoint.pt",  # Updated checkpoint name
    'dataset_path': "hf://datasets/applied-ai-018/pretraining_v1-omega_books/CC-MAIN-2013-20/train-*.parquet",
    'mode': 'pretrain',               # Force pretrain mode
    'continue_training': True,        # Whether to continue training from a checkpoint
    'system_prompt': """just think before answer."""
}

# ==========================================
# 1.1) Select device
# ==========================================
device = "mps" if torch.backends.mps.is_available() else \
         ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1.2) Data Loading and Preprocessing for Omega Books Dataset
# ==========================================
def load_omega_books_data_as_bytes():
    """
    Load Omega Books dataset and convert text to bytes for byte-level tokenization.
    Returns raw bytes and tensor versions of train and validation data.
    """
    print("Loading Omega Books dataset as bytes...")

    try:
        # Load data from Hugging Face using Dask for distributed processing
        ddf = dd.read_parquet(hyperparams['dataset_path'])

        # Basic data analysis
        print(f"Dataset partitions: {ddf.npartitions}")
        print(f"Column names: {ddf.columns.tolist()}")

        # Get a sample to understand the data structure
        sample = ddf.head(5)
        print("\nFirst 5 rows (sample):")
        print(sample)

        # Check for missing values in sample
        print("\nMissing values in sample:")
        print(sample.isnull().sum())

        # Try to identify content columns based on common names in text datasets
        columns = ddf.columns.tolist()
        content_cols = [col for col in columns if col.lower() in ['text', 'content', 'body', 'document']]

        if not content_cols:
            print("Could not identify text content columns, using first column")
            content_col = columns[0]
        else:
            content_col = content_cols[0]

        print(f"Using '{content_col}' as content column")

        # Process in chunks - use the sample_size parameter to control memory usage
        sample_size = hyperparams.get('sample_size', 500000)

        # Try with Dask first for distributed processing
        try:
            train_ddf = ddf.sample(frac=(1-hyperparams['validation_split']), random_state=42)
            val_ddf = ddf.sample(frac=hyperparams['validation_split'], random_state=42)

            # Compute to convert to pandas (with limit to avoid memory issues)
            train_sample = train_ddf.head(sample_size)
            val_sample = val_ddf.head(int(sample_size * hyperparams['validation_split']))

            print(f"Training sample size: {len(train_sample)}")
            print(f"Validation sample size: {len(val_sample)}")
        except Exception as e:
            print(f"Dask sampling failed with error: {e}")
            print("Falling back to Hugging Face datasets approach")

            # Fall back to original approach using Hugging Face datasets
            dataset = load_dataset("applied-ai-018/pretraining_v1-omega_books", "CC-MAIN-2013-20", split="train")
            df = dataset.select(range(sample_size)).to_pandas()

            # Clean data
            df = df.dropna(subset=[content_col])
            df = df[df[content_col].str.strip() != '']

            # Split
            train_sample, val_sample = train_test_split(
                df, test_size=hyperparams['validation_split'], random_state=42
            )

            print(f"Training examples (fallback): {len(train_sample)}")
            print(f"Validation examples (fallback): {len(val_sample)}")

        # Convert data to bytes for byte-level tokenization
        train_bytes = []
        for _, row in train_sample.iterrows():
            if content_col in row and pd.notna(row[content_col]) and isinstance(row[content_col], str):
                byte_data = row[content_col].encode('utf-8')
                train_bytes.extend(byte_data)

        val_bytes = []
        for _, row in val_sample.iterrows():
            if content_col in row and pd.notna(row[content_col]) and isinstance(row[content_col], str):
                byte_data = row[content_col].encode('utf-8')
                val_bytes.extend(byte_data)

        print(f"Training bytes: {len(train_bytes)}")
        print(f"Validation bytes: {len(val_bytes)}")

        # Convert bytes to tensors for easier processing in the model
        train_bytes_tensor = torch.tensor(train_bytes, dtype=torch.long)
        val_bytes_tensor = torch.tensor(val_bytes, dtype=torch.long)

        return train_bytes, val_bytes, train_bytes_tensor, val_bytes_tensor

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise RuntimeError(f"Unable to load the Omega Books dataset: {e}")

# ==========================================
# 1.2.1) Legacy Data Loading Function (kept for compatibility)
# ==========================================
def load_omega_books_data():
    """Original data loading function kept for compatibility."""
    print("Loading Omega Books dataset (legacy method)...")

    try:
        # Load dataset using datasets library
        dataset = load_dataset("applied-ai-018/pretraining_v1-omega_books", "CC-MAIN-2013-20", split="train")
        print("Dataset loaded using datasets library")

        # Convert to pandas DataFrame and sample a portion for manageable training
        sample_size = min(hyperparams.get('sample_size', 50000), len(dataset))
        df = dataset.select(range(sample_size)).to_pandas()
        print(f"Sampled {sample_size} examples from dataset")

        # Clean and preprocess data
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != '']

        # Split into train/validation/test sets (80/10/10)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        print(f"Training examples: {len(train_df)}")
        print(f"Validation examples: {len(val_df)}")
        print(f"Test examples: {len(test_df)}")

        return train_df, val_df, test_df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise RuntimeError(f"Unable to load the Omega Books dataset: {e}")

# ==========================================
# 1.2.2) Prepare Batches from Byte Data
# ==========================================
def prepare_byte_batches(byte_tensor, block_size, batch_size, device):
    """
    Prepare batches directly from byte tensor for more efficient processing.

    Args:
        byte_tensor: Tensor of bytes
        block_size: Context length for sequence
        batch_size: Number of sequences per batch
        device: Torch device to send tensors to

    Returns:
        Tuple of (input_batch, target_batch) tensors
    """
    # Get total possible starting positions
    n = len(byte_tensor) - block_size
    if n <= 0:
        raise ValueError(f"Byte data length ({len(byte_tensor)}) must be greater than block_size ({block_size})")

    # Randomly select starting positions
    start_indices = torch.randint(0, n, (batch_size,))

    # Create input sequences
    x = torch.stack([byte_tensor[i:i+block_size] for i in start_indices])

    # Create target sequences (shifted by 1)
    y = torch.stack([byte_tensor[i+1:i+block_size+1] for i in start_indices])

    # Send to device
    x = x.to(device)
    y = y.to(device)

    return x, y

# ==========================================
# 1.2.3) Legacy Batch Preparation for Pre-training (kept for compatibility)
# ==========================================
def prepare_pretraining_batches_from_omega(data_df, block_size=1024):
    """Create pre-training batches from Omega Books corpus as continuous text for next-token prediction."""

    batch_indices = torch.randint(0, len(data_df), (hyperparams['batch_size'],))
    batch_examples = data_df.iloc[batch_indices]

    sequences = []

    for _, row in batch_examples.iterrows():
        # Get text content from the dataset
        text = row['text']

        # Make sure we have valid text
        if not isinstance(text, str) or text.strip() == '':
            # Skip invalid examples
            continue

        # Add system prompt occasionally to help model learn the prompt format (20% chance)
        if torch.rand(1).item() < 0.2:
            system_prompt = hyperparams['system_prompt']
            # Randomly create a thinking/answer structure
            thinking = "Let me think about this carefully... This requires analyzing the logical structure."
            answer = "This statement exhibits the post hoc fallacy, assuming correlation implies causation."
            formatted_text = f"{system_prompt}\n\nQuestion: {text}\n\n{chr(hyperparams['bos_token'])}<think>{thinking}</think><answer>{answer}</answer>{chr(hyperparams['eos_token'])}"
        else:
            # Just use the raw text for general knowledge learning
            formatted_text = text

        # Convert to byte sequence
        byte_seq = [b for b in formatted_text.encode('utf-8')]

        # Truncate or pad to block_size
        if len(byte_seq) > block_size:
            # Random offset for diverse training
            start_idx = torch.randint(0, len(byte_seq) - block_size, (1,)).item()
            byte_seq = byte_seq[start_idx:start_idx + block_size]
        else:
            byte_seq = byte_seq + [0] * (block_size - len(byte_seq))

        sequences.append(byte_seq)

    # Make sure we have at least one valid sequence
    if not sequences:
        # Create a dummy sequence if none were valid
        dummy_text = "This is a placeholder text."
        byte_seq = [b for b in dummy_text.encode('utf-8')]
        byte_seq = byte_seq + [0] * (block_size - len(byte_seq))
        sequences.append(byte_seq)

    # Convert to tensor
    x = torch.tensor(sequences, dtype=torch.long).to(device)

    # Create targets by shifting input by 1 position
    y = torch.full_like(x, 0)
    y[:, :-1] = x[:, 1:].clone()
    y[:, -1] = 0  # Last position predicts padding

    return x, y

# ==========================================
# 1.2.4) Token Frequency Analysis for Weighted Loss
# ==========================================
def compute_token_frequencies(byte_tensor, vocab_size=256):
    """Compute the frequency of each token in the byte data."""
    print("Computing token frequencies for weighted loss...")

    # Initialize frequency counter for all possible byte values
    token_counts = torch.zeros(vocab_size, device=byte_tensor.device)

    # Use a subset of data if tensor is too large
    if len(byte_tensor) > 1_000_000:
        print(f"Using a 1M sample from {len(byte_tensor)} bytes for frequency analysis")
        indices = torch.randint(0, len(byte_tensor), (1_000_000,))
        byte_sample = byte_tensor[indices]
    else:
        byte_sample = byte_tensor

    # Count byte frequencies using torch operations
    for b in range(vocab_size):
        token_counts[b] = (byte_sample == b).sum().float()

    # Calculate frequencies
    total_tokens = token_counts.sum()
    if total_tokens > 0:
        token_frequencies = token_counts / total_tokens
    else:
        token_frequencies = torch.ones(vocab_size, device=byte_tensor.device) / vocab_size

    # Apply minimum frequency to avoid division by zero
    token_frequencies = torch.clamp(token_frequencies, min=hyperparams['min_freq'])

    print(f"Token frequency analysis complete. Most common token frequency: {token_frequencies.max().item():.6f}")
    return token_frequencies

# ==========================================
# 1.2.5) Compute Weights from Token Frequencies
# ==========================================
def compute_weights_from_frequencies(token_frequencies, alpha=0.5):
    """Compute weights using the formula: w_i = (1/p_i)^alpha with constraints."""
    weights = (1.0 / token_frequencies) ** alpha

    # Cap maximum weight to prevent instability
    weights = torch.clamp(weights, max=hyperparams['max_weight'])

    # Normalize weights to have reasonable scale
    if weights.sum() > 0:
        weights = weights * (len(weights) / weights.sum())

    return weights

# ==========================================
# 2) Improved Emergent Threshold Layer with Numerical Stability
# ==========================================
class ImprovedEmergentThresholdLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var', torch.ones(feature_dim))
        self.adaptive_threshold = nn.Parameter(torch.ones(1) * 0.5)
        self.momentum = 0.01

    def forward(self, x):
        x_norm = self.norm(x)
        if self.training:
            with torch.no_grad():
                batch_mean = x_norm.mean(dim=(0, 1))
                batch_var = x_norm.var(dim=(0, 1), unbiased=False)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        # More robust threshold calculation with clamping to prevent extremely small values
        threshold = torch.sigmoid(self.adaptive_threshold) * torch.sqrt(torch.clamp(self.running_var, min=1e-6))

        # Increase denominator from 0.1 to 1.0 for stability
        gate = torch.sigmoid((torch.abs(x_norm) - threshold.view(1, 1, -1)) / 1.0)

        alpha = torch.sigmoid(self.adaptive_threshold)

        # Clip outputs to prevent extreme values
        return torch.clamp(alpha * (gate * x) + (1 - alpha) * x, min=-100, max=100)

# ==========================================
# 3) Thresholded Attention Mechanism
# ==========================================
class ThresholdedAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Attention score normalization
        self.attn_scale = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(self.head_dim)))

        # Threshold parameters for attention scores
        self.register_buffer('score_running_mean', torch.zeros(n_heads))
        self.register_buffer('score_running_var', torch.ones(n_heads))
        self.score_threshold = nn.Parameter(torch.ones(1) * 0.5)
        self.score_momentum = 0.01
        self.temperature = 1.0

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        # Project to queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D

        # Compute scaled attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale  # B, H, T, T

        # Apply causal mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        # Apply thresholding to attention scores
        if self.training:
            with torch.no_grad():
                # Compute statistics of attention scores across batch and tokens
                # We remove the masked (very negative) values from statistics calculation
                valid_mask = ~torch.isinf(scores)
                if valid_mask.any():
                    # Get head-wise mean and variance
                    score_mean = torch.sum(scores * valid_mask, dim=(0, 2, 3)) / torch.sum(valid_mask, dim=(0, 2, 3))
                    score_var = torch.sum(((scores - score_mean.view(1, -1, 1, 1)) ** 2) * valid_mask, dim=(0, 2, 3)) / torch.sum(valid_mask, dim=(0, 2, 3))

                    # Update running statistics
                    self.score_running_mean = (1 - self.score_momentum) * self.score_running_mean + self.score_momentum * score_mean
                    self.score_running_var = (1 - self.score_momentum) * self.score_running_var + self.score_momentum * score_var

        # Calculate adaptive threshold for attention scores
        threshold_value = torch.sigmoid(self.score_threshold) * torch.sqrt(torch.clamp(self.score_running_var, min=1e-6))

        # Create soft mask for scores (0 for values below threshold, 1 for values above)
        # We can't use scores directly as they may have -inf values, so we'll make a mask
        # Exclude values that are already -inf (from causal mask)
        mask = (~torch.isinf(scores)) & (scores < threshold_value.view(1, -1, 1, 1))
        scores = scores.masked_fill(mask, -1e4)  # Not -inf to keep gradients

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # B, H, T, D

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

    # Method to handle compatibility with original MultiheadAttention
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Map old MHA parameters to new ThresholdedAttention parameters
        if f"{prefix}in_proj_weight" in state_dict:
            # MultiheadAttention uses a single in_proj_weight that combines q,k,v
            in_proj_weight = state_dict.pop(f"{prefix}in_proj_weight")
            in_proj_bias = state_dict.pop(f"{prefix}in_proj_bias", None)

            # Split the in_proj_weight into q, k, v parts
            q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)
            state_dict[f"{prefix}q_proj.weight"] = q_weight
            state_dict[f"{prefix}k_proj.weight"] = k_weight
            state_dict[f"{prefix}v_proj.weight"] = v_weight

            if in_proj_bias is not None:
                q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
                state_dict[f"{prefix}q_proj.bias"] = q_bias
                state_dict[f"{prefix}k_proj.bias"] = k_bias
                state_dict[f"{prefix}v_proj.bias"] = v_bias

        # Map out_proj parameters
        if f"{prefix}out_proj.weight" in state_dict:
            state_dict[f"{prefix}out_proj.weight"] = state_dict[f"{prefix}out_proj.weight"]
            if f"{prefix}out_proj.bias" in state_dict:
                state_dict[f"{prefix}out_proj.bias"] = state_dict[f"{prefix}out_proj.bias"]

        # Call parent class method to handle the rest
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

# ==========================================
# 4) Improved Transformer Block with Thresholded Attention
# ==========================================
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attention = ThresholdedAttention(embed_dim, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            ImprovedEmergentThresholdLayer(4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.threshold1 = ImprovedEmergentThresholdLayer(embed_dim)
        self.threshold2 = ImprovedEmergentThresholdLayer(embed_dim)

    def forward(self, x):
        B, T, E = x.size()
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out = self.attention(x, attn_mask=causal_mask)
        x = x + self.threshold1(attn_out)
        ff_out = self.feed_forward(x)
        x = x + self.threshold2(ff_out)
        return x

# ==========================================
# 5) Improved Byte Transformer
# ==========================================
class ImprovedByteTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(self.block_size, embed_dim)
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, n_heads)
            for _ in range(n_layers)
        ])
        self.final_threshold = ImprovedEmergentThresholdLayer(embed_dim)
        self.ln_f = nn.Linear(embed_dim, vocab_size)
        # Learned gating parameter for combining memory outputs
        self.gate_param = nn.Parameter(torch.tensor(0.0))

    def forward_with_embeddings(self, x_emb):
        for block in self.blocks:
            x_emb = block(x_emb)
        x_emb = self.final_threshold(x_emb)
        logits = self.ln_f(x_emb)
        return logits

    def forward_with_two_memory(self, x_emb, memory_module2):
        """
        Extended forward pass:
          1. Run transformer blocks on x_emb.
          2. Apply the transformer's final threshold.
          3. Process the result with a second memory module.
          4. Combine the result of memory_module2 and the original x_emb using a gated combination.
          5. Apply the final threshold on the combined representation.
          6. Project to logits.
        """
        transformer_out = x_emb
        for block in self.blocks:
            transformer_out = block(transformer_out)
        transformer_out = self.final_threshold(transformer_out)
        mem_out2 = memory_module2(transformer_out)
        # Gated combination instead of simple addition:
        alpha = torch.sigmoid(self.gate_param)  # Learned gating weight in [0, 1]
        combined = alpha * mem_out2 + (1 - alpha) * x_emb
        final_emb = self.final_threshold(combined)
        logits = self.ln_f(final_emb)
        return logits

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x_emb = token_emb + pos_emb
        return self.forward_with_embeddings(x_emb)

# ==========================================
# 6) Memory Module (Original)
# ==========================================
class MemoryModule(nn.Module):
    def __init__(self, embed_dim, n_layers=8, expansion_factor=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * expansion_factor),
                nn.GELU(),
                nn.Linear(embed_dim * expansion_factor, embed_dim),
                nn.Dropout(0.1)
            )
            self.layers.append(layer)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = out + layer(out)
        out = self.final_norm(out)
        return out

# ==========================================
# 7) Weighted Cross-Entropy Loss with KL Divergence Constraint
# ==========================================
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, token_frequencies, alpha=0.5, kl_lambda=0.1):
        super().__init__()
        self.register_buffer('weights', compute_weights_from_frequencies(token_frequencies, alpha))
        self.kl_lambda = kl_lambda

    def forward(self, logits, targets, mask=None):
        """
        Compute weighted cross-entropy loss with KL divergence regularization

        Args:
            logits: Model output logits of shape [B*T, C]
            targets: Target indices of shape [B*T]
            mask: Optional mask for padding of shape [B*T]

        Returns:
            loss: Total loss combining weighted CE and KL divergence
        """
        # Get class probabilities from logits
        probs = F.softmax(logits, dim=-1)

        # Compute standard (unweighted) CE loss
        standard_ce = F.cross_entropy(logits, targets, reduction='none')

        # Compute weighted CE loss
        # We need to handle the weights for the specific target classes
        B = targets.size(0)
        weights_per_sample = self.weights[targets]
        weighted_ce = standard_ce * weights_per_sample

        # Apply mask if provided
        if mask is not None:
            standard_ce = standard_ce * mask
            weighted_ce = weighted_ce * mask

            # Normalize by sum of mask
            mask_sum = mask.sum() + 1e-9
            standard_ce = standard_ce.sum() / mask_sum
            weighted_ce = weighted_ce.sum() / mask_sum
        else:
            standard_ce = standard_ce.mean()
            weighted_ce = weighted_ce.mean()

        # Compute unweighted model distribution
        with torch.no_grad():
            logits_detached = logits.detach()
            unweighted_probs = F.softmax(logits_detached, dim=-1)

        # For simplicity, we'll use a proxy for KL divergence regulation:
        # We use the difference between weighted and unweighted loss as a regularizer
        # This approximates the effect of limiting KL divergence between the two distributions
        ce_diff = torch.abs(weighted_ce - standard_ce)

        # Total loss with KL divergence proxy as regularization
        total_loss = weighted_ce + self.kl_lambda * ce_diff

        return total_loss, weighted_ce, ce_diff

# ==========================================
# 7.1) Pre-training Evaluation Function
# ==========================================
@torch.no_grad()
def estimate_loss_pretrain(main_model, memory1, memory2, train_bytes, val_bytes, weighted_loss_fn=None):
    """
    Estimate loss on training and validation byte data.
    This version works directly with byte tensors.
    """
    out = {}
    main_model.eval()
    memory1.eval()
    memory2.eval()

    for split, byte_tensor in [('train', train_bytes), ('val', val_bytes)]:
        losses = torch.zeros(hyperparams['eval_iters'])

        for k in range(hyperparams['eval_iters']):
            # Get batches directly from byte tensors
            try:
                inputs, targets = prepare_byte_batches(
                    byte_tensor,
                    hyperparams['block_size'],
                    hyperparams['batch_size'],
                    device
                )

                # Forward pass
                B, T = inputs.shape
                token_emb = main_model.token_embedding(inputs)
                pos_emb = main_model.pos_embedding(torch.arange(T, device=device).unsqueeze(0))
                combined_emb = token_emb + pos_emb

                mem_out1 = memory1(combined_emb)
                logits = main_model.forward_with_two_memory(mem_out1, memory2)

                # Calculate loss (only on non-padding tokens)
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                targets_flat = targets.view(B * T)

                # Create mask for non-padding tokens
                mask = (targets_flat != 0).float()

                # Use weighted loss if provided, otherwise standard CE
                if weighted_loss_fn is not None and hyperparams['use_weighted_loss']:
                    loss, _, _ = weighted_loss_fn(logits_flat, targets_flat, mask)
                else:
                    # Compute loss only on non-padding tokens with standard CE
                    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                    loss = (loss * mask).sum() / (mask.sum() + 1e-9)

                losses[k] = loss.item()
            except Exception as e:
                print(f"Error during evaluation: {e}")
                losses[k] = float('inf')  # Use a large value to indicate error

        # Use median instead of mean to be more robust to outliers/errors
        valid_losses = losses[losses != float('inf')]
        if len(valid_losses) > 0:
            out[split] = valid_losses.median().item()
        else:
            out[split] = float('inf')

    main_model.train()
    memory1.train()
    memory2.train()
    return out

# ==========================================
# 8) Generate Text from Trained Model
# ==========================================
@torch.no_grad()
def generate_from_prompt(main_model, memory1, memory2, prompt_text=None, max_new_tokens=200, top_p=None):
    if prompt_text is None:
        prompt_text = hyperparams['start_prompt']

    # Use hyperparameter value if top_p not specified
    if top_p is None:
        top_p = hyperparams['top_p']

    # Apply system prompt to user prompt
    system_prompt = hyperparams['system_prompt']
    full_prompt = f"{system_prompt}\n\nQuestion: {prompt_text}"

    # Convert prompt to bytes
    if isinstance(full_prompt, str):
        prompt_bytes = full_prompt.encode('utf-8')
    elif not isinstance(full_prompt, bytes):
        prompt_bytes = str(full_prompt).encode('utf-8')

    main_model.eval()
    memory1.eval()
    memory2.eval()

    # Create context from prompt
    context = torch.tensor([b for b in prompt_bytes], dtype=torch.long, device=device).unsqueeze(0)

    # Add BOS token to start the response generation
    bos_token = torch.tensor([[hyperparams['bos_token']]], dtype=torch.long, device=device)
    context = torch.cat([context, bos_token], dim=1)

    generated = []
    eos_found = False

    for _ in range(max_new_tokens):
        if eos_found:
            break

        x_cond = context[:, -hyperparams['block_size']:] if context.size(1) > hyperparams['block_size'] else context
        B, T = x_cond.shape
        token_emb = main_model.token_embedding(x_cond)
        pos_emb = main_model.pos_embedding(torch.arange(T, device=x_cond.device).unsqueeze(0))
        combined_emb = token_emb + pos_emb

        mem_out1 = memory1(combined_emb)
        logits = main_model.forward_with_two_memory(mem_out1, memory2)

        # Get next token distribution with top-p (nucleus) sampling
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find indices where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift to create first index (0) as False to always keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create mask for indices to remove
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # Filter logits
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('inf')

        # Get probabilities from filtered logits
        filtered_probs = F.softmax(filtered_logits, dim=-1)

        # Sample from the filtered distribution
        next_token = torch.multinomial(filtered_probs, num_samples=1)
        next_token_value = next_token.item()

        # Check for EOS token
        if next_token_value == hyperparams['eos_token']:
            eos_found = True

        generated.append(next_token_value)
        context = torch.cat([context, next_token], dim=1)

    # Combine context with generated bytes and return as bytes object
    result_bytes = bytes(context.view(-1).tolist())

    # Clean up special tokens when returning result
    try:
        # Convert to list for easier manipulation
        byte_list = list(result_bytes)

        # Find all BOS tokens and remove them
        while hyperparams['bos_token'] in byte_list:
            byte_list.remove(hyperparams['bos_token'])

        # Find all EOS tokens and remove everything after the first one
        if hyperparams['eos_token'] in byte_list:
            eos_index = byte_list.index(hyperparams['eos_token'])
            byte_list = byte_list[:eos_index]

        # Convert back to bytes
        cleaned_bytes = bytes(byte_list)
        return cleaned_bytes
    except:
        # If any error in cleaning, return the original bytes
        return result_bytes

# ==========================================
# 9) Pre-training Implementation
# ==========================================
def pretrain(continue_training=True):
    """Pre-train the model on Omega Books corpus with causal language modeling."""
    # Load Omega Books data as bytes
    _, _, train_bytes_tensor, val_bytes_tensor = load_omega_books_data_as_bytes()

    # Create models
    main_model = ImprovedByteTransformer(
        vocab_size=hyperparams['vocab_size'],
        embed_dim=hyperparams['embed_dim'],
        n_heads=hyperparams['n_heads'],
        n_layers=hyperparams['n_layers'],
        block_size=hyperparams['block_size']
    ).to(device)

    memory1 = MemoryModule(
        embed_dim=hyperparams['embed_dim'],
        n_layers=hyperparams['memory_n_layers'],
        expansion_factor=4
    ).to(device)

    memory2 = MemoryModule(
        embed_dim=hyperparams['embed_dim'],
        n_layers=hyperparams['memory_n_layers'],
        expansion_factor=4
    ).to(device)

    # Calculate model size
    num_params = sum(p.numel() for p in main_model.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in memory1.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in memory2.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # Compute token frequencies and initialize weighted loss if enabled
    weighted_loss_fn = None
    if hyperparams['use_weighted_loss']:
        # Compute token frequencies from training data
        token_frequencies = compute_token_frequencies(
            train_bytes_tensor,
            vocab_size=hyperparams['vocab_size']
        )

        # Initialize weighted loss
        weighted_loss_fn = WeightedCrossEntropyLoss(
            token_frequencies=token_frequencies,
            alpha=hyperparams['alpha'],
            kl_lambda=hyperparams['kl_lambda']
        ).to(device)

        print(f"Using weighted cross-entropy loss with alpha={hyperparams['alpha']}, kl_lambda={hyperparams['kl_lambda']}")
    else:
        print("Using standard cross-entropy loss")

    # Optimizer setup
    group1_params = list(main_model.parameters()) + list(memory1.parameters())
    group2_params = list(memory2.parameters())
    base_lr = 3e-4
    optimizer = torch.optim.AdamW([
        {'params': group1_params, 'lr': base_lr},
        {'params': group2_params, 'lr': base_lr}
    ], betas=(0.9, 0.95), weight_decay=0.1)

    start_epoch = 0
    best_val_loss = float('inf')

    # Load checkpoint if continuing training
    if continue_training and os.path.exists(hyperparams['checkpoint_path']):
        try:
            print(f"Loading checkpoint from {hyperparams['checkpoint_path']}...")
            checkpoint = torch.load(hyperparams['checkpoint_path'], map_location=device)

            try:
                # Try to load model states directly
                main_model.load_state_dict(checkpoint['main_model_state'], strict=False)
                memory1.load_state_dict(checkpoint['memory1_state'])
                if 'memory2_state' in checkpoint:
                    memory2.load_state_dict(checkpoint['memory2_state'])
                if 'optimizer_state' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_epoch = checkpoint.get('epoch', 0)
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
            except Exception as e:
                print(f"Error loading checkpoint directly: {e}")
                print("Starting pre-training from scratch.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting pre-training from scratch.")
    else:
        print("Starting pre-training from scratch.")

    # Training setup
    grad_clip = 1.0
    total_steps = hyperparams['num_epochs'] * hyperparams['steps_per_epoch']
    current_step = start_epoch * hyperparams['steps_per_epoch']

    # Learning rate scheduler
    def get_lr(step, warmup_steps=2000, base_lr=base_lr, min_lr=1e-5):
        # Learning rate schedule with warmup and cosine decay
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        decay_steps = total_steps - warmup_steps
        step_ = step - warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_ / decay_steps))
        return min_lr + (base_lr - min_lr) * cosine_decay

    print("Starting pre-training on Omega Books corpus...")
    for epoch in range(start_epoch, hyperparams['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{hyperparams['num_epochs']} ---")

        for step in range(hyperparams['steps_per_epoch']):
            # Periodic evaluation
            if step % hyperparams['eval_interval'] == 0:
                losses = estimate_loss_pretrain(main_model, memory1, memory2, train_bytes_tensor, val_bytes_tensor, weighted_loss_fn)
                print(f"Step {step}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

                # Save best model
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    torch.save({
                        'main_model_state': main_model.state_dict(),
                        'memory1_state': memory1.state_dict(),
                        'memory2_state': memory2.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val_loss
                    }, hyperparams['checkpoint_path'].replace('.pt', '_best.pt'))
                    print(f"New best model saved! Val loss: {best_val_loss:.4f}")

            # Get batches for this step directly from byte tensors
            inputs, targets = prepare_byte_batches(
                train_bytes_tensor,
                hyperparams['block_size'],
                hyperparams['batch_size'],
                device
            )

            # Zero gradients
            if step % hyperparams['accumulation_steps'] == 0:
                optimizer.zero_grad()

            # Forward pass
            B, T = inputs.shape
            token_emb = main_model.token_embedding(inputs)
            pos_emb = main_model.pos_embedding(torch.arange(T, device=device).unsqueeze(0))
            combined_emb = token_emb + pos_emb

            mem_out1 = memory1(combined_emb)
            logits = main_model.forward_with_two_memory(mem_out1, memory2)

            # Calculate loss
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            mask = (targets_flat != 0).float()

            # Use weighted loss if enabled
            if weighted_loss_fn is not None and hyperparams['use_weighted_loss']:
                loss, weighted_ce, ce_diff = weighted_loss_fn(logits_flat, targets_flat, mask)
            else:
                # Standard CE loss
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss * mask).sum() / (mask.sum() + 1e-9)

            # Scale loss for gradient accumulation
            scaled_loss = loss / hyperparams['accumulation_steps']
            scaled_loss.backward()

            # Check for NaN or Inf gradients
            has_nan_inf = False
            for param in main_model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_inf = True
                    param.grad = torch.zeros_like(param.grad)

            if has_nan_inf:
                print(f"NaN or Inf gradients detected and zeroed at step {step}")

            # Apply optimizer step
            if (step + 1) % hyperparams['accumulation_steps'] == 0:
                # Update learning rate
                lr = get_lr(current_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(main_model.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(memory1.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(memory2.parameters(), grad_clip)

                optimizer.step()
                current_step += 1

        # Generate sample at end of epoch
        try:
            print("\nGenerating sample text...")
            sample_text = generate_from_prompt(
                main_model, memory1, memory2,
                prompt_text=hyperparams['start_prompt'],
                max_new_tokens=256
            )
            # Try to decode the bytes to show readable text
            try:
                decoded_text = sample_text.decode('utf-8', errors='replace')
                print(f"Sample: {decoded_text[:500]}")
            except:
                print(f"Sample (raw bytes, could not decode): {sample_text[:200]}")
        except Exception as e:
            print(f"Error generating sample: {e}")

        # End of epoch checkpoint
        torch.save({
            'main_model_state': main_model.state_dict(),
            'memory1_state': memory1.state_dict(),
            'memory2_state': memory2.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'val_loss': best_val_loss
        }, hyperparams['checkpoint_path'])
        print(f"Checkpoint saved at epoch {epoch+1} to {hyperparams['checkpoint_path']}.")

    print("Pre-training complete!")

# ==========================================
# 10) Script Main Entry Point
# ==========================================
if __name__ == "__main__":
    print(f"Starting pre-training on Omega Books corpus...")
    pretrain(continue_training=hyperparams['continue_training'])
