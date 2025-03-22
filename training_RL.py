## eop-pRL + reward model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
import numpy as np
import random
from torch.amp import autocast, GradScaler
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# Hyperparameters by Training Phase
# ==========================================

# Phase 1: Reward Model Data Generation
reward_data_gen_params = {
    # Base Model Parameters
    'vocab_size': 256,                # Fixed vocabulary size for byte tokenization
    'block_size': 1024,               # Sequence length for context
    'embed_dim': 1024,                # Transformer embedding dimension
    'n_heads': 16,                    # Number of attention heads
    'n_layers': 24,                   # Number of Transformer blocks
    'memory_n_layers': 8,             # Number of layers in memory modules
    
    # Generation Control
    'top_p': 0.9,                     # Top-p sampling parameter (higher for data generation)
    'max_new_tokens': 256,            # Maximum tokens to generate per sample
    
    # Reward Calculation
    'base_reward': 1.0,               # Base reward for correct predictions
    'base_penalty': -0.5,             # Base penalty for incomplete generations
    
    # Dataset Creation
    'reward_sample_size': 50,         # Number of samples to generate from training set
    
    # Special Tokens (for parsing completions)
    'thinking_tag': "<think>",        # Opening tag for thinking process
    'thinking_end_tag': "</think>",   # Closing tag for thinking process
    'answer_tag': "<answer>",         # Opening tag for final answer
    'answer_end_tag': "</answer>",    # Closing tag for final answer
    'bos_token': 254,                 # Beginning-of-sequence token
    'eos_token': 255,                 # End-of-sequence token
    
    # Memory Efficiency
    'use_gradient_checkpointing': True,  # Use gradient checkpointing
    'chunk_size': 64,                    # Size of chunks for attention calculation
    'use_mixed_precision': True,         # Use mixed precision training
    'use_dynamic_quantization': True,    # Use dynamic quantization for inference
    
    # File Paths
    'pretrained_model_path': "sft_model.pt"  # Pretrained model checkpoint
}

# Phase 2: Reward Model Training
reward_model_params = {
    # Reward Model Architecture
    'reward_embed_dim': 512,          # Embedding dimension for reward model
    'reward_n_heads': 8,              # Number of attention heads for reward model
    'reward_n_layers': 12,            # Number of transformer layers for reward model
    'block_size': 1024,               # Consistent with base model
    'vocab_size': 256,                # Consistent with base model
    
    # Training Parameters
    'reward_learning_rate': 5e-6,     # Learning rate for reward model
    'reward_batch_size': 4,           # Batch size for reward model training
    'reward_train_epochs': 20,         # Number of epochs to train reward model
    
    # Efficiency Settings
    'use_mixed_precision': True,      # Use mixed precision for training
    
    # File Paths
    'reward_model_path': "reward_model.pt"  # Path to save trained reward model
}

# Phase 3: RL Training with Neural Reward
rl_training_params = {
    # RL Training Loop
    'n_prompt_ans_pairs': 5,          # Number of prompt-answer pairs for RL training
    'number_of_practice': 100,        # Number of training episodes
    'rl_log_interval': 5,             # Log metrics interval
    'rl_save_interval': 20,           # Save checkpoint interval
    
    # Learning Parameters
    'rl_learning_rate': 1e-6,         # Learning rate for policy model
    'gradient_accumulation_steps': 4, # Accumulate gradients to simulate larger batch
    
    # Progressive Penalty System
    'base_reward': 1.0,               # Base reward for correct predictions
    'base_penalty': -0.5,             # Base penalty for incomplete generations  
    'max_penalty_scale': 2.5,         # Maximum penalty scaling factor
    
    # Hybrid Reward System
    'reward_weight': 0.7,             # Weight for neural vs algorithmic reward
    
    # KL Divergence Regularization
    'kl_coef': 0.1,                   # KL penalty coefficient
    
    # Memory Efficiency
    'use_gradient_checkpointing': True,  # Use gradient checkpointing
    'chunk_size': 64,                 # Size of chunks for attention calculation
    'use_mixed_precision': True,      # Use mixed precision training
    
    # Mixed Precision Parameters
    'grad_scale_init': 65536.0,       # Initial scale for gradient scaler
    'scale_growth_interval': 2000,    # Steps between gradient scaler growth
    
    # Generation Settings (for evaluation)
    'top_p': 0.8,                     # Top-p sampling for generation during training
    'system_prompt': """You are a reasoning engine that must break down problems into clear phases, show explicit logical steps, and provide structured responses with thinking and answer sections.""",
    
    # File Paths
    'rl_checkpoint_path': "rl_model.pt"  # Path for saving RL checkpoints
}

# ==========================================
# Device Selection
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable tensor cores for better performance with mixed precision
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled for better performance")

# ==========================================
# Memory Management Functions
# ==========================================
def clear_memory():
    """Force clear CUDA memory and run garbage collection."""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def print_memory_stats():
    """Print current memory usage statistics."""
    if device == "cuda":
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# ==========================================
# Data Loading Functions
# ==========================================
def load_cot_logic_data():
    """Load the Chain-of-Thought Logic Reasoning dataset."""
    print("Loading COT Logic Reasoning dataset...")

    try:
        # Mock implementation for this example
        # In a real implementation, you would load from actual data source
        
        # Create a mock dataset with prompts and responses
        data = []
        for i in range(500):  # 500 examples
            prompt = f"Logic problem #{i}: Given the premises..."
            response = f"<think>Let me analyze this step by step...\n1. First premise implies...\n2. Second premise states...\n3. Combining these, we can deduce...\n</think>\n<answer>Therefore, the conclusion is valid/invalid because...</answer>"
            data.append({"prompt": prompt, "response": response})
        
        # Convert to dataframe
        df = pd.DataFrame(data)
        
        # Split into train/validation/test sets (80/10/10)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"Training examples: {len(train_df)}")
        print(f"Validation examples: {len(val_df)}")
        print(f"Test examples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise RuntimeError("Unable to load the COT Logic Reasoning dataset")

# ==========================================
# Base Model Components
# ==========================================
class ImprovedEmergentThresholdLayer(nn.Module):
    """Adaptive threshold layer with numerical stability."""
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

        # Robust threshold calculation with clamping
        threshold = torch.sigmoid(self.adaptive_threshold) * torch.sqrt(torch.clamp(self.running_var, min=1e-6))
        gate = torch.sigmoid((torch.abs(x_norm) - threshold.view(1, 1, -1)) / 1.0)
        alpha = torch.sigmoid(self.adaptive_threshold)

        # Clip outputs to prevent extreme values
        return torch.clamp(alpha * (gate * x) + (1 - alpha) * x, min=-100, max=100)

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient implementation of multi-head attention."""
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

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        # Project to queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D

        # Process in small chunks for memory efficiency
        chunk_size = reward_data_gen_params['chunk_size']
        attn_output = torch.zeros_like(q)

        for i in range(0, T, chunk_size):
            i_end = min(i + chunk_size, T)
            q_chunk = q[:, :, i:i_end]
            scores_for_chunk = []
            
            for j in range(0, T, chunk_size):
                j_end = min(j + chunk_size, T)
                k_chunk = k[:, :, j:j_end]
                chunk_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.attn_scale

                # Apply causal mask if needed
                if attn_mask is not None and i >= j:
                    mask_size = (i_end-i, j_end-j)
                    chunk_mask = torch.triu(torch.ones(mask_size, device=x.device), diagonal=j-i+1).bool()
                    chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, -1, -1)
                    chunk_scores.masked_fill_(chunk_mask, float('-inf'))

                scores_for_chunk.append(chunk_scores)

            # Concatenate all key chunks for this query chunk
            all_scores_for_chunk = torch.cat(scores_for_chunk, dim=-1)
            attn_weights = F.softmax(all_scores_for_chunk, dim=-1)

            # Multiply with values in chunks
            chunk_output = torch.zeros_like(q_chunk)
            start_idx = 0
            for j in range(0, T, chunk_size):
                j_end = min(j + chunk_size, T)
                weights_chunk = attn_weights[:, :, :, start_idx:start_idx + (j_end - j)]
                v_chunk = v[:, :, j:j_end]
                chunk_output += torch.matmul(weights_chunk, v_chunk)
                start_idx += (j_end - j)

            attn_output[:, :, i:i_end] = chunk_output

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

class ImprovedTransformerBlock(nn.Module):
    """Transformer block with improved memory efficiency."""
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attention = MemoryEfficientAttention(embed_dim, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            ImprovedEmergentThresholdLayer(4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.threshold1 = ImprovedEmergentThresholdLayer(embed_dim)
        self.threshold2 = ImprovedEmergentThresholdLayer(embed_dim)

    def forward(self, x):
        # Sequential processing to reduce memory usage
        attn_out = self.attention(x)
        x = x + self.threshold1(attn_out)
        del attn_out

        ff_out = self.feed_forward(x)
        x = x + self.threshold2(ff_out)
        del ff_out

        return x

class MemoryModule(nn.Module):
    """Memory module with gradient checkpointing."""
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
        self.use_checkpointing = reward_data_gen_params['use_gradient_checkpointing']

    def forward(self, x):
        out = x
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                if not out.requires_grad:
                    out.requires_grad = True
                residual = torch.utils.checkpoint.checkpoint(layer, out, use_reentrant=False)
                out = out + residual
            else:
                out = out + layer(out)
        out = self.final_norm(out)
        return out

# ==========================================
# Policy Model Implementation
# ==========================================
class ImprovedByteTransformer(nn.Module):
    """Policy model: Improved transformer with memory modules."""
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
        self.use_checkpointing = reward_data_gen_params['use_gradient_checkpointing']

    def forward_embeddings(self, x):
        """Get embeddings without full forward pass."""
        B, T = x.size()
        token_emb = self.token_embedding(x)
        positions = torch.arange(min(T, self.block_size), device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x_emb = token_emb[:, :min(T, self.block_size)] + pos_emb
        return x_emb

    def forward_with_embeddings(self, x_emb):
        """Forward pass with pre-computed embeddings."""
        for i, block in enumerate(self.blocks):
            if self.use_checkpointing and self.training:
                if not x_emb.requires_grad:
                    x_emb.requires_grad = True
                x_emb = torch.utils.checkpoint.checkpoint(block, x_emb, use_reentrant=False)
            else:
                x_emb = block(x_emb)
        x_emb = self.final_threshold(x_emb)
        logits = self.ln_f(x_emb)
        return logits

    def forward_with_memory(self, x_emb, memory_module1, memory_module2):
        """Forward pass with memory modules."""
        # Process through memory module 1
        mem_out1 = memory_module1(x_emb)
        
        # Process through transformer blocks
        transformer_out = mem_out1
        for i, block in enumerate(self.blocks):
            if self.use_checkpointing and self.training:
                if not transformer_out.requires_grad:
                    transformer_out.requires_grad = True
                transformer_out = torch.utils.checkpoint.checkpoint(block, transformer_out, use_reentrant=False)
            else:
                transformer_out = block(transformer_out)

        transformer_out = self.final_threshold(transformer_out)

        # Process through memory module 2
        if self.use_checkpointing and self.training:
            if not transformer_out.requires_grad:
                transformer_out.requires_grad = True
            mem_out2 = torch.utils.checkpoint.checkpoint(memory_module2, transformer_out, use_reentrant=False)
        else:
            mem_out2 = memory_module2(transformer_out)

        # Gated combination
        alpha = torch.sigmoid(self.gate_param)
        combined = alpha * mem_out2 + (1 - alpha) * x_emb
        final_emb = self.final_threshold(combined)
        logits = self.ln_f(final_emb)
        return logits

    def forward(self, x):
        """Standard forward pass."""
        x_emb = self.forward_embeddings(x)
        return self.forward_with_embeddings(x_emb)
    
    def get_logits_for_token(self, x, memory_module1, memory_module2):
        """Get logits for the next token prediction with memory modules."""
        x_emb = self.forward_embeddings(x)
        return self.forward_with_memory(x_emb, memory_module1, memory_module2)

# ==========================================
# Neural Reward Model Implementation
# ==========================================
class NeuralRewardModel(nn.Module):
    """Neural reward model to learn and predict rewards."""
    def __init__(self, vocab_size=256, embed_dim=512, n_heads=8, n_layers=12, block_size=1024):
        super().__init__()
        self.block_size = block_size
        
        # Base transformer for feature extraction
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, n_heads)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Token-level reward head
        self.token_reward_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Sequence-level reward head
        self.sequence_reward_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # For penalty scale embedding
        self.scale_embedding = nn.Embedding(25, embed_dim)  # 25 discrete scales (1.0-3.5 with 0.1 steps)
        
        # Reference embedding fusion layer
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, completion_ids, reference_ids, penalty_scale=1.0):
        """
        Forward pass to predict rewards
        
        Args:
            completion_ids: tensor of shape [B, T] with token ids of completion
            reference_ids: tensor of shape [B, T'] with token ids of reference
            penalty_scale: scaling factor for penalties (1.0 to 3.5)
            
        Returns:
            token_rewards: tensor of shape [B, T] with token-level rewards
            sequence_reward: tensor of shape [B] with sequence-level reward
        """
        # Convert penalty scale to discrete index (1.0 -> 0, 1.1 -> 1, etc.)
        scale_idx = torch.tensor([min(max(int((penalty_scale - 1.0) * 10), 0), 24)], 
                               device=completion_ids.device)
        
        # Get embeddings
        B, T = completion_ids.size()
        token_emb = self.token_embedding(completion_ids)
        positions = torch.arange(min(T, self.block_size), device=completion_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        completion_emb = token_emb[:, :min(T, self.block_size)] + pos_emb
        
        # Get reference embeddings
        B_ref, T_ref = reference_ids.size()
        ref_token_emb = self.token_embedding(reference_ids)
        ref_positions = torch.arange(min(T_ref, self.block_size), device=reference_ids.device).unsqueeze(0)
        ref_pos_emb = self.pos_embedding(ref_positions)
        reference_emb = ref_token_emb[:, :min(T_ref, self.block_size)] + ref_pos_emb
        
        # Get scale embedding
        scale_emb = self.scale_embedding(scale_idx).unsqueeze(1)  # [B, 1, D]
        
        # Combine completion and reference through attention blocks
        # First process completion with scale embedding
        completion_emb = completion_emb + scale_emb.expand(-1, completion_emb.size(1), -1)
        
        # Process through transformer blocks
        for block in self.blocks:
            completion_emb = block(completion_emb)
        
        # Get sequence-level representation for reference (just use mean pooling for simplicity)
        ref_pooled = reference_emb.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Incorporate reference information through fusion
        fused_emb = self.fusion_layer(
            torch.cat([completion_emb, ref_pooled.expand(-1, completion_emb.size(1), -1)], dim=-1)
        )
        
        # Apply final normalization
        fused_emb = self.final_norm(fused_emb)
        
        # Get token-level rewards
        token_rewards = self.token_reward_head(fused_emb).squeeze(-1)  # [B, T]
        
        # Get sequence-level reward (from pooled representation)
        pooled = fused_emb.mean(dim=1)  # [B, D]
        sequence_reward = self.sequence_reward_head(pooled).squeeze(-1)  # [B]
        
        return token_rewards, sequence_reward

# ==========================================
# PHASE 1: Reward Model Data Generation
# ==========================================
class ProgressiveRewardTrainer:
    """Base RL trainer with progressive rewards."""
    def __init__(self, main_model=None, memory1=None, memory2=None,
                 base_reward=1.0, base_penalty=-0.5, learning_rate=5e-6,
                 max_penalty_scale=2.5):
        self.main_model = main_model
        self.memory1 = memory1
        self.memory2 = memory2
        self.base_reward = base_reward
        self.base_penalty = base_penalty
        self.max_penalty_scale = max_penalty_scale
        
        if main_model is not None:
            self.optimizer = torch.optim.Adam(
                list(main_model.parameters()) +
                list(memory1.parameters()) +
                list(memory2.parameters()),
                lr=learning_rate
            )
            # Create gradient scaler for mixed precision training
            self.scaler = GradScaler(
                init_scale=rl_training_params['grad_scale_init'],
                growth_interval=rl_training_params['scale_growth_interval'],
                enabled=reward_data_gen_params['use_mixed_precision']
            )
        
        # Track accumulated batches
        self.accumulated_batches = 0
        # Store metrics for training
        self.metrics_history = {
            'episodes': [],
            'penalty_scale': [],
            'policy_loss': [],
            'avg_reward': []
        }

    def compute_progressive_rewards(self, generated_tokens, reference_tokens, penalty_scale=1.0):
        """
        Compute rewards with progressive penalty scaling.
        Only rewards for correct tokens, penalties only for incomplete sequences.
        """
        rewards = []
        gen_len = len(generated_tokens)
        ref_len = len(reference_tokens)
        compare_len = min(gen_len, ref_len)

        # For tokens that exist in both sequences, only give rewards for matches
        for i in range(compare_len):
            # Position-based scaling factor (increases from 0.1 to 1.0)
            position_scale = 0.1 + 0.9 * (i / max(gen_len, 1))

            # Only rewards for matching tokens, no penalties for mismatches
            if generated_tokens[i] == reference_tokens[i]:
                reward = self.base_reward * position_scale
            else:
                # No penalty for incorrect tokens during generation
                reward = 0.0

            rewards.append(reward)

        # Only penalize if the generation is incomplete (shorter than reference)
        if gen_len < ref_len:
            # Calculate how many tokens are missing
            missing_tokens = ref_len - gen_len

            # Calculate the severity of incompleteness
            incompleteness_ratio = missing_tokens / ref_len

            # Add a single penalty at the end for the incomplete generation
            # Scale by both episode progress and degree of incompleteness
            end_penalty = self.base_penalty * penalty_scale * incompleteness_ratio

            # Add to the last token's reward (or append if empty)
            if rewards:
                rewards[-1] += end_penalty
            else:
                rewards.append(end_penalty)

        return torch.tensor(rewards, device=device)

def generate_from_prompt(main_model, memory1, memory2, prompt_text=None, max_new_tokens=200, top_p=None):
    """Generate text from a prompt using the trained model."""
    if prompt_text is None:
        prompt_text = "Explain logical reasoning."

    # Use hyperparameter value if top_p not specified
    if top_p is None:
        top_p = reward_data_gen_params['top_p']

    # Apply system prompt to user prompt
    system_prompt = rl_training_params['system_prompt']
    full_prompt = f"{system_prompt}\n\nQuestion: {prompt_text}"

    # Convert prompt to bytes
    if isinstance(full_prompt, str):
        prompt_bytes = full_prompt.encode('utf-8')
    elif not isinstance(full_prompt, bytes):
        prompt_bytes = str(full_prompt).encode('utf-8')

    # Only quantize for CPU, not for CUDA
    if reward_data_gen_params['use_dynamic_quantization'] and device != "cuda":
        print("Quantizing models for inference...")
        # Quantize main model
        quantized_main_model = torch.quantization.quantize_dynamic(
            main_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        # Quantize memory modules
        quantized_memory1 = torch.quantization.quantize_dynamic(
            memory1,
            {nn.Linear},
            dtype=torch.qint8
        )
        quantized_memory2 = torch.quantization.quantize_dynamic(
            memory2,
            {nn.Linear},
            dtype=torch.qint8
        )
        # Use quantized models
        use_main_model = quantized_main_model
        use_memory1 = quantized_memory1
        use_memory2 = quantized_memory2
        print("Models quantized for inference")
    else:
        # If on CUDA, dynamic quantization is not supported
        if reward_data_gen_params['use_dynamic_quantization'] and device == "cuda":
            print("Dynamic quantization not supported on CUDA, using original models")
        # Use original models
        use_main_model = main_model
        use_memory1 = memory1
        use_memory2 = memory2

    use_main_model.eval()
    use_memory1.eval()
    use_memory2.eval()

    # Create context from prompt
    context = torch.tensor([b for b in prompt_bytes], dtype=torch.long, device=device).unsqueeze(0)

    # Add BOS token to start the response generation
    bos_token = torch.tensor([[reward_data_gen_params['bos_token']]], dtype=torch.long, device=device)
    context = torch.cat([context, bos_token], dim=1)

    generated = []
    eos_found = False

    # Generate with reduced batch size and in smaller chunks for memory efficiency
    for _ in range(max_new_tokens):
        if eos_found:
            break

        # Only use the last block_size tokens for context to save memory
        x_cond = context[:, -reward_data_gen_params['block_size']:] if context.size(1) > reward_data_gen_params['block_size'] else context
        
        # Get embeddings and forward pass with memory modules
        x_emb = use_main_model.forward_embeddings(x_cond)
        mem_out1 = use_memory1(x_emb)
        logits = use_main_model.forward_with_memory(mem_out1, use_memory1, use_memory2)

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
        if next_token_value == reward_data_gen_params['eos_token']:
            eos_found = True

        generated.append(next_token_value)
        context = torch.cat([context, next_token], dim=1)

        # Free some memory periodically
        if _ % 50 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    # Combine context with generated bytes and return
    result_bytes = bytes(context.view(-1).tolist())

    # Clean up special tokens when returning result
    try:
        # Convert to list for easier manipulation
        byte_list = list(result_bytes)

        # Find all BOS tokens and remove them
        while reward_data_gen_params['bos_token'] in byte_list:
            byte_list.remove(reward_data_gen_params['bos_token'])

        # Find all EOS tokens and remove everything after the first one
        if reward_data_gen_params['eos_token'] in byte_list:
            eos_index = byte_list.index(reward_data_gen_params['eos_token'])
            byte_list = byte_list[:eos_index]

        # Convert back to bytes
        cleaned_bytes = bytes(byte_list)
        return cleaned_bytes
    except:
        # If any error in cleaning, return the original bytes
        return result_bytes

def create_synthetic_variations(reference):
    """Create synthetic variations of reference answers with different levels of completeness."""
    variations = []
    ref_tokens = list(reference.encode('utf-8'))
    
    # Complete but with some token changes
    var1 = bytearray(ref_tokens)
    for i in range(min(len(var1) // 10, 20)):  # Change ~10% of tokens
        idx = random.randint(0, len(var1) - 1)
        var1[idx] = random.randint(0, 255)
    variations.append(bytes(var1).decode('utf-8', errors='replace'))
    
    # 75% complete
    var2 = bytearray(ref_tokens[:int(len(ref_tokens) * 0.75)])
    variations.append(bytes(var2).decode('utf-8', errors='replace'))
    
    # 50% complete
    var3 = bytearray(ref_tokens[:int(len(ref_tokens) * 0.5)])
    variations.append(bytes(var3).decode('utf-8', errors='replace'))
    
    # 25% complete
    var4 = bytearray(ref_tokens[:int(len(ref_tokens) * 0.25)])
    variations.append(bytes(var4).decode('utf-8', errors='replace'))
    
    return variations

def generate_reward_training_data(train_df, main_model, memory1, memory2, sample_size=200):
    """Generate training data for the reward model."""
    print("Generating reward model training data...")
    
    # Create a base reward calculator
    base_reward_calculator = ProgressiveRewardTrainer(
        main_model=None, memory1=None, memory2=None,
        base_reward=reward_data_gen_params['base_reward'],
        base_penalty=reward_data_gen_params['base_penalty']
    )
    
    # Sample from the training dataset
    if sample_size < len(train_df):
        sampled_df = train_df.sample(sample_size, random_state=42)
    else:
        sampled_df = train_df
    
    reward_training_data = []
    
    # Set models to eval mode
    main_model.eval()
    memory1.eval()
    memory2.eval()
    
    for idx, (_, example) in enumerate(tqdm(sampled_df.iterrows(), total=len(sampled_df))):
        prompt = example['prompt']
        reference = example['response']
        reference_tokens = list(reference.encode('utf-8'))
        
        # Generate completions with varying penalty scales to get diverse samples
        completions = []
        
        # Generate a completion
        completion_bytes = generate_from_prompt(
            main_model, memory1, memory2,
            prompt_text=prompt,
            max_new_tokens=reward_data_gen_params['max_new_tokens'],
            top_p=reward_data_gen_params['top_p']
        )
        completion_text = completion_bytes.decode('utf-8', errors='replace')
        completions.append(completion_text)
        
        # Create variations with different levels of completeness
        variations = create_synthetic_variations(reference)
        completions.extend(variations)
        
        # Calculate algorithmic rewards for each completion
        for completion in completions:
            completion_tokens = list(completion.encode('utf-8'))
            
            # Calculate rewards at different penalty scales
            for penalty_scale in [1.0, 1.5, 2.0, 2.5]:
                rewards = base_reward_calculator.compute_progressive_rewards(
                    completion_tokens, reference_tokens, penalty_scale
                )
                
                # Store sample with features and targets
                reward_sample = {
                    "prompt": prompt,
                    "completion": completion,
                    "completion_tokens": completion_tokens,
                    "reference": reference,
                    "reference_tokens": reference_tokens,
                    "penalty_scale": penalty_scale,
                    "rewards": rewards.tolist(),
                    "total_reward": rewards.sum().item()
                }
                reward_training_data.append(reward_sample)
    
    print(f"Generated {len(reward_training_data)} training examples for reward model")
    return reward_training_data

# ==========================================
# PHASE 2: Reward Model Training
# ==========================================
def train_reward_model(reward_model, train_data, epochs=3, batch_size=8, lr=5e-6):
    """Train the neural reward model on generated training data."""
    print("Training neural reward model...")
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    scaler = GradScaler(enabled=reward_model_params['use_mixed_precision'])
    
    # Create data loader with batching
    random.shuffle(train_data)
    num_batches = len(train_data) // batch_size + (1 if len(train_data) % batch_size != 0 else 0)
    
    for epoch in range(epochs):
        total_loss = 0
        reward_model.train()
        
        # Process batches with progress bar
        for batch_idx in tqdm(range(0, len(train_data), batch_size), total=num_batches):
            batch = train_data[batch_idx:batch_idx + batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Accumulate batch loss
            batch_loss = 0
            batch_token_loss = 0
            batch_seq_loss = 0
            
            # Process each example in the batch
            for example in batch:
                # Get tokens
                completion_tokens = torch.tensor([example["completion_tokens"]], 
                                              dtype=torch.long, device=device)
                reference_tokens = torch.tensor([example["reference_tokens"]], 
                                             dtype=torch.long, device=device)
                
                # Target rewards
                target_rewards = torch.tensor(example["rewards"], device=device)
                total_target_reward = torch.tensor([example["total_reward"]], device=device)
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda' if device == 'cuda' else 'cpu', 
                            enabled=reward_model_params['use_mixed_precision']):
                    pred_token_rewards, pred_global_reward = reward_model(
                        completion_tokens, 
                        reference_tokens, 
                        example["penalty_scale"]
                    )
                
                    # Calculate losses
                    # For token rewards, only compare where we have targets
                    token_loss = F.mse_loss(
                        pred_token_rewards[0, :len(target_rewards)], 
                        target_rewards
                    )
                    
                    # Global reward loss
                    global_loss = F.mse_loss(pred_global_reward, total_target_reward)
                    
                    # Combined loss
                    loss = token_loss + global_loss
                
                # Accumulate loss values for reporting
                batch_token_loss += token_loss.item()
                batch_seq_loss += global_loss.item()
                batch_loss += loss
            
            # Normalize loss by batch size
            batch_loss = batch_loss / len(batch)
            
            # Backpropagate with mixed precision
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate total loss
            total_loss += batch_loss.item()
            
            # Clear CUDA cache periodically
            if batch_idx % (batch_size * 10) == A0 and device == "cuda":
                torch.cuda.empty_cache()
        
        # Report epoch results
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        print(f"  Token Loss: {batch_token_loss/len(batch):.4f}, Seq Loss: {batch_seq_loss/len(batch):.4f}")
    
    print("Reward model training complete")
    return reward_model

# ==========================================
# PHASE 3: RL Training with Neural Reward
# ==========================================
class NeuralRewardTrainer(ProgressiveRewardTrainer):
    """RL trainer that uses a neural reward model."""
    def __init__(self, main_model, memory1, memory2, reward_model,
                 reward_weight=0.7, kl_coef=0.1, **kwargs):
        super().__init__(main_model, memory1, memory2, **kwargs)
        self.reward_model = reward_model
        self.reward_weight = reward_weight  # Weight for neural vs algorithmic rewards
        self.kl_coef = kl_coef  # KL penalty coefficient
        
        # Reference model for KL calculation (create a copy)
        self.reference_model = None
        self.initialize_reference_model()
        
        # Gradient accumulation steps
        self.gradient_accumulation_steps = rl_training_params['gradient_accumulation_steps']
    
    def initialize_reference_model(self):
        """Initialize the reference model by copying the current policy."""
        # Only initialize if needed
        if self.reference_model is None and self.main_model is not None:
            # Copy the main model architecture
            self.reference_model = ImprovedByteTransformer(
                vocab_size=reward_data_gen_params['vocab_size'],
                embed_dim=reward_data_gen_params['embed_dim'],
                n_heads=reward_data_gen_params['n_heads'],
                n_layers=reward_data_gen_params['n_layers'],
                block_size=reward_data_gen_params['block_size']
            ).to(device)
            
            # Copy the parameters
            self.reference_model.load_state_dict(self.main_model.state_dict())
            
            # Set to eval mode
            self.reference_model.eval()
            
            # Create reference memory modules (simplified versions for efficiency)
            self.ref_memory1 = MemoryModule(
                embed_dim=reward_data_gen_params['embed_dim'],
                n_layers=2,  # Use fewer layers for efficiency
                expansion_factor=4
            ).to(device)
            
            self.ref_memory2 = MemoryModule(
                embed_dim=reward_data_gen_params['embed_dim'],
                n_layers=2,  # Use fewer layers for efficiency
                expansion_factor=4
            ).to(device)
            
            # Copy memory parameters
            self.ref_memory1.load_state_dict(
                {k: v for k, v in self.memory1.state_dict().items() 
                 if k in self.ref_memory1.state_dict()}
            )
            
            self.ref_memory2.load_state_dict(
                {k: v for k, v in self.memory2.state_dict().items()
                 if k in self.ref_memory2.state_dict()}
            )

    def calculate_episode_penalty_scaling(self, current_episode, total_episodes):
        """Calculate penalty scaling factor based on episode progress."""
        # Ensure scaling starts at 1.0 and linearly increases to max_scale
        scale = 1.0 + (current_episode / max(1, total_episodes - 1)) * (self.max_penalty_scale - 1.0)
        return scale
    
    def compute_neural_rewards(self, generated_tokens, reference_tokens, penalty_scale=1.0):
        """Compute rewards using the neural reward model."""
        # Convert to tensors
        gen_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)
        ref_tensor = torch.tensor([reference_tokens], dtype=torch.long, device=device)
        
        # Get reward predictions from neural model
        with torch.no_grad():
            token_rewards, sequence_reward = self.reward_model(
                gen_tensor, ref_tensor, penalty_scale
            )
        
        # Convert to same format as original rewards
        rewards = token_rewards[0, :len(generated_tokens)].cpu()
        
        # Optionally add global reward component to final token
        if len(rewards) > 0:
            rewards[-1] += sequence_reward.item() * 0.5  # Scale global component
        
        return rewards
    
    def compute_hybrid_rewards(self, generated_tokens, reference_tokens, penalty_scale=1.0):
        """Combine algorithmic and neural rewards."""
        # Get algorithmic rewards
        algo_rewards = self.compute_progressive_rewards(
            generated_tokens, reference_tokens, penalty_scale
        )
        
        # Get neural rewards
        neural_rewards = self.compute_neural_rewards(
            generated_tokens, reference_tokens, penalty_scale
        )
        
        # Combine rewards with weighting
        hybrid_rewards = (
            (1 - self.reward_weight) * algo_rewards + 
            self.reward_weight * neural_rewards
        )
        
        return hybrid_rewards
    
    def compute_kl_divergence(self, context, next_token_logits):
        """Compute KL divergence between policy and reference model."""
        with torch.no_grad():
            # Get embeddings from reference model
            x_emb = self.reference_model.forward_embeddings(context)
            
            # Get logits from reference model with memory modules
            ref_logits = self.reference_model.forward_with_memory(
                x_emb, self.ref_memory1, self.ref_memory2
            )
            
            # Get next token logits from reference
            ref_next_token_logits = ref_logits[:, -1, :]
        
        # Compute KL divergence
        policy_probs = F.softmax(next_token_logits, dim=-1)
        ref_probs = F.softmax(ref_next_token_logits, dim=-1)
        
        # Compute KL: sum(policy_probs * log(policy_probs / ref_probs))
        kl_div = F.kl_div(
            policy_probs.log(), ref_probs,
            reduction='batchmean',
            log_target=False
        )
        
        return kl_div
    
    def train_step(self, prompt, reference_answer, current_episode=0, total_episodes=1):
        """Execute one REINFORCE training step with neural rewards."""
        # Calculate penalty scaling based on episode progress
        penalty_scale = self.calculate_episode_penalty_scaling(current_episode, total_episodes)

        # Ensure models are in training mode
        self.main_model.train()
        self.memory1.train()
        self.memory2.train()
        self.reward_model.eval()  # Keep reward model in eval mode

        # Only zero gradients at the start of accumulation
        if self.accumulated_batches == 0:
            self.optimizer.zero_grad()

        # Prepare input context
        system_prompt = rl_training_params['system_prompt']
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}"
        prompt_bytes = full_prompt.encode('utf-8')

        # Add BOS token to start generation
        prompt_tokens = list(prompt_bytes) + [reward_data_gen_params['bos_token']]
        context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Reference answer tokens
        ref_tokens = list(reference_answer.encode('utf-8'))

        # Storage for generation
        log_probs = []
        kl_penalties = []
        generated_tokens = []

        # Auto-regressive generation with gradient tracking and mixed precision
        max_tokens = min(4096, len(ref_tokens) * 2)  # Reduced for memory

        # Enable mixed precision for forward passes
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu', 
                    enabled=rl_training_params['use_mixed_precision']):
            for _ in range(max_tokens):
                # Clear CUDA cache periodically during generation
                if _ % 50 == 0 and device == "cuda":
                    torch.cuda.empty_cache()

                # Get context within block size limit
                x_cond = context[:, -reward_data_gen_params['block_size']:] if context.size(1) > reward_data_gen_params['block_size'] else context

                # Forward pass through memory1, main model, and memory2
                x_emb = self.main_model.forward_embeddings(x_cond)
                mem_out1 = self.memory1(x_emb)
                logits = self.main_model.forward_with_memory(mem_out1, self.memory1, self.memory2)

                # Get probabilities for next token
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                log_prob_dist = F.log_softmax(next_token_logits, dim=-1)

                # Calculate KL penalty with reference model
                kl_penalty = self.compute_kl_divergence(x_cond, next_token_logits)
                kl_penalties.append(kl_penalty)

                # Sample token
                next_token = torch.multinomial(probs, num_samples=1)
                token_value = next_token.item()

                # Record log probability for policy gradient
                token_log_prob = log_prob_dist.gather(1, next_token).squeeze()
                log_probs.append(token_log_prob)

                # Add token to generated sequence
                generated_tokens.append(token_value)
                context = torch.cat([context, next_token], dim=1)

                # Stop conditions
                if token_value == reward_data_gen_params['eos_token']:
                    break

                # Check for answer end tag
                try:
                    last_tokens = [t for t in context[0, -30:].tolist() if t != 0]
                    recent_text = bytes(last_tokens).decode('utf-8', errors='replace')

                    if reward_data_gen_params['answer_end_tag'] in recent_text:
                        full_text = bytes([t for t in context[0].tolist() if t != 0]).decode('utf-8', errors='replace')
                        if (reward_data_gen_params['thinking_end_tag'] in full_text and
                            reward_data_gen_params['answer_end_tag'] in full_text):
                            break
                except:
                    pass

        # Calculate hybrid rewards with penalty scaling
        rewards = self.compute_hybrid_rewards(generated_tokens, ref_tokens, penalty_scale)

        # Match rewards to log_probs length
        if len(rewards) > len(log_probs):
            rewards = rewards[:len(log_probs)]
        elif len(log_probs) > len(rewards):
            log_probs = log_probs[:len(rewards)]
            kl_penalties = kl_penalties[:len(rewards)]

        # REINFORCE policy gradient loss with mixed precision and KL penalty
        loss_metrics = {"policy_loss": 0.0, "kl_loss": 0.0, "avg_reward": 0.0}

        if len(log_probs) > 0 and len(rewards) > 0:
            # Standard policy gradient loss
            policy_loss = -torch.sum(torch.stack(log_probs) * rewards)
            
            # Add KL penalty
            kl_loss = self.kl_coef * torch.sum(torch.stack(kl_penalties))
            total_loss = policy_loss + kl_loss
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / self.gradient_accumulation_steps

            # Use scaler for mixed precision backpropagation
            self.scaler.scale(total_loss).backward()

            # Record metrics
            loss_metrics["policy_loss"] = policy_loss.item()  # Unscaled for reporting
            loss_metrics["kl_loss"] = kl_loss.item()  # Unscaled for reporting
            loss_metrics["avg_reward"] = rewards.mean().item()

            # Increment accumulated batches
            self.accumulated_batches += 1

            # Update parameters if we've accumulated enough gradients
            if self.accumulated_batches >= self.gradient_accumulation_steps:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.memory1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.memory2.parameters(), 1.0)

                # Update with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Reset accumulation counter
                self.accumulated_batches = 0

        # Clear memory
        if device == "cuda":
            torch.cuda.empty_cache()

        # Return training metrics
        return {
            'policy_loss': loss_metrics["policy_loss"],
            'kl_loss': loss_metrics["kl_loss"],
            'avg_reward': loss_metrics["avg_reward"],
            'generated_length': len(generated_tokens),
            'reference_length': len(ref_tokens),
            'scaler_scale': self.scaler.get_scale(),
            'optimizer_step_taken': self.accumulated_batches == 0,
            'penalty_scale': penalty_scale
        }

    def train(self, train_df, num_prompt_pairs=10, num_episodes=100, log_interval=5, save_interval=20):
        """Run full RL training procedure with neural rewards."""
        # Sample a fixed set of prompt-answer pairs for training
        if len(train_df) < num_prompt_pairs:
            print(f"Warning: Requested {num_prompt_pairs} pairs but dataset only has {len(train_df)} examples")
            selected_indices = list(range(len(train_df)))
        else:
            selected_indices = random.sample(range(len(train_df)), num_prompt_pairs)

        selected_pairs = train_df.iloc[selected_indices]
        print(f"Selected {len(selected_indices)} prompt-answer pairs for RL training")

        # Initialize reference model for KL divergence
        self.initialize_reference_model()

        # Main training loop
        for episode in range(num_episodes):
            # Sample random prompt-answer pair from our selected pairs
            idx = random.randint(0, len(selected_pairs) - 1)
            prompt = selected_pairs.iloc[idx]['prompt']
            reference = selected_pairs.iloc[idx]['response']

            # Print memory stats before training step
            if (episode + 1) % log_interval == 0 and device == "cuda":
                print_memory_stats()

            # Execute training step with episode information
            metrics = self.train_step(prompt, reference,
                                     current_episode=episode,
                                     total_episodes=num_episodes)

            # Store metrics for tracking
            self.metrics_history['episodes'].append(episode + 1)
            self.metrics_history['penalty_scale'].append(metrics['penalty_scale'])
            self.metrics_history['policy_loss'].append(metrics['policy_loss'])
            self.metrics_history['avg_reward'].append(metrics['avg_reward'])

            # Logging
            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{num_episodes}, Metrics: {metrics}")

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                save_path = rl_training_params['rl_checkpoint_path'].replace('.pt', f'_ep{episode+1}.pt')
                torch.save({
                    'main_model_state': self.main_model.state_dict(),
                    'memory1_state': self.memory1.state_dict(),
                    'memory2_state': self.memory2.state_dict(),
                    'episode': episode + 1,
                    'scaler': self.scaler.state_dict(),
                    'metrics_history': self.metrics_history
                }, save_path)
                print(f"Checkpoint saved to {save_path}")

                # Update reference model periodically
                if (episode + 1) % (save_interval * 2) == 0:
                    print("Updating reference model...")
                    self.update_reference_model()

                # Force cleanup after checkpoint
                clear_memory()
    
    def update_reference_model(self):
        """Update the reference model with current policy weights."""
        if self.reference_model is not None:
            self.reference_model.load_state_dict(self.main_model.state_dict())
            self.ref_memory1.load_state_dict(
                {k: v for k, v in self.memory1.state_dict().items() 
                 if k in self.ref_memory1.state_dict()}
            )
            self.ref_memory2.load_state_dict(
                {k: v for k, v in self.memory2.state_dict().items()
                 if k in self.ref_memory2.state_dict()}
            )
            print("Reference model updated with current policy weights")

# ==========================================
# Evaluation Function
# ==========================================
def evaluate_model(main_model, memory1, memory2, test_df, reward_model=None, num_examples=5):
    """Evaluate the model on test examples."""
    print(f"Evaluating model on {num_examples} test examples...")
    
    # Select examples to test
    test_examples = test_df.sample(min(num_examples, len(test_df)))
    
    results = []
    for i, (_, example) in enumerate(test_examples.iterrows()):
        prompt = example['prompt']
        reference = example['response']
        
        print(f"\n--- Test Example {i+1} ---")
        print(f"Prompt: {prompt[:100]}...")
        
        # Generate response
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if device == "cuda":
            start_time.record()
            
        generated_bytes = generate_from_prompt(
            main_model, memory1, memory2,
            prompt_text=prompt,
            max_new_tokens=512
        )
        
        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            generation_time = 0
        
        # Process generated text
        try:
            generated_text = generated_bytes.decode('utf-8', errors='replace')
            print(f"\nGenerated Response (truncated): {generated_text[:500]}...")
            
            # Find tags in response
            thinking_start = generated_text.find(reward_data_gen_params['thinking_tag'])
            thinking_end = generated_text.find(reward_data_gen_params['thinking_end_tag'])
            answer_start = generated_text.find(reward_data_gen_params['answer_tag'])
            answer_end = generated_text.find(reward_data_gen_params['answer_end_tag'])
            
            # Compute rewards if reward model is available
            reward_scores = None
            if reward_model is not None:
                gen_tokens = list(generated_bytes)
                ref_tokens = list(reference.encode('utf-8'))
                
                # Create tensors
                gen_tensor = torch.tensor([gen_tokens], dtype=torch.long, device=device)
                ref_tensor = torch.tensor([ref_tokens], dtype=torch.long, device=device)
                
                # Get reward predictions
                with torch.no_grad():
                    _, global_reward = reward_model(gen_tensor, ref_tensor, 2.0)
                    reward_scores = global_reward.item()
            
            # Store results
            result = {
                'prompt': prompt,
                'reference': reference,
                'generated': generated_text,
                'has_thinking': thinking_start >= 0 and thinking_end > thinking_start,
                'has_answer': answer_start >= 0 and answer_end > answer_start,
                'generation_time': generation_time,
                'reward_score': reward_scores
            }
            results.append(result)
            
            # Print analysis
            if thinking_start >= 0 and thinking_end > thinking_start:
                print("\nThinking Process:")
                print(generated_text[thinking_start:min(thinking_end + len(reward_data_gen_params['thinking_end_tag']), 
                                                     thinking_start + 300)] + "...")
            
            if answer_start >= 0 and answer_end > answer_start:
                print("\nFinal Answer:")
                print(generated_text[answer_start:min(answer_end + len(reward_data_gen_params['answer_end_tag']),
                                                   answer_start + 300)] + "...")
            
            if reward_scores is not None:
                print(f"\nReward Score: {reward_scores:.4f}")
            
        except Exception as e:
            print(f"Error processing response: {e}")
    
    # Compute aggregate metrics
    if results:
        thinking_rate = sum(1 for r in results if r['has_thinking']) / len(results)
        answer_rate = sum(1 for r in results if r['has_answer']) / len(results)
        avg_generation_time = sum(r['generation_time'] for r in results) / len(results)
        
        print("\n=== Evaluation Summary ===")
        print(f"Thinking section present: {thinking_rate:.2f}")
        print(f"Answer section present: {answer_rate:.2f}")
        print(f"Average generation time: {avg_generation_time:.2f} seconds")
        
        if all(r['reward_score'] is not None for r in results):
            avg_reward = sum(r['reward_score'] for r in results) / len(results)
            print(f"Average reward score: {avg_reward:.4f}")
    
    return results

# ==========================================
# Main Training Process
# ==========================================
def main():
    """Main function to run the complete training process."""
    print("Starting End-Only Penalty Reinforcement Learning with Neural Rewards...")
    
    # 1. Load dataset
    train_df, val_df, test_df = load_cot_logic_data()
    
    # 2. Create models
    print("Creating models...")
    main_model = ImprovedByteTransformer(
        vocab_size=reward_data_gen_params['vocab_size'],
        embed_dim=reward_data_gen_params['embed_dim'],
        n_heads=reward_data_gen_params['n_heads'],
        n_layers=reward_data_gen_params['n_layers'],
        block_size=reward_data_gen_params['block_size']
    ).to(device)
    
    memory1 = MemoryModule(
        embed_dim=reward_data_gen_params['embed_dim'],
        n_layers=reward_data_gen_params['memory_n_layers'],
        expansion_factor=4
    ).to(device)
    
    memory2 = MemoryModule(
        embed_dim=reward_data_gen_params['embed_dim'],
        n_layers=reward_data_gen_params['memory_n_layers'],
        expansion_factor=4
    ).to(device)
    
    reward_model = NeuralRewardModel(
        vocab_size=reward_model_params['vocab_size'],
        embed_dim=reward_model_params['reward_embed_dim'],
        n_heads=reward_model_params['reward_n_heads'],
        n_layers=reward_model_params['reward_n_layers'],
        block_size=reward_model_params['block_size']
    ).to(device)
    
    # Calculate model sizes
    policy_params = sum(p.numel() for p in main_model.parameters() if p.requires_grad)
    memory_params = sum(p.numel() for p in memory1.parameters() if p.requires_grad)
    memory_params += sum(p.numel() for p in memory2.parameters() if p.requires_grad)
    reward_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
    
    print(f"Policy model parameters: {policy_params:,}")
    print(f"Memory modules parameters: {memory_params:,}")
    print(f"Reward model parameters: {reward_params:,}")
    print(f"Total parameters: {policy_params + memory_params + reward_params:,}")
    
    # 3. Load pretrained policy model if available
    if os.path.exists(reward_data_gen_params['pretrained_model_path']):
        print(f"Loading pretrained model from {reward_data_gen_params['pretrained_model_path']}...")
        try:
            checkpoint = torch.load(reward_data_gen_params['pretrained_model_path'], map_location=device)
            main_model.load_state_dict(checkpoint['main_model_state'], strict=False)
            memory1.load_state_dict(checkpoint['memory1_state'])
            if 'memory2_state' in checkpoint:
                memory2.load_state_dict(checkpoint['memory2_state'])
            print("Pretrained model loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Starting with randomly initialized weights")
    else:
        print("No pretrained model found. Starting with randomly initialized weights")
    
    # Check if reward model is already available
    reward_model_available = os.path.exists(reward_model_params['reward_model_path'])
    
    if reward_model_available:
        print(f"\nFound existing reward model at {reward_model_params['reward_model_path']}")
        try:
            # Load the reward model
            reward_model.load_state_dict(torch.load(reward_model_params['reward_model_path'], map_location=device))
            print("Reward model loaded successfully")
            
            # Skip phases 1 and 2
            print("Skipping Phases 1 and 2 since reward model is already available")
        except Exception as e:
            print(f"Error loading reward model: {e}")
            reward_model_available = False
            print("Will generate and train reward model from scratch")
    
    if not reward_model_available:
        # ==========================================
        # PHASE 1: Generate Reward Model Training Data
        # ==========================================
        print("\n========== PHASE 1: REWARD MODEL DATA GENERATION ==========")
        reward_training_data = generate_reward_training_data(
            train_df, main_model, memory1, memory2, 
            sample_size=reward_data_gen_params['reward_sample_size']
        )
        
        # Clear memory after Phase 1
        clear_memory()
        print("Memory cleared after Phase 1")
        
        # ==========================================
        # PHASE 2: Train Reward Model
        # ==========================================
        print("\n========== PHASE 2: REWARD MODEL TRAINING ==========")
        reward_model = train_reward_model(
            reward_model, 
            reward_training_data,
            epochs=reward_model_params['reward_train_epochs'],
            batch_size=reward_model_params['reward_batch_size'],
            lr=reward_model_params['reward_learning_rate']
        )
        
        # Save the trained reward model
        torch.save(reward_model.state_dict(), reward_model_params['reward_model_path'])
        print(f"Reward model saved to {reward_model_params['reward_model_path']}")
        
        # Clear memory after Phase 2
        clear_memory()
        print("Memory cleared after Phase 2")
    
    # ==========================================
    # PHASE 3: RL Training with Neural Reward
    # ==========================================
    print("\n========== PHASE 3: RL TRAINING WITH NEURAL REWARD ==========")
    neural_trainer = NeuralRewardTrainer(
        main_model=main_model,
        memory1=memory1,
        memory2=memory2,
        reward_model=reward_model,
        reward_weight=rl_training_params['reward_weight'],
        kl_coef=rl_training_params['kl_coef'],
        base_reward=rl_training_params['base_reward'],
        base_penalty=rl_training_params['base_penalty'],
        learning_rate=rl_training_params['rl_learning_rate'],
        max_penalty_scale=rl_training_params['max_penalty_scale']
    )
    
    # Run RL training with neural rewards
    neural_trainer.train(
        train_df=train_df,
        num_prompt_pairs=rl_training_params['n_prompt_ans_pairs'],
        num_episodes=rl_training_params['number_of_practice'],
        log_interval=rl_training_params['rl_log_interval'],
        save_interval=rl_training_params['rl_save_interval']
    )
    
    # Save final model
    final_model_path = "eop_prl_final_model.pt"
    torch.save({
        'main_model_state': main_model.state_dict(),
        'memory1_state': memory1.state_dict(),
        'memory2_state': memory2.state_dict(),
        'reward_model_state': reward_model.state_dict(),
        'metrics_history': neural_trainer.metrics_history
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate final model
    evaluate_model(main_model, memory1, memory2, test_df, reward_model, num_examples=5)
    
    # Clear memory after Phase 3
    clear_memory()
    print("Memory cleared after Phase 3")
    
    print("End-Only Penalty Reinforcement Learning with Neural Rewards complete!")

if __name__ == "__main__":
    main()
