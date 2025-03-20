## eop pRL

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from torch.amp import autocast, GradScaler  # For mixed precision
import gc  # For garbage collection
import matplotlib.pyplot as plt

# ==========================================
# 1) Hyperparameters
# ==========================================
hyperparams = {
    # Model Architecture
    'block_size': 1024,               # Sequence length for context
    'batch_size': 1,                  # Batch size (reduced from 2 to save memory)
    'embed_dim': 1024,                # Transformer embedding dimension
    'n_heads': 16,                    # Number of attention heads
    'n_layers': 24,                   # Number of Transformer blocks
    'memory_n_layers': 8,             # Number of layers in the original MemoryModule
    'vocab_size': 256,                # Fixed vocabulary size for byte tokenization

    # Memory Efficiency Settings
    'use_gradient_checkpointing': True,  # Use gradient checkpointing
    'gradient_accumulation_steps': 4,    # Accumulate gradients for X steps
    'chunk_size': 64,                    # Size of chunks for attention calculation
    'use_dynamic_quantization': True,    # Use dynamic quantization
    'limit_attention_memory': True,      # Use memory-efficient attention implementation

    # RL Training Parameters
    'n_prompt_ans_pairs': 5,          # Number of prompt-answer pairs to use for RL training
    'number_of_practice': 100,        # Number of practice episodes for RL training
    'rl_log_interval': 5,             # Log metrics every X episodes during RL training
    'rl_save_interval': 20,           # Save checkpoint every X episodes during RL training
    'base_reward': 1.0,               # Base reward value for correct predictions
    'base_penalty': -0.5,             # Base penalty value for incorrect predictions
    'rl_learning_rate': 1e-6,         # Learning rate for RL fine-tuning
    'max_penalty_scale': 2.5,         # Maximum penalty scaling factor for episode progression

    # Mixed Precision Parameters
    'use_mixed_precision': True,      # Whether to use mixed precision training
    'grad_scale_init': 65536.0,       # Initial scale for gradient scaler
    'scale_growth_interval': 2000,    # Steps between gradient scaler growth

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

    # File Paths
    'pretrained_model_path': "threshold_transformer_checkpoint.pt",  # Path to load pretrained model
    'rl_checkpoint_path': "rl_transformer_checkpoint.pt",        # RL checkpoint path

    # System Prompt
    'system_prompt': """just think before answer."""
}

# ==========================================
# 1.1) Select device and optimize settings
# ==========================================
device = "mps" if torch.backends.mps.is_available() else \
         ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable tensor cores for better performance with mixed precision
if device == "cuda" and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled for better performance")

    # Set up GPU for maximum memory efficiency
    torch.cuda.empty_cache()
    print(f"CUDA memory allocated before starting: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"CUDA memory reserved before starting: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# ==========================================
# 1.2) Memory Management Functions
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
# 1.3) Data Loading and Preprocessing for COT Logic Reasoning
# ==========================================
def load_cot_logic_data():
    print("Loading COT Logic Reasoning dataset...")

    try:
        # Try standard pandas read_parquet first
        df = pd.read_parquet("isaiahbjork/cot-logic-reasoning/cot-logic-reasoning.parquet")
        print("Dataset loaded using standard path")
    except Exception as e:
        print(f"Error loading dataset with standard path: {e}")
        try:
            # Try with datasets library if available
            try:
                from datasets import load_dataset
                dataset = load_dataset("isaiahbjork/cot-logic-reasoning")
                df = dataset["train"].to_pandas()
                print("Dataset loaded using datasets library")
            except:
                # If all else fails, use the original path format
                df = pd.read_parquet("hf://datasets/isaiahbjork/cot-logic-reasoning/cot-logic-reasoning.parquet")
                print("Dataset loaded using hf:// protocol")
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            raise RuntimeError("Unable to load the COT Logic Reasoning dataset")

    print(f"Data size: {len(df)}")

    # Split into train/validation/test sets (80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Test examples: {len(test_df)}")

    return train_df, val_df, test_df

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
# 3) Memory-Efficient Attention Mechanism
# ==========================================
class MemoryEfficientAttention(nn.Module):
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

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        # Project to queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, T, D

        # Super memory-efficient attention implementation
        # Process in small chunks for both query and key sequences
        chunk_size = hyperparams['chunk_size']
        attn_output = torch.zeros_like(q)

        for i in range(0, T, chunk_size):
            i_end = min(i + chunk_size, T)

            # Get current query chunk
            q_chunk = q[:, :, i:i_end]

            # Compute scores for this chunk against all keys, in smaller sub-chunks
            scores_for_chunk = []
            for j in range(0, T, chunk_size):
                j_end = min(j + chunk_size, T)

                # Get key chunk and compute scores
                k_chunk = k[:, :, j:j_end]
                chunk_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.attn_scale

                # Apply causal mask if needed - only allow attention to previous positions
                if attn_mask is not None and i >= j:
                    # Generate mask just for this chunk
                    mask_size = (i_end-i, j_end-j)
                    chunk_mask = torch.triu(torch.ones(mask_size, device=x.device), diagonal=j-i+1).bool()
                    chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, -1, -1)
                    chunk_scores.masked_fill_(chunk_mask, float('-inf'))

                scores_for_chunk.append(chunk_scores)

            # Concatenate all key chunks for this query chunk
            all_scores_for_chunk = torch.cat(scores_for_chunk, dim=-1)

            # Apply softmax across the full key dimension
            attn_weights = F.softmax(all_scores_for_chunk, dim=-1)

            # Multiply with values in chunks
            chunk_output = torch.zeros_like(q_chunk)
            start_idx = 0
            for j in range(0, T, chunk_size):
                j_end = min(j + chunk_size, T)

                # Get weights for this chunk and the corresponding values
                weights_chunk = attn_weights[:, :, :, start_idx:start_idx + (j_end - j)]
                v_chunk = v[:, :, j:j_end]

                # Accumulate the output for this chunk
                chunk_output += torch.matmul(weights_chunk, v_chunk)
                start_idx += (j_end - j)

            # Place the output for this query chunk in the right position
            attn_output[:, :, i:i_end] = chunk_output

        # Reshape output back to original dimensions
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
# 4) Improved Transformer Block with Memory Efficiency
# ==========================================
class ImprovedTransformerBlock(nn.Module):
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
        # Use sequential processing to reduce memory usage
        attn_out = self.attention(x)
        x = x + self.threshold1(attn_out)

        # Explicitly delete to free memory
        del attn_out

        ff_out = self.feed_forward(x)
        x = x + self.threshold2(ff_out)

        # Explicitly delete to free memory
        del ff_out

        return x

# ==========================================
# 5) Improved Byte Transformer with Gradient Checkpointing
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
        self.use_checkpointing = hyperparams['use_gradient_checkpointing']

    def forward_with_embeddings(self, x_emb):
        for i, block in enumerate(self.blocks):
            if self.use_checkpointing and self.training:
                # Ensure tensor requires gradients for checkpointing
                if not x_emb.requires_grad:
                    x_emb.requires_grad = True
                x_emb = torch.utils.checkpoint.checkpoint(block, x_emb, use_reentrant=False)
            else:
                x_emb = block(x_emb)
        x_emb = self.final_threshold(x_emb)
        logits = self.ln_f(x_emb)
        return logits

    def forward_with_two_memory(self, x_emb, memory_module2):
        """
        Extended forward pass with memory modules and gradient checkpointing
        """
        transformer_out = x_emb
        for i, block in enumerate(self.blocks):
            if self.use_checkpointing and self.training:
                # Ensure tensor requires gradients for checkpointing
                if not transformer_out.requires_grad:
                    transformer_out.requires_grad = True
                transformer_out = torch.utils.checkpoint.checkpoint(block, transformer_out, use_reentrant=False)
            else:
                transformer_out = block(transformer_out)

        transformer_out = self.final_threshold(transformer_out)

        if self.use_checkpointing and self.training:
            # Ensure tensor requires gradients for checkpointing
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
        B, T = x.size()
        token_emb = self.token_embedding(x)
        positions = torch.arange(min(T, self.block_size), device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x_emb = token_emb[:, :min(T, self.block_size)] + pos_emb
        return self.forward_with_embeddings(x_emb)

# ==========================================
# 6) Memory Module with Gradient Checkpointing
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
        self.use_checkpointing = hyperparams['use_gradient_checkpointing']

    def forward(self, x):
        out = x
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Ensure tensor requires gradients for checkpointing
                if not out.requires_grad:
                    out.requires_grad = True
                residual = torch.utils.checkpoint.checkpoint(layer, out, use_reentrant=False)
                out = out + residual
            else:
                out = out + layer(out)
        out = self.final_norm(out)
        return out

# ==========================================
# 8) Progressive Reward RL Training with Gradient Accumulation and End-Only Penalty
# ==========================================
class ProgressiveRewardTrainer:
    def __init__(self, main_model, memory1, memory2,
                 base_reward=1.0, base_penalty=-0.5,
                 learning_rate=5e-6, max_penalty_scale=2.5):
        self.main_model = main_model
        self.memory1 = memory1
        self.memory2 = memory2
        self.base_reward = base_reward
        self.base_penalty = base_penalty
        self.max_penalty_scale = max_penalty_scale
        self.optimizer = torch.optim.Adam(
            list(main_model.parameters()) +
            list(memory1.parameters()) +
            list(memory2.parameters()),
            lr=learning_rate
        )
        # Create gradient scaler for mixed precision training
        self.scaler = GradScaler(
            init_scale=hyperparams['grad_scale_init'],
            growth_interval=hyperparams['scale_growth_interval'],
            enabled=hyperparams['use_mixed_precision']
        )
        # Gradient accumulation steps
        self.gradient_accumulation_steps = hyperparams['gradient_accumulation_steps']
        # Track accumulated batches
        self.accumulated_batches = 0
        # Store metrics for visualization
        self.metrics_history = {
            'episodes': [],
            'penalty_scale': [],
            'policy_loss': [],
            'avg_reward': []
        }

    def calculate_episode_penalty_scaling(self, current_episode, total_episodes):
        """Calculate the penalty scaling factor based on episode progress."""
        # Ensure the scaling starts at 1.0 and linearly increases to max_scale
        scale = 1.0 + (current_episode / max(1, total_episodes - 1)) * (self.max_penalty_scale - 1.0)
        return scale

    def compute_progressive_rewards(self, generated_tokens, reference_tokens, penalty_scale=1.0):
        """
        Compute rewards that only penalize for incomplete completions at the end.
        Matching tokens still receive rewards, but non-matching tokens don't get penalties.
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
            
            # Calculate the severity of incompleteness (higher if more is missing)
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

    def train_step(self, prompt, reference_answer, current_episode=0, total_episodes=1):
        """Execute one REINFORCE training step with progressive rewards and gradient accumulation."""
        # Calculate penalty scaling based on episode progress
        penalty_scale = self.calculate_episode_penalty_scaling(current_episode, total_episodes)

        # Ensure models are in training mode
        self.main_model.train()
        self.memory1.train()
        self.memory2.train()

        # Only zero gradients at the start of accumulation
        if self.accumulated_batches == 0:
            self.optimizer.zero_grad()

        # Prepare input context
        system_prompt = hyperparams['system_prompt']
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}"
        prompt_bytes = full_prompt.encode('utf-8')

        # Add BOS token to start generation
        prompt_tokens = list(prompt_bytes) + [hyperparams['bos_token']]
        context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Reference answer tokens
        ref_tokens = list(reference_answer.encode('utf-8'))

        # Storage for generation
        log_probs = []
        generated_tokens = []

        # Auto-regressive generation with gradient tracking and mixed precision
        # Using a smaller max_tokens to save memory
        max_tokens = min(4096, len(ref_tokens) * 2)  # Reduced from 512

        # Enable mixed precision for forward passes
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=hyperparams['use_mixed_precision']):
            for _ in range(max_tokens):
                # Clear CUDA cache periodically during generation
                if _ % 50 == 0 and device == "cuda":
                    torch.cuda.empty_cache()

                # Get context within block size limit
                x_cond = context[:, -hyperparams['block_size']:] if context.size(1) > hyperparams['block_size'] else context

                # Get embeddings
                B, T = x_cond.shape
                token_emb = self.main_model.token_embedding(x_cond)

                # Handle the case where T > block_size
                effective_T = min(T, self.main_model.block_size)
                pos_indices = torch.arange(effective_T, device=x_cond.device).unsqueeze(0)
                pos_emb = self.main_model.pos_embedding(pos_indices)

                combined_emb = token_emb[:, :effective_T] + pos_emb

                # Forward pass through model
                mem_out1 = self.memory1(combined_emb)
                logits = self.main_model.forward_with_two_memory(mem_out1, self.memory2)

                # Get probabilities for next token
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                log_prob_dist = F.log_softmax(next_token_logits, dim=-1)

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
                if token_value == hyperparams['eos_token']:
                    break

                # Check for answer end tag
                try:
                    last_tokens = [t for t in context[0, -30:].tolist() if t != 0]
                    recent_text = bytes(last_tokens).decode('utf-8', errors='replace')

                    if hyperparams['answer_end_tag'] in recent_text:
                        full_text = bytes([t for t in context[0].tolist() if t != 0]).decode('utf-8', errors='replace')
                        if (hyperparams['thinking_end_tag'] in full_text and
                            hyperparams['answer_end_tag'] in full_text):
                            break
                except:
                    pass

        # Calculate progressive rewards with penalty scaling
        rewards = self.compute_progressive_rewards(generated_tokens, ref_tokens, penalty_scale)

        # Match rewards to log_probs length
        if len(rewards) > len(log_probs):
            rewards = rewards[:len(log_probs)]
        elif len(log_probs) > len(rewards):
            log_probs = log_probs[:len(rewards)]

        # REINFORCE policy gradient loss with mixed precision handling
        loss_metrics = {"policy_loss": 0.0, "avg_reward": 0.0}

        if len(log_probs) > 0 and len(rewards) > 0:
            # Use full precision for loss calculation
            policy_loss = -torch.sum(torch.stack(log_probs) * rewards)

            # Scale loss for gradient accumulation
            policy_loss = policy_loss / self.gradient_accumulation_steps

            # Use scaler for mixed precision backpropagation
            self.scaler.scale(policy_loss).backward()

            # Record metrics
            loss_metrics["policy_loss"] = policy_loss.item() * self.gradient_accumulation_steps  # Unscale for reporting
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

        return {
            'policy_loss': loss_metrics["policy_loss"],
            'avg_reward': loss_metrics["avg_reward"],
            'generated_length': len(generated_tokens),
            'reference_length': len(ref_tokens),
            'scaler_scale': self.scaler.get_scale(),
            'optimizer_step_taken': self.accumulated_batches == 0,  # True if we just took an optimizer step
            'penalty_scale': penalty_scale  # Track the penalty scaling for monitoring
        }

    def train(self, train_df, num_prompt_pairs=10, num_episodes=100, log_interval=5, save_interval=20):
        """Run full RL training procedure."""
        # Sample a fixed set of prompt-answer pairs for training
        if len(train_df) < num_prompt_pairs:
            print(f"Warning: Requested {num_prompt_pairs} pairs but dataset only has {len(train_df)} examples")
            selected_indices = list(range(len(train_df)))
        else:
            selected_indices = random.sample(range(len(train_df)), num_prompt_pairs)

        selected_pairs = train_df.iloc[selected_indices]
        print(f"Selected {len(selected_indices)} prompt-answer pairs for RL training")

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

            # Store metrics for visualization
            self.metrics_history['episodes'].append(episode + 1)
            self.metrics_history['penalty_scale'].append(metrics['penalty_scale'])
            self.metrics_history['policy_loss'].append(metrics['policy_loss'])
            self.metrics_history['avg_reward'].append(metrics['avg_reward'])

            # Logging
            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{num_episodes}, Metrics: {metrics}")

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                save_path = hyperparams['rl_checkpoint_path'].replace('.pt', f'_ep{episode+1}.pt')
                torch.save({
                    'main_model_state': self.main_model.state_dict(),
                    'memory1_state': self.memory1.state_dict(),
                    'memory2_state': self.memory2.state_dict(),
                    'episode': episode + 1,
                    'scaler': self.scaler.state_dict(),  # Save scaler state
                    'metrics_history': self.metrics_history  # Save metrics for visualization
                }, save_path)
                print(f"Checkpoint saved to {save_path}")

                # Visualize penalty effect after saving checkpoint
                if episode + 1 >= log_interval:
                    self.visualize_penalty_effect()

                # Force cleanup after checkpoint
                clear_memory()

    def visualize_penalty_effect(self):
        """Generate plots to visualize the effect of the progressive penalty scaling."""
        # Only create visualization if we have enough data points
        if len(self.metrics_history['episodes']) < 2:
            return

        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Plot 1: Penalty Scale over episodes
        ax1.plot(self.metrics_history['episodes'], self.metrics_history['penalty_scale'],
                marker='o', linestyle='-', color='red')
        ax1.set_ylabel('Penalty Scale')
        ax1.set_title('Progressive Penalty Scaling over Episodes')
        ax1.grid(True)

        # Plot 2: Policy Loss over episodes
        ax2.plot(self.metrics_history['episodes'], self.metrics_history['policy_loss'],
                marker='x', linestyle='-', color='blue')
        ax2.set_ylabel('Policy Loss')
        ax2.set_title('Policy Loss over Episodes')
        ax2.grid(True)

        # Plot 3: Average Reward over episodes
        ax3.plot(self.metrics_history['episodes'], self.metrics_history['avg_reward'],
                marker='s', linestyle='-', color='green')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Reward')
        ax3.set_title('Average Reward over Episodes')
        ax3.grid(True)

        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(f"penalty_effect_visualization_ep{max(self.metrics_history['episodes'])}.png")
        plt.close(fig)
        print(f"Visualization saved to penalty_effect_visualization_ep{max(self.metrics_history['episodes'])}.png")

# ==========================================
# 7) Generate Text from Trained Model with Dynamic Quantization
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

    # Only quantize for CPU, not for CUDA (to fix the error)
    if hyperparams['use_dynamic_quantization'] and device != "cuda":
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
        # If on CUDA, dynamic quantization is not supported, so use original models
        if hyperparams['use_dynamic_quantization'] and device == "cuda":
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
    bos_token = torch.tensor([[hyperparams['bos_token']]], dtype=torch.long, device=device)
    context = torch.cat([context, bos_token], dim=1)

    generated = []
    eos_found = False

    # Generate with reduced batch size and in smaller chunks for memory efficiency
    for _ in range(max_new_tokens):
        if eos_found:
            break

        # Only use the last block_size tokens for context to save memory
        x_cond = context[:, -hyperparams['block_size']:] if context.size(1) > hyperparams['block_size'] else context
        B, T = x_cond.shape
        token_emb = use_main_model.token_embedding(x_cond)

        # Handle the case where T > block_size
        effective_T = min(T, use_main_model.block_size)
        pos_indices = torch.arange(effective_T, device=x_cond.device).unsqueeze(0)
        pos_emb = use_main_model.pos_embedding(pos_indices)

        combined_emb = token_emb[:, :effective_T] + pos_emb

        # Forward pass with memory modules
        mem_out1 = use_memory1(combined_emb)
        logits = use_main_model.forward_with_two_memory(mem_out1, use_memory2)

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

        # Free some memory periodically
        if _ % 50 == 0 and device == "cuda":
            torch.cuda.empty_cache()

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
# 9) Main RL Training Function
# ==========================================
def train_with_progressive_rewards():
    """Main function to run RL training with progressive rewards."""
    # Create models with original architecture
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

    # Set all models to training mode explicitly
    main_model.train()
    memory1.train()
    memory2.train()

    # Calculate model size
    num_params = sum(p.numel() for p in main_model.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in memory1.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in memory2.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # Load pretrained model if available
    if os.path.exists(hyperparams['pretrained_model_path']):
        print(f"Loading pretrained model from {hyperparams['pretrained_model_path']}...")
        try:
            checkpoint = torch.load(hyperparams['pretrained_model_path'], map_location=device)
            main_model.load_state_dict(checkpoint['main_model_state'], strict=False)
            memory1.load_state_dict(checkpoint['memory1_state'])
            if 'memory2_state' in checkpoint:
                memory2.load_state_dict(checkpoint['memory2_state'])
            print("Pretrained model loaded successfully.")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Starting with randomly initialized weights.")
    else:
        print(f"Warning: No pretrained model found at {hyperparams['pretrained_model_path']}.")
        print("Starting with randomly initialized weights.")

    # Enabled gradient checkpointing if requested (saves memory during training)
    if hyperparams['use_gradient_checkpointing']:
        print("Gradient checkpointing enabled for memory efficiency")

    # Load dataset
    train_df, val_df, _ = load_cot_logic_data()

    # Create RL trainer with the max penalty scale
    trainer = ProgressiveRewardTrainer(
        main_model=main_model,
        memory1=memory1,
        memory2=memory2,
        base_reward=hyperparams['base_reward'],
        base_penalty=hyperparams['base_penalty'],
        learning_rate=hyperparams['rl_learning_rate'],
        max_penalty_scale=hyperparams['max_penalty_scale']
    )

    # Run training
    trainer.train(
        train_df=train_df,
        num_prompt_pairs=hyperparams['n_prompt_ans_pairs'],
        num_episodes=hyperparams['number_of_practice'],
        log_interval=hyperparams['rl_log_interval'],
        save_interval=hyperparams['rl_save_interval']
    )

    # Save final model
    torch.save({
        'main_model_state': main_model.state_dict(),
        'memory1_state': memory1.state_dict(),
        'memory2_state': memory2.state_dict(),
        'metrics_history': trainer.metrics_history
    }, "rl_final_model.pt")

    # Generate final visualization
    trainer.visualize_penalty_effect()

    print("RL training complete!")

# ==========================================
# 10) Test RL-trained Model
# ==========================================
def test_rl_model(model_path="rl_final_model.pt"):
    """Test the RL-trained model on a few examples."""
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

    # Load trained model
    if os.path.exists(model_path):
        print(f"Loading RL-trained model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        main_model.load_state_dict(checkpoint['main_model_state'], strict=False)
        memory1.load_state_dict(checkpoint['memory1_state'])
        memory2.load_state_dict(checkpoint['memory2_state'])

        # If metrics history is available, visualize it
        if 'metrics_history' in checkpoint:
            visualize_saved_metrics(checkpoint['metrics_history'])
    else:
        print(f"Model path {model_path} not found. Exiting test.")
        return

    # Load dataset
    _, _, test_df = load_cot_logic_data()

    # Select a few examples to test
    test_examples = test_df.sample(3)

    for i, (_, example) in enumerate(test_examples.iterrows()):
        prompt = example['prompt']
        reference = example['response']

        print(f"\n--- Test Example {i+1} ---")
        print(f"Prompt: {prompt[:100]}...")

        # Generate response
        generated_bytes = generate_from_prompt(
            main_model, memory1, memory2,
            prompt_text=prompt,
            max_new_tokens=512
        )

        try:
            generated_text = generated_bytes.decode('utf-8', errors='replace')
            print(f"\nGenerated Response: {generated_text[:2000]}...")

            # Find tags in response
            thinking_start = generated_text.find(hyperparams['thinking_tag'])
            thinking_end = generated_text.find(hyperparams['thinking_end_tag'])
            answer_start = generated_text.find(hyperparams['answer_tag'])
            answer_end = generated_text.find(hyperparams['answer_end_tag'])

            if thinking_start >= 0 and thinking_end > thinking_start:
                print("\nThinking Process:")
                print(generated_text[thinking_start:thinking_end + len(hyperparams['thinking_end_tag'])])

            if answer_start >= 0 and answer_end > answer_start:
                print("\nFinal Answer:")
                print(generated_text[answer_start:answer_end + len(hyperparams['answer_end_tag'])])

        except Exception as e:
            print(f"Error decoding response: {e}")

def visualize_saved_metrics(metrics_history):
    """Visualize metrics from a saved checkpoint."""
    if not metrics_history or len(metrics_history['episodes']) < 2:
        print("Not enough data to visualize metrics")
        return

    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Plot 1: Penalty Scale over episodes
    ax1.plot(metrics_history['episodes'], metrics_history['penalty_scale'],
            marker='o', linestyle='-', color='red')
    ax1.set_ylabel('Penalty Scale')
    ax1.set_title('Progressive Penalty Scaling over Episodes')
    ax1.grid(True)

    # Plot 2: Policy Loss over episodes
    ax2.plot(metrics_history['episodes'], metrics_history['policy_loss'],
            marker='x', linestyle='-', color='blue')
    ax2.set_ylabel('Policy Loss')
    ax2.set_title('Policy Loss over Episodes')
    ax2.grid(True)

    # Plot 3: Average Reward over episodes
    ax3.plot(metrics_history['episodes'], metrics_history['avg_reward'],
            marker='s', linestyle='-', color='green')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Average Reward over Episodes')
    ax3.grid(True)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig("loaded_model_metrics_visualization.png")
    plt.close(fig)
    print("Visualization of loaded model metrics saved to loaded_model_metrics_visualization.png")

# ==========================================
# 11) Main Entry Point
# ==========================================
if __name__ == "__main__":
    print("Starting Progressive Reward RL Training with Memory Optimizations...")
    print(f"Gradient checkpointing enabled: {hyperparams['use_gradient_checkpointing']}")
    print(f"Gradient accumulation steps: {hyperparams['gradient_accumulation_steps']}")
    print(f"Batch size: {hyperparams['batch_size']}")
    print(f"Using dynamic quantization: {hyperparams['use_dynamic_quantization']}")
    print(f"Maximum penalty scale: {hyperparams['max_penalty_scale']}")

    # Clear CUDA memory before starting
    if device == "cuda":
        torch.cuda.empty_cache()
        print_memory_stats()

    try:
        print("\nRunning RL training...")
        train_with_progressive_rewards()

        print("\nTesting RL-trained model...")
        test_rl_model()

        print("\nTraining complete!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
