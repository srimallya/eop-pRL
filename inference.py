## inference V2

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1) Define Hyperparameters for Inference
# ==========================================
hyperparams = {
    'block_size': 1024,               # Sequence length for context
    'embed_dim': 1024,                # Transformer embedding dimension
    'n_heads': 16,                    # Number of attention heads
    'n_layers': 24,                   # Number of Transformer blocks
    'memory_n_layers': 8,             # Number of layers in the Memory modules
    'vocab_size': 256,                # Fixed vocabulary size for byte tokenization
    'bos_token': 254,                 # Beginning-of-sequence token (byte value)
    'eos_token': 255,                 # End-of-sequence token (byte value)
    'checkpoint_path': "threshold_transformer_checkpoint.pt", # Path to model checkpoint
    'top_p': 0.6,                     # Top-p sampling parameter (0-1)
    'system_prompt': """IMPORTANT: Your response format should have two parts:
    1. First, explain your thinking process in detail between <think> </think> tags.
    2. Then, provide your final answer between <answer> </answer> tags.
    For example: <think> Let me think about this problem carefully...
    [detailed reasoning process] </think> <answer> [concise answer] </answer> """
}

# ==========================================
# 2) Select device for inference
# ==========================================
device = "mps" if torch.backends.mps.is_available() else \
         ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 3) NEW: Thresholded Attention Implementation
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

        # Apply thresholding to attention scores (only in training)
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
# 4) Model Architecture with Thresholded Attention
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

class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        # Use ThresholdedAttention instead of nn.MultiheadAttention
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
# 5) Model Conversion Function (for compatibility)
# ==========================================
def convert_original_model_to_thresholded(original_checkpoint_path, device_type=device):
    """
    Function to convert a standard model checkpoint to use the thresholded attention.
    This is a fallback if loading directly fails.
    """
    print(f"Converting original model to thresholded version: {original_checkpoint_path}")

    # Load original checkpoint
    checkpoint = torch.load(original_checkpoint_path, map_location=device_type)

    # Create new models
    new_main_model = ImprovedByteTransformer(
        vocab_size=hyperparams['vocab_size'],
        embed_dim=hyperparams['embed_dim'],
        n_heads=hyperparams['n_heads'],
        n_layers=hyperparams['n_layers'],
        block_size=hyperparams['block_size']
    ).to(device_type)

    new_memory1 = MemoryModule(
        embed_dim=hyperparams['embed_dim'],
        n_layers=hyperparams['memory_n_layers'],
        expansion_factor=4
    ).to(device_type)

    new_memory2 = MemoryModule(
        embed_dim=hyperparams['embed_dim'],
        n_layers=hyperparams['memory_n_layers'],
        expansion_factor=4
    ).to(device_type)

    # Load non-attention parts directly
    # Embeddings, Layer Norms, Feed Forward, and Memory Modules should have identical keys
    state_dict = checkpoint['main_model_state']
    new_state_dict = {}

    # Copy all parts that can be directly copied
    for k, v in state_dict.items():
        if 'attention' not in k:
            new_state_dict[k] = v

    # Load memory modules directly
    new_memory1.load_state_dict(checkpoint['memory1_state'])
    if 'memory2_state' in checkpoint:
        new_memory2.load_state_dict(checkpoint['memory2_state'])

    # Process attention parts
    for layer_idx in range(hyperparams['n_layers']):
        prefix = f"blocks.{layer_idx}.attention."

        # Handle MultiheadAttention parameters specially
        if f"{prefix}in_proj_weight" in state_dict:
            in_proj_weight = state_dict[f"{prefix}in_proj_weight"]
            in_proj_bias = state_dict.get(f"{prefix}in_proj_bias", None)

            # Split the weights/biases for q, k, v
            embed_dim = hyperparams['embed_dim']
            q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)

            # Set in new state dict
            new_state_dict[f"{prefix}q_proj.weight"] = q_weight
            new_state_dict[f"{prefix}k_proj.weight"] = k_weight
            new_state_dict[f"{prefix}v_proj.weight"] = v_weight

            if in_proj_bias is not None:
                q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
                new_state_dict[f"{prefix}q_proj.bias"] = q_bias
                new_state_dict[f"{prefix}k_proj.bias"] = k_bias
                new_state_dict[f"{prefix}v_proj.bias"] = v_bias

        # Copy output projection
        if f"{prefix}out_proj.weight" in state_dict:
            new_state_dict[f"{prefix}out_proj.weight"] = state_dict[f"{prefix}out_proj.weight"]
            if f"{prefix}out_proj.bias" in state_dict:
                new_state_dict[f"{prefix}out_proj.bias"] = state_dict[f"{prefix}out_proj.bias"]

    # Load the modified state dict
    new_main_model.load_state_dict(new_state_dict, strict=False)

    return new_main_model, new_memory1, new_memory2

# ==========================================
# 6) Inference Function with Top-p Sampling
# ==========================================
@torch.no_grad()
def generate_from_prompt(main_model, memory1, memory2, prompt_text=None, max_new_tokens=200, top_p=None):
    if prompt_text is None:
        prompt_text = "Explain why the statement 'I wore my lucky socks today, and I got an A on my test, so my socks must be lucky' is a logical fallacy."

    # Use default top_p if not specified
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

    # Set models to evaluation mode
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
            break  # Immediately break out of the loop when EOS is found

        # Only append non-EOS tokens to the generated sequence
        generated.append(next_token_value)
        context = torch.cat([context, next_token], dim=1)

    # Convert generated tokens to bytes
    result_bytes = bytes(generated)

    # Clean up special tokens when returning result
    try:
        # Convert to string
        response_str = result_bytes.decode('utf-8')
        return response_str
    except:
        # If decoding fails, return a message
        return "Error: Unable to decode generated response."

# ==========================================
# 7) Main Inference Function
# ==========================================
def inference(prompt_text, max_tokens=512, top_p=None):
    """Generate a response for a given prompt using the pre-trained model with thresholded attention"""
    # Use default top_p if not specified
    if top_p is None:
        top_p = hyperparams['top_p']

    # Initialize models
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

    # Check if thresholded checkpoint exists
    thresholded_checkpoint_path = hyperparams['checkpoint_path'].replace('.pt', '_thresholded.pt')
    if os.path.exists(thresholded_checkpoint_path):
        checkpoint_path = thresholded_checkpoint_path
        print(f"Using thresholded model checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = hyperparams['checkpoint_path']
        print(f"Thresholded checkpoint not found, using original: {checkpoint_path}")

    # Load pre-trained model weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Try to load model states directly
        try:
            # Load main model state
            main_model.load_state_dict(checkpoint['main_model_state'], strict=False)
            print("Main model loaded with some parameters ignored (normal for model conversion)")

            # Load memory modules
            memory1.load_state_dict(checkpoint['memory1_state'])
            if 'memory2_state' in checkpoint:
                memory2.load_state_dict(checkpoint['memory2_state'])

            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint directly: {e}")
            print("Trying model conversion approach...")

            if checkpoint_path == hyperparams['checkpoint_path']:
                # If direct loading fails with original checkpoint, try conversion
                main_model, memory1, memory2 = convert_original_model_to_thresholded(checkpoint_path)
                print("Model converted successfully")
            else:
                # If even the thresholded checkpoint fails, something is wrong
                raise Exception("Failed to load both regular and thresholded checkpoints")

    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        raise Exception(f"Error loading checkpoint: {e}")

    # Generate response with top-p sampling
    response = generate_from_prompt(
        main_model, memory1, memory2,
        prompt_text=prompt_text,
        max_new_tokens=max_tokens,
        top_p=top_p
    )

    return response

# ==========================================
# 8) Example Usage
# ==========================================
if __name__ == "__main__":
    import os
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate responses using thresholded attention model')
    parser.add_argument('--prompt', type=str, default="Why sea is salty?",
                        help='Input prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--top_p', type=float, default=hyperparams['top_p'],
                        help='Top-p sampling parameter (0-1)')
    parser.add_argument('--checkpoint', type=str, default=hyperparams['checkpoint_path'],
                        help='Path to model checkpoint')

    # Check if we're in a notebook environment
    in_notebook = False
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        pass

    if in_notebook:
        # Running in notebook, use default parameters
        test_prompt = "Why sea is salty?"
        max_tokens = 2048
        top_p = hyperparams['top_p']
        hyperparams['checkpoint_path'] = hyperparams['checkpoint_path']  # Keep default
    else:
        # Running as script, parse arguments
        args = parser.parse_args()
        test_prompt = args.prompt
        max_tokens = args.max_tokens
        top_p = args.top_p
        hyperparams['checkpoint_path'] = args.checkpoint

    print(f"Input prompt: {test_prompt}")
    print(f"Using top-p: {top_p}")
    print("\nGenerating response...")

    try:
        response = inference(test_prompt, max_tokens=max_tokens, top_p=top_p)
        print(f"\nResponse:\n{response}")
    except Exception as e:
        print(f"Error during inference: {e}")
