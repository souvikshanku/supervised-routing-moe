
import os

import glob
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 3072,                 # Embedding dimension
    "n_heads": 24,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "n_experts": 3                   # Number of experts
}

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx



class Llama3MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.layers = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx, expert_map):
        tok_embeds = self.embed_tokens(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        kl_loss = 0

        for block in self.layers:
            x, block_kl_loss = block(x, mask, self.cos, self.sin, expert_map)
            kl_loss += block_kl_loss

        x = self.norm(x)
        logits = self.lm_head(x.to(self.cfg["dtype"]))
        return logits, kl_loss / self.cfg['n_layers']


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        # self.mlp = FeedForward(cfg)
        self.moe = MoE(cfg)
        self.input_layernorm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.post_attention_layernorm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin, expert_map):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.post_attention_layernorm(x)
        # x = self.mlp(x)
        x, kl_loss = self.moe(x, expert_map)
        x = x + shortcut  # Add the original input back

        return x, kl_loss


class ExpertFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.router = nn.Linear(cfg["emb_dim"], cfg["n_experts"], dtype=cfg["dtype"], bias=False)
        self.experts = nn.ModuleList([ExpertFFN(cfg) for _ in range(cfg["n_experts"])])

    def forward(self, x, expert_map, eps: float = 1e-9):
        b, sl, emb_dim = x.shape
        device = x.device

        logits = self.router(x)
        r = torch.softmax(logits, dim=-1)               # (b, sl, E), fp32
        outs = [expert(x) for expert in self.experts]   # list of (b, sl, emb_dim)
        outs = torch.stack(outs, dim=-1)                # (b, sl, emb_dim, E)
        out = (outs * r.unsqueeze(-2)).sum(dim=-1)      # (b, sl, emb_dim)

        masked = r * expert_map.unsqueeze(dim=1)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(eps)
        masked_target = masked / denom
        masked_target = masked_target.detach()  # detach the target so gradients do not flow into it
        kl_loss = F.kl_div(
            torch.log(r.clamp_min(eps)),
            masked_target.clamp_min(eps),
            reduction="batchmean"
        )
        return out, kl_loss


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, num_heads, num_kv_groups, dtype=None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.k_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.q_proj = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin):
        b, num_tokens, d_in = x.shape

        queries = self.q_proj(x)  # Shape: (b, num_tokens, d_out)
        keys = self.k_proj(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.v_proj(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Use the mask to fill attention scores
        attn_scores = attn_scores.masked_fill(mask[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.o_proj(context_vec)  # optional projection

        return context_vec




cfg = LLAMA32_CONFIG_3B

safetensors_path = "/home/jovyan/z_acl/.latent/llama3_3b_local"
if os.path.isdir(safetensors_path):
    safetensor_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
    hf_state_dict = {}
    for f in safetensor_files:
        hf_state_dict.update(load_file(f))
else:
    hf_state_dict = load_file(safetensors_path)

new_state_dict = {}
for k, v in hf_state_dict.items():
    if k.startswith("model."):
        new_state_dict[k[6:]] = v  # Remove 'model.' prefix
    else:
        new_state_dict[k] = v

# Copy embed_tokens weight to lm_head
new_state_dict["lm_head.weight"] = new_state_dict["embed_tokens.weight"]


for l in range(cfg['n_layers']):
    router_weight = torch.empty(cfg['n_experts'], cfg['emb_dim'], dtype=cfg['dtype'])
    nn.init.kaiming_uniform_(router_weight, a=math.sqrt(5))
    new_state_dict[f"layers.{l}.moe.router.weight"] = router_weight

    for e in range(cfg['n_experts']):
        new_state_dict[f"layers.{l}.moe.experts.{e}.fc1.weight"] = hf_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"]
        new_state_dict[f"layers.{l}.moe.experts.{e}.fc2.weight"] = hf_state_dict[f"model.layers.{l}.mlp.up_proj.weight"]
        new_state_dict[f"layers.{l}.moe.experts.{e}.fc3.weight"] = hf_state_dict[f"model.layers.{l}.mlp.down_proj.weight"]

    del (
        new_state_dict[f"layers.{l}.mlp.gate_proj.weight"],
        new_state_dict[f"layers.{l}.mlp.up_proj.weight"],
        new_state_dict[f"layers.{l}.mlp.down_proj.weight"]
    )



from transformers import AutoTokenizer
model_dir = "/home/jovyan/z_acl/.latent/llama3_3b_local"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = Llama3MoE(LLAMA32_CONFIG_3B)
model.load_state_dict(new_state_dict)
model.to("cuda:1")


prmopts = [
    "hey can you prove this theorem?",
    "write a python function to reverse a string.",
    "what's 2 raised to the power of 10?",
    "haha no way"
]

expert_map = torch.tensor([
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 0]
]).to("cuda:1")

model_inputs = tokenizer(prmopts, padding=True, return_tensors="pt").to("cuda:1")

logits, kl_loss = model(
    in_idx=model_inputs["input_ids"],
    expert_map=expert_map
)

logits.shape, kl_loss


# from transformers import AutoTokenizer
# model_dir = "/home/jovyan/z_acl/.latent/llama3_3b_local"
# tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# MAX_NEW_TOKENS = 32
# TEMPERATURE = 0.
# TOP_K = 1

# PROMPT = 'What do llamas eat?'
# token_ids = generate(
#     model=model.to("cuda"),
#     idx=text_to_token_ids(PROMPT, tokenizer).to("cuda"),
#     max_new_tokens=MAX_NEW_TOKENS,
#     context_size=LLAMA32_CONFIG_3B["context_length"],
#     top_k=TOP_K,
#     temperature=TEMPERATURE
# )

# output_text = token_ids_to_text(token_ids, tokenizer)

# print("Output text:\n\n", output_text)



