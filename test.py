# %%
import os
import json
import glob
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gc
from tqdm import trange

gc.collect()
torch.cuda.empty_cache()


LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 2048,                 # Embedding dimension
    "n_heads": 32,                   # Number of attention heads
    "n_layers": 16,                  # Number of layers
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
    "n_experts": 4,                  # Number of experts
    "top_k": 2                       # Number of active experts
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


def generate(model, idx, expert_map, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond, expert_map)
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


# %%
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
        self.k = cfg["top_k"]

    def forward(self, x, expert_map, eps: float = 1e-9):
        b, sl, emb_dim = x.shape
        device = x.device

        logits = self.router(x)
        r = torch.softmax(logits, dim=-1)  # (b, sl, E)

        # first, let's get the output of top-k routing
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)   # (b, sl, k)
        topk_mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
        topk_mask.scatter_(-1, topk_idx, True)                  # bool mask for top-k
        masked = r * topk_mask                                  # keep probs for top-k
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(eps)
        r_topk = masked / denom                            # renormalized top-k (used for forward)
        outs = [expert(x) for expert in self.experts]      # list of (b, sl, emb_dim)
        outs = torch.stack(outs, dim=-1)                   # (b, sl, emb_dim, E)
        out = (outs * r_topk.unsqueeze(-2)).sum(dim=-1) 

        # calc KL-div to minimize
        expert_masked = r * expert_map.unsqueeze(dim=1)
        denom = expert_masked.sum(dim=-1, keepdim=True).clamp_min(eps)
        expert_masked_target = expert_masked / denom
        expert_masked_target = expert_masked_target.detach()  # detach the target so gradients do not flow into it
        kl_loss = F.kl_div(
            torch.log(r.clamp_min(eps)),
            expert_masked_target.clamp_min(eps),
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



# %% [markdown]
# #### Upcycling

# %%
# cfg = LLAMA32_CONFIG_3B
cfg = LLAMA32_CONFIG_1B
DEVICE = "cuda:3"


# safetensors_path = "/home/jovyan/z_acl/.latent/llama3_3b_local"
safetensors_path = "/home/jovyan/z_acl/.latent/llama3_1b_local"
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


# %%
from transformers import AutoTokenizer

model = Llama3MoE(cfg)
model.load_state_dict(new_state_dict)
model.to(DEVICE)
print("param count:", sum(p.numel() for p in model.parameters()))

# model_dir = "/home/jovyan/z_acl/.latent/llama3_3b_local"
model_dir = "/home/jovyan/z_acl/.latent/llama3_1b_local"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %% [markdown]
# #### Train

# %%
import json
from torch.utils.data import Dataset, DataLoader


class Code_Math_Data(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.datadict = json.load(f)

    def __len__(self):
        return len(self.datadict)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self.datadict[str(i)] for i in indices]
        return self.datadict[str(idx)]

# %%
MAX_SEQ_LEN = 512
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 8
LEARING_RATE = 5e-5

dataset = Code_Math_Data("data/train.json")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=LEARING_RATE)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
loss_history = {"nll": [], "kl": [], "total": []}


for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        model_inputs = tokenizer(
            batch['question'],
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        expert_map = torch.tensor([
            [1, 1, 0, 0] if t == 'math'
            else [0, 0, 1, 1]
            for t in batch['type']
        ]).to(DEVICE)

        input_ids = model_inputs['input_ids']
        pad_token_id = tokenizer.pad_token_id
        pad_column = torch.full((input_ids.shape[0], 1), pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad_column], dim=1)

        seq_indices = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
        start_token = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
        pad_token_id = tokenizer.pad_token_id
        start = (input_ids == start_token).int().argmax(dim=1)
        end = (input_ids == pad_token_id).int().argmax(dim=1)
        loss_mask = (
            (seq_indices >= (start).unsqueeze(1))
            & (seq_indices < (end).unsqueeze(1))
        )  # [B, S]

        logits, kl_loss = model(in_idx=input_ids, expert_map=expert_map)

        log_probs = F.log_softmax(logits, dim=-1)  # [B, S, V]
        targets = input_ids[:, 1:].unsqueeze(-1)   # [B, S-1, 1]
        token_log_probs = log_probs[:, :-1, :].gather(dim=-1, index=targets).squeeze(-1)  # [B, S-1]

        num_tokens = (seq_indices < (end).unsqueeze(1)).sum()
        nll = - (token_log_probs * loss_mask[:, :-1])  # [B, S-1]
        nll_loss = nll.sum() / num_tokens
        # model.forward returns per layer total kl-loss, so we divide by `num_tokens` here
        kl_per_token = kl_loss / num_tokens
        loss = nll_loss + kl_per_token

        loss_history["nll"].append(nll_loss.item())
        loss_history["kl"].append(kl_per_token.item())
        loss_history["total"].append(loss.item())

        loss = loss / GRAD_ACCUMULATION_STEPS
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step & zero grad every `GRAD_ACCUMULATION_STEPS` steps
        if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            true_loss = loss.item() * GRAD_ACCUMULATION_STEPS

        if (step + 1) % 20 == 0:
            print(f"{nll_loss.item():.6f}", f"{kl_per_token.item():.6f}")

    # catch remaining gradients if dataset size not divisible by accumulation steps
    if (step + 1) % GRAD_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()


with open("moe_sft_logs.txt", "w") as f:
    json.dump(loss_history, f)

PATH = "llama3_1b_moe/model.pth"
torch.save(model.state_dict(), PATH)

# %%
import matplotlib.pyplot as plt

with open("moe_sft_logs.txt", "r") as f:
    loss_history = json.load(f)

plt.plot(loss_history['kl'])
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss (KL-div)")
plt.show()

plt.plot(loss_history['total'])
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss (CE + KL-div)")
plt.show()

# %%
"""
TODO:
1. ✅ Proper masking of questions when doing SFT
    * changes in data?
    * start mask in loss calc

4. ✅ Implement top-k routing

3. ✅ To have or not have shared expert?
    -- not to have. let's add 1 more expert and then do 2 way split

4. ✅ Eval on a small set? -- just do on gsm8k test set
    * create test set for code and math?
    * nll for code for now?

2. ✅ Figure out a way to measure expert spec? -- read some

5. May be train with higher lr
    -- kinda terrible. need to see if 'experts' work or not

6. ✅ may be instad of code, do something else, like
    - science MCQ or medical data?
    - easier to show perf comparison b/w base and sft
"""

# %% [markdown]
# #### Eval

# %%
dataset = Code_Math_Data("data/test.json")

math_acc = 0
medical_acc = 0

for i in range(len(dataset)):
    data = dataset[i]
    prompt = data['question']
    expert_map = torch.tensor([
        [1, 1, 0, 0] 
        if data['type'] == 'math'
        else [0, 0, 1, 1]
    ]).to(DEVICE)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(DEVICE),
        expert_map=expert_map,
        max_new_tokens=500,
        context_size=cfg["context_length"],
        eos_id=tokenizer.pad_token_id,
        temperature=0.0
    )

    output_text = token_ids_to_text(token_ids, tokenizer)

    if data['type'] == 'math':
        try:
            gen_answer = float(output_text.split("#####")[1])
        except Exception:
            gen_answer = float('inf')
        actual_answer = float(data['answer'].split("#####")[1])
        if gen_answer == actual_answer:
            math_acc += 1 
    else:
        gen_answer = output_text[-1]
        actual_answer = data['answer']
        if gen_answer == actual_answer:
            medical_acc += 1

print(f"Math: {math_acc} / {len(dataset) // 2} ({math_acc / (len(dataset) // 2)})")
print(f"Medical: {medical_acc} / {len(dataset) // 2} ({medical_acc / (len(dataset) // 2)})")

# %%
# MoE - GSM8K Test Set

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", 'main')

math_acc = 0
for i in range(len(dataset['test'])):
    try:
        data = dataset['test'][i]
        expert_map = torch.tensor([
            [1, 1, 0, 0] 
            # if data['type'] == 'math'
            # else [0, 0, 1, 1]
        ]).to(DEVICE)

        question = (
            f"Question:\n{data['question']}\n"
            + f"Answer:\n"
            + "<|reserved_special_token_0|>"
        )
        actual_answer = float(data['answer'].split("####")[1].replace(",", ""))

        token_ids = generate(
                model=model,
                idx=text_to_token_ids(question, tokenizer).to(DEVICE),
                expert_map=expert_map,
                max_new_tokens=500,
                context_size=cfg["context_length"],
                eos_id=tokenizer.pad_token_id,
                temperature=0.0
            )

        output_text = token_ids_to_text(token_ids, tokenizer)
        try:
            gen_answer = float(output_text.split("#####")[1])
        except Exception:
            gen_answer = float('inf')

        if gen_answer == actual_answer:
            math_acc += 1

    except Exception as e:
        print(e)

print(f"GSM8k test: {math_acc} / {len(dataset['test'])} ({math_acc / len(dataset['test'])})")

# %% [markdown]
# #### Work

# %%
PROMPT = "Question:\nAdam has $500 in his account. He spends 3 dollars every time he rides a bus and he takes the bus 5 times a week. How much money, in dollars, will be left in his account after a month, assuming a month has 4 weeks?\nAnswer:\n<|reserved_special_token_0|>"

expert_map = torch.tensor([
    [0, 0, 1, 1]
    # [1, 1, 0, 0]
]).to(DEVICE)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(DEVICE),
    expert_map=expert_map,
    max_new_tokens=150,
    context_size=cfg["context_length"],
    eos_id=tokenizer.pad_token_id,
    temperature=0.4
)

output_text = token_ids_to_text(token_ids, tokenizer)

print(output_text)


# %%
model.layers[11].moe#.router.weight
model.layers[11].moe.experts[0].fc1.weight, model.layers[11].moe.experts[3].fc1.weight

# %%


# %% [markdown]
# #### Eval Real

# %% [markdown]
# ##### Eval Base

# %%
import json
import random
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/home/jovyan/z_acl/.latent/llama3_1b_local"
DEVICE = "cuda:3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
model.to(DEVICE)

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", 'main')


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond).logits
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

np.random.seed(42)
random.seed(42)

with open("data/train.json") as f:
    train_data = json.load(f)

valid_indices = [i for i in train_data if train_data[i]['type'] == "math"]
ids = np.random.choice(valid_indices, size=8, replace=False).tolist()
few_shot = ("\n".join([train_data[i]['question']+'<|eot_id|>' for i in ids]))

print(few_shot)

# %%
# base - aug-nl

from tqdm import trange
with open("data/test.json", "r") as f:
    test_data = json.load(f)

# test_data = {k: test_data[k] for k in test_data if test_data[k]['type'] == 'math'}
indices = [k for k in test_data if test_data[k]['type'] == 'math']

math_acc = 0
# count = 0

for j in trange(len(indices)):
    i = indices[j]
    try:
        data = test_data[i]
        expert_map = torch.tensor([
            [1, 1, 0, 0] 
            # if data['type'] == 'math'
            # else [0, 0, 1, 1]
        ]).to(DEVICE)

        question = (
            few_shot
            + f"\nQuestion:\n{data['question']}\n"
            + f"Answer:\n"
            + "<|reserved_special_token_0|>"
        )
        actual_answer = float(data['answer'].split("#####")[1])

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(question, tokenizer).to(DEVICE),
            max_new_tokens=200,
            context_size=LLAMA32_CONFIG_1B["context_length"],
            eos_id=tokenizer.convert_tokens_to_ids('<|eot_id|>'),
            temperature=0.0
        )

        output_text = token_ids_to_text(token_ids, tokenizer)
        try:
            gen_answer = float(output_text.split("#####")[-1])
        except Exception:
            gen_answer = float('inf')

        if gen_answer == actual_answer:
            math_acc += 1

    except Exception as e:
        print(e)

    # count += 1
    # if count == 20:
    #     break

# print(f"{math_acc} / {count}")

print(f"GSM8k-Aug NL test: {math_acc} / {(len(test_data) // 2)} ({math_acc / (len(test_data) // 2)})")

# %%
# base gsm-8k

math_acc = 0
count = 0

for i in range(len(dataset['test'])):
    count += 1
    try:
        data = dataset['test'][i]
        expert_map = torch.tensor([
            [1, 1, 0, 0] 
            # if data['type'] == 'math'
            # else [0, 0, 1, 1]
        ]).to(DEVICE)

        question = (
            few_shot
            + f"\nQuestion:\n{data['question']}\n"
            + f"Answer:\n"
            + "<|reserved_special_token_0|>"
        )
        actual_answer = float(data['answer'].split("####")[1].replace(",", ""))

        token_ids = generate(
                model=model,
                idx=text_to_token_ids(question, tokenizer).to(DEVICE),
                max_new_tokens=200,
                context_size=LLAMA32_CONFIG_1B["context_length"],
                eos_id=tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                temperature=0.0
            )

        output_text = token_ids_to_text(token_ids, tokenizer)
        try:
            gen_answer = float(output_text.split("#####")[-1])
        except Exception:
            gen_answer = float('inf')

        if gen_answer == actual_answer:
            math_acc += 1

    except Exception as e:
        print(e)

    if count == 100:
        break

print(f"{math_acc} / {count}")
# print(f"GSM8k test: {math_acc} / {len(dataset['test'])} ({math_acc / len(dataset['test'])})")

# %% [markdown]
# ##### Eval MoE - AUG-NL

# %%
import json
import random
import numpy as np

np.random.seed(42)
random.seed(42)

with open("data/train.json") as f:
    train_data = json.load(f)

valid_indices = [i for i in train_data if train_data[i]['type'] == "math"]
ids = np.random.choice(valid_indices, size=8, replace=False).tolist()
few_shot = ("\n".join([train_data[i]['question']+'<|eot_id|>' for i in ids]))

# print(few_shot)

# %%
# base - aug-nl

from tqdm import trange
with open("data/test.json", "r") as f:
    test_data = json.load(f)

# test_data = {k: test_data[k] for k in test_data if test_data[k]['type'] == 'math'}
indices = [k for k in test_data if test_data[k]['type'] == 'math']

math_acc = 0
# count = 0

for j in trange(len(indices)):
    i = indices[j]
    try:
        data = test_data[i]

        question = (
            few_shot
            + f"\n{data['question']}"
        )
        actual_answer = float(data['answer'].split("#####")[1])

        expert_map = torch.tensor([
            [1, 1, 0, 0] 
            # if data['type'] == 'math'
            # else [0, 0, 1, 1]
        ]).to(DEVICE)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(question, tokenizer).to(DEVICE),
            expert_map=expert_map,
            max_new_tokens=500,
            context_size=cfg["context_length"],
            eos_id=tokenizer.pad_token_id,
            temperature=0.0
        )

        output_text = token_ids_to_text(token_ids, tokenizer)
        try:
            gen_answer = float(output_text.split("#####")[-1])
        except Exception:
            gen_answer = float('inf')

        if gen_answer == actual_answer:
            math_acc += 1

    except Exception as e:
        print(e)

    # count += 1
    # if count == 100:
    #     break

print(f"MoE GSM8k-Aug NL test: {math_acc} / {(len(test_data) // 2)} ({math_acc / (len(test_data) // 2)})")

# %% [markdown]
# ##### Eval MoE - GSM OG TEST

# %%
import json
import random
import numpy as np
from datasets import load_dataset
from tqdm import trange

np.random.seed(42)
random.seed(42)

with open("data/train.json") as f:
    train_data = json.load(f)

valid_indices = [i for i in train_data if train_data[i]['type'] == "math"]
ids = np.random.choice(valid_indices, size=8, replace=False).tolist()
few_shot = ("\n".join([train_data[i]['question']+'<|eot_id|>' for i in ids]))

dataset = load_dataset("openai/gsm8k", 'main')

# %%
# base gsm-8k

math_acc = 0
count = 0

for i in trange(len(dataset['test'])):
    try:
        data = dataset['test'][i]
        question = (
            few_shot
            + f"\nQuestion:\n{data['question']}\n"
            + f"Answer:\n"
            + "<|reserved_special_token_0|>"
        )
        actual_answer = float(data['answer'].split("####")[1].replace(",", ""))
        expert_map = torch.tensor([
            [1, 1, 0, 0] 
            # if data['type'] == 'math'
            # else [0, 0, 1, 1]
        ]).to(DEVICE)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(question, tokenizer).to(DEVICE),
            expert_map=expert_map,
            max_new_tokens=500,
            context_size=cfg["context_length"],
            eos_id=tokenizer.pad_token_id,
            temperature=0.0
        )

        output_text = token_ids_to_text(token_ids, tokenizer)
        try:
            gen_answer = float(output_text.split("#####")[-1])
        except Exception:
            gen_answer = float('inf')

        if gen_answer == actual_answer:
            math_acc += 1

    except Exception as e:
        print(e)

    count += 1
    # if count == 10:
    #     break
    if count % 50 == 0:
        print(math_acc / count)

# print(f"{math_acc} / {count}")
print(f"MoE GSM8k test: {math_acc} / {len(dataset['test'])} ({math_acc / len(dataset['test'])})")

# %% [markdown]
# #### Eval - Medical

# %%
import json
import random
import numpy as np
from datasets import load_dataset
from tqdm import trange

np.random.seed(42)
random.seed(42)

with open("data/train.json") as f:
    train_data = json.load(f)

valid_indices = [i for i in train_data if train_data[i]['type'] != "math"]
ids = np.random.choice(valid_indices, size=8, replace=False).tolist()
few_shot = ("\n".join([train_data[i]['question']+'<|eot_id|>' for i in ids]))

# %%
# MoE - medical

from tqdm import trange
with open("data/test.json", "r") as f:
    test_data = json.load(f)

# test_data = {k: test_data[k] for k in test_data if test_data[k]['type'] == 'math'}
indices = [k for k in test_data if test_data[k]['type'] != 'math']

medical_acc = 0
count = 0

for j in trange(len(indices)):
    i = indices[j]
    try:
        data = test_data[i]

        question = (
            few_shot
            + f"\n{data['question']}"
        )
        actual_answer = data['answer']

        expert_map = torch.tensor([
            # [1, 1, 0, 0] 
            # if data['type'] == 'math'
            # else [0, 0, 1, 1]
            [0, 0, 1, 1]
        ]).to(DEVICE)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(question, tokenizer).to(DEVICE),
            expert_map=expert_map,
            max_new_tokens=500,
            context_size=cfg["context_length"],
            eos_id=tokenizer.pad_token_id,
            temperature=0.0
        )

        output_text = token_ids_to_text(token_ids, tokenizer)
        try:
            gen_answer = output_text[-1]
        except Exception:
            gen_answer = float('inf')

        if gen_answer == actual_answer:
            medical_acc += 1

    except Exception as e:
        print(e)

    # count += 1
    # if count == 20:
    #     break

print(f"MoE Medical test: {medical_acc} / {(len(test_data) // 2)} ({medical_acc / (len(test_data) // 2)})")

# %%
# base - medical


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond).logits
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


from tqdm import trange
with open("data/test.json", "r") as f:
    test_data = json.load(f)

# test_data = {k: test_data[k] for k in test_data if test_data[k]['type'] == 'math'}
indices = [k for k in test_data if test_data[k]['type'] != 'math']

medical_acc = 0
count = 0

for j in trange(len(indices)):
    i = indices[j]
    try:
        data = test_data[i]

        question = (
            few_shot
            + f"\n{data['question']}"
        )
        actual_answer = data['answer']

        expert_map = torch.tensor([
            # [1, 1, 0, 0] 
            # if data['type'] == 'math'
            # else [0, 0, 1, 1]
            [0, 0, 1, 1]
        ]).to(DEVICE)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(question, tokenizer).to(DEVICE),
            # expert_map=expert_map,
            max_new_tokens=1,
            context_size=LLAMA32_CONFIG_1B["context_length"],
            eos_id=tokenizer.pad_token_id,
            temperature=0.0
        )

        output_text = token_ids_to_text(token_ids, tokenizer)
        try:
            gen_answer = output_text[-1]
        except Exception:
            gen_answer = float('inf')

        if gen_answer == actual_answer:
            medical_acc += 1

    except Exception as e:
        print(e)

    # count += 1
    # if count == 20:
    #     break

print(f"Base Medical test: {medical_acc} / {(len(test_data) // 2)} ({medical_acc / (len(test_data) // 2)})")


