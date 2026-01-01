import gc
import os
import json
import math

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from transformers import AutoTokenizer
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader

from model import LLAMA32_CONFIG_1B, Llama3MoE


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


if __name__ == "__main__":

    cfg = LLAMA32_CONFIG_1B
    DEVICE = "cuda:1"

    # -------------------------------- Upcycling --------------------------------
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


    model = Llama3MoE(cfg)
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    # print("param count:", sum(p.numel() for p in model.parameters()))

    # model_dir = "/home/jovyan/z_acl/.latent/llama3_3b_local"
    model_dir = "/home/jovyan/z_acl/.latent/llama3_1b_local"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only MoE routers
    for layer in model.layers:
        for p in layer.moe.router.parameters():
            p.requires_grad = True

    print("param count:", sum(p.numel() for p in model.parameters()))
    print("param to finetune:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # -------------------------------- Training --------------------------------
    MAX_SEQ_LEN = 512
    NUM_EPOCHS = 1
    BATCH_SIZE = 4
    GRAD_ACCUMULATION_STEPS = 8
    LEARING_RATE = 5e-4

    dataset = Code_Math_Data("data/train.json")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    router_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(router_params, lr=LEARING_RATE)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARING_RATE)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_history = {"nll": [], "total": []}

    model.train()

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

            logits, expert_loss = model(in_idx=input_ids, expert_map=expert_map, tokenizer=tokenizer)

            # seq_indices = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
            # start_token = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
            # pad_token_id = tokenizer.pad_token_id
            # start = (input_ids == start_token).int().argmax(dim=1)
            # end = (input_ids == pad_token_id).int().argmax(dim=1)
            # loss_mask = (
            #     (seq_indices >= (start).unsqueeze(1))
            #     & (seq_indices < (end).unsqueeze(1))
            # )  # [B, S]
            # log_probs = F.log_softmax(logits, dim=-1)  # [B, S, V]
            # targets = input_ids[:, 1:].unsqueeze(-1)   # [B, S-1, 1]
            # token_log_probs = log_probs[:, :-1, :].gather(dim=-1, index=targets).squeeze(-1)  # [B, S-1]
            # num_tokens = (seq_indices < (end).unsqueeze(1)).sum()
            # nll = - (token_log_probs * loss_mask[:, :-1])  # [B, S-1]
            # nll_loss = nll.sum() / num_tokens

            loss = expert_loss

            loss_history["total"].append(loss.item())

            loss = loss / GRAD_ACCUMULATION_STEPS
            loss.backward()

            # step & zero grad every `GRAD_ACCUMULATION_STEPS` steps
            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                true_loss = loss.item() * GRAD_ACCUMULATION_STEPS

            if (step + 1) % 20 == 0:
                print(loss.item())

        # catch remaining gradients if dataset size not divisible by accumulation steps
        if (step + 1) % GRAD_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    with open("moe_sft_logs.txt", "w") as f:
        json.dump(loss_history, f)

    PATH = "llama3_1b_moe/model.pth"
    torch.save(model.state_dict(), PATH)
