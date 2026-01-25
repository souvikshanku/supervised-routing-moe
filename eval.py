import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

DEVICE = "cuda:3"


model_dir = "llama3_1b_local"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto"
)

model.to(DEVICE)



with open("data/train.json", "r") as f:
    train_data = json.load(f)

with open("data/test.json", "r") as f:
    test_data = json.load(f)

print(len(train_data), len(test_data))
countdown_indices = [i for i in test_data if test_data[i]['type'] == 'countdown']


prompt = (
    train_data['1']['question']
    + "\n"
    + train_data['2']['question']
    + "\n"
)

question = test_data[countdown_indices[10]]['question']
gold = test_data[countdown_indices[10]]['answer']
gold_eq = re.search(r"<answer>\s*(.*?)\s*</answer>", gold, re.S).group(1)
actual_numbers = list(map(int, re.findall(r"\d+", gold_eq.split("=")[0])))

model_inputs = tokenizer([prompt + question], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    stop_strings=['</answer>'],
    temperature=0.2,
    # top_p=1.0,
    # do_sample=False
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
out = tokenizer.decode(output_ids)
print(out)


try:
    equation = re.search(r"<answer>\s*(.*?)\s*</answer>", out, re.S).group(1)
    lhs, rhs = map(str.strip, equation.split("="))
    numbers = list(map(int, re.findall(r"\d+", lhs)))
    is_correct = (
        (eval(lhs) == int(rhs))
        and set(numbers) == set(actual_numbers)
    )

except:
    print("falied to parse")
    is_correct = False
