import re
import json
import numpy as np
import random
from datasets import load_dataset, load_from_disk


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    dataset = load_dataset("zen-E/GSM8k-Aug-NL")

    math_data = {}
    size = len(dataset["train"])
    ids = np.random.choice(np.arange(size), size=10000, replace=False).tolist()

    for i, index in enumerate(ids):
        qna = (
            f"Question:\n{dataset['train'][i]['question']}\n"
            + f"Answer:\n{dataset['train'][i]['cot']}\n"
            + "#####"
            + dataset['train'][i]['answer']
        )
        math_data[i] = qna

    with open("data/math_data.json", "w") as f:
        json.dump(math_data, f)


    dataset = load_dataset("PsiPi/CodeAlpaca_20k_NoBlanks")
    code_data = {}
    size = len(dataset["train"])
    ids = np.random.choice(np.arange(size), size=10000, replace=False).tolist()

    for i, index in enumerate(ids):
        qna = (
            f"Question:\n{dataset['train'][i]['instruction']}\n"
            + (f"Input:\n{dataset['train'][i]['input']}\n" if dataset['train'][i]['input'] else "")
            + "Answer:\n"
            + dataset['train'][i]['output']
        )
        code_data[i] = qna

    with open("data/code_data.json", "w") as f:
        json.dump(code_data, f)

    print("datasets have been created!")
