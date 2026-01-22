import json
import numpy as np
import random
from datasets import load_dataset


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    train_size_per_type = 10000
    test_size_per_type = 1000
    dataset_size = train_size_per_type + test_size_per_type

    # Load and process countdown data
    countdown_train = []
    countdown_test = []
    dataset = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "verified_Qwen2.5-7B-Instruct")
    size = len(dataset["train"])
    ids = np.random.choice(np.arange(size), size=dataset_size, replace=False).tolist()

    for i, index in enumerate(ids):
        question = "\n".join([d['content'] for d in dataset['train'][index]['messages'][:-1]])
        answer = dataset['train'][index]['messages'][-1]['content']
        if i < train_size_per_type:
            text = f"User:\n{question}\nAssistant:\n{answer}"
            countdown_train.append({"question": text, "type": "countdown"})
        else:
            countdown_test.append({
                "question": f"User:\n{question}\nAssistant:\n",
                "answer": answer,
                "type": "countdown"
            })

    # Load and process tldr data
    tldr_train = []
    tldr_test = []
    dataset = load_dataset("trl-lib/tldr")
    size = len(dataset["train"])
    ids = np.random.choice(np.arange(size), size=dataset_size, replace=False).tolist()
    
    prompt = "User:\nSummarize the post into a concise TL;DR."
    for i, index in enumerate(ids):
        post = dataset['train'][index]['prompt']
        summary = dataset['train'][index]['completion']
    
        if i < train_size_per_type:
            text = f"{prompt}\n{post}\nAssistant:\n{summary.strip()}"
            tldr_train.append({"question": text, "type": "tldr"})
        else:
            tldr_test.append({
                "question": f"{prompt}\n{post}\nAssistant:\n",
                "answer": summary.strip(),
                "type": "tldr"
            })

    train_items = countdown_train + tldr_train
    test_items = countdown_test + tldr_test
    random.shuffle(train_items)
    random.shuffle(test_items)

    train_data = {i: item for i, item in enumerate(train_items)}
    test_data = {i: item for i, item in enumerate(test_items)}

    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w") as f:
        json.dump(train_data, f)

    with open("data/test.json", "w") as f:
        json.dump(test_data, f)

    print(f"Train dataset created: {len(train_data)} samples")
    print(f"Test dataset created: {len(test_data)} samples")
