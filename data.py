import json
import numpy as np
import random
from datasets import load_dataset


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    train_size_per_type = 5000
    test_size_per_type = 500
    dataset_size = train_size_per_type + test_size_per_type

    # Load and process math data
    math_train = []
    math_test = []
    dataset = load_dataset("zen-E/GSM8k-Aug-NL")
    size = len(dataset["train"])
    ids = np.random.choice(np.arange(size), size=dataset_size, replace=False).tolist()

    for i, index in enumerate(ids):
        if i < train_size_per_type:
            qna = (
                f"Question:\n{dataset['train'][index]['question']}\n"
                + f"\nAnswer:\n<|reserved_special_token_0|>"
                + f"{dataset['train'][index]['cot']}\n"
                + "#####"
                + dataset['train'][index]['answer']
            )
            math_train.append({"question": qna, "type": "math"})
        else:
            question = (
                f"Question:\n{dataset['train'][index]['question']}\n"
                + f"\nAnswer:\n<|reserved_special_token_0|>"
            )
            answer = (
                f"{dataset['train'][index]['cot']}\n"
                + "#####"
                + dataset['train'][index]['answer']
            )
            math_test.append({"question": question, "answer": answer, "type": "math"})

    # Load and process medical data
    medical_train = []
    medical_test = []
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    size = len(dataset["train"])
    ids = np.random.choice(np.arange(size), size=dataset_size, replace=False).tolist()

    for i, index in enumerate(ids):
        context = "\n".join(list(dataset['train'][index]['context'].values())[0])
        question = dataset['train'][index]['question']
        answer = dataset['train'][index]['long_answer']
        final_decision = dataset['train'][index]['final_decision']

        if i < train_size_per_type:
            qna = (
                f"Context:\n{context}"
                + f"\nQuestion:\n{question}"
                + f"\nAnswer:\n<|reserved_special_token_0|>"
                + f"{answer}\n#####{final_decision}"
            )
            medical_train.append({"question": qna, "type": "medical"})
        else:
            question = (
                f"Context:\n{context}"
                + f"\nQuestion:\n{question}"
                + "\nAnswer:\n<|reserved_special_token_0|>"
            )

            answer = f"{answer}\n#####{final_decision}"
            medical_test.append({"question": question, "answer": answer, "type": "medical"})

    # Load and process code data
    code_train = []
    code_test = []
    dataset = load_dataset("PsiPi/CodeAlpaca_20k_NoBlanks")
    size = len(dataset["train"])
    ids = np.random.choice(np.arange(size), size=dataset_size, replace=False).tolist()

    for i, index in enumerate(ids):
        instruction = dataset['train'][index]['instruction']
        inp = dataset['train'][index]['input']
        output = dataset['train'][index]['output']

        if i < train_size_per_type:
            qna = (
                f"Question:\n{instruction}\n"
                + (f"Input:\n{inp}\n" if inp else "")
                + "\nAnswer:\n<|reserved_special_token_0|>"
                + output
            )
            code_train.append({"question": qna, "type": "code"})
        else:
            question = (
                f"Question:\n{instruction}\n"
                + (f"Input:\n{inp}\n" if inp else "")
                + "\nAnswer:\n<|reserved_special_token_0|>"
            )
            answer = output
            code_test.append({"question": question, "answer": answer, "type": "code"})

    # Merge into lists
    train_items = math_train + medical_train + code_train
    test_items = math_test + medical_test + code_test

    # Shuffle
    random.shuffle(train_items)
    random.shuffle(test_items)

    # Convert to dict with sequential keys
    train_data = {i: item for i, item in enumerate(train_items)}
    test_data = {i: item for i, item in enumerate(test_items)}

    with open("data/train.json", "w") as f:
        json.dump(train_data, f)

    with open("data/test.json", "w") as f:
        json.dump(test_data, f)

    print(f"Train dataset created: {len(train_data)} samples")
    print(f"Test dataset created: {len(test_data)} samples")
