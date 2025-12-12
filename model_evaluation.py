import pandas as pd
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import evaluate

from collections import defaultdict
import random


class Metrics:
    def __init__(self):
        self.acc = evaluate.load("accuracy")

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = self.acc.compute(predictions=preds, references=labels)["accuracy"]
        mae = float(np.mean(np.abs((preds + 1) - (labels + 1))))

        return {"accuracy": acc, "mae_stars": mae}


class ReviewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_uniform_dataset() -> list[dict]:
    df = pd.read_parquet('data/professors.parquet')
    
    df["num_reviews"] = df["reviews"].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
    df = df[df["num_reviews"] < 100]
    combined = np.concatenate(df["reviews"].to_numpy())
    data = list(combined)
    groups = defaultdict(list)
    for item in data:
        groups[item["rating"]].append(item)

    min_count = min(len(groups[r]) for r in groups)

    balanced_data = []
    for rating in range(1, 6):  # ratings 1–5
        balanced_data.extend(random.sample(groups[rating], min_count))

    print(f"Uniform count per rating: {min_count}")
    print(f"Total items: {len(balanced_data)}")

    cleaned: list[dict] = []
    for r in balanced_data:
        text = (r.get("review") or "").strip()
        rating = r.get("rating", None)
        if not text or rating is None:
            continue
        # ensure rating is between 1 and 5
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except:
                continue
        if not (1 <= rating <= 5):
            continue
        stars = int(round(rating))  # 1-5
        label = stars - 1  # 0-4
        cleaned.append({
            "text": text,
            "label": label
        })
    print(len(cleaned))
    return cleaned

def get_data() -> list[dict]:
    df = pd.read_parquet(
        'data/reviewed_professors.parquet', columns=["reviews"])
    # df = df.tail(5)
    combined = np.concatenate(df["reviews"].to_numpy())

    cleaned: list[dict] = []
    for r in combined:
        text = (r.get("review") or "").strip()
        rating = r.get("rating", None)
        if not text or rating is None:
            continue
        # ensure rating is between 1 and 5
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except:
                continue
        if not (1 <= rating <= 5):
            continue
        stars = int(round(rating))  # 1-5
        label = stars - 1  # 0-4
        cleaned.append({
            "text": text,
            "label": label
        })
    print("Total usable reviews:", len(cleaned))
    return cleaned


def main():
    new_data = get_uniform_dataset()
    # all_data = get_data()
    # train_data, eval_data = train_test_split(
    #     all_data, test_size=0.1, random_state=42, stratify=[d["label"] for d in all_data]
    # )

    checkpoint_dir = "data/FineTuned/prof-review-llama-1b-v3/checkpoint-908"
    base_dir = "data/FineTuned/prof-review-llama-1b-v3"

    tokenizer = AutoTokenizer.from_pretrained(
        base_dir,
        dtype="auto",
        device_map={"": "cuda:0"},
        local_files_only=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    # train_dataset = ReviewsDataset(train_data, tokenizer, max_length=512)
    # eval_dataset = ReviewsDataset(eval_data,  tokenizer, max_length=512)
    eval_dataset = ReviewsDataset(new_data, tokenizer, max_length=512)


    id2label = {i: f"{i+1}_stars" for i in range(5)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=5,
        id2label=id2label,
        label2id=label2id,
        dtype="auto",
        device_map={"": "cuda:0"},
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    metrics = Metrics()

    training_args = TrainingArguments(
        output_dir="data/FineTuned/eval_tmp",
        per_device_eval_batch_size=12,
        do_train=False,
        do_eval=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=metrics,
    )

    predictions = trainer.predict(eval_dataset)

    metrics = predictions.metrics
    print("Eval metrics:", metrics)

    logits = predictions.predictions
    labels = predictions.label_ids

    preds = np.argmax(logits, axis=-1)
    stars = preds + 1

    cm = confusion_matrix(labels + 1, stars)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="rocket",
                xticklabels=[1,2,3,4,5],
                yticklabels=[1,2,3,4,5])
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.xlabel("Predicted Stars", color="white")
    plt.ylabel("True Stars", color="white")
    plt.title("Model B - Confusion Matrix", color="white")
    plt.savefig(fname="./data/FineTuned/eval_tmp/cm_a_evalset.png", transparent=True)


if __name__ == "__main__":
    main()