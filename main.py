import pandas as pd
import numpy as np
import os
os.environ["HF_HOME"] = "./data/HuggingFace"

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import evaluate


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


def get_data() -> list[dict]:
    df = pd.read_parquet(
        'data/reviewed_professors.parquet', columns=["reviews"])
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

    all_data = get_data()
    train_data, eval_data = train_test_split(
        all_data, test_size=0.1, random_state=42, stratify=[d["label"] for d in all_data]
    )

    # model_name = "meta-llama/Llama-3.2-3B"
    model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        dtype="auto",
        device_map={"": "cuda:0"},
        local_files_only=True
    )


    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ReviewsDataset(train_data, tokenizer, max_length=512)
    eval_dataset = ReviewsDataset(eval_data,  tokenizer, max_length=512)

    id2label = {i: f"{i+1}_stars" for i in range(5)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,
        id2label=id2label,
        label2id=label2id,
        dtype="auto",
        device_map={"": "cuda:0"},
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = "data/FineTuned/prof-review-llama-1b-v3"
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            learning_rate=2e-5,
            warmup_ratio=0.05,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="mae_stars",
            greater_is_better=False,

            gradient_checkpointing=True,
        )

    compute_metrics = Metrics()
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
    
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = trainer.evaluate(eval_dataset)
    print(results)

    # predictions = trainer.predict(eval_dataset)
    # logits = predictions.predictions
    # labels = predictions.label_ids

    # preds = np.argmax(logits, axis=-1)
    # stars = preds + 1

    # cm = confusion_matrix(labels + 1, stars)

    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=[1,2,3,4,5],
    #             yticklabels=[1,2,3,4,5])
    # plt.xlabel("Predicted Stars")
    # plt.ylabel("True Stars")
    # plt.title("Confusion Matrix - Star Ratings")
    # plt.show()


if __name__ == "__main__":
    main()
