import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
        cleaned.append({
            "text": text,
            "rating": int(round(rating))
        })
    print("Total usable reviews:", len(cleaned))
    return cleaned

def main():

    dataset = get_data()

    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = ReviewsDataset(dataset, tokenizer)


if __name__ == "__main__":
    main()
