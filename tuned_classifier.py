from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = "prof-review-llama-1b"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

def predict_stars(text: str) -> int:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()   # 0-4
        stars = pred_label + 1                             # 1–5
    return stars
