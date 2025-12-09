import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "JungZoona/T3Q-qwen2.5-14b-v1.2-e2"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True)

with torch.no_grad():
    model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        streamer=streamer,
    )