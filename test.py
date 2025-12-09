import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.utils.attention_visualizer import AttentionMaskVisualizer
from dotenv import load_dotenv
import os

load_dotenv()
# model_name = "meta-llama/Llama-3.2-3B"
# model_name = "meta-llama/Llama-3.2-1B"
model_name = "meta-llama/Llama-2-7b-hf"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True)

with torch.no_grad():
    model.generate(
        **input_ids,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        streamer=streamer,
    )