import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from dotenv import load_dotenv
import os
import time

class TimingStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.token_timestamps = []
        self.start_time = None

    def put(self, value):
        # On the first token, record start time
        if self.start_time is None:
            self.start_time = time.time()

        # Record timestamp for every token
        self.token_timestamps.append(time.time())

        # Process token normally
        super().put(value)

    def finish(self):
        super().finish()


load_dotenv()
model_name = "meta-llama/Llama-3.2-3B"
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map={"": "cuda:0"},
    token=os.getenv("HF_TOKEN"),
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("Hi, I am ", return_tensors="pt").to(model.device)

streamer = TimingStreamer(tokenizer, skip_prompt=True)

with torch.no_grad():
    model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        streamer=streamer,
    )

timestamps = streamer.token_timestamps

# Number of streamed tokens
num_tokens = len(timestamps)

# First token latency
first_token_latency = timestamps[0] - streamer.start_time

# Inter-token intervals
if num_tokens > 1:
    intervals = [
        t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])
    ]
    avg_time_per_token = sum(intervals) / len(intervals)
else:
    avg_time_per_token = float("nan")

print("Tokens generated:", num_tokens)
print("First token latency:", first_token_latency, "seconds")
print("Avg time per token:", avg_time_per_token, "seconds")
print("Throughput:", 1 / avg_time_per_token if num_tokens > 1 else float("nan"), "tokens/sec")
