import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_BATCH_SIZE = 4
BATCH_WINDOW_SEC = 0.15
MAX_NEW_TOKENS_DEFAULT = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[server] loading {MODEL_NAME} on {device} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
model.eval()
print("[server] model loaded.")

app = FastAPI(title="Jetson MLOps Lab - Day3 Serving")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT


class GenerateResponse(BaseModel):
    text: str
    batch_size: int
    latency_sec: float


@dataclass
class PendingRequest:
    prompt: str
    max_new_tokens: int
    future: "asyncio.Future"


request_queue: Optional[asyncio.Queue] = None


@app.on_event("startup")
async def startup():
    global request_queue
    request_queue = asyncio.Queue()
    asyncio.create_task(batch_worker())


def generate_batch(prompts, max_new_tokens):
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    chat_prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_only = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen_only, skip_special_tokens=True)


async def run_batch(batch):
    start = time.monotonic()
    prompts = [item.prompt for item in batch]
    max_new = max(item.max_new_tokens for item in batch)
    texts = await asyncio.to_thread(generate_batch, prompts, max_new)
    elapsed = time.monotonic() - start
    for item, text in zip(batch, texts):
        if not item.future.done():
            item.future.set_result((text, len(batch), elapsed))


async def batch_worker():
    while True:
        first = await request_queue.get()
        batch = [first]
        deadline = time.monotonic() + BATCH_WINDOW_SEC
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(request_queue.get(), timeout=remaining)
                batch.append(nxt)
            except asyncio.TimeoutError:
                break
        await run_batch(batch)


@app.get("/health")
async def health():
    return {"status": "ok", "device": device}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    pending = PendingRequest(prompt=req.prompt, max_new_tokens=req.max_new_tokens, future=fut)
    await request_queue.put(pending)
    text, batch_size, elapsed = await fut
    return GenerateResponse(text=text, batch_size=batch_size, latency_sec=elapsed)


@app.post("/generate_nobatch", response_model=GenerateResponse)
async def generate_nobatch(req: GenerateRequest):
    start = time.monotonic()
    texts = await asyncio.to_thread(generate_batch, [req.prompt], req.max_new_tokens)
    elapsed = time.monotonic() - start
    return GenerateResponse(text=texts[0], batch_size=1, latency_sec=elapsed)
