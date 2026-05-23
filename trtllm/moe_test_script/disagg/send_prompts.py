#!/usr/bin/env python
"""Replay prompts.txt through the disagg router concurrently.

Used during offline EPLB stats collection (stage 1) to drive workload
through the gen server so EXPERT_STATISTIC_* counters accumulate.

Defaults:
  PROMPTS_FILE=./prompts.txt
  ROUTER_URL=http://localhost:8000/v1/completions
  MODEL=/llm-models/DeepSeek-V3-Lite/fp8
  MAX_TOKENS=128
  CONCURRENCY=32

Usage (from TRT-LLM root):
  python moe_test_script/disagg/send_prompts.py
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp


async def send_one(session, url, payload, idx):
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            text = data["choices"][0]["text"]
            return idx, True, text[:80]
    except Exception as e:
        return idx, False, repr(e)[:160]


async def run(prompts, url, model, max_tokens, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def bounded_send(session, p, idx):
        async with sem:
            return await send_one(session, url, {
                "model": model,
                "prompt": p,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
            }, idx)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_send(session, p, i) for i, p in enumerate(prompts)]
        start = time.time()
        ok_cnt = 0
        for fut in asyncio.as_completed(tasks):
            idx, ok, info = await fut
            if ok:
                ok_cnt += 1
                if idx % 20 == 0:
                    print(f"  [{idx:>4}] OK   {info!r}", flush=True)
            else:
                print(f"  [{idx:>4}] FAIL {info}", flush=True)
        elapsed = time.time() - start
        print(f"\n[INFO] done. {ok_cnt}/{len(prompts)} succeeded in {elapsed:.1f}s "
              f"({ok_cnt / max(elapsed, 1e-6):.1f} req/s)")
        return ok_cnt == len(prompts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file",
                        default=os.environ.get("PROMPTS_FILE", "./prompts.txt"))
    parser.add_argument("--router-url",
                        default=os.environ.get("ROUTER_URL",
                                               "http://localhost:8000/v1/completions"))
    parser.add_argument("--model",
                        default=os.environ.get("MODEL",
                                               "/llm-models/DeepSeek-V3-Lite/fp8"))
    parser.add_argument("--max-tokens",
                        type=int,
                        default=int(os.environ.get("MAX_TOKENS", "128")))
    parser.add_argument("--concurrency",
                        type=int,
                        default=int(os.environ.get("CONCURRENCY", "32")))
    args = parser.parse_args()

    if not os.path.isfile(args.prompts_file):
        sys.exit(f"[ERROR] prompts file not found: {args.prompts_file}")

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"[INFO] loaded {len(prompts)} prompts from {args.prompts_file}")
    print(f"[INFO] POST to {args.router_url}")
    print(f"[INFO] model={args.model} max_tokens={args.max_tokens} "
          f"concurrency={args.concurrency}")

    ok = asyncio.run(run(prompts, args.router_url, args.model,
                         args.max_tokens, args.concurrency))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
