"""
synthetic_gen.py
Generates contested synthetic queries from SQuAD passages using GPT-4o.
These are HIGH VARIANCE queries that make DAPO learn best —
questions where role assignment genuinely changes the verdict.
"""

import json
import asyncio
import os
import httpx
from pathlib import Path
from datasets import load_dataset  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
from src.prompts import GENERATION_PROMPT
OPENROUTER_URL = os.getenv("OPENROUTER_URL")
ROUTER_KEY     = os.getenv("ROUTER_KEY")

GENERATION_PROMPT = GENERATION_PROMPT


class SyntheticGenerator:
    """
    Generates contested queries from SQuAD passages via GPT-4o.
    Output: data/synthetic/queries.jsonl
    Each line = one TrainingQuery-compatible JSON object.
    """

    def __init__(self, model: str = "openai/gpt-4o", concurrency: int = 5):
        self.model       = model
        self.concurrency = concurrency
        self.semaphore   = asyncio.Semaphore(concurrency)

    async def _call(self, prompt: str) -> str:
        """Single OpenRouter call via httpx."""
        headers = {
            "Authorization": f"Bearer {ROUTER_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/nexomnis",
        }
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.8,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or "[]"

    async def generate_from_passage(
        self,
        passage_id: str,
        passage:    str,
    ) -> list[dict]:
        """Generate 3 contested queries from one passage."""
        async with self.semaphore:
            try:
                raw = await self._call(
                    GENERATION_PROMPT.format(excerpt=passage[:1200])
                )

                # Strip markdown fences if present
                raw = raw.strip()
                if "```" in raw:
                    raw = raw.split("```")[1].lstrip("json")

                items = json.loads(raw)
                results = []
                for i, item in enumerate(items[:3]):
                    results.append({
                        "query_id":     f"synthetic_{passage_id}_{i}",
                        "query":        item["query"],
                        "evidence":     item["evidence"],
                        "ground_truth": item["ground_truth"],
                        "pro_argument": item.get("pro_argument", ""),
                        "con_argument": item.get("con_argument", ""),
                        "doc_type":     "research_paper",
                        "source":       "synthetic",
                    })
                return results

            except Exception as e:
                print(f"[SyntheticGen] Failed for passage {passage_id}: {e}")
                return []

    async def generate(
        self,
        n_papers:    int = 100,
        output_path: str = "data/synthetic/queries.jsonl",
    ) -> int:
        """
        Generate synthetic queries from n_papers SQuAD passages.
        Target: ~300 queries from 100 passages (3 per passage).
        """
        print(f"[SyntheticGen] Loading SQuAD for {n_papers} passages...")
        dataset  = load_dataset("rajpurkar/squad", split="train")

        # Collect unique contexts (passages) — SQuAD has many questions per context
        seen     = set()
        passages = []
        for i, example in enumerate(dataset):
            ctx = example["context"]
            if ctx not in seen and len(ctx) > 200:
                seen.add(ctx)
                passages.append((str(i), ctx))
            if len(passages) >= n_papers:
                break

        print(f"[SyntheticGen] Generating from {len(passages)} passages...")

        tasks   = [
            self.generate_from_passage(pid, passage)
            for pid, passage in passages
        ]
        results = await asyncio.gather(*tasks)

        # Flatten and write to JSONL
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(output_path, "w") as f:
            for batch in results:
                for item in batch:
                    f.write(json.dumps(item) + "\n")
                    total += 1

        print(f"[SyntheticGen] Wrote {total} synthetic queries → {output_path}")
        return total