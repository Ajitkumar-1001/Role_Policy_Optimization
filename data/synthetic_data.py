"""
synthetic_gen.py
Generates contested synthetic queries from QASPER papers using GPT-4o.
These are the HIGH VARIANCE queries that make DAPO learn best —
questions where role assignment genuinely changes the verdict.
"""

import json
import asyncio
from pathlib import Path
import litellm # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from datasets import load_dataset  # pyright: ignore[reportMissingImports]
from tqdm.asyncio import tqdm  # pyright: ignore[reportMissingModuleSource]


GENERATION_PROMPT = """You are generating training data for a multi-agent adversarial reasoning system.

Given this academic paper excerpt:
---
{excerpt}
---

Generate 3 contested questions about this excerpt. A contested question is one where:
- Reasonable people could disagree on the answer
- Both a "yes" and "no" side have defensible arguments from the text
- The answer is NOT immediately obvious from a single sentence

For each question provide:
- A strong argument FOR a positive/affirmative answer (citing specific evidence)
- A strong argument AGAINST (citing specific contradictions or gaps)
- The most defensible ground truth answer

Return ONLY valid JSON array, no markdown:
[
  {{
    "query": "<the contested question>",
    "pro_argument": "<strongest case for yes>",
    "con_argument": "<strongest case for no>",
    "ground_truth": "<most defensible answer with brief reasoning>",
    "evidence": "<the most relevant 2-3 sentences from the excerpt>"
  }},
  ...3 items total
]"""


class SyntheticGenerator:
    """
    Generates contested queries from QASPER papers.
    Output: data/synthetic/queries.jsonl
    Each line = one TrainingQuery-compatible JSON object.
    """

    def __init__(self, model: str = "openai/gpt-4o", concurrency: int = 5):
        self.model       = model
        self.concurrency = concurrency
        self.semaphore   = asyncio.Semaphore(concurrency)

    async def generate_from_paper(
        self,
        paper_id: str,
        abstract: str,
    ) -> list[dict]:
        """Generate 3 contested queries from one paper abstract."""
        async with self.semaphore:
            try:
                response = await litellm.acompletion(
                    model       = self.model,
                    messages    = [{
                        "role": "user",
                        "content": GENERATION_PROMPT.format(
                            excerpt = abstract[:1200]
                        )
                    }],
                    temperature = 0.8,   # some diversity in generation
                    timeout     = 60,
                )
                raw = response.choices[0].message.content or "[]"

                # Strip markdown fences if present
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                items = json.loads(raw)
                results = []
                for i, item in enumerate(items[:3]):
                    results.append({
                        "query_id":     f"synthetic_{paper_id}_{i}",
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
                print(f"[SyntheticGen] Failed for paper {paper_id}: {e}")
                return []

    async def generate(
        self,
        n_papers:   int = 100,
        output_path: str = "data/synthetic/queries.jsonl",
    ) -> int:
        """
        Generate synthetic queries from n_papers QASPER abstracts.
        Target: ~300 queries from 100 papers (3 per paper).
        """
        print(f"[SyntheticGen] Loading QASPER for {n_papers} papers...")
        dataset = load_dataset("allenai/qasper", split="train", trust_remote_code=True)

        papers = []
        for i, example in enumerate(dataset):
            if len(papers) >= n_papers:
                break
            abstract = example.get("abstract", "")
            if abstract and len(abstract) > 200:
                papers.append((str(i), abstract))

        print(f"[SyntheticGen] Generating from {len(papers)} papers...")

        # Run all generations concurrently (rate-limited by semaphore)
        tasks = [
            self.generate_from_paper(pid, abstract)
            for pid, abstract in papers
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

        print(f"[SyntheticGen] Wrote {total} synthetic queries to {output_path}")
        return total