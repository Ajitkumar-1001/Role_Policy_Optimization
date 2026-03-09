"""
generate_synthetic.py
Generate synthetic contested queries from SQuAD passages.
Run this BEFORE training to populate data/synthetic/queries.jsonl.

Usage:
  python scripts/generate_synthetic.py
  python scripts/generate_synthetic.py --n_papers 50
  python scripts/generate_synthetic.py --n_papers 100 --model openai/gpt-4o
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
load_dotenv()

import argparse
from data.synthetic_data import SyntheticGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_papers", type=int, default=100,
                        help="Number of SQuAD passages to use (default: 100)")
    parser.add_argument("--output",   type=str, default="data/synthetic/queries.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--model",    type=str, default="openai/gpt-4o",
                        help="LLM model for generation")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel API calls (default: 5)")
    args = parser.parse_args()

    async def run():
        gen   = SyntheticGenerator(model=args.model, concurrency=args.concurrency)
        total = await gen.generate(n_papers=args.n_papers, output_path=args.output)
        print(f"\n✅ Done. Generated {total} synthetic queries → {args.output}")

    asyncio.run(run())


if __name__ == "__main__":
    main()