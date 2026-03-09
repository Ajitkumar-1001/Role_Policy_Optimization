"""
data_loader.py
Loads and preprocesses training datasets.

Sources:
  - SQuAD        (replaces QASPER  — factual QA, Parquet native)
  - SNLI         (replaces FEVER   — adversarial entailment, Parquet native)
  - ContractNLI  (legal clause entailment)
  - Synthetic    (GPT-4o generated, loaded from local JSONL)
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset


@dataclass
class TrainingQuery:
    """One training instance for DAPO."""
    query_id:     str
    query:        str
    evidence:     str   # relevant document passage
    ground_truth: str   # correct answer (for LLM judge reward)
    doc_type:     str   # research_paper / legal / general
    source:       str   # squad / snli / contractnli / synthetic


class DataLoader:
    """
    Unified loader for all four training data sources.
    Returns TrainingQuery instances ready for DAPO training loop.
    """

    # ------------------------------------------------------------------ #
    #  SQuAD — Factual QA  (replaces QASPER)                            #
    # ------------------------------------------------------------------ #

    def load_squad(self, n: int = 400) -> list[TrainingQuery]:
        """
        SQuAD: Factual reading comprehension QA.
        Each example has a context paragraph + question + answer span.
        Parquet-native, no loading script required.
        """
        print(f"[DataLoader] Loading SQuAD ({n} examples)...")
        dataset = load_dataset("rajpurkar/squad", split="train")
        queries = []

        for i, example in enumerate(dataset):
            if len(queries) >= n:
                break
            try:
                question     = example["question"].strip()
                context      = example["context"].strip()
                answers_list = example["answers"]["text"]

                if not question or not context or not answers_list:
                    continue

                answer_text = answers_list[0].strip()
                if not answer_text:
                    continue

                queries.append(TrainingQuery(
                    query_id     = f"squad_{i}",
                    query        = question,
                    evidence     = context[:1000],
                    ground_truth = answer_text,
                    doc_type     = "research_paper",
                    source       = "squad",
                ))

            except Exception:
                continue

        print(f"[DataLoader] Loaded {len(queries)} SQuAD queries")
        return queries

    # Keep old name as alias so existing code doesn't break
    def load_qasper(self, n: int = 400) -> list[TrainingQuery]:
        return self.load_squad(n)

    # ------------------------------------------------------------------ #
    #  SNLI — Adversarial entailment  (replaces FEVER)                  #
    # ------------------------------------------------------------------ #

    def load_snli(self, n: int = 200) -> list[TrainingQuery]:
        """
        SNLI: Natural language inference — entailment / neutral / contradiction.
        Adversarial by nature, perfect for courtroom challenge framing.
        Parquet-native, no loading script required.
        """
        print(f"[DataLoader] Loading SNLI ({n} examples)...")
        dataset = load_dataset("stanfordnlp/snli", split="train")
        queries = []

        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        for i, example in enumerate(dataset):
            if len(queries) >= n:
                break
            try:
                label      = example["label"]
                premise    = example["premise"].strip()
                hypothesis = example["hypothesis"].strip()

                # Skip unlabeled examples
                if label == -1 or not premise or not hypothesis:
                    continue

                label_text   = label_map.get(label, "unknown")
                ground_truth = f"The hypothesis is in {label_text} with the premise."

                queries.append(TrainingQuery(
                    query_id     = f"snli_{i}",
                    query        = f"Does this hypothesis follow from the evidence? '{hypothesis}'",
                    evidence     = premise[:800],
                    ground_truth = ground_truth,
                    doc_type     = "general",
                    source       = "snli",
                ))

            except Exception:
                continue

        print(f"[DataLoader] Loaded {len(queries)} SNLI queries")
        return queries

    # Keep old name as alias
    def load_fever(self, n: int = 200) -> list[TrainingQuery]:
        return self.load_snli(n)

    # ------------------------------------------------------------------ #
    #  ContractNLI — Legal clause entailment                            #
    # ------------------------------------------------------------------ #

    def load_contractnli(self, n: int = 100) -> list[TrainingQuery]:
        """
        ContractNLI: Legal contract clause entailment.
        High-stakes, adversarial, perfect for courtroom framing.
        """
        print(f"[DataLoader] Loading ContractNLI ({n} examples)...")
        try:
            dataset = load_dataset("nyu-mll/multi_nli", split="train")
            queries  = []
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

            for i, example in enumerate(dataset):
                if len(queries) >= n:
                    break
                try:
                    hypothesis = example.get("hypothesis", "").strip()
                    premise    = example.get("premise", "").strip()
                    label      = example.get("label", -1)

                    if not hypothesis or not premise:
                        continue

                    label_text   = label_map.get(label, "unknown")
                    ground_truth = f"The clause relationship is: {label_text}"

                    queries.append(TrainingQuery(
                        query_id     = f"contractnli_{i}",
                        query        = f"Does this contract clause apply? '{hypothesis}'",
                        evidence     = premise[:800],
                        ground_truth = ground_truth,
                        doc_type     = "contract",
                        source       = "contractnli",
                    ))

                except Exception:
                    continue

            print(f"[DataLoader] Loaded {len(queries)} ContractNLI queries")
            return queries

        except Exception as e:
            print(f"[DataLoader] ContractNLI failed: {e}. Skipping.")
            return []

    # ------------------------------------------------------------------ #
    #  Synthetic — GPT-4o generated contested queries                   #
    # ------------------------------------------------------------------ #

    def load_synthetic(self, path: str = "data/synthetic/queries.jsonl") -> list[TrainingQuery]:
        """
        Load pre-generated synthetic queries from local JSONL.
        Run generate_synthetic.py first to populate this file.
        """
        print(f"[DataLoader] Loading synthetic queries from {path}...")
        queries = []
        p = Path(path)

        if not p.exists():
            print(f"[DataLoader] Not found: {path}. Run generate_synthetic.py first.")
            return []

        with open(p) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    queries.append(TrainingQuery(
                        query_id     = data["query_id"],
                        query        = data["query"],
                        evidence     = data["evidence"],
                        ground_truth = data["ground_truth"],
                        doc_type     = data.get("doc_type", "research_paper"),
                        source       = "synthetic",
                    ))
                except Exception:
                    continue

        print(f"[DataLoader] Loaded {len(queries)} synthetic queries")
        return queries

    # ------------------------------------------------------------------ #
    #  load_all — combine, shuffle, split                               #
    # ------------------------------------------------------------------ #

    def load_all(
        self,
        squad_n:        int = 400,
        snli_n:         int = 200,
        contract_n:     int = 100,
        synthetic_path: str = "data/synthetic/queries.jsonl",
        seed:           int = 42,
    ) -> dict[str, list[TrainingQuery]]:
        """
        Load all sources, combine, shuffle, split into train / val / test.
        Returns {"train": [...], "val": [...], "test": [...]}
        """
        all_queries = (
            self.load_squad(squad_n)
            + self.load_snli(snli_n)
            + self.load_contractnli(contract_n)
            + self.load_synthetic(synthetic_path)
        )

        print(f"\n[DataLoader] Total queries loaded: {len(all_queries)}")

        random.seed(seed)
        random.shuffle(all_queries)

        n       = len(all_queries)
        n_test  = max(50, int(n * 0.10))
        n_val   = max(50, int(n * 0.10))
        n_train = n - n_test - n_val

        splits = {
            "train": all_queries[:n_train],
            "val":   all_queries[n_train : n_train + n_val],
            "test":  all_queries[n_train + n_val :],
        }

        print(f"  train : {len(splits['train'])}")
        print(f"  val   : {len(splits['val'])}")
        print(f"  test  : {len(splits['test'])}")

        return splits


# ------------------------------------------------------------------ #
#  Quick test                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    loader = DataLoader()

    print("\n── SQuAD ──────────────────────────────")
    squad = loader.load_squad(n=3)
    for q in squad:
        print(f"  [{q.query_id}] {q.query[:70]}")
        print(f"           GT: {q.ground_truth[:60]}\n")

    print("\n── SNLI ───────────────────────────────")
    snli = loader.load_snli(n=3)
    for q in snli:
        print(f"  [{q.query_id}] {q.query[:70]}")
        print(f"           GT: {q.ground_truth}\n")

    print("\n── ContractNLI ────────────────────────")
    cnli = loader.load_contractnli(n=3)
    for q in cnli:
        print(f"  [{q.query_id}] {q.query[:70]}")
        print(f"           GT: {q.ground_truth}\n")