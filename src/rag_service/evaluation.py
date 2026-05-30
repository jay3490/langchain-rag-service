"""Retrieval evaluation: hit rate@k and mean reciprocal rank over a JSONL question set.

Each line of the question file is a JSON object::

    {"question": "...", "relevant": ["bm25.md"]}

A retrieved chunk counts as relevant when its ``source`` is listed in ``relevant``.
Ranks are chunk ranks, starting at 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.retrievers import BaseRetriever

from rag_service.config import RetrieverMode, Settings
from rag_service.ingestion import load_directory, split_documents
from rag_service.providers import build_embeddings
from rag_service.retrieval import RetrievalOptions, build_retriever
from rag_service.service import embedding_id
from rag_service.store import FaissIndex

MODES: tuple[RetrieverMode, ...] = ("dense", "bm25", "hybrid")


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    relevant: frozenset[str]


@dataclass
class QuestionResult:
    question: str
    retrieved: list[str]
    first_relevant_rank: int | None


@dataclass
class EvalReport:
    mode: str
    search_type: str
    k: int
    questions: int
    hit_rate: float
    mrr: float
    results: list[QuestionResult] = field(default_factory=list)


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read a JSONL question set; blank lines are skipped."""
    questions: list[EvalQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        relevant = record.get("relevant")
        if not isinstance(record.get("question"), str) or not relevant:
            raise ValueError(f"{path}:{number}: expected 'question' and a non-empty 'relevant'")
        questions.append(EvalQuestion(record["question"], frozenset(map(str, relevant))))
    return questions


def first_relevant_rank(retrieved: Sequence[str], relevant: frozenset[str]) -> int | None:
    """1-based rank of the first relevant item, or ``None`` if there is none."""
    for rank, source in enumerate(retrieved, start=1):
        if source in relevant:
            return rank
    return None


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of questions with a relevant item within the top ``k``."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Mean of ``1 / rank`` over questions; a missing relevant item contributes 0."""
    if not ranks:
        return 0.0
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def evaluate(
    retriever: BaseRetriever,
    questions: Sequence[EvalQuestion],
    *,
    mode: str,
    search_type: str,
    k: int,
) -> EvalReport:
    """Run every question through ``retriever`` and score the top ``k`` chunks."""
    results: list[QuestionResult] = []
    for item in questions:
        documents = retriever.invoke(item.question)[:k]
        retrieved = [str(d.metadata.get("source", "")) for d in documents]
        results.append(
            QuestionResult(item.question, retrieved, first_relevant_rank(retrieved, item.relevant))
        )
    ranks = [r.first_relevant_rank for r in results]
    return EvalReport(
        mode=mode,
        search_type=search_type,
        k=k,
        questions=len(results),
        hit_rate=hit_rate(ranks, k),
        mrr=mean_reciprocal_rank(ranks),
        results=results,
    )


def run(
    settings: Settings,
    docs: Path,
    questions_path: Path,
    *,
    k: int,
    modes: Sequence[RetrieverMode],
) -> list[EvalReport]:
    """Index ``docs`` into a temporary directory and evaluate each retriever mode."""
    questions = load_questions(questions_path)
    embeddings = build_embeddings(settings)
    chunks = split_documents(load_directory(docs), settings.chunk_size, settings.chunk_overlap)
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        index = FaissIndex(Path(tmp), embeddings, embedding_id(settings))
        index.add(chunks)
        base = RetrievalOptions.from_settings(settings).with_overrides(
            k=k, fetch_k=max(k, settings.fetch_k)
        )
        return [
            evaluate(
                build_retriever(index.snapshot, base.with_overrides(mode=mode)),
                questions,
                mode=mode,
                search_type=base.search_type if mode != "bm25" else "-",
                k=k,
            )
            for mode in modes
        ]


def format_table(reports: Sequence[EvalReport]) -> str:
    header = f"{'mode':<8} {'search':<10} {'k':>3} {'questions':>9} {'hit_rate@k':>10} {'mrr':>6}"
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r.mode:<8} {r.search_type:<10} {r.k:>3} {r.questions:>9} "
            f"{r.hit_rate:>10.3f} {r.mrr:>6.3f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Measure retrieval hit rate@k and MRR on a JSONL question set.",
    )
    parser.add_argument("--docs", type=Path, default=Path("data/docs"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("-k", type=int, default=None, help="cut-off (default: RAG_TOP_K)")
    parser.add_argument("--mode", choices=MODES, action="append", dest="modes")
    parser.add_argument("--search-type", choices=("similarity", "mmr"), default=None)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--misses", action="store_true", help="list questions with no hit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {"search_type": args.search_type} if args.search_type else {}
    settings = Settings(**overrides)
    k = args.k or settings.top_k
    reports = run(settings, args.docs, args.questions, k=k, modes=args.modes or MODES)

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
        return 0
    embedder = (
        f"fake (hashing embeddings, dim {settings.fake_embedding_dim})"
        if settings.provider == "fake"
        else f"openai-compatible ({settings.embedding_model})"
    )
    print(f"provider: {embedder}")
    print(f"chunks: size {settings.chunk_size}, overlap {settings.chunk_overlap}\n")
    print(format_table(reports))
    if args.misses:
        for report in reports:
            for result in report.results:
                if result.first_relevant_rank is None or result.first_relevant_rank > k:
                    print(f"\n[{report.mode}] miss: {result.question}\n  got: {result.retrieved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
