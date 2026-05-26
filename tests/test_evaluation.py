import json
from pathlib import Path

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_service.config import Settings
from rag_service.evaluation import (
    EvalQuestion,
    evaluate,
    first_relevant_rank,
    hit_rate,
    load_questions,
    main,
    mean_reciprocal_rank,
    run,
)
from tests.helpers import REPO_ROOT, SAMPLE_DOCS

QUESTIONS = REPO_ROOT / "data" / "eval" / "questions.jsonl"


def test_first_relevant_rank() -> None:
    relevant = frozenset({"b.md"})
    assert first_relevant_rank(["a.md", "b.md", "b.md"], relevant) == 2
    assert first_relevant_rank(["a.md"], relevant) is None
    assert first_relevant_rank([], relevant) is None


def test_hit_rate_counts_ranks_within_k() -> None:
    assert hit_rate([1, 3, None, 5], k=3) == pytest.approx(0.5)
    assert hit_rate([1, 3, None, 5], k=5) == pytest.approx(0.75)
    assert hit_rate([], k=3) == 0.0


def test_mean_reciprocal_rank() -> None:
    assert mean_reciprocal_rank([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert mean_reciprocal_rank([]) == 0.0


class FixedRetriever(BaseRetriever):
    answers: dict[str, list[str]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [Document(page_content=s, metadata={"source": s}) for s in self.answers[query]]


def test_evaluate_truncates_to_k_and_aggregates() -> None:
    retriever = FixedRetriever(
        answers={"q1": ["x", "a", "b"], "q2": ["b", "b", "a"], "q3": ["x", "y", "a"]}
    )
    questions = [
        EvalQuestion("q1", frozenset({"a"})),
        EvalQuestion("q2", frozenset({"a", "b"})),
        EvalQuestion("q3", frozenset({"a"})),
    ]

    report = evaluate(retriever, questions, mode="test", search_type="-", k=2)

    assert [r.first_relevant_rank for r in report.results] == [2, 1, None]
    assert report.results[2].retrieved == ["x", "y"]
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((0.5 + 1) / 3)


def test_load_questions_validates_records(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text('{"question": "q", "relevant": ["a.md"]}\n\n', encoding="utf-8")
    assert load_questions(good) == [EvalQuestion("q", frozenset({"a.md"}))]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "q", "relevant": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        load_questions(bad)


def test_bundled_question_set_references_existing_documents() -> None:
    sources = {path.name for path in SAMPLE_DOCS.iterdir()}
    questions = load_questions(QUESTIONS)
    assert len(questions) >= 15
    assert all(q.relevant <= sources for q in questions)


def test_run_scores_every_mode_on_the_sample_corpus() -> None:
    reports = run(Settings(_env_file=None), SAMPLE_DOCS, QUESTIONS, k=4, modes=["dense", "bm25"])

    assert [r.mode for r in reports] == ["dense", "bm25"]
    for report in reports:
        assert report.questions == len(load_questions(QUESTIONS))
        assert 0.0 <= report.mrr <= report.hit_rate <= 1.0


def test_cli_prints_a_table_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--docs", str(SAMPLE_DOCS), "--questions", str(QUESTIONS), "-k", "3"]

    assert main([*args, "--mode", "bm25"]) == 0
    table = capsys.readouterr().out
    assert "provider: fake" in table
    assert "hit_rate@k" in table
    assert "bm25" in table

    assert main([*args, "--mode", "hybrid", "--json"]) == 0
    (report,) = json.loads(capsys.readouterr().out)
    assert report["mode"] == "hybrid"
    assert report["k"] == 3
