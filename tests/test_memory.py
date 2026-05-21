from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from rag_service.chain import build_rag_chain, build_rewrite_chain
from rag_service.memory import SessionStore
from rag_service.prompts import format_conversation
from rag_service.providers import ExtractiveFakeChatModel


class RecordingRetriever(BaseRetriever):
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return [Document(page_content="BM25 sets b to 0.75.", metadata={"source": "bm25.md"})]


def test_session_store_appends_and_isolates_sessions() -> None:
    store = SessionStore(max_messages=10)
    store.append_turn("a", "q1", "a1")
    store.append_turn("b", "other", "reply")

    messages = store.get("a")
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert [m.content for m in messages] == ["q1", "a1"]
    assert store.get("missing") == []
    assert len(store) == 2


def test_session_store_trims_oldest_turns() -> None:
    store = SessionStore(max_messages=4)
    for turn in range(5):
        store.append_turn("s", f"q{turn}", f"a{turn}")

    assert [m.content for m in store.get("s")] == ["q3", "a3", "q4", "a4"]


def test_session_store_with_odd_limit_still_starts_on_a_user_turn() -> None:
    store = SessionStore(max_messages=3)
    store.append_turn("s", "q0", "a0")
    store.append_turn("s", "q1", "a1")
    assert [m.content for m in store.get("s")] == ["q1", "a1"]


def test_session_store_evicts_least_recently_used_session() -> None:
    store = SessionStore(max_sessions=2)
    store.append_turn("old", "q", "a")
    store.append_turn("new", "q", "a")
    store.get("old")  # touching a session marks it as recently used
    store.append_turn("newest", "q", "a")

    assert store.get("new") == []
    assert store.get("old") != []


def test_session_store_disabled_and_clear() -> None:
    disabled = SessionStore(max_messages=0)
    disabled.append_turn("s", "q", "a")
    assert disabled.get("s") == []

    store = SessionStore()
    store.append_turn("s", "q", "a")
    assert store.clear("s") is True
    assert store.clear("s") is False


def test_format_conversation() -> None:
    history = [HumanMessage(content="How  does\nBM25 work?"), AIMessage(content="Well.")]
    assert format_conversation(history) == "user: How does BM25 work?\nassistant: Well."


def test_rewrite_is_skipped_without_history() -> None:
    llm = GenericFakeChatModel(messages=iter([]))  # raises if it is ever called
    chain = build_rewrite_chain(llm)
    assert chain.invoke({"question": "What is BM25?", "chat_history": []}) == "What is BM25?"
    assert chain.invoke({"question": "What is BM25?"}) == "What is BM25?"


def test_rewrite_uses_the_model_when_there_is_history() -> None:
    llm = GenericFakeChatModel(
        messages=iter([AIMessage(content="  What value is typical for BM25 b?\n")])
    )
    history = [HumanMessage(content="What is b in BM25?"), AIMessage(content="Length norm.")]

    rewritten = build_rewrite_chain(llm).invoke(
        {"question": "What value is typical for it?", "chat_history": history}
    )

    assert rewritten == "What value is typical for BM25 b?"


def test_follow_up_is_retrieved_with_the_standalone_question() -> None:
    retriever = RecordingRetriever()
    chain = build_rag_chain(ExtractiveFakeChatModel(), retriever)
    history = [
        HumanMessage(content="How does BM25 normalise document length?"),
        AIMessage(content="With the b parameter."),
    ]

    result = chain.invoke({"question": "What value is usual?", "chat_history": history})

    expected = "How does BM25 normalise document length? What value is usual?"
    assert result["question"] == "What value is usual?"
    assert result["standalone_question"] == expected
    assert retriever.queries == [expected]
