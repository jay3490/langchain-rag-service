import math

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_service.config import Settings
from rag_service.providers import (
    NO_ANSWER,
    ExtractiveFakeChatModel,
    HashingEmbeddings,
    build_chat_model,
    build_embeddings,
)
from rag_service.text import split_sentences, tokenize


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_tokenize_drops_stop_words_and_keeps_identifiers() -> None:
    assert tokenize("What is the top_k of BM25?") == ["top_k", "bm25"]


def test_split_sentences() -> None:
    assert split_sentences("One.  Two!\nThree?") == ["One.", "Two!", "Three?"]


def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    embeddings = HashingEmbeddings(dim=64)
    first = embeddings.embed_query("reciprocal rank fusion")
    second = HashingEmbeddings(dim=64).embed_documents(["reciprocal rank fusion"])[0]

    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


def test_hashing_embeddings_reflect_shared_vocabulary() -> None:
    embeddings = HashingEmbeddings()
    query = embeddings.embed_query("BM25 length normalisation")
    related = embeddings.embed_query("BM25 applies document length normalisation")
    unrelated = embeddings.embed_query("Server-sent events stream tokens to browsers")

    assert _cosine(query, related) > _cosine(query, unrelated)


def test_empty_text_embeds_to_zero_vector() -> None:
    assert HashingEmbeddings(dim=16).embed_query("the of and") == [0.0] * 16


PROMPT = (
    '<context>\n<source id="1" name="a.md">Cats sleep a lot. Dogs bark at night.</source>\n'
    '<source id="2" name="b.md">BM25 uses the parameter k1 for saturation.</source>\n'
    "</context>\n<question>Which parameter controls saturation in BM25?</question>"
)


def test_fake_model_extracts_and_cites_best_sentence() -> None:
    model = ExtractiveFakeChatModel(max_sentences=1)
    answer = model.invoke([SystemMessage(content="Be grounded."), HumanMessage(content=PROMPT)])
    assert answer.content == "BM25 uses the parameter k1 for saturation. [2]"


def test_fake_model_declines_without_overlap() -> None:
    prompt = '<context><source id="1">Cats sleep.</source></context><question>Quantum?</question>'
    assert ExtractiveFakeChatModel().invoke(prompt).content == NO_ANSWER


def test_fake_model_rewrites_follow_up_questions() -> None:
    prompt = (
        "<conversation>\nuser: How does BM25 normalise length?\nassistant: With b.\n"
        "</conversation>\n<question>What value is typical?</question>"
    )
    rewritten = ExtractiveFakeChatModel().invoke(prompt).content
    assert rewritten == "How does BM25 normalise length? What value is typical?"


def test_fake_model_streams_the_same_text() -> None:
    model = ExtractiveFakeChatModel()
    streamed = "".join(str(chunk.content) for chunk in model.stream(PROMPT))
    assert streamed == model.invoke(PROMPT).content
    assert len(list(model.stream(PROMPT))) > 1


def test_factory_returns_fake_models_by_default() -> None:
    settings = Settings(_env_file=None)
    assert isinstance(build_embeddings(settings), HashingEmbeddings)
    assert isinstance(build_chat_model(settings), ExtractiveFakeChatModel)


def test_factory_configures_openai_compatible_endpoint() -> None:
    settings = Settings(
        _env_file=None,
        provider="openai",
        openai_base_url="http://localhost:11434/v1",
        chat_model="llama3.2",
        embedding_model="nomic-embed-text",
    )

    chat = build_chat_model(settings)
    embeddings = build_embeddings(settings)

    assert isinstance(chat, ChatOpenAI)
    assert chat.model_name == "llama3.2"
    assert chat.openai_api_base == "http://localhost:11434/v1"
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "nomic-embed-text"
    assert embeddings.check_embedding_ctx_length is False
