"""Prompt templates and the helpers that render retrieved context into them."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

NO_ANSWER = "I could not find the answer in the provided documents."

ANSWER_SYSTEM = (
    "You answer questions using only the numbered sources inside <context>.\n"
    "Cite each statement with the number of the source it came from in square brackets, "
    "for example [2]. Do not cite sources that you did not use.\n"
    "If the sources do not contain the answer, reply exactly: " + NO_ANSWER + "\n"
    "Do not use outside knowledge."
)

ANSWER_HUMAN = "<context>\n{context}\n</context>\n\n<question>{question}</question>"


def answer_prompt() -> ChatPromptTemplate:
    """System instructions, optional chat history, then the context and question."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", ANSWER_HUMAN),
        ]
    )


REWRITE_SYSTEM = (
    "You rewrite follow-up questions for a search engine.\n"
    "Given the conversation and the latest question, write a single standalone question "
    "that can be understood without the conversation. Resolve pronouns and references to "
    "earlier turns. Do not answer the question. Reply with the question only."
)

REWRITE_HUMAN = "<conversation>\n{conversation}\n</conversation>\n\n<question>{question}</question>"


def rewrite_prompt() -> ChatPromptTemplate:
    """Prompt that turns a follow-up question into a standalone search query."""
    return ChatPromptTemplate.from_messages([("system", REWRITE_SYSTEM), ("human", REWRITE_HUMAN)])


RERANK_SYSTEM = (
    "You judge how useful a passage is for answering a search query.\n"
    "Reply with a single integer from 0 to 10, where 0 means the passage is unrelated "
    "and 10 means it directly answers the query. Reply with the number only."
)

RERANK_HUMAN = "<query>{query}</query>\n\n<passage>\n{passage}\n</passage>"


def rerank_prompt() -> ChatPromptTemplate:
    """Prompt that asks the chat model for a 0-10 relevance grade of one passage."""
    return ChatPromptTemplate.from_messages([("system", RERANK_SYSTEM), ("human", RERANK_HUMAN)])


def format_conversation(messages: Sequence[BaseMessage]) -> str:
    """Render chat history as ``user:``/``assistant:`` lines, one message per line."""
    roles = {"human": "user", "ai": "assistant"}
    return "\n".join(
        f"{roles.get(message.type, message.type)}: {' '.join(message.text.split())}"
        for message in messages
    )


def source_label(document: Document) -> str:
    """Human-readable origin of a chunk, e.g. ``manual.pdf (page 3)``."""
    label = str(document.metadata.get("source", "unknown"))
    page = document.metadata.get("page")
    return f"{label} (page {page})" if page is not None else label


def format_context(documents: Sequence[Document]) -> str:
    """Render documents as numbered ``<source>`` blocks; numbers start at 1."""
    return "\n".join(
        f'<source id="{number}" name="{source_label(document)}">\n'
        f"{document.page_content.strip()}\n</source>"
        for number, document in enumerate(documents, start=1)
    )
