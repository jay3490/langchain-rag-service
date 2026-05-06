# Conversational retrieval

In a chat, users ask follow-up questions that only make sense with the earlier turns. After
asking "How does BM25 normalise for document length?", a user might ask "What value is
usually used for it?". The follow-up alone contains no searchable terms: "it" refers to the
`b` parameter mentioned in the previous turn.

## History-aware query rewriting

A history-aware retriever first asks the language model to rewrite the latest question
into a standalone question, using the conversation so far. The rewritten question, such as
"What value is usually used for the BM25 length normalisation parameter b?", is what gets
sent to the retriever. The original question is still shown to the model when it writes
the final answer.

When there is no history yet, the rewrite step is skipped and the question is used as it
is. This saves a model call on the first turn of every conversation.

## Storing history

Each conversation is identified by a session id. The service keeps the messages of each
session and appends the user question and the assistant answer after every turn. Keeping
only the most recent messages bounds the size of the rewrite prompt; older turns are
dropped first.

In-memory storage is lost when the process restarts and is not shared between replicas. A
deployment with several workers would move the history into a shared store such as Redis or
a database.
