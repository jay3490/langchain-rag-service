# Chunking documents

Retrieval works on chunks rather than whole documents. Embedding models have a maximum
input length, and a chunk that covers one topic produces a sharper embedding than a long
document that covers many.

## Chunk size and overlap

Chunk size is measured in characters or tokens. Small chunks, around 200 to 400 characters,
retrieve precisely but often lose the surrounding context needed to answer a question.
Large chunks, above 1500 characters, keep context but dilute the embedding and waste the
prompt budget of the language model. Values between 500 and 1000 characters are a common
starting point for prose.

Overlap repeats the end of one chunk at the start of the next, so that a sentence cut at a
boundary still appears whole in at least one chunk. An overlap of 10 to 20 percent of the
chunk size is typical. The overlap must always be smaller than the chunk size.

## Recursive splitting

A recursive character splitter tries a list of separators in order. It first splits on
blank lines to keep paragraphs together, then on single newlines, then on spaces, and only
splits inside a word as a last resort. Each piece that is still too large is split again
with the next separator.

For Markdown, the separator list starts with heading markers, so that a chunk tends to
begin at a section heading. This keeps each chunk aligned with the structure of the source
document.

## Metadata

Every chunk should carry the name of the file it came from and its position within that
file. That metadata is what allows an answer to cite its sources and lets a re-ingested
file replace its old chunks instead of duplicating them.
