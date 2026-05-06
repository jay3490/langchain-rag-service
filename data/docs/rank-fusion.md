# Reciprocal rank fusion

A hybrid retriever runs several retrievers for the same query and merges their result
lists. The scores produced by different retrievers are not comparable: a BM25 score is an
unbounded sum, while a cosine similarity lies between -1 and 1. Reciprocal rank fusion
(RRF) sidesteps the problem by ignoring scores and using only ranks.

## The formula

For every document, RRF adds `1 / (k + rank)` for each result list in which the document
appears, where `rank` starts at 1 for the top result. The constant `k` is commonly set to
60, the value used in the original paper by Cormack, Clarke and Buettcher (2009). A larger
`k` flattens the curve so that lower-ranked results still contribute; a smaller `k` lets
the top few positions dominate.

Documents are then sorted by their fused score. A document that appears near the top of
both lists beats a document that is first in one list and absent from the other.

## Weights

A weighted variant multiplies each list's contribution by a weight, for example to trust
the dense retriever more than the lexical one. With equal weights the method is symmetric.
A weight of zero removes a retriever from the fusion without changing any other code.

## Deduplication

The same chunk can be returned by both retrievers, so fusion needs a stable identity for
each chunk. Using a content hash or an explicit chunk id as the key avoids counting one
chunk as two different results.
