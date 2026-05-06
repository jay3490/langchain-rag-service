# Maximal marginal relevance

Plain similarity search returns the k chunks closest to the query. When a corpus contains
near-duplicate passages, those k chunks often say the same thing, and the language model
receives repeated context instead of complementary evidence.

## How MMR selects results

Maximal marginal relevance (MMR), introduced by Carbonell and Goldstein in 1998, selects
results one at a time. At each step it picks the candidate that maximises

`lambda * sim(query, candidate) - (1 - lambda) * max sim(candidate, already selected)`

The first term rewards relevance to the query. The second term penalises similarity to
results that were already chosen.

## Parameters

The `lambda` parameter, often called `lambda_mult`, ranges from 0 to 1. A value of 1
reduces MMR to plain similarity search, while a value of 0 maximises diversity and ignores
relevance after the first pick. A value of 0.5 balances the two.

MMR does not search the whole index. It first fetches `fetch_k` candidates by similarity,
for example 20, and then re-ranks that pool down to the final `k`. A pool that is too small
leaves nothing diverse to choose from; a pool that is too large spends time comparing
candidates that were never likely to be selected.

## When to use it

MMR helps when documents overlap heavily, such as versioned manuals or support articles
that repeat boilerplate. On a small corpus of distinct documents it changes little.
