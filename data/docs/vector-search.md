# Dense vector search

Dense retrieval represents every chunk of text as an embedding: a fixed-length vector of
floating point numbers produced by an embedding model. A query is embedded with the same
model, and the chunks whose vectors lie closest to the query vector are returned.

## Distance metrics

Two metrics are common. Euclidean (L2) distance measures the straight-line distance
between two vectors; smaller is closer. Inner product measures alignment; larger is closer.
When every vector is normalised to unit length, ranking by L2 distance and ranking by inner
product (cosine similarity) produce the same order, so normalising embeddings before
indexing removes the need to choose.

## FAISS index types

FAISS is a library for similarity search over dense vectors. Its simplest index,
`IndexFlatL2`, compares the query against every stored vector. Flat search is exact and
needs no training, which makes it the right default for collections of up to a few hundred
thousand vectors.

Larger collections use approximate indexes. An IVF index clusters vectors into inverted
lists and only scans the lists nearest to the query; the `nprobe` parameter trades recall
for speed. HNSW builds a navigable small-world graph and walks it greedily from an entry
point. Approximate indexes can miss a true nearest neighbour, so recall should be measured
against a flat index before switching.

## Persistence

A FAISS index lives in memory. To survive a restart it must be written to disk and loaded
again. The LangChain FAISS wrapper stores the raw index in `index.faiss` and the mapping
from vector positions to documents in `index.pkl`. The pickle file can execute code when
loaded, so it should only ever be loaded from a directory the service wrote itself.

## Weaknesses

Embeddings capture meaning but blur exact tokens. A query for an error code such as
`E1042` or a function name may retrieve passages about similar errors rather than the one
that contains the identifier. Lexical retrieval such as BM25 covers that gap.
