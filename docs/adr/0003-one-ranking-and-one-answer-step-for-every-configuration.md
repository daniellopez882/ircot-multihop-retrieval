# ADR 0003: One ranking rule and one answer step for every configuration

Status: accepted

## Context

A one-shot retriever returns a ranked list; an iterative loop returns a pool assembled by several
queries. Recall@2 and Recall@5 need a single ranked list, and the way hops are merged can decide
the comparison on its own: concatenating hops in order makes later hops invisible at small k,
while sorting by raw score mixes scores of different queries that are not on one scale.

Answer accuracy has the same problem. If each configuration had its own answering procedure, an
EM/F1 difference could come from the answering rather than from what was retrieved.

## Decision

- The pool becomes a ranking round-robin by hop: every hop's best new passage, then every hop's
  second, and so on; hop order breaks ties. Each loop step retrieves `k` passages that are not
  already pooled, so every hop contributes. Recall over the whole pool is reported next to
  Recall@k.
- Every configuration is answered the same way: the same SQuAD 2.0 reader reads the top five
  passages of the configuration's ranking with the original question, and the highest-margin span
  is the answer.

## Consequences

- EM/F1 differences between configurations are retrieval differences seen through one fixed
  reader. They say nothing about how an LLM reader would use the same passages.
- Round-robin favours breadth: a loop's Recall@5 is built from its first two or three hops.
- The reader answers the original multi-hop question, so a passage that holds the bridge entity
  (not the final answer) can win; the README lists this under its known gaps.

## Alternatives rejected

- Hop-major concatenation: every loop's Recall@2 would equal its first hop's.
- A score-sorted merge: scores from different queries and from min-max fused lists are not
  comparable.
- A loop-specific answer step (for example reading only the last hop): it would mix an answering
  heuristic into what is meant to be a retrieval comparison.
