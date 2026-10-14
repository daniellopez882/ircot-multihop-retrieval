# ADR 0001: Retrieve over a shared corpus; tune only on a separate training sample

Status: accepted

## Context

MuSiQue and HotpotQA ship each question with its own small paragraph set (20 and 10 paragraphs).
Retrieving inside that set is close to trivial for a dense model: two to four of the paragraphs
are supporting and the rest are distractors picked for that one question. The IRCoT paper instead
retrieves from a corpus built across many questions, which is what makes a bad first hop costly.

The loop also has knobs that change its results: the dense weight of the hybrid retriever, the
query-formulation policy and the stop controller's weights. Picking any of them by looking at the
evaluation questions would turn the reported comparison into a tuned one.

## Decision

- Each benchmark sample retrieves over one corpus: the deduplicated union of the paragraphs of all
  its questions. Gold passages are mapped to corpus ids once, when the corpus is built.
- Two disjoint fixed-seed samples are drawn per dataset from one permutation: the evaluation
  sample (the first 1,000) and a training sample (the next 200). The training sample gets its own
  union corpus.
- Everything that is chosen is chosen on the training sample, in this order: the hybrid dense
  weight (a grid of three, by one-shot Recall@5), the query policy (by pool recall of the 4-step
  loop), then the stop controller (fitted on that policy's per-hop traces). Each configuration
  runs over the evaluation sample once.

## Consequences

- Retrieval is harder than in the distractor setting, and differences between one-shot and
  iterative retrieval become visible. The absolute numbers are not comparable with papers that
  use the full Wikipedia corpus or the per-question paragraph sets.
- Corpus size, and therefore difficulty, depends on the sample size. The results file records the
  number of passages and the corpus sha256 with every run.
- Embedding two corpora per dataset costs encoder time; embeddings are cached by exact passage
  list, so repeated runs do not pay it again.

## Alternatives rejected

- Per-question paragraph sets: too easy to separate the methods, and the loop's "retrieve k new
  passages" step would run out of passages after two hops.
- Full Wikipedia (as the paper does for HotpotQA): far beyond the disk and CPU budget of this
  project, and MuSiQue has no official full corpus.
- Tuning on the evaluation sample: the comparison against one-shot retrieval would favour
  whichever side got tuned.
