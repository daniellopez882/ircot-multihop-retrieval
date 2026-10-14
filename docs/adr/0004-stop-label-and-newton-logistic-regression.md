# ADR 0004: What the stop controller predicts, and how it is fitted

Status: accepted

## Context

IRCoT stops when the language model writes "answer is" or when it reaches a step limit. The
offline reasoner never says it is done, so stopping has to be learned from signals the loop
already has: reader confidence, whether the answer moved, whether the bridge linked to a title,
how many new passages the hop added, and how much of the question the pool covers.

Two things needed deciding: the training target and the fitting method. Answer correctness is a
tempting target, but with a fixed extractive reader it is mostly a property of the reader and is
rarely reached on 3- and 4-hop questions, so it would teach the controller to stop early almost
always.

## Decision

- Training data comes from the selected query policy's 4-step loop on the training sample, run
  without stopping. At each decision point (after hops 1 to 3) the label is 1 when no later hop
  adds a gold passage, that is, when stopping now would cost no recall.
- The model is an L2-regularised logistic regression on nine standardised features, fitted with
  Newton's method (IRLS) from a zero start. It has no random state, so the same traces always give
  the same weights. scikit-learn's `LogisticRegression` is used only as a parity check in tests.
- The loop stops when the predicted probability reaches 0.5. Hard budgets (steps, pool tokens,
  reasoner tokens) are checked first and always win.

## Consequences

- The controller trades retrieval cost against retrieval recall, which is what the benchmark
  measures; it does not try to predict answer correctness.
- The label is relative to a 4-step horizon: passages a fifth hop would have found do not count.
- The coefficients are few and readable; they are stored in the results file with the training
  row count and positive rate.

## Alternatives rejected

- Answer EM as the label: dominated by reader quality, and nearly always 0 on long chains.
- A tree ensemble: more capacity than nine features and a few hundred rows can support, and
  harder to read in a results file.
- A fixed threshold on the reader margin alone: one signal, and its scale differs between
  datasets.
