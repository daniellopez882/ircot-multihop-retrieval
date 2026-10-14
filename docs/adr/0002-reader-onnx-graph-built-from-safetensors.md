# ADR 0002: Build the reader's ONNX graph from its safetensors weights

Status: accepted

## Context

The extractive reader is `deepset/minilm-uncased-squad2`, a 12-layer, 384-wide BERT fine-tuned on
SQuAD 2.0. Its model repository publishes PyTorch, Flax and safetensors weights but no ONNX file.
The usual export path (`torch.onnx` or `optimum`) needs PyTorch, which is several times larger
than every other dependency of this project combined, for a one-off conversion. The ONNX copies
on the Hub are third-party uploads whose provenance cannot be checked against the original
weights.

The reader is also the benchmark's main cost: every configuration reads passages with it, so its
speed on a six-core CPU decides whether the full run takes one hour or several.

## Decision

`ircot.onnx_bert` writes the graph with `onnx.helper` from the pinned, checksum-verified
safetensors file: embeddings and their LayerNorm from standard opset-17 operators, and each
encoder layer from onnxruntime's fused CPU operators (`Attention` with one packed query/key/value
projection, `SkipLayerNormalization` for bias + residual + norm, `BiasGelu` for the exact erf
GELU). `ircot fetch` runs the export once into the data directory, and every results file records
the exported graph's sha256.

Both onnxruntime sessions (reader and encoder) run with worker-thread spinning turned off.

The graph is tested against an independent NumPy implementation of the BERT forward pass on
random weights, including padded batches. The exported reader is checked in the integration
suite on reading-comprehension questions and on a passage long enough to need a second window.

## Consequences

- No PyTorch anywhere: the runtime is onnxruntime, tokenizers and NumPy, and the container stays
  small.
- The graph runs on onnxruntime only (the fused operators are not part of the ONNX standard), and
  the exporter covers the BERT architecture only.
- During development, an export written with standard operators only produced the same logits
  to within 1e-4 and the same spans on a 100-passage check, and was roughly 30% slower.
  Spinning threads were the larger cost on a machine shared with other processes.
- The exported bytes depend on the installed `onnx` version, so the graph's sha256 is recorded per
  run rather than pinned, and the reader's cache key includes it.

## Alternatives rejected

- PyTorch-based export: correct and standard, but a heavy dependency for one conversion.
- A third-party ONNX upload: unverifiable provenance.
- Dynamic int8 quantisation: on a 100-passage spot check it was barely faster on this CPU and
  picked a different top span for most passages, so the reader stays fp32.
- Standard operators only: portable to other ONNX runtimes, but slower, and nothing else in the
  project needs another runtime.
- A NumPy forward pass at run time: simple, but much slower than onnxruntime's kernels.
