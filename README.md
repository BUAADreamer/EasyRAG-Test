# EasyRAG

Easy-to-Use Retrieval-Augmented Generation Tool of LLM.

## Use

### build

```shell
python3 builder.py --config configs/bge_small.yaml
```

### test

```shell
python3 test.py --config configs/bge_small.yaml
```

## Supported RAG Algorithms

- [x] ICRALM [1]
- [x] REPLUG [2]
- [x] Self-RAG [3]

## References

1. Ram O, Levine Y, Dalmedigos I, et al. In-context retrieval-augmented language models[J]. arXiv preprint arXiv:2302.00083, 2023.
2. Shi W, Min S, Yasunaga M, et al. Replug: Retrieval-augmented black-box language models[J]. arXiv preprint arXiv:2301.12652, 2023.
3. Asai A, Wu Z, Wang Y, et al. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection[J]. arXiv preprint arXiv:2310.11511, 2023.