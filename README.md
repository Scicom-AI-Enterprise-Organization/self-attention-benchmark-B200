# causal-self-attention-benchmark-B200

Benchmark causal self-attention (BFloat16, forward+backward) across attention backends and GPU hardware.

## Motivation

MFU was reported low (< 50%) on B200 from [Scicom-AI-Enterprise-Organization/small-malaysian-lm-B200](https://github.com/Scicom-AI-Enterprise-Organization/small-malaysian-lm-B200) during full-parameter Qwen3 1.7B finetuning at 4K with proper multipacking. Profiling showed significant time spent in attention backward:

<img src="profiling.png" width="50%">

Before running full experiments, we benchmark attention backends across different sequence lengths and document packing scenarios.

## Benchmark scenarios

1. **Single document** (full length): `[1024, 2048, 4096, 8192, 12288, 16384, 32768, 65536]`
2. **Consistent multi-doc splits**: same total length split into equal chunks, splits `[4, 5, 6, 7, 8, 9]`
3. **Randomized varlen**: random chunk sizes summing to total length, splits `[4, 5, 6, 7, 8, 9]`

## How to run

Generate reproducible random sequences (or use pre-generated ones in [randomize/](randomize/)):

```bash
python3 generate_random.py
```

Run benchmarks (v1, up to 12K tokens):

```bash
python3 run.py --attention "fa3"       > outputs/out-fa3
python3 run.py --attention "fa2"       > outputs/out-fa2
python3 run.py --attention "flex"      > outputs/out-flex
python3 run.py --attention "fa_triton" > outputs/out-fa-triton
```

Run benchmarks (v2, up to 65K tokens, includes FA4):

```bash
python3 run.py --attention "fa4"       > outputs-v2/out-fa4
python3 run.py --attention "fa3"       > outputs-v2/out-fa3
python3 run.py --attention "fa2"       > outputs-v2/out-fa2
python3 run.py --attention "flex"      > outputs-v2/out-flex
python3 run.py --attention "fa_triton" > outputs-v2/out-fa-triton
```

## Results

All times are `fwd+bwd` latency in milliseconds, single batch.

### v1 — up to 12K tokens (`outputs/`)

#### Single document

| total_len | FA3 H100 | FA2 H100 | Flex H100 | FA2 B200 | Flex B200 | FA-Triton B200 | FA2 5090 | Flex 5090 |
|----------:|:--------:|:--------:|:---------:|:--------:|:---------:|:--------------:|:--------:|:---------:|
| 1024      | 0.309    | 0.523    | 2.308     | 0.346    | 1.378     | 0.412          | 0.377    | 1.520     |
| 2048      | 0.462    | 0.790    | 3.086     | 0.672    | 1.916     | 1.097          | 0.895    | 2.676     |
| 4096      | 1.145    | 2.081    | 5.570     | 1.843    | 4.166     | 2.538          | 2.760    | 6.495     |
| 8192      | 3.412    | 6.526    | 15.105    | 6.031    | 7.362     | 8.198          | 9.640    | 21.576    |
| 12288     | 7.240    | 13.469   | 30.319    | 12.754   | 14.560    | 17.275         | 20.819   | 46.471    |

**Hardware:** H100 = NVIDIA H100 80GB HBM3 (SXM), 5090 = NVIDIA GeForce RTX 5090

Full multi-doc results: [outputs/](outputs/)

---

### v2 — up to 65K tokens (`outputs-v2/`)

Adds Flash Attention 4 (FA4) and extends sequence lengths to 65536.

#### Single document

| total_len | FA4 B200 | FA4 H100 | FA3 H100 | FA2 H100 | FA2 B200 | Flex B200 | FA-Triton B200 | Flex H100 |
|----------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------:|:--------------:|:---------:|
| 1024      | 1.850    | 0.386    | 0.361    | 0.399    | 0.352    | 8.544     | 2.462          | 1.871     |
| 2048      | 2.894    | 0.507    | 0.510    | 0.740    | 0.669    | 9.729     | 3.144          | 2.686     |
| 4096      | 2.917    | 1.166    | 1.100    | 2.017    | 1.838    | 10.260    | 4.137          | 5.198     |
| 8192      | 3.293    | 3.502    | 3.399    | 6.464    | 6.035    | 12.592    | 8.777          | 14.673    |
| 12288     | 4.385    | 7.767    | 7.478    | 13.763   | 12.685   | 17.847    | 17.822         | 30.039    |
| 16384     | 6.725    | 13.164   | 12.739   | 23.421   | 21.808   | 26.347    | 30.393         | 51.673    |
| 32768     | 24.783   | 52.578   | 49.992   | 92.290   | 82.974   | 91.169    | 118.652        | 199.195   |
| 65536     | **95.604**  | 202.336  | 195.760  | 369.234  | 325.841  | 353.303   | 485.893        | 791.442   |

**Key findings:**
- **FA4 on B200 is the clear winner at long context** — 2.1× faster than FA4 on H100 at 65K tokens, and ~3.4× faster than FA2 on B200
- **FA3/FA4 on H100 are comparable** at all sequence lengths
- **Flex Attention is significantly slower** across all hardware — roughly 4–8× slower than FA3/FA4 at large sequences
- **FA-Triton on B200** falls between FA2 B200 and Flex B200 at long context

#### Consistent multi-doc (splits=4, single representative)

| total_len | FA4 B200 | FA4 H100 | FA3 H100 | FA2 B200 |
|----------:|:--------:|:--------:|:--------:|:--------:|
| 8192      | 2.766    | 1.471    | 1.426    | 1.935    |
| 16384     | 3.460    | 4.212    | 3.972    | 6.235    |
| 32768     | 7.640    | 14.740   | 13.881   | 22.294   |
| 65536     | 26.649   | 54.958   | 52.737   | 84.385   |

Multi-doc packing dramatically reduces attention cost vs single-doc by reducing the effective max sequence length per sample.

Full multi-doc results: [outputs-v2/](outputs-v2/)
