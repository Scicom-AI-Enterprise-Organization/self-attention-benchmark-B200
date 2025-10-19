# self-attention-benchmark-B200

Benchmark causal self-attention in Bfloat16 forward-backward using B200.

MFU reported is super low < 20% on B200 from [Scicom-AI-Enterprise-Organization/small-malaysian-lm-B200](https://github.com/Scicom-AI-Enterprise-Organization/small-malaysian-lm-B200) for full parameter Qwen3 1.7B finetuning 4k proper multipacking, I think we need to fine better attention backend before run the entire experiments, so we are going to benchmark on various cases,

1. Single document (full length), eg, [1024], [2048]
2. Consistent multi-doc splits, eg, [256, 256, 256, 256] for maxlen 1024
3. Randomized varlen variable, but sum = total length (eg, [300, 200, 500, 24]), keeping total tokens consistent

## How to

1. First generate the same random sequences to be use for the benchmark,

```bash
python3 generate_random.py
```

But do not worry, we uploaded the randomize lengths in [randomize](randomize) so we make sure the lengths use across benchmarks are consistent.

2. Run benchmark,

Flash Attention 2,

```bash
python3 run.py --attention "fa2" > outputs/out-fa2
```

Flex Attention,

```bash
python3 run.py --attention "flex" > outputs/out-flex
```