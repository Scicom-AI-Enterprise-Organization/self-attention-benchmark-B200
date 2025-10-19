import importlib
import json
import click
from glob import glob
from collections import defaultdict
from helper import clear_cuda_cache, make_varlen_inputs

def print_line(cu_seq_lens_q, query_lens, max_seqlen_q, avg_time):
    cu_seq_lens_q = cu_seq_lens_q.cpu().tolist()
    print(f"{cu_seq_lens_q} | total_len={cu_seq_lens_q[-1]:<5} | splits={len(query_lens):<3} | max_len={max_seqlen_q:<4} | time(fwd+bwd)={avg_time*1000:.3f} ms")

@click.option('--attention', default='fa2', help='attention name.')
@click.option('--lengths', default='[1024, 2048, 4096, 8192, 12288]', help='lengths.')
@click.option('--splits', default='[4, 5, 6, 7, 8, 9]', help='splits.')
@click.command()
def main(attention, lengths, splits):

    module = importlib.import_module(attention)
    benchmark_fn = getattr(module, "benchmark_varlen_bwd")
    lengths = eval(lengths)
    splits = eval(splits)
    print('Single Document,')
    for total_len in lengths:
        clear_cuda_cache()
        query_lens = make_varlen_inputs(total_len, "single", return_tensors=False)
        q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens = make_varlen_inputs(query_lens=query_lens)
        avg_time, cu_seq_lens_q, query_lens, max_seqlen_q = benchmark_fn(q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens)
        print_line(cu_seq_lens_q, query_lens, max_seqlen_q, avg_time)
    print()

    for s in splits:
        print(f'Consistent multi-doc splits {s},')
        for total_len in lengths:
            clear_cuda_cache()
            query_lens = make_varlen_inputs(total_len, "even", num_splits=s, return_tensors=False)
            if query_lens is None:
                continue
            q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens = make_varlen_inputs(query_lens=query_lens)
            avg_time, cu_seq_lens_q, query_lens, max_seqlen_q = benchmark_fn(q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens)
            print_line(cu_seq_lens_q, query_lens, max_seqlen_q, avg_time)
    print()

    files = glob('randomize/*.json')
    splits = defaultdict(dict)
    for f in files:
        splitted = f.split('-')
        s = int(f.split('-')[-1].replace('.json', ''))
        l = int(f.split('-')[1])
        with open(f) as fopen:
            d = json.load(fopen)
            
        splits[s][l] = d

    for s in splits.keys():
        print(f'Randomize multi-doc splits {s},')
        lengths = sorted(splits[s])
        for l in lengths:
            clear_cuda_cache()
            query_lens = splits[s][l]
            q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens = make_varlen_inputs(query_lens=query_lens)
            avg_time, cu_seq_lens_q, query_lens, max_seqlen_q = benchmark_fn(q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens)
            print_line(cu_seq_lens_q, query_lens, max_seqlen_q, avg_time)

if __name__ == '__main__':
    main()