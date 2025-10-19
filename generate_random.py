from helper import make_varlen_inputs
import json

for total_len in [1024, 2048, 4096, 8192, 12288]:
    for i in range(4, 10, 1):
        query_lens = make_varlen_inputs(total_len, "random", num_splits=i, return_tensors=False)
        with open(f'random-{total_len}-{i}.json', 'w') as fopen:
            json.dump(query_lens.tolist(), fopen)
