import random
import numpy as np
import torch

print('Device:', torch.cuda.get_device_name(0))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def generate_list_sum_n(n, length=5, min_val=5):
    numbers = [min_val] * length
    remaining = n - min_val * length
    for _ in range(remaining):
        numbers[random.randint(0, length - 1)] += 1
    random.shuffle(numbers)
    return numbers

def make_varlen_inputs(total_len=None, split_type="single", num_splits=4, dim=128, nheads=32, return_tensors=True, query_lens=None):
    if query_lens is None:
        if split_type == "single":
            query_lens = np.array([total_len], dtype=np.int64)
        elif split_type == "even":
            try:
                assert total_len % num_splits == 0, "total_len must divide evenly"
                query_lens = np.array([total_len // num_splits] * num_splits, dtype=np.int64)
            except:
                return
        elif split_type == "random":
            query_lens = np.array(
                generate_list_sum_n(total_len, length=num_splits, min_val=max(1, total_len // (num_splits * 2))),
                dtype=np.int64
            )
        else:
            raise ValueError("Unknown split_type")
    else:
        total_len = sum(query_lens)

    if return_tensors:
        cumsum = [0] + np.cumsum(query_lens).tolist()
        cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32).cuda()
        max_seqlen_q = int(np.max(query_lens))

        q = torch.randn(1, total_len, nheads, dim, dtype=torch.bfloat16, requires_grad=True, device="cuda")
        k = torch.randn(1, total_len, nheads, dim, dtype=torch.bfloat16, requires_grad=True, device="cuda")
        v = torch.randn(1, total_len, nheads, dim, dtype=torch.bfloat16, requires_grad=True, device="cuda")

        return q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens
    else:
        return query_lens

@torch.no_grad()
def clear_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

if __name__ == '__main__':

    for total_len in [1024, 2048, 4096, 8192, 12288]:
        make_varlen_inputs(total_len, "single")
        make_varlen_inputs(total_len, "even", num_splits=4)
        make_varlen_inputs(total_len, "random", num_splits=4)