import fa_triton_kernel
import torch
import time

def benchmark_varlen_bwd(q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens, warmup=3, repeat=10):

    for _ in range(warmup):
        out = fa_triton_kernel.flash_attn_varlen_func(
            q=q[0],
            k=k[0],
            v=v[0],
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            causal=True
        )
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        out = fa_triton_kernel.flash_attn_varlen_func(
            q=q[0],
            k=k[0],
            v=v[0],
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            causal=True
        )
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None

    end = time.time()

    avg_time = (end - start) / repeat
    return avg_time, cu_seq_lens_q, query_lens, max_seqlen_q
