import time
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

torch._dynamo.config.cache_size_limit = 1000

flex_attention = torch.compile(flex_attention, dynamic=False)

kernel_options = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}

def generate_list_sum_n(n, length=5, min_val=5):

    numbers = [min_val] * length
    remaining = n - min_val * length

    for _ in range(remaining):
        numbers[random.randint(0, length - 1)] += 1

    random.shuffle(numbers)
    return numbers

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    offsets = offsets[offsets != -1]
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )

def length_to_offsets(lengths, device):
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def generate_doc_mask_mod(offsets):
    
    offsets = pad_sequence(offsets, batch_first = True, padding_value = -1)
    docs = [_offsets_to_doc_ids_tensor(offsets[i]) for i in range(offsets.shape[0])]
    docs = torch.stack(docs, 0)
    
    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[b, q_idx] == docs[b, kv_idx]
        return causal_mask & document_mask
    
    return document_causal_mask

def benchmark_varlen_bwd(q, k, v, cu_seq_lens_q, max_seqlen_q, query_lens, warmup=3, repeat=10):

    seq_len = q.shape[1]
    device = q.device

    for _ in range(warmup):
        attention_mask = cu_seq_lens_q
        document_causal_mask = generate_doc_mask_mod(attention_mask[None])
        attention_mask = create_block_mask(
            document_causal_mask, None, None, seq_len, seq_len, device, _compile = True)
        
        out = flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), score_mod=None, block_mask=attention_mask,
            kernel_options=kernel_options,
        )
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        attention_mask = cu_seq_lens_q
        document_causal_mask = generate_doc_mask_mod(attention_mask[None])
        attention_mask = create_block_mask(
            document_causal_mask, None, None, seq_len, seq_len, device, _compile = True)
        
        out = flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), score_mod=None, block_mask=attention_mask,
            kernel_options=kernel_options,
        )
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None

    end = time.time()

    avg_time = (end - start) / repeat
    return avg_time, cu_seq_lens_q, query_lens, max_seqlen_q