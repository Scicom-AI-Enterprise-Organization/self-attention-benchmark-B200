"""
Modified from https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import torch
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


# Varlen version of the inner attention kernel
@triton.jit
def _attn_fwd_inner_varlen(acc, l_i, m_i, q,  #
                           K, V,  #
                           stride_kt, stride_kd, stride_vt, stride_vd,  #
                           start_k_idx, end_k_idx,  #
                           dtype: tl.constexpr, start_m, qk_scale,  #
                           BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                           STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                           seqlen_k: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, seqlen_k
    
    # Clamp to actual sequence length
    lo = tl.minimum(lo, seqlen_k)
    hi = tl.minimum(hi, seqlen_k)
    
    offs_k = tl.arange(0, HEAD_DIM)
    
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Calculate actual indices in K/V tensors
        k_indices = start_k_idx + start_n + offs_n
        valid_k = (start_n + offs_n) < seqlen_k
        
        # Load K and V
        k_ptrs = K + k_indices[:, None] * stride_kt + offs_k[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=valid_k[:, None], other=0.0)
        
        # Compute QK
        qk = tl.dot(q, tl.trans(k))
        
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            mask = mask & valid_k[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = tl.where(valid_k[None, :], qk, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        
        p = tl.math.exp2(qk)
        
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        
        # Load V
        v_ptrs = V + k_indices[:, None] * stride_vt + offs_k[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=valid_k[:, None], other=0.0)
        
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


# Varlen forward kernel
@triton.jit
def _attn_fwd_varlen(sm_scale, M,  #
                     Q, K, V, O,  #
                     cu_seqlens_q, cu_seqlens_k,  #
                     stride_qh, stride_qt, stride_qd,  #
                     stride_kh, stride_kt, stride_kd,  #
                     stride_vh, stride_vt, stride_vd,  #
                     stride_oh, stride_ot, stride_od,  #
                     H, max_seqlen_q,  #
                     HEAD_DIM: tl.constexpr,  #
                     BLOCK_M: tl.constexpr,  #
                     BLOCK_N: tl.constexpr,  #
                     STAGE: tl.constexpr):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    
    # Load sequence boundaries
    start_q_idx = tl.load(cu_seqlens_q + off_b)
    end_q_idx = tl.load(cu_seqlens_q + off_b + 1)
    start_k_idx = tl.load(cu_seqlens_k + off_b)
    end_k_idx = tl.load(cu_seqlens_k + off_b + 1)
    
    seqlen_q = end_q_idx - start_q_idx
    seqlen_k = end_k_idx - start_k_idx
    
    # Early exit if this block is beyond the sequence
    if start_m * BLOCK_M >= seqlen_q:
        return
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Mask for valid queries in this block
    valid_q = offs_m < seqlen_q
    
    # Load Q - indices are absolute positions in the flattened tensor
    q_indices = start_q_idx + offs_m
    q_ptrs = Q + off_h * stride_qh + q_indices[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Scale factor
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    
    # Adjust K and V pointers to head (no batch dimension in varlen)
    K = K + off_h * stride_kh
    V = V + off_h * stride_vh
    
    # Attention computation
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_varlen(acc, l_i, m_i, q,  #
                                               K, V,  #
                                               stride_kt, stride_kd, stride_vt, stride_vd,  #
                                               start_k_idx, end_k_idx,  #
                                               dtype, start_m, qk_scale,  #
                                               BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                               4 - STAGE, offs_m, offs_n, seqlen_k)
    
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner_varlen(acc, l_i, m_i, q,  #
                                               K, V,  #
                                               stride_kt, stride_kd, stride_vt, stride_vd,  #
                                               start_k_idx, end_k_idx,  #
                                               dtype, start_m, qk_scale,  #
                                               BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                               2, offs_m, offs_n, seqlen_k)
    
    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    
    # Store M - use local sequence position (offs_m) not global indices
    # M layout is (batch_size * nheads, max_seqlen_q) to handle variable lengths
    m_base = off_bh * max_seqlen_q  # max_seqlen_q needs to be passed as parameter
    m_ptrs = M + m_base + offs_m
    tl.store(m_ptrs, m_i, mask=valid_q)
    
    # Store O - indices are absolute positions in the flattened tensor
    o_indices = start_q_idx + offs_m
    o_ptrs = O + off_h * stride_oh + o_indices[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(dtype), mask=valid_q[:, None])


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# Varlen backward preprocessing kernel
@triton.jit
def _attn_bwd_preprocess_varlen(O, DO,  #
                                Delta,  #
                                cu_seqlens_q,  #
                                stride_oh, stride_ot, stride_od,  #
                                stride_doh, stride_dot, stride_dod,  #
                                H, max_seqlen_q,  #
                                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                                ):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    
    # Load sequence boundaries
    start_q_idx = tl.load(cu_seqlens_q + off_b)
    end_q_idx = tl.load(cu_seqlens_q + off_b + 1)
    seqlen_q = end_q_idx - start_q_idx
    
    # Early exit if beyond sequence
    if start_m * BLOCK_M >= seqlen_q:
        return
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    valid_m = offs_m < seqlen_q
    
    # Load O and DO
    q_indices = start_q_idx + offs_m
    o_ptrs = O + off_h * stride_oh + q_indices[:, None] * stride_ot + offs_d[None, :] * stride_od
    do_ptrs = DO + off_h * stride_doh + q_indices[:, None] * stride_dot + offs_d[None, :] * stride_dod
    
    o = tl.load(o_ptrs, mask=valid_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=valid_m[:, None], other=0.0).to(tl.float32)
    
    # Compute delta
    delta = tl.sum(o * do, axis=1)
    
    # Store delta
    delta_base = off_bh * max_seqlen_q
    delta_ptrs = Delta + delta_base + offs_m
    tl.store(delta_ptrs, delta, mask=valid_m)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# Varlen backward dkdv inner loop
@triton.jit
def _attn_bwd_dkdv_varlen(dk, dv,  #
                          Q, k, v, sm_scale,  #
                          DO,  #
                          M, D,  #
                          stride_qt, stride_qd,  #
                          stride_dot, stride_dod,  #
                          start_q_idx, seqlen_q,  #
                          H, BLOCK_M1: tl.constexpr,  #
                          BLOCK_N1: tl.constexpr,  #
                          HEAD_DIM: tl.constexpr,  #
                          start_n, start_m, num_steps,  #
                          MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    
    for blk_idx in range(num_steps):
        offs_m_curr = curr_m + tl.arange(0, BLOCK_M1)
        valid_m = offs_m_curr < seqlen_q
        
        # Load Q^T
        q_indices = start_q_idx + offs_m_curr
        qT_ptrs = Q + q_indices[None, :] * stride_qt + offs_k[:, None] * stride_qd
        qT = tl.load(qT_ptrs, mask=valid_m[None, :], other=0.0)
        
        # Load m
        m = tl.load(M + offs_m_curr, mask=valid_m, other=0.0)
        
        # Compute attention weights
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        
        # Autoregressive masking
        if MASK:
            mask = (offs_m_curr[None, :] >= offs_n[:, None]) & valid_m[None, :]
            pT = tl.where(mask, pT, 0.0)
        else:
            pT = tl.where(valid_m[None, :], pT, 0.0)
        
        # Load DO
        do_ptrs = DO + q_indices[:, None] * stride_dot + offs_k[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=valid_m[:, None], other=0.0)
        
        # Compute dV
        ppT = pT.to(tl.float16)
        dv += tl.dot(ppT, do)
        
        # Load D (delta)
        Di = tl.load(D + offs_m_curr, mask=valid_m, other=0.0)
        
        # Compute dP and dS
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        
        # Increment
        curr_m += step_m
    
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


# Varlen backward dq inner loop
@triton.jit
def _attn_bwd_dq_varlen(dq, q, K, V,  #
                        do, m, D,  #
                        stride_kt, stride_kd,  #
                        stride_vt, stride_vd,  #
                        start_k_idx, seqlen_k,  #
                        H,  #
                        BLOCK_M2: tl.constexpr,  #
                        BLOCK_N2: tl.constexpr,  #
                        HEAD_DIM: tl.constexpr,  #
                        start_m, start_n, num_steps,  #
                        MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # D (delta) should already be loaded before calling this function
    # But we need Di as a column vector for proper broadcasting
    Di = D  # D should be shape [BLOCK_M2, 1] when passed in
    
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    
    for blk_idx in range(num_steps):
        offs_n_curr = curr_n + tl.arange(0, BLOCK_N2)
        valid_n = offs_n_curr < seqlen_k
        
        # Load K^T and V^T
        k_indices = start_k_idx + offs_n_curr
        kT_ptrs = K + k_indices[None, :] * stride_kt + offs_k[:, None] * stride_kd
        vT_ptrs = V + k_indices[None, :] * stride_vt + offs_k[:, None] * stride_vd
        
        kT = tl.load(kT_ptrs, mask=valid_n[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=valid_n[None, :], other=0.0)
        
        # Compute attention weights
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        
        # Autoregressive masking
        if MASK:
            mask = (offs_m[:, None] >= offs_n_curr[None, :]) & valid_n[None, :]
            p = tl.where(mask, p, 0.0)
        else:
            p = tl.where(valid_n[None, :], p, 0.0)
        
        # Compute dP and dS
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di)  # Di is already [BLOCK_M2, 1] from the caller
        ds = ds.to(tl.float16)
        
        # Compute dQ
        dq += tl.dot(ds, tl.trans(kT))
        
        # Increment
        curr_n += step_n
    
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, k, v, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


# Varlen backward kernel
@triton.jit
def _attn_bwd_varlen(Q, K, V, sm_scale,  #
                     DO, DQ, DK, DV,  #
                     M, D,  #
                     cu_seqlens_q, cu_seqlens_k,  #
                     stride_qh, stride_qt, stride_qd,  #
                     stride_kh, stride_kt, stride_kd,  #
                     stride_vh, stride_vt, stride_vd,  #
                     stride_doh, stride_dot, stride_dod,  #
                     H, max_seqlen_q,  #
                     BLOCK_M1: tl.constexpr,  #
                     BLOCK_N1: tl.constexpr,  #
                     BLOCK_M2: tl.constexpr,  #
                     BLOCK_N2: tl.constexpr,  #
                     BLK_SLICE_FACTOR: tl.constexpr,  #
                     HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    
    pid = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    
    # Load sequence boundaries
    start_q_idx = tl.load(cu_seqlens_q + off_b)
    end_q_idx = tl.load(cu_seqlens_q + off_b + 1)
    start_k_idx = tl.load(cu_seqlens_k + off_b)
    end_k_idx = tl.load(cu_seqlens_k + off_b + 1)
    
    seqlen_q = end_q_idx - start_q_idx
    seqlen_k = end_k_idx - start_k_idx
    
    # Early exit if both dK/dV and dQ sections would be beyond their respective sequences
    # dK/dV section processes: start_n = pid * BLOCK_N1
    # dQ section processes: start_m = pid * BLOCK_M2
    start_n_check = pid * BLOCK_N1
    start_m_check = pid * BLOCK_M2
    if start_n_check >= seqlen_k and start_m_check >= seqlen_q:
        return
    
    # Early exit if beyond sequence
    if pid * BLOCK_N1 >= seqlen_k:
        return
    
    # Adjust pointers for head
    Q = Q + off_h * stride_qh
    K = K + off_h * stride_kh
    V = V + off_h * stride_vh
    DO = DO + off_h * stride_doh
    DQ = DQ + off_h * stride_qh
    DK = DK + off_h * stride_kh
    DV = DV + off_h * stride_vh
    
    # M and D indexing
    m_base = off_bh * max_seqlen_q
    M = M + m_base
    D = D + m_base
    
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Compute dK and dV
    start_n = pid * BLOCK_N1
    start_m = start_n
    
    # Early exit if this block is entirely beyond the K sequence
    if start_n >= seqlen_k:
        # Skip dK/dV computation, but might still need to do dQ
        # Continue to dQ section below
        pass
    else:
        MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        valid_n = offs_n < seqlen_k
        
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        
        # Load K and V
        k_indices = start_k_idx + offs_n
        k_ptrs = K + k_indices[:, None] * stride_kt + offs_k[None, :] * stride_kd
        v_ptrs = V + k_indices[:, None] * stride_vt + offs_k[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)
        
        # Diagonal (masked) blocks
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv_varlen(dk, dv,  #
                                       Q, k, v, sm_scale,  #
                                       DO,  #
                                       M, D,  #
                                       stride_qt, stride_qd,  #
                                       stride_dot, stride_dod,  #
                                       start_q_idx, seqlen_q,  #
                                       H,  #
                                       MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                                       start_n, start_m, num_steps,  #
                                       MASK=True)
        
        start_m += num_steps * MASK_BLOCK_M1
        num_steps = (seqlen_q - start_m) // BLOCK_M1
        
        # Non-masked blocks
        if num_steps > 0:
            dk, dv = _attn_bwd_dkdv_varlen(dk, dv,  #
                                           Q, k, v, sm_scale,  #
                                           DO,  #
                                           M, D,  #
                                           stride_qt, stride_qd,  #
                                           stride_dot, stride_dod,  #
                                           start_q_idx, seqlen_q,  #
                                           H,  #
                                           BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                                           start_n, start_m, num_steps,  #
                                           MASK=False)
        
        # Store dV and dK
        dv_ptrs = DV + k_indices[:, None] * stride_vt + offs_k[None, :] * stride_vd
        tl.store(dv_ptrs, dv, mask=valid_n[:, None])
        
        dk *= sm_scale
        dk_ptrs = DK + k_indices[:, None] * stride_kt + offs_k[None, :] * stride_kd
        tl.store(dk_ptrs, dk, mask=valid_n[:, None])
    
    # Compute dQ
    start_m = pid * BLOCK_M2
    if start_m >= seqlen_q:
        return

    # Clamp to key length, not query length
    end_n = start_m + BLOCK_M2
    end_n = tl.minimum(end_n, seqlen_k)

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    valid_m = offs_m < seqlen_q

    # Load Q, DO, m, D as before
    q_indices = start_q_idx + offs_m
    q_ptrs = Q + q_indices[:, None] * stride_qt + offs_k[None, :] * stride_qd
    do_ptrs = DO + q_indices[:, None] * stride_dot + offs_k[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=valid_m[:, None], other=0.0)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    m = tl.load(M + offs_m, mask=valid_m, other=0.0)[:, None]
    Di = tl.load(D + offs_m, mask=valid_m, other=0.0)[:, None]

    # --- FIXED traversal for causal masked region ---
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    start_n_masked = end_n - num_steps * MASK_BLOCK_N2
    start_n_masked = tl.maximum(start_n_masked, 0)

    dq = _attn_bwd_dq_varlen(
        dq, q, K, V,
        do, m, Di,
        stride_kt, stride_kd,
        stride_vt, stride_vd,
        start_k_idx, seqlen_k,
        H,
        BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
        start_m, start_n_masked, num_steps,
        MASK=True
    )

    end_n = start_n_masked

    # --- Unmasked region ---
    num_steps = end_n // BLOCK_N2
    if num_steps > 0:
        dq = _attn_bwd_dq_varlen(
            dq, q, K, V,
            do, m, Di,
            stride_kt, stride_kd,
            stride_vt, stride_vd,
            start_k_idx, seqlen_k,
            H,
            BLOCK_M2, BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * BLOCK_N2, num_steps,
            MASK=False
        )

    # Store dQ
    dq *= LN2
    dq_ptrs = DQ + q_indices[:, None] * stride_qt + offs_k[None, :] * stride_qd
    tl.store(dq_ptrs, dq, mask=valid_m[:, None])
    dq_ptrs = DQ + q_indices[:, None] * stride_qt + offs_k[None, :] * stride_qd
    tl.store(dq_ptrs, dq, mask=valid_m[:, None])


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        return dq, dk, dv, None, None, None, None


class _attention_varlen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, sm_scale):
        # Shape: q, k, v: (total_tokens, nheads, headdim)
        # cu_seqlens_q, cu_seqlens_k: (batch_size + 1,)
        
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert q.shape[1] == k.shape[1]  # Same number of heads
        
        HEAD_DIM = q.shape[-1]
        assert HEAD_DIM in {16, 32, 64, 128, 256}
        
        total_q = q.shape[0]
        total_k = k.shape[0]
        nheads = q.shape[1]
        batch_size = cu_seqlens_q.shape[0] - 1
        
        o = torch.empty_like(q)
        # M needs to store max values for each query token across all sequences
        # Total size should be (batch_size * nheads, total_q) but we store per-sequence
        # Actually, simpler to allocate (batch_size * nheads, max_seqlen_q) and index properly
        M = torch.empty((batch_size * nheads, max_seqlen_q), device=q.device, dtype=torch.float32)
        
        stage = 3 if causal else 1
        
        BLOCK_M = 128
        BLOCK_N = 64
        
        def grid(META):
            return (triton.cdiv(max_seqlen_q, BLOCK_M), batch_size * nheads, 1)
        
        # For varlen format (total_tokens, nheads, headdim):
        # stride(0) is for heads, stride(1) is for tokens (0 for flattened), stride(2) is for headdim
        _attn_fwd_varlen[grid](
            sm_scale, M,  #
            q, k, v, o,  #
            cu_seqlens_q, cu_seqlens_k,  #
            q.stride(1), q.stride(0), q.stride(2),  # Q strides: (head, token, headdim)
            k.stride(1), k.stride(0), k.stride(2),  # K strides
            v.stride(1), v.stride(0), v.stride(2),  # V strides
            o.stride(1), o.stride(0), o.stride(2),  # O strides
            nheads, max_seqlen_q,  #
            HEAD_DIM=HEAD_DIM,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  #
            STAGE=stage,
        )
        
        ctx.save_for_backward(q, k, v, o, M, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        do = do.contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        batch_size = cu_seqlens_q.shape[0] - 1
        nheads = q.shape[1]
        HEAD_DIM = ctx.HEAD_DIM
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        
        # Backward pass constants
        BLOCK_M = 128
        BLOCK_M1, BLOCK_N1 = 32, 128
        BLOCK_M2, BLOCK_N2 = 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        
        # Scale K for backward pass
        arg_k = k * (ctx.sm_scale * RCP_LN2)
        
        # Preprocessing: compute delta
        delta = torch.empty((batch_size * nheads, max_seqlen_q), device=q.device, dtype=torch.float32)
        
        def pre_grid(META):
            return (triton.cdiv(max_seqlen_q, BLOCK_M), batch_size * nheads)
        
        _attn_bwd_preprocess_varlen[pre_grid](
            o, do,  #
            delta,  #
            cu_seqlens_q,  #
            o.stride(1), o.stride(0), o.stride(2),  # head, token, headdim
            do.stride(1), do.stride(0), do.stride(2),  #
            nheads, max_seqlen_q,  #
            BLOCK_M=BLOCK_M, HEAD_DIM=HEAD_DIM
        )
        
        # Main backward pass
        def grid(META):
            return (triton.cdiv(max_seqlen_k, BLOCK_N1), batch_size * nheads)
        
        _attn_bwd_varlen[grid](
            q, arg_k, v, ctx.sm_scale,  #
            do, dq, dk, dv,  #
            M, delta,  #
            cu_seqlens_q, cu_seqlens_k,  #
            q.stride(1), q.stride(0), q.stride(2),  # Q strides
            k.stride(1), k.stride(0), k.stride(2),  # K strides
            v.stride(1), v.stride(0), v.stride(2),  # V strides
            do.stride(1), do.stride(0), do.stride(2),  # DO strides
            nheads, max_seqlen_q,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=HEAD_DIM,  #
            num_warps=4,  #
            num_stages=1  #
        )
        
        return dq, dk, dv, None, None, None, None, None, None


attention = _attention.apply


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    """Variable-length Flash Attention function.
    
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    
    Args:
        q: (total_q, nheads, headdim) - Query tensor
        k: (total_k, nheads_k, headdim) - Key tensor
        v: (total_k, nheads_k, headdim) - Value tensor
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32 - Cumulative sequence lengths for queries
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32 - Cumulative sequence lengths for keys
        max_seqlen_q: int - Maximum query sequence length in the batch
        max_seqlen_k: int - Maximum key sequence length in the batch
        dropout_p: float - Dropout probability (not yet supported)
        softmax_scale: float - Scaling factor for QK^T. Default: 1/sqrt(headdim)
        causal: bool - Whether to apply causal masking
    
    Returns:
        out: (total_q, nheads, headdim) - Output tensor
    """
    
    # Handle MQA/GQA by expanding K and V
    nheads_q = q.shape[1]
    nheads_kv = k.shape[1]
    
    if nheads_q != nheads_kv:
        assert nheads_q % nheads_kv == 0, "nheads_q must be divisible by nheads_kv"
        # Expand K and V to match Q's head count
        expand_ratio = nheads_q // nheads_kv
        k = k.repeat_interleave(expand_ratio, dim=1)
        v = v.repeat_interleave(expand_ratio, dim=1)
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    return _attention_varlen.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, 
                                   max_seqlen_q, max_seqlen_k, causal, softmax_scale)