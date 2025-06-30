import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from math import sqrt
from einops import einsum

class FlashAttentionTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        batch_size, Nq, d = Q.shape
        Nk = K.size(1)

        # 分块大小
        Bq = 16 
        Bk = 16
        
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk
        
        # 缩放因子
        scale = 1 / (d ** 0.5)

        # 初始化输出和logsumexp
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, Nq), device=Q.device)

        for i in range(Tq):
            # 计算当前查询块的起始/结束索引
            q_start = i * Bq
            q_end = min((i + 1) * Bq, Nq)
            q_len = q_end - q_start

            Q_tile = Q[:, q_start:q_end, :]
                        
            O_i = torch.zeros_like(Q_tile)
            m_i = torch.full((batch_size, q_len), -float('inf'), device=Q.device)
            l_i = torch.zeros_like(m_i)

            for j in range(Tk):
                # 计算当前键/值块的起始/结束索引
                k_start = j * Bk
                k_end = min((j + 1) * Bk, Nk)

                # 读取K，V
                K_tile = K[:, k_start:k_end, :]
                V_tile = V[:, k_start:k_end, :]
                
                # 计算注意力分数
                S_ij = einsum(Q_tile, K_tile, "... n d, ... m d -> ... n m") * scale

                # 更新行最大值
                c_m_i = torch.maximum(m_i, S_ij.max(dim=-1).values)

                # 计算相关性矩阵
                P_ij = torch.exp(S_ij - c_m_i.unsqueeze(-1))

                # 缩放补偿因子
                s = torch.exp(m_i - c_m_i)

                m_i = c_m_i
                l_i = s * l_i + torch.sum(P_ij, dim=-1)
                O_i = O_i * s.unsqueeze(-1) + einsum(P_ij, V_tile, "... n m, ... m d -> ... n d")

            
            # 保存结果到全局内存
            O[:, q_start:q_end, :] = O_i / (l_i.unsqueeze(-1) + 1e-6)
            L[:, q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(O, L)
        
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")
    

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr):

    # 块索引
    # 每个block负责处理一个query_tile和batch。
    # 所以内部只循环key_title
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 块信息
    # base‌: 指向张量第一个元素的地址
    # shape:‌ 张量整体形状‌，用于处理越界访问的情况
    # strides: ‌各维度步长‌，确保正确使用内存布局
    # offsets‌: 起始块的ND坐标‌
    # ‌block_shape 单块的形状
    # order: 内存维度顺序
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (query_tile_index * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,))

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero") 
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    if is_causal:
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE

    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_valid_len = min(Q_TILE_SIZE, N_QUERIES - query_tile_index * Q_TILE_SIZE)

    for i in range(n_k_tiles):
        # 读取K，V
        # boundary_check‌ 指定需执行边界检查的维度，只检查第0维
        K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        k_valid_len = min(K_TILE_SIZE, N_KEYS - i * K_TILE_SIZE)

        # 计算注意力分数
        # tl.dot是矩阵乘法
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # 边界掩码
        q_mask = tl.arange(0, Q_TILE_SIZE) < q_valid_len
        k_mask = tl.arange(0, K_TILE_SIZE) < k_valid_len
        boundary_mask = q_mask[:, None] & k_mask[None, :]
        S_ij = tl.where(boundary_mask, S_ij, float('-inf'))

        if is_causal:
            # 当前K块在context的位置
            k_pos = tl.arange(0, K_TILE_SIZE) + i * K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :] 
            S_ij = tl.where(mask, S_ij, float('-inf'))

        # 更新行最大值
        c_m_i = tl.maximum(m_i, tl.max(S_ij, axis=-1))

        # 计算相关性矩阵
        P_ij = tl.exp(S_ij - c_m_i[:, None])

        # 缩放补偿因子
        s = tl.exp(m_i - c_m_i)

        m_i = c_m_i
        l_i = s * l_i + tl.sum(P_ij, axis=-1)
        O_i = O_i * s[:, None] + tl.dot(P_ij.to(V_tile.dtype), V_tile)

        # 移动指针到下一个块
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    O_i = O_i / (l_i[:, None] + 1e-6)
    l_i = m_i + tl.log(l_i)

    # 保存
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, l_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))
    
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):

        batch_size, Nq, d = Q.shape
        Nk = K.size(1)

        # 分块大小
        Bq = 16 
        Bk = 16
        
        Tq = (Nq + Bq - 1) // Bq

        # 缩放因子
        scale = 1 / (d ** 0.5)      

        # 初始化输出和logsumexp
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, Nq), device=Q.device)

        flash_fwd_kernel[(Tq, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=Nq, N_KEYS=Nk,
            scale=scale,
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal)

        ctx.save_for_backward(O, L)
        ctx.is_causal = is_causal
        return O
