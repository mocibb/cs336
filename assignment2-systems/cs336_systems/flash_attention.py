import torch
import triton
import triton.language as tl

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

    log2e: tl.constexpr = 1.44269504

    if is_causal:
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE

    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_valid_len = min(Q_TILE_SIZE, N_QUERIES - query_tile_index * Q_TILE_SIZE)
    q_mask = tl.arange(0, Q_TILE_SIZE) < q_valid_len

    # 这里面耗时主要在两次dot操作
    for i in range(n_k_tiles):
        # 读取K，V
        # boundary_check‌ 指定需执行边界检查的维度，只检查第0维
        K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        k_valid_len = min(K_TILE_SIZE, N_KEYS - i * K_TILE_SIZE)

        # 计算注意力分数
        # 边界掩码
        k_mask = tl.arange(0, K_TILE_SIZE) < k_valid_len
        boundary_mask = q_mask[:, None] & k_mask[None, :]
        # tl.dot是矩阵乘法
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale + tl.where(boundary_mask, 0, -1.0e6)

        if is_causal:
            # 当前K块在context的位置
            k_pos = tl.arange(0, K_TILE_SIZE) + i * K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :] 
            S_ij = tl.where(mask, S_ij, float('-inf'))

        # 更新行最大值
        m_curr = tl.maximum(m_i, tl.max(S_ij, axis=-1))

        # 计算相关性矩阵
        P_ij = tl.math.exp2((S_ij - m_curr[:, None])*log2e)

        # 缩放补偿因子
        alpha = tl.math.exp2((m_i - m_curr)*log2e)

        m_i = m_curr
        l_i = alpha * l_i + tl.sum(P_ij, axis=-1)
        O_i = O_i * alpha[:, None] + tl.dot(P_ij.to(V_tile.dtype), V_tile)

        # 移动指针到下一个块
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / (l_i[:, None] + 1e-6)
    l_i = m_i + tl.math.log2(l_i)/log2e

    # 保存
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, l_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))

@triton.jit
def flash_fwd_casual_kernel(
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
    DTYPE: tl.constexpr,
    TOTAL_Q_BLOCKS: int,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    # 块索引
    # 每个block负责处理一个Q矩阵的两行和一个batch。
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    log2e: tl.constexpr = 1.44269504

    off_q = tl.arange(0, Q_TILE_SIZE)
    off_k = tl.arange(0, K_TILE_SIZE)
    off_d = tl.arange(0, D)
    
    # batch指针
    Q_batch_ptr = Q_ptr + batch_index * stride_qb
    K_batch_ptr = K_ptr + batch_index * stride_kb
    V_batch_ptr = V_ptr + batch_index * stride_vb
    O_batch_ptr = O_ptr + batch_index * stride_ob
    L_batch_ptr = L_ptr + batch_index * stride_lb

    # 顶部指针 (e.g., tile 0, 1, 2...)
    q_top_offset = query_tile_index * Q_TILE_SIZE
    offs_q_top = q_top_offset + off_q
    Q_top_ptr = Q_batch_ptr + offs_q_top[:, None] * stride_qq + off_d[None, :] * stride_qd
    O_top_ptr = O_batch_ptr + offs_q_top[:, None] * stride_oq + off_d[None, :] * stride_od
    L_top_ptr = L_batch_ptr + offs_q_top * stride_lq
    q_top_mask = offs_q_top < N_QUERIES

    # 顶部处理
    m_top = tl.full([Q_TILE_SIZE], float('-inf'), dtype=tl.float32)
    l_top = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc_top = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    
    q_top = tl.load(Q_top_ptr, mask=q_top_mask[:, None], other=0.0)
    end_k_pos_top = (query_tile_index + 1) * Q_TILE_SIZE
    k_tiles_top = tl.cdiv(min(N_KEYS, end_k_pos_top), K_TILE_SIZE)
    
    for k_tile_index in range(k_tiles_top):
        k_start_pos = k_tile_index * K_TILE_SIZE
        offs_k_curr = k_start_pos + off_k
        k_mask = offs_k_curr < N_KEYS

        # 加载K和V
        K_tile_ptr = K_batch_ptr + offs_k_curr[:, None] * stride_kk + off_d[None, :] * stride_kd
        V_tile_ptr = V_batch_ptr + offs_k_curr[:, None] * stride_vk + off_d[None, :] * stride_vd
        k = tl.load(K_tile_ptr, mask=k_mask[:, None], other=0.0)
        v = tl.load(V_tile_ptr, mask=k_mask[:, None], other=0.0)

        # 计算相关性矩阵
        S_ij = tl.dot(q_top, tl.trans(k)) * scale
        causal_mask = offs_q_top[:, None] >= offs_k_curr[None, :]
        S_ij = tl.where(causal_mask, S_ij, float('-inf'))

        m_curr = tl.maximum(m_top, tl.max(S_ij, axis=1))
        P_ij = tl.math.exp2((S_ij - m_curr[:, None])*log2e)
        
        alpha = tl.math.exp2((m_top - m_curr)*log2e)
        m_top = m_curr
        l_top = alpha * l_top + tl.sum(P_ij, axis=-1)
        acc_top = acc_top * alpha[:, None] + tl.dot(P_ij.to(DTYPE), v)

    # 保存处理
    O_top = (acc_top / (l_top[:, None] + 1e-6)).to(DTYPE)
    L_top = (m_top + tl.math.log2(l_top)/log2e).to(DTYPE)
    tl.store(O_top_ptr, O_top, mask=q_top_mask[:, None])
    tl.store(L_top_ptr, L_top, mask=q_top_mask)

    if query_tile_index >= TOTAL_Q_BLOCKS - 1 - query_tile_index:
        return

    # 底部指针(e.g., tile N-1, N-2, N-3...)
    q_bottom_offset = (TOTAL_Q_BLOCKS - 1 - query_tile_index) * Q_TILE_SIZE
    offs_q_bottom = q_bottom_offset + off_q
    Q_bottom_ptr = Q_batch_ptr + offs_q_bottom[:, None] * stride_qq + off_d[None, :] * stride_qd
    O_bottom_ptr = O_batch_ptr + offs_q_bottom[:, None] * stride_oq + off_d[None, :] * stride_od
    L_bottom_ptr = L_batch_ptr + offs_q_bottom * stride_lq
    q_bottom_mask = offs_q_bottom < N_QUERIES

    # 底部处理
    m_bottom = tl.full([Q_TILE_SIZE], float('-inf'), dtype=tl.float32)
    l_bottom = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc_bottom = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    
    q_bottom = tl.load(Q_bottom_ptr, mask=q_bottom_mask[:, None], other=0.0)
    end_k_pos_bottom = (TOTAL_Q_BLOCKS - query_tile_index) * Q_TILE_SIZE
    k_tiles_bottom = tl.cdiv(min(N_KEYS, end_k_pos_bottom), K_TILE_SIZE)
    
    for k_tile_index in range(k_tiles_bottom):
        k_start_pos = k_tile_index * K_TILE_SIZE
        offs_k_curr = k_start_pos + off_k
        k_mask = offs_k_curr < N_KEYS

        # 加载K和V
        K_tile_ptr = K_batch_ptr + offs_k_curr[:, None] * stride_kk + off_d[None, :] * stride_kd
        V_tile_ptr = V_batch_ptr + offs_k_curr[:, None] * stride_vk + off_d[None, :] * stride_vd
        k = tl.load(K_tile_ptr, mask=k_mask[:, None], other=0.0)
        v = tl.load(V_tile_ptr, mask=k_mask[:, None], other=0.0)

        # 计算相关性矩阵
        S_ij = tl.dot(q_bottom, tl.trans(k)) * scale
        causal_mask = offs_q_bottom[:, None] >= offs_k_curr[None, :]
        S_ij = tl.where(causal_mask, S_ij, float('-inf'))

        m_curr = tl.maximum(m_bottom, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_curr[:, None])

        alpha = tl.math.exp2((m_bottom - m_curr)*log2e)
        m_bottom = m_curr
        l_bottom = alpha * l_bottom + tl.sum(P_ij, axis=-1)
        acc_bottom = acc_bottom * alpha[:, None] + tl.dot(P_ij.to(DTYPE), v)

    # 保存处理
    O_bottom = (acc_bottom / (l_bottom[:, None] + 1e-6)).to(DTYPE)
    L_bottom = (m_bottom + tl.math.log2(l_bottom)/log2e).to(DTYPE)
    tl.store(O_bottom_ptr, O_bottom, mask=q_bottom_mask[:, None])
    tl.store(L_bottom_ptr, L_bottom, mask=q_bottom_mask)

@triton.jit
def flash_bwd_preprocess(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    M, N,
    M_TILE_SIZE: tl.constexpr,
    N_TILE_SIZE: tl.constexpr):
    
    # 获取当前线程块索引
    m_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 创建块指针替代直接偏移计算
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(M, N),
        strides=(stride_oq, stride_od),
        offsets=(m_tile_idx * M_TILE_SIZE, 0),
        block_shape=(M_TILE_SIZE, N_TILE_SIZE),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(M, N),
        strides=(stride_doq, stride_dod),
        offsets=(m_tile_idx * M_TILE_SIZE, 0),
        block_shape=(M_TILE_SIZE, N_TILE_SIZE),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(M,),
        strides=(stride_dq,),
        offsets=(m_tile_idx * M_TILE_SIZE,),
        block_shape=(M_TILE_SIZE,),
        order=(0,)
    )

    # 初始化累加缓冲区
    D = tl.zeros((M_TILE_SIZE,), dtype=tl.float32)

    # 分块处理N维度
    for i in range(tl.cdiv(N, N_TILE_SIZE)):
        o = tl.load(O_block_ptr, boundary_check=(0,1), padding_option="zero")
        do = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        D += tl.sum(o * do, axis=1)
        
        # 移动指针到下一个N分块
        O_block_ptr = O_block_ptr.advance((0, N_TILE_SIZE))
        dO_block_ptr = dO_block_ptr.advance((0, N_TILE_SIZE))

    # 存储结果
    tl.store(D_block_ptr, D, boundary_check=(0,))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, dO_ptr, 
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

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

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES,),
        strides = (stride_dq,),
        offsets = (query_tile_index * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (query_tile_index * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,))
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape = (N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32) 
    D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero") 
    dO_tile = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    kv_idx = tl.arange(0, K_TILE_SIZE)
    d_idx = tl.arange(0, D)
    v_offset = kv_idx[:, None] * stride_dvk + d_idx[None, :] * stride_dvd
    k_offset = kv_idx[:, None] * stride_dkk + d_idx[None, :] * stride_dkd

    if is_causal:
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE


    # 外循环为i，内循环为j 
    for j in range(n_k_tiles):
        # 读取K，V
        K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        kv_begin = j*K_TILE_SIZE

        # 计算注意力分数
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        if is_causal:
            # 当前K块在context的位置
            k_pos = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :] 
            S_ij = tl.where(mask, S_ij, float('-inf'))
        P_ij = tl.exp(S_ij - L_tile[:, None])

        # 计算dV, dP, dS, dK, dQ
        dV = tl.dot(tl.trans(P_ij), dO_tile)
        dP = tl.dot(dO_tile, tl.trans(V_tile))
        dS = P_ij * (dP - D_tile[:, None])
        dK = tl.dot(tl.trans(dS), Q_tile) * scale
        dQ += tl.dot(dS, K_tile) * scale
        
        # 原子地将计算出的dV和dK加到全局内存中
        # tl.atomic_add(
        #     dV_ptr + batch_index * stride_dvb + kv_begin*stride_dvk
        #     + v_offset,
        #     dV,
        #     sem="relaxed"
        # )

        # tl.atomic_add(
        #     dK_ptr + batch_index * stride_dkb + kv_begin*stride_dkk
        #     + k_offset,
        #     dK,
        #     sem="relaxed"
        # )

        # 移动指针到下一个块
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ.to(dQ_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, dO_ptr, 
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr):

    # 先循环key
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape = (N_QUERIES,),
        strides = (stride_dq,),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,))
    
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape = (N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    # 初始化 dK 和 dV 
    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    n_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    
    if is_causal:
        k_pos = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE    
    
    for j in range(n_q_tiles):
        # 读取Q, dO, L, D
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        # 计算注意力分数
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        if is_causal:
            # 当前Q块在context的位置
            q_pos = tl.arange(0, Q_TILE_SIZE) + j * Q_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :] 
            S_ij = tl.where(mask, S_ij, float('-inf'))
        P_ij = tl.exp(S_ij - L_tile[:, None])
            
        # 计算dV, dP, dS, dK, dQ
        dV += tl.dot(tl.trans(P_ij), dO_tile)
        dP = tl.dot(dO_tile, tl.trans(V_tile))
        dS = P_ij * (dP - D_tile[:, None])
        dK += tl.dot(tl.trans(dS), Q_tile) * scale
        
        # 移动指针到下一个块
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
    

    tl.store(dK_block_ptr, dK.to(dK_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(dV_block_ptr, dV.to(dV_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_dq_casual_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, dO_ptr, 
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    DTYPE: tl.constexpr,
    TOTAL_Q_BLOCKS: int,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    off_q = tl.arange(0, Q_TILE_SIZE)
    off_k = tl.arange(0, K_TILE_SIZE)
    off_d = tl.arange(0, D)
    
    # batch指针
    Q_batch_ptr = Q_ptr + batch_index * stride_qb
    K_batch_ptr = K_ptr + batch_index * stride_kb
    V_batch_ptr = V_ptr + batch_index * stride_vb
    D_batch_ptr = D_ptr + batch_index * stride_db
    L_batch_ptr = L_ptr + batch_index * stride_lb
    dO_batch_ptr = dO_ptr + batch_index * stride_dob
    dQ_batch_ptr = dQ_ptr + batch_index * stride_dqb

    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    
    # 顶部指针
    q_top_offset = query_tile_index * Q_TILE_SIZE
    offs_q_top = q_top_offset + off_q
    Q_top_ptr = Q_batch_ptr + (q_top_offset + off_q[:, None]) * stride_qq + off_d[None, :] * stride_qd
    D_top_ptr = D_batch_ptr + offs_q_top * stride_dq
    L_top_ptr = L_batch_ptr + offs_q_top * stride_lq
    dO_top_ptr = dO_batch_ptr + (q_top_offset + off_q[:, None]) * stride_doq + off_d[None, :] * stride_dod
    dQ_top_ptr = dQ_batch_ptr + (q_top_offset + off_q[:, None]) * stride_dqq + off_d[None, :] * stride_dqd
    q_top_mask = offs_q_top < N_QUERIES

    q_top = tl.load(Q_top_ptr, mask=q_top_mask[:, None], other=0.0)
    d_top = tl.load(D_top_ptr, mask=q_top_mask, other=0.0)
    l_top = tl.load(L_top_ptr, mask=q_top_mask, other=0.0)
    dO_top = tl.load(dO_top_ptr, mask=q_top_mask[:, None], other=0.0)
    
    end_k_pos_top = (query_tile_index + 1) * Q_TILE_SIZE
    k_tiles_top = tl.cdiv(min(N_KEYS, end_k_pos_top), K_TILE_SIZE)

    # 外循环为按照Q的行，内循环为按照K的行 
    for k_tile_index in range(k_tiles_top):
        k_start_pos = k_tile_index * K_TILE_SIZE
        offs_k_curr = k_start_pos + off_k
        k_mask = offs_k_curr < N_KEYS

        # 加载K和V
        K_tile_ptr = K_batch_ptr + offs_k_curr[:, None] * stride_kk + off_d[None, :] * stride_kd
        V_tile_ptr = V_batch_ptr + offs_k_curr[:, None] * stride_vk + off_d[None, :] * stride_vd
        k = tl.load(K_tile_ptr, mask=k_mask[:, None], other=0.0)
        v = tl.load(V_tile_ptr, mask=k_mask[:, None], other=0.0)

        # 计算注意力分数
        S_ij = tl.dot(q_top, tl.trans(k)) * scale

        # 当前K块在context的位置
        causal_mask = offs_q_top[:, None] >= offs_k_curr[None, :]
        S_ij = tl.where(causal_mask, S_ij, float('-inf'))
        P_ij = tl.exp(S_ij - l_top[:, None])

        # 计算dV, dP, dS, dK, dQ
        dP = tl.dot(dO_top, tl.trans(v))
        dS = P_ij * (dP - d_top[:, None])
        dQ += tl.dot(dS.to(DTYPE), k) * scale
        
    # 保存dQ 
    tl.store(dQ_top_ptr, dQ.to(DTYPE), mask=q_top_mask[:, None])

    if query_tile_index >= TOTAL_Q_BLOCKS - 1 - query_tile_index:
        return
    
    # 底部指针
    q_bottom_offset = (TOTAL_Q_BLOCKS - 1 - query_tile_index) * Q_TILE_SIZE
    offs_q_bottom = q_bottom_offset + off_q
    Q_bottom_ptr = Q_batch_ptr + (q_bottom_offset + off_q[:, None]) * stride_qq + off_d[None, :] * stride_qd
    D_bottom_ptr = D_batch_ptr + offs_q_bottom * stride_dq
    L_bottom_ptr = L_batch_ptr + offs_q_bottom * stride_lq
    dO_bottom_ptr = dO_batch_ptr + (q_bottom_offset + off_q[:, None]) * stride_doq + off_d[None, :] * stride_dod
    dQ_bottom_ptr = dQ_batch_ptr + (q_bottom_offset + off_q[:, None]) * stride_dqq + off_d[None, :] * stride_dqd
    q_bottom_mask = offs_q_bottom < N_QUERIES

    q_bottom = tl.load(Q_bottom_ptr, mask=q_bottom_mask[:, None], other=0.0)
    d_bottom = tl.load(D_bottom_ptr, mask=q_bottom_mask, other=0.0)
    l_bottom = tl.load(L_bottom_ptr, mask=q_bottom_mask, other=0.0)
    dO_bottom = tl.load(dO_bottom_ptr, mask=q_bottom_mask[:, None], other=0.0)

    end_k_pos_bottom = (TOTAL_Q_BLOCKS - query_tile_index) * Q_TILE_SIZE
    k_tiles_bottom = tl.cdiv(min(N_KEYS, end_k_pos_bottom), K_TILE_SIZE)
    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for k_tile_index in range(k_tiles_bottom):
        k_start_pos = k_tile_index * K_TILE_SIZE
        offs_k_curr = k_start_pos + off_k
        k_mask = offs_k_curr < N_KEYS

        # 加载K和V
        K_tile_ptr = K_batch_ptr + offs_k_curr[:, None] * stride_kk + off_d[None, :] * stride_kd
        V_tile_ptr = V_batch_ptr + offs_k_curr[:, None] * stride_vk + off_d[None, :] * stride_vd
        k = tl.load(K_tile_ptr, mask=k_mask[:, None], other=0.0)
        v = tl.load(V_tile_ptr, mask=k_mask[:, None], other=0.0)

        # 计算注意力分数
        S_ij = tl.dot(q_bottom, tl.trans(k)) * scale

        # 当前K块在context的位置
        causal_mask = offs_q_bottom[:, None] >= offs_k_curr[None, :]
        S_ij = tl.where(causal_mask, S_ij, float('-inf'))
        P_ij = tl.exp(S_ij - l_bottom[:, None])

        # 计算dV, dP, dS, dK, dQ
        dP = tl.dot(dO_bottom, tl.trans(v))
        dS = P_ij * (dP - d_bottom[:, None])
        dQ += tl.dot(dS.to(DTYPE), k) * scale

    # 保存dQ 
    tl.store(dQ_bottom_ptr, dQ.to(DTYPE), mask=q_bottom_mask[:, None])


@triton.jit
def flash_bwd_dkdv_casual_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, dO_ptr, 
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    DTYPE: tl.constexpr,
    TOTAL_K_BLOCKS: int,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr):
    
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    off_q = tl.arange(0, Q_TILE_SIZE)
    off_k = tl.arange(0, K_TILE_SIZE)
    off_d = tl.arange(0, D)
    
    # batch指针
    Q_batch_ptr = Q_ptr + batch_index * stride_qb
    K_batch_ptr = K_ptr + batch_index * stride_kb
    V_batch_ptr = V_ptr + batch_index * stride_vb
    D_batch_ptr = D_ptr + batch_index * stride_db
    L_batch_ptr = L_ptr + batch_index * stride_lb
    dO_batch_ptr = dO_ptr + batch_index * stride_dob
    dK_batch_ptr = dK_ptr + batch_index * stride_dkb
    dV_batch_ptr = dV_ptr + batch_index * stride_dvb

    # 顶部指针
    k_top_offset = key_tile_index * K_TILE_SIZE
    offs_k_top = k_top_offset + off_k
    
    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    K_top_ptr = K_batch_ptr + offs_k_top[:, None] * stride_kk + off_d[None, :] * stride_kd
    V_top_ptr = V_batch_ptr + offs_k_top[:, None] * stride_vk + off_d[None, :] * stride_vd
    dK_top_ptr = dK_batch_ptr + offs_k_top[:, None] * stride_dkk + off_d[None, :] * stride_dkd
    dV_top_ptr = dV_batch_ptr + offs_k_top[:, None] * stride_dvk + off_d[None, :] * stride_dvd
    k_top_mask = offs_k_top < N_KEYS

    k_top = tl.load(K_top_ptr, mask=k_top_mask[:, None], other=0.0).to(tl.float32)
    v_top = tl.load(V_top_ptr, mask=k_top_mask[:, None], other=0.0).to(tl.float32)

    # 当前是第k行，end_q_pos_bottom是所有比当前k大的行
    q_start_top = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
    q_tiles_top = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    # 外循环为按照K的行，内循环为按照Q的行 
    for q_tile_index in range(q_start_top, q_tiles_top):
        q_offset = q_tile_index * Q_TILE_SIZE
        offs_q_curr = q_offset + off_q
        q_mask = offs_q_curr < N_QUERIES
        
        # 读取Q, dO, L, D
        Q_tile_ptr = Q_batch_ptr + offs_q_curr[:, None] * stride_qq + off_d[None, :] * stride_qd
        dO_tile_ptr = dO_batch_ptr + offs_q_curr[:, None] * stride_doq + off_d[None, :] * stride_dod
        L_tile_ptr = L_batch_ptr + offs_q_curr * stride_lq
        D_tile_ptr = D_batch_ptr + offs_q_curr * stride_dq

        q = tl.load(Q_tile_ptr, mask=q_mask[:, None], other=0.0).to(tl.float32)
        dO = tl.load(dO_tile_ptr, mask=q_mask[:, None], other=0.0).to(tl.float32)
        l = tl.load(L_tile_ptr, mask=q_mask, other=0.0)
        d = tl.load(D_tile_ptr, mask=q_mask, other=0.0)

        # 计算注意力分数
        S_ij = tl.dot(q, tl.trans(k_top)) * scale
        causal_mask = offs_q_curr[:, None] >= offs_k_top[None, :]
        S_ij = tl.where(causal_mask, S_ij, float('-inf'))
        P_ij = tl.exp(S_ij - l[:, None])
    
        # 计算dV, dP, dS, dK, dQ
        dV += tl.dot(tl.trans(P_ij), dO)
        dP = tl.dot(dO, tl.trans(v_top))
        dS = P_ij * (dP - d[:, None])
        dK += tl.dot(tl.trans(dS), q) * scale

    # 保存dK和dV
    tl.store(dK_top_ptr, dK.to(DTYPE), mask=k_top_mask[:, None])
    tl.store(dV_top_ptr, dV.to(DTYPE), mask=k_top_mask[:, None])

    if key_tile_index >= TOTAL_K_BLOCKS - 1 - key_tile_index:
        return

    # 底部指针
    k_bottom_offset = (TOTAL_K_BLOCKS - 1 - key_tile_index) * K_TILE_SIZE
    offs_k_bottom = k_bottom_offset + off_k
    
    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    K_bottom_ptr = K_batch_ptr + offs_k_bottom[:, None] * stride_kk + off_d[None, :] * stride_kd
    V_bottom_ptr = V_batch_ptr + offs_k_bottom[:, None] * stride_vk + off_d[None, :] * stride_vd
    dK_bottom_ptr = dK_batch_ptr + offs_k_bottom[:, None] * stride_dkk + off_d[None, :] * stride_dkd
    dV_bottom_ptr = dV_batch_ptr + offs_k_bottom[:, None] * stride_dvk + off_d[None, :] * stride_dvd
    k_bottom_mask = offs_k_bottom < N_KEYS

    k_bottom = tl.load(K_bottom_ptr, mask=k_bottom_mask[:, None], other=0.0).to(tl.float32)
    v_bottom = tl.load(V_bottom_ptr, mask=k_bottom_mask[:, None], other=0.0).to(tl.float32)

    q_start_bottom = ((TOTAL_K_BLOCKS- 1 - key_tile_index) * K_TILE_SIZE) // Q_TILE_SIZE
    q_tiles_bottom = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    # 外循环为按照K的行，内循环为按照Q的行 
    for q_tile_index in range(q_start_bottom, q_tiles_bottom):
        q_offset = q_tile_index * Q_TILE_SIZE
        offs_q_curr = q_offset + off_q
        q_mask = offs_q_curr < N_QUERIES
        
        # 读取Q, dO, L, D
        Q_tile_ptr = Q_batch_ptr + offs_q_curr[:, None] * stride_qq + off_d[None, :] * stride_qd
        dO_tile_ptr = dO_batch_ptr + offs_q_curr[:, None] * stride_doq + off_d[None, :] * stride_dod
        L_tile_ptr = L_batch_ptr + offs_q_curr * stride_lq
        D_tile_ptr = D_batch_ptr + offs_q_curr * stride_dq

        q = tl.load(Q_tile_ptr, mask=q_mask[:, None], other=0.0).to(tl.float32)
        dO = tl.load(dO_tile_ptr, mask=q_mask[:, None], other=0.0).to(tl.float32)
        l = tl.load(L_tile_ptr, mask=q_mask, other=0.0)
        d = tl.load(D_tile_ptr, mask=q_mask, other=0.0)

        # 计算注意力分数
        S_ij = tl.dot(q, tl.trans(k_bottom)) * scale
        causal_mask = offs_q_curr[:, None] >= offs_k_bottom[None, :]
        S_ij = tl.where(causal_mask, S_ij, float('-inf'))
        P_ij = tl.exp(S_ij - l[:, None])
    
        # 计算dV, dP, dS, dK, dQ
        dV += tl.dot(tl.trans(P_ij), dO)
        dP = tl.dot(dO, tl.trans(v_bottom))
        dS = P_ij * (dP - d[:, None])
        dK += tl.dot(tl.trans(dS), q) * scale

    # 保存dK和dV
    tl.store(dK_bottom_ptr, dK.to(DTYPE), mask=k_bottom_mask[:, None])
    tl.store(dV_bottom_ptr, dV.to(DTYPE), mask=k_bottom_mask[:, None])

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):

        batch_size, Nq, d = Q.shape
        Nk = K.size(1)
        assert Nq == Nk

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

        if is_causal:
            # 在causal时flash_fwd_casual_kernel会快接近一倍
            flash_fwd_casual_kernel[((Tq+1)//2, batch_size)](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                N_QUERIES=Nq, N_KEYS=Nk,
                scale=scale,
                D=d,
                DTYPE=(tl.float32 if Q.dtype == torch.float32 else tl.float16), 
                TOTAL_Q_BLOCKS=Tq,
                Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
                num_warps=1,  #
                num_stages=3)
        else:
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
                D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
                num_warps=1,  #在4060Ti上最佳
                num_stages=3)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Bq = Bq
        ctx.Bk = Bk
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        # 分块大小
        Bq = ctx.Bq 
        Bk = ctx.Bk

        batch_size, Nq, d = Q.shape
        Nk = K.size(1)

        assert Nq % Bq == 0
        assert Nk % Bk == 0
        
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk
        
        # 使用triton计算
        # D = torch.sum(O*dO, dim = -1)
        D = torch.empty_like(L)
        M_TILE_SIZE = 64
        N_TILE_SIZE = 64
        Tm = (Nq + M_TILE_SIZE - 1) // M_TILE_SIZE
        flash_bwd_preprocess[(Tm, batch_size)](
            O, dO, D,
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            M=Nq, N=d,
            M_TILE_SIZE=M_TILE_SIZE, N_TILE_SIZE=N_TILE_SIZE)

        # 缩放因子
        scale = 1 / (d ** 0.5)
        # 初始化输出
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        if is_causal:
            flash_bwd_dq_casual_kernel[((Tq+1)//2, batch_size)](
                Q, K, V,
                D, L, dO,
                dQ,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                D.stride(0), D.stride(1),
                L.stride(0), L.stride(1),
                dO.stride(0), dO.stride(1), dO.stride(2),
                dQ.stride(0), dQ.stride(1), dQ.stride(2),
                N_QUERIES=Nq, N_KEYS=Nk,
                scale=scale, D=d,
                DTYPE=(tl.float32 if Q.dtype == torch.float32 else tl.float16), 
                TOTAL_Q_BLOCKS=Tq, 
                Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
                num_warps=2)
            
            flash_bwd_dkdv_casual_kernel[((Tk+1)//2, batch_size)](
                Q, K, V,
                D, L, dO,
                dK, dV,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                D.stride(0), D.stride(1),
                L.stride(0), L.stride(1),
                dO.stride(0), dO.stride(1), dO.stride(2),
                dK.stride(0), dK.stride(1), dK.stride(2),
                dV.stride(0), dV.stride(1), dV.stride(2),
                N_QUERIES=Nq, N_KEYS=Nk,
                scale=scale, D=d,
                DTYPE=(tl.float32 if Q.dtype == torch.float32 else tl.float16), 
                TOTAL_K_BLOCKS=Tk, 
                Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
                num_warps=2)
        else:
            flash_bwd_dq_kernel[(Tq, batch_size)](
                Q, K, V,
                D, L, dO,
                dQ, dK, dV,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                D.stride(0), D.stride(1),
                L.stride(0), L.stride(1),
                dO.stride(0), dO.stride(1), dO.stride(2),
                dQ.stride(0), dQ.stride(1), dQ.stride(2),
                dK.stride(0), dK.stride(1), dK.stride(2),
                dV.stride(0), dV.stride(1), dV.stride(2),
                N_QUERIES=Nq, N_KEYS=Nk,
                scale=scale,
                D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
                num_warps=2)
            
            flash_bwd_dkdv_kernel[(Tk, batch_size)](
                Q, K, V,
                D, L, dO,
                dK, dV,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                D.stride(0), D.stride(1),
                L.stride(0), L.stride(1),
                dO.stride(0), dO.stride(1), dO.stride(2),
                dK.stride(0), dK.stride(1), dK.stride(2),
                dV.stride(0), dV.stride(1), dV.stride(2),
                N_QUERIES=Nq, N_KEYS=Nk,
                scale=scale,
                D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
                num_warps=2)
            
        return dQ, dK, dV, None


