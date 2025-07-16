import torch
from einops import einsum

def flash_backward(Q, K, V, O, L, dO, is_causal = False):
    _, Nq, d = Q.shape
    Nk = K.size(1)
    scale = 1 / (d ** 0.5)
    D = torch.sum(O * dO, dim=-1)
    S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        mask = torch.triu(torch.ones(Nq, Nk, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.exp(S - L.unsqueeze(-1))

    dV = einsum(P, dO, "... q k, ... q d -> ... k d")

    dP = einsum(dO, V, "... q d, ... k d -> ... q k")
    dS = P * (dP - D[..., None])

    dQ = einsum(dS, K, "... q k, ... k d -> ... q d") * scale
    dK = einsum(dS, Q, "... q k, ... q d -> ... k d") * scale

    return dQ, dK, dV, None

compiled_backward = torch.compile(flash_backward)

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
                m_curr = torch.maximum(m_i, S_ij.max(dim=-1).values)

                # 计算相关性矩阵
                P_ij = torch.exp(S_ij - m_curr.unsqueeze(-1))

                # 缩放补偿因子
                s = torch.exp(m_i - m_curr)

                m_i = m_curr
                l_i = s * l_i + torch.sum(P_ij, dim=-1)
                O_i = O_i * s.unsqueeze(-1) + einsum(P_ij, V_tile, "... n m, ... m d -> ... n d")

            
            # 保存结果到全局内存
            O[:, q_start:q_end, :] = O_i / (l_i.unsqueeze(-1) + 1e-6)
            L[:, q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors  
        is_causal = ctx.is_causal
        return compiled_backward(Q, K, V, O, L, dO, is_causal)
    
