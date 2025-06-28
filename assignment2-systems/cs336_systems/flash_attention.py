import torch
import torch.nn.functional as F
import triton.language as tl
from einops import einsum

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        batch_size, N_q, d_k = Q.shape
        N_k = K.size(1)

        # 分块大小
        B_q = 16 
        B_k = 16
        
        T_q = (N_q + B_q - 1) // B_q
        T_k = (N_k + B_k - 1) // B_k
        
        # 缩放因子
        scale = 1 / (d_k ** 0.5)

        # 初始化输出和logsumexp
        O = torch.empty_like(Q)
        L = torch.empty((Q.size(0), Q.size(1)), device=Q.device)

        for i in range(T_q):
            # 计算当前查询块的起始/结束索引
            q_start = i * B_q
            q_end = (i + 1) * B_q
            q_len = q_end - q_start

            Q_tile = Q[:, q_start:q_end, :]
                        
            O_i = torch.zeros_like(Q_tile)
            m_i = torch.full((batch_size, q_len), -float('inf'), device=Q.device)
            l_i = torch.zeros_like(m_i)

            for j in range(T_k):
                # 计算当前键/值块的起始/结束索引
                k_start = j * B_k
                k_end = min((j + 1) * B_k, N_k)

                # 读取K，V
                K_tile = K[:, k_start:k_end, :]
                V_tile = V[:, k_start:k_end, :]
                
                # 计算注意力分数
                S_ij = einsum(Q_tile, K_tile, "... n d_k, ... m d_k -> ... n m") * scale

                # 更新行最大值
                c_m_i = torch.maximum(m_i, S_ij.max(dim=-1).values)

                # 计算相关性矩阵
                P_ij = torch.exp(S_ij - c_m_i.unsqueeze(-1))

                # 缩放补偿因子
                s = torch.exp(m_i - c_m_i)

                m_i = c_m_i
                l_i = s * l_i + torch.sum(P_ij, dim=-1)
                O_i = O_i * s.unsqueeze(-1) + einsum(P_ij, V_tile, "... n m, ... m d_k -> ... n d_k")

            
            # 保存结果到全局内存
            O[:, q_start:q_end, :] = O_i / (l_i.unsqueeze(-1) + 1e-6)
            L[:, q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(O, L)
        
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")