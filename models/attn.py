# write a cross attention module:
from math import sqrt
import torch
import torch.nn as nn

class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2)) # (B, H, S1, W) x (B, H, W, S2) -> (B, H, S1, S2)
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)  # (B, H, S1, S2)
        attention = torch.matmul(attention,V)  # (B, H, S1, S2) x (B, H, S2, W) -> (B, H, S1, W)
        return attention

class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算value和key
    """
    def __init__(self,hidden_size_q, hidden_size_kv, all_head_size,head_num):
        super().__init__()
        self.hidden_size_q = hidden_size_q       # 输入维度
        self.hidden_size_kv = hidden_size_kv     # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size_q, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size_kv, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size_kv, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size_q)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,x,y,attention_mask=None):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为v和k的输入
        """
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length_1, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length_2, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length_2, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        if attention_mask is None:
            attention_mask = torch.ones(x.size(0),x.size(1),y.size(1)).to(x.device).bool()
        attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output