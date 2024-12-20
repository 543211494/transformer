import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    # query, key, value 代表注意力的三个输入张量
    # mask       掩码张量
    # dropout    传入的Dropout实例化对象
    # 取query的最后一个维度，代表词嵌入的维度
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt((d_k))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 归一化
    p_attn = F.softmax(scores, dim=-1)
    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    # head             头数
    # embedding_dim    词嵌入的维度
    # dropout          置0比率
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0
        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head
        # 头数
        self.head = head
        # 获得线性层对象，内部变换矩阵为embedding_dim * embedding_dim 方阵
        # Q、K、V各需要一个，最后拼接的矩阵还需要一个
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # 最后得到的注意力张量
        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)

        # 获得一个batch_size的变量，代表有多少条样本
        batch_size = query.size(0)

        # 多头处理
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)

# if __name__ == '__main__':
#     vocab = 15
#     d_model = 512
#     max_len = 60
#     dropout = 0.2
#     head = 8
#
#     x = torch.LongTensor([[1,2,4,5],[4,3,2,9]]).detach()
#     emb = Embeddings(vocab, d_model)
#     embr = emb(x)
#     pe = PositionalEncoding(d_model, dropout, max_len)
#     pe_result = pe(embr)
#     print(pe_result)
#
#     query = pe_result
#     key = pe_result
#     value = pe_result
#     mask = torch.zeros(head,4,4).detach()
#
#     mha = MultiHeadedAttention(head,d_model,dropout)
#     mha_result = mha(query,key,value,mask)
#     print(mha_result)
#     print(mha_result.shape)
