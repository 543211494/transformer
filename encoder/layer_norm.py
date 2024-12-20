import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    # features 词嵌入的维度
    # eps      一个很小的数，防止分母为0
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # nn.Parameter表示是模型的参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    # x 来自上一层的输出
    def forward(self, x):
        # 对x的最后一个维度求均值
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度求标准差
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

# if __name__ == '__main__':
#     vocab = 15
#     d_model = 512
#     max_len = 60
#     dropout = 0.2
#     head = 8
#     d_ff = 64
#
#     x = torch.LongTensor([[1,2,4,5],[4,3,2,9]]).detach()
#     emb = Embeddings(vocab, d_model)
#     embr = emb(x)
#     pe = PositionalEncoding(d_model, dropout, max_len)
#     pe_result = pe(embr)
#
#     query = pe_result
#     key = pe_result
#     value = pe_result
#     mask = torch.zeros(head,4,4).detach()
#
#     mha = MultiHeadedAttention(head,d_model,dropout)
#     mha_result = mha(query,key,value,mask)
#
#     ff = PositionwiseFeedForward(d_model,d_ff,dropout)
#     ff_result = ff(mha_result)
#     print(ff_result)
#
#     ln = LayerNorm(d_model)
#     ln_result = ln(ff_result)
#     print(ln_result)
