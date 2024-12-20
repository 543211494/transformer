import torch
import torch.nn as nn

from encoder.layer_norm import LayerNorm


# 实现子层连接结构的类
class SublayerConnection(nn.Module):
    # size 词嵌入维度大小
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    # x          上一层或子层的输入
    # sublayer   该子层连接中的子层函数
    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(self.norm(sublayer(x)))

# if __name__ == '__main__':
#     vocab = 15
#     d_model = 512
#     max_len = 60
#     dropout = 0.2
#     head = 8
#
#     x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).detach()
#     emb = Embeddings(vocab, d_model)
#     embr = emb(x)
#     pe = PositionalEncoding(d_model, dropout, max_len)
#     pe_result = pe(embr)
#
#     mask = torch.zeros(head,4,4).detach()
#
#     self_attn = MultiHeadedAttention(head, d_model, dropout)
#     sublayer = lambda x: self_attn(x,x,x,mask)
#
#     sc = SublayerConnection(d_model, dropout)
#     sc_result = sc(pe_result,sublayer)
#     print(sc_result)
#     print(sc_result.shape)
