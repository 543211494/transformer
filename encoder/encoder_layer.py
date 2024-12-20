import torch
import torch.nn as nn

from encoder.attention import clones
from encoder.sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    # size          词嵌入维度大小
    # self_attn     自注意力对象
    # feed_forward  前馈全连接层对象
    # dropout       置0比率
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# if __name__ == '__main__':
#     vocab = 15
#     d_model = 512
#     max_len = 60
#     dropout = 0.2
#     head = 8
#     d_ff = 64
#
#     x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).detach()
#     emb = Embeddings(vocab, d_model)
#     embr = emb(x)
#     pe = PositionalEncoding(d_model, dropout, max_len)
#     pe_result = pe(embr)
#
#     self_attn = MultiHeadedAttention(head,d_model)
#     ff = PositionwiseFeedForward(d_model,d_ff,dropout)
#     mask = torch.zeros(8, 4, 4).detach()
#
#     el = EncoderLayer(d_model, self_attn, ff, dropout)
#     el_result = el(pe_result, mask)
#     print(el_result)
#     print(el_result.shape)
