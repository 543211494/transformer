import torch
import torch.nn as nn

from encoder.attention import clones
from encoder.layer_norm import LayerNorm


class Encoder(nn.Module):
    # layer 编码器层
    # N     编码器层个数
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 深拷贝N个编码器层
        self.layers = clones(layer, N)
        # 初始化一个规范化层，用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# if __name__ == '__main__':
#     vocab = 15
#     d_model = 512
#     max_len = 60
#     dropout = 0.2
#     head = 8
#     d_ff = 64
#     N = 8
#
#     x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).detach()
#     emb = Embeddings(vocab, d_model)
#     embr = emb(x)
#     pe = PositionalEncoding(d_model, dropout, max_len)
#     pe_result = pe(embr)
#
#     c = copy.deepcopy
#     self_attn = MultiHeadedAttention(head,d_model)
#     ff = PositionwiseFeedForward(d_model,d_ff,dropout)
#     mask = torch.zeros(8, 4, 4).detach()
#
#     layer = EncoderLayer(d_model, c(self_attn), c(ff), dropout)
#     en = Encoder(layer, N)
#     en_result = en(pe_result, mask)
#     print(en_result)
#     print(en_result.shape)
