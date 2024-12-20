import torch
import torch.nn as nn

from encoder.attention import clones
from encoder.sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    # size          词嵌入的维度大小
    # self_attn     多头自注意力对象，Q=K=V
    # src_attn      多头注意力对象,Q!=K=V
    # feed_forward  前馈全连接层对象
    # dropout       置0比率
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # x            上一层输入的张量
    # memory       来自编码器层的语义存储变量
    # source_mask  源数据掩码张量,遮蔽对结果没有意义的字符而产生的注意力值
    # target_mask  目标数据掩码张量
    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, source_mask))
        return self.sublayer[2](x, self.feed_forward)

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
#
#     source_mask = mask
#     target_mask = mask
#     print(source_mask)
#     print(target_mask)
#     d1 = DecoderLayer(d_model, c(self_attn),c(self_attn),c(ff),dropout)
#     d1_result = d1(pe_result, en_result, source_mask, target_mask)
#     print(d1_result)
#     print(d1_result.shape)
