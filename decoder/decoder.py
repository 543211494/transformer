import torch
import torch.nn as nn

from encoder.attention import clones
from encoder.layer_norm import LayerNorm


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
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
#
#     source_mask = mask
#     target_mask = mask
#     layer = DecoderLayer(d_model, c(self_attn), c(self_attn), c(ff),dropout)
#     de = Decoder(layer,N)
#     de_result = de(pe_result, en_result, source_mask, target_mask)
#     print(de_result)
#     print(de_result.shape)
