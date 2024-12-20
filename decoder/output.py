import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    # d_model   词嵌入维度
    # vocab     词表大小
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab)

    # x  上一层的输出张量x
    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)

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
#
#     vocab = 512
#     gen = Generator(d_model,vocab)
#     gen_result = gen(de_result)
#     print(gen_result)
#     print(gen_result.shape)
