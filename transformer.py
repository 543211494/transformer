import copy
import torch
import torch.nn as nn

from decoder.decoder_layer import DecoderLayer
from encoder.encoder_layer import EncoderLayer
from decoder.output import Generator
from embedding.embedding import Embeddings
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from embedding.position_encoding import PositionalEncoding
from encoder.attention import MultiHeadedAttention
from encoder.positionwise_feed_forward import PositionwiseFeedForward


# source_vocab     源数据特征总数
# target_vocab     目标数据特征
# N                编码器和解码器堆叠数
# d_model          词嵌入维度大小
# d_ff             前馈全连接网络中变换矩阵的维度
# head             多头注意力结构中的多头数
# dropout          置零比率
def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化多头注意力类
    attn = MultiHeadedAttention(head, d_model)
    # 实例化全馈全连接类
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化位置编码类
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(source_vocab, d_model), c(position)),
        nn.Sequential(Embeddings(target_vocab, d_model), c(position)),
        Generator(d_model, target_vocab)
    )

    # 参数的维度大于1，将其初始化为一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class EncoderDecoder(nn.Module):
    # encoder       编码器对象
    # decoder       解码器对象
    # source_embed  源数据嵌入函数
    # target_embed  目标数据嵌入函数
    # generator     输出部分的类别生成器对象
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    # source        源数据
    # taregt        目标数据
    # source_mask   源数据的掩码张量
    # target_mask   目标数据的掩码张量
    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask,
                           target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

# if __name__ == '__main__':
#     source_vocab = 15
#     target_vocab = 15
#     N = 6
#     res = make_model(source_vocab,target_vocab,N)
#     print(res)
