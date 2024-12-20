import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # d_model   词嵌入维度
    # dropout   置0比率
    # max_len   每个句子的最大长度
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 实例化nn中预定义的Dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，大小为max_len * d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵,position为一个连续自然数组成的max_len * 1矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 将max_len * 1矩阵转换为max_len * d_model矩阵
        # 需要一个1 * d_model的变换矩阵div_term
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # 给所有行的偶数列赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        # 给所有行的奇数列赋值
        pe[:, 1::2] = torch.cos(position * div_term)

        # 拓展维度,从max_len * d_model变为1 * max_len * d_model
        pe = pe.unsqueeze(0)

        # 注册到buffer
        self.register_buffer("pe", pe)

    # x 文本序列的词嵌入表示
    def forward(self, x: torch.Tensor):
        # 截取文本长度的位置编码
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

# from embedding import Embeddings
# if __name__ == '__main__':
#     vocab = 15
#     d_model = 4
#     max_len = 60
#     dropout = 0.1
#     x = torch.LongTensor([[1,2,4,5]]).detach()
#     emb = Embeddings(vocab,d_model)
#     embr = emb(x)
#     pe = PositionalEncoding(d_model,dropout,max_len)
#     pe_result = pe(embr)
#     print(pe_result)
