import torch
import torch.nn as nn

import math

'''

'''


class Embeddings(nn.Module):
    # vocab     词表的大小
    # d_model   词嵌入的维度
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        # 获得词嵌入对象
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    # 前向传播
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# if __name__ == '__main__':
#     d_model = 3
#     vocab = 10
#     x = torch.LongTensor([[1,2,4,5],[4,3,2,9]]).detach()
#     emb = Embeddings(vocab,d_model)
#     embr = emb(x)
#     print(embr)
#     print(embr.shape)
