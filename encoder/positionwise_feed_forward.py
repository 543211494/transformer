import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    # d_model 第一个线性层的输入维度
    # d_ff    第一个线性层的输出维度
    # dropout 置0比率
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

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
