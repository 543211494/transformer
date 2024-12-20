import numpy as np

import torch


# size 掩码张量最后两个维度的大小
def subsequent_mask(size):
    # 定义掩码张量的形状
    attn_shape = (1, size, size)

    # 使用np.ones向这个形状中添加1元素形成上三角阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 对三角阵进行反转，即1变0,0变1
    return torch.from_numpy(1 - subsequent_mask)

# if __name__ == '__main__':
#     sm = subsequent_mask(5)
#     print(sm)
