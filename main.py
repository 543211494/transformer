import torch
import numpy as np

from pyitcast.transformer_utils import Batch
# 基于Adam的标准优化器，使其对序列到序列的任务更有效
from pyitcast.transformer_utils import get_std_opt
# 标签平滑工具包，用于标签平滑，作用是小幅度改变原有标签的值域
# 因为在理论上即使是人工标注的数据也可能并非完全正确
# 因此使用标签平滑来弥补这种偏差，放在过拟合
from pyitcast.transformer_utils import LabelSmoothing
# 损失计算工具包
from pyitcast.transformer_utils import SimpleLossCompute

from pyitcast.transformer_utils import run_epoch
from pyitcast.transformer_utils import greedy_decode

from transformer import make_model
from transformer import EncoderDecoder


# V            随机生成数字的最大值+1
# batch        每次输送给模型更新一次参数的数据量
# num_batch    一个输送num_batch次数据
def data_generator(V, batch, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # 数据矩阵的第一列数字都为1,作为起始标志
        data[:, 0] = 1
        source = data.detach()
        target = data.detach()
        # 使用Batch对source和target进行对应批次的掩码张量生成
        yield Batch(source, target)


def run(model: EncoderDecoder, loss, epochs=50):
    for epoch in range(epochs):
        # 使用训练模式，所有参数将被更新
        model.train()
        # 训练
        run_epoch(data_generator(10, 8, 20), model, loss)

        # 模型使用评估模式，参数将不会发生变化
        model.eval()
        run_epoch(data_generator(10, 8, 5), model, loss)
    model.eval()
    source = torch.LongTensor([[1, 3, 2, 4, 5, 6, 7, 8, 9, 5]])
    source_mask = torch.ones(1, 1, 10).detach()
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == '__main__':
    print(torch.__version__)
    V = 20
    model = make_model(V, V, N=2)
    model_optimizer = get_std_opt(model)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
    run(model, loss)
