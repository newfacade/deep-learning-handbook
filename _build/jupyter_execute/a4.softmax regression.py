# 单层神经网络

```{note}
本节用最简单的单层神经网络（实际上就是softmax regression）来实现图像分类，主要目的是为了跑通流程
```

## 定义三要素

![jupyter](images/softmax.svg)

import torch
from torch import nn
import d2l


def init_weights(m):
    """initialize at random"""
    if type(m) == nn.Linear:
        # 赋值操作都带_
        nn.init.normal_(m.weight, std=0.01)
        

# softmax模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# 循环各层调用init_weights
net.apply(init_weights)

## 训练

# 1.获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 2.训练
lr, num_epochs = 0.01, 10
d2l.train_image_classifier(net, train_iter, test_iter, lr, num_epochs)

