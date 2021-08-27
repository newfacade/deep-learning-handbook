# 图像分类Pipeline

```{note}
图像分类的pipeline都是类似的，只是模型不同<br/>
所以我们可以先定义好训练图像分类的函数，模型作为其参数
```

## 一些辅助函数和类

import torch
import d2l


#@save
def try_gpu():
    """尽量使用gpu"""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#@save
class Accumulator:
    """累计n个数据"""
    
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#@save
def correct_predictions(y_hat, y):
    """
    :param y_hat: (n_samples, n_categories)
    :param y: (n_samples, )
    :return: 正确预测的个数
    """
    y_hat = y_hat.argmax(axis=1)  # across columns
    is_correct = y_hat.type(y.dtype) == y
    return float(is_correct.type(y.dtype).sum())

#@save
def accuracy(net, data_iter):
    """
    :param net: 模型
    :param data_iter: 图像分类数据集
    :return: 模型的准确率，这里使用了Accumulator和correct_predictions
    """
    net.eval()  # Set the model to evaluation mode
    metric = d2l.Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        # y.numel()表示y中的数据数
        metric.add(d2l.correct_predictions(net(X), y), y.numel())
    return metric[0] / metric[1]

## 动画

为了让我们的训练过程更加直观，我们实现一个展示训练过程中各项数据动态变化的类

from IPython import display
import matplotlib.pyplot as plt


#@save
def use_svg_display():
    """使用svg格式"""
    display.set_matplotlib_formats('svg')

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置坐标轴"""
    # 设置坐标标签
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    # 设置比例尺，{`linear`, `log`, ...}
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    # 设置x轴和y轴的显示范围
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    # 加上图例、网格
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
class Animator:
    """动态画折线图"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5)):
        """参数都是 matplotlib 画图的参数"""
        # 使用svg格式
        d2l.use_svg_display()
        # 获得画布和坐标轴
        self.fig, self.axes = plt.subplots(figsize=figsize)
        # config_axes() 即 d2l.set_axes(self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.config_axes = lambda: d2l.set_axes(self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure"""
        if not hasattr(y, "__len__"):
            y = [y]
        # Total n curves
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        # initialization
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        # 添加数据
        for i, (a, b) in enumerate(zip(x, y)):
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes.cla()  # 清除子图目前状态，防止重叠
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        # 不是多图而是动态
        display.clear_output(wait=True)

## 训练图像分类的函数

#@save
def train_image_classifier(net, train_iter, test_iter, learning_rate, num_epochs):
    """
    训练图像分类器，记录数据并以动画展示
    e.g. training FashionMNIST
    """
    device = d2l.try_gpu()
    # 需模型和数据均转向device
    net = net.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 记录误差和、正确预测样本数、总样本数
    metric = d2l.Accumulator(3)
    # 画训练误差、训练准确率、测试准确率
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs],
                            legend=["train_loss", "train_acc", "test_acc"])
    for epoch in range(num_epochs):
        net.train()  # 因为计算accuracy会使net转向eval模式
        metric.reset()
        for x, y in train_iter:
            # Compute prediction error
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            loss = loss_fn(y_hat, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录数据
            metric.add(float(loss) * len(y), d2l.correct_predictions(y_hat, y), y.numel())
        # 画图
        animator.add(epoch + 1, 
                     (metric[0] / metric[2], metric[1] / metric[2], d2l.accuracy(net, test_iter)))
    # 打印最终的数据
    print(f"loss {animator.Y[0][-1]:.3f}, "
          f"train acc {animator.Y[1][-1]:3f}, "
          f"test acc {animator.Y[2][-1]: 3f}")

分类问题的损失函数CrossEntropyLoss的计算公式:

$$\mbox{loss}(x, class) = -\mbox{log}\left(\frac{\mbox{exp}(x[class])}{\sum_{j}\mbox{exp}(x[j])}\right) = -x[class] + \log\left ({\sum_{j}\exp({x[j]})}\right )$$

