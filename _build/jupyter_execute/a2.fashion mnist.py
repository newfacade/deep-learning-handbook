# Fashion-MNIST数据集

```{note}
Fashion-MNIST是一个替代MNIST的图像分类数据集（因为MNIST太简单了）<br/>
它的的大小、格式和训练集/测试集划分与MNIST完全一致：60000/10000的训练测试数据划分，28x28的灰度图片，分成10个类别<br/>
我们可以使用torchvision下载并预处理Fashion-MNIST，后面我们会多次使用到这个数据集
```

## 加载数据

from torch.utils import data
from torchvision import datasets, transforms


#@save
def load_data_fashion_mnist(batch_size, resize=None):
    """加载Fashion-MNIST."""
    # 定义transforms，肯定要ToTensor，Resize is optional
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # 下载数据
    train_set = datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    test_set = datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    # dataset to data_iter
    return (data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4))


train_iter, test_iter = load_data_fashion_mnist(batch_size=128)

## 探索

Fashion-MNIST中包含的10个类别分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。以下函数用于在数字标签索引及其文本名称之间进行转换。

def get_fashion_mnist_labels(labels):
    """label to name"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

import matplotlib.pyplot as plt


def show_images(images, num_rows, num_columns, scale=2, titles=None):
    """
    展示图片
    """
    # 一纸多图
    _, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * scale, num_rows * scale))
    axes = axes.flatten()
    names = get_fashion_mnist_labels(titles)
    # 一个image一个ax
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img.squeeze(0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles is not None:
            ax.set_title(names[i])
    plt.show()

for X, y in train_iter:
    print(X.shape)
    print(y.shape)
    show_images(X[: 10], num_rows=2, num_columns=5, titles=y[: 10])
    break

