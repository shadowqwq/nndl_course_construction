# 代码修改自：https://github.com/mandeer/Classifier
import math
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn

def getDataLoader():
    transform = transforms.Compose([ transforms.Resize(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataTrain = datasets.CIFAR10(root='CIFAR10/', train=True, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(dataset=dataTrain, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    return trainLoader

def makeMlpConv(in_channel, hid_channel, out_channel, kernel_size=3, stride=1, padding=1):
    assert isinstance(hid_channel, int) or len(hid_channel) == 2
    if isinstance(hid_channel, int):
        hid1 = hid_channel
        hid2 = hid_channel
    else:
        hid1 = hid_channel[0]
        hid2 = hid_channel[1]
    return nn.Sequential(
        nn.Conv2d(in_channel, hid1, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(hid1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hid1, hid2, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(hid2),
        nn.ReLU(inplace=True),
        nn.Conv2d(hid2, out_channel, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )

class NIN(nn.Module):
    def __init__(self, num_classes):
        super(NIN, self).__init__()
        self.n_class = num_classes
        self.MlpConv1 = makeMlpConv(3, (192, 160), 96, kernel_size=5, stride=1, padding=2)
        self.MlpConv2 = makeMlpConv(96, 192, 192, kernel_size=5, stride=1, padding=2)
        self.MlpConv3 = makeMlpConv(192, 192, self.n_class, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.MlpConv1(x)
        x = self.pool1(x)
        x = self.MlpConv2(x)
        x = self.pool2(x)
        x = self.MlpConv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), self.n_class)
        return x

def train(total_epoch, model, trainLoader):
    model = model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(total_epoch):
        model.train()
        train_loss = 0.0
        for ii, (datas, labels) in enumerate(trainLoader):
            datas, labels = datas.to('cpu'), labels.to('cpu')
            optimizer.zero_grad()
            score = model(datas)
            loss = criterion(score, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
            if (ii + 1) % 100 == 0 or (ii + 1) == len(trainLoader):
                current_lr = optimizer.param_groups[0]['lr']
                print('epoch=%d, [%d/%d], loss=%.6f, lr=%.6f' % (epoch + 1, ii + 1, len(trainLoader), train_loss / (ii + 1), current_lr))
    return

if __name__ == '__main__':
    trainLoader = getDataLoader()
    model = NIN(num_classes=10)
    print(model)
    epoch = 20
    train(epoch, model, trainLoader)