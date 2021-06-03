# 代码修改自：https://github.com/mandeer/Classifier
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn

def getDataLoader():
    transform = transforms.Compose([ transforms.Resize(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataTrain = datasets.CIFAR10(root='CIFAR10/', train=True, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(dataset=dataTrain, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    return trainLoader

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

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
    model = SqueezeNet()
    print(model)
    epoch = 20
    train(epoch, model, trainLoader)