# 代码修改自：https://github.com/mandeer/Classifier
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn

def getDataLoader():
    transform = transforms.Compose([ transforms.Resize(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataTrain = datasets.CIFAR10(root='CIFAR10/', train=True, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(dataset=dataTrain, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    return trainLoader

def make_layers():
    layers = []
    in_channels = 3
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
    model = VGG()
    print(model)
    epoch = 20
    train(epoch, model, trainLoader)