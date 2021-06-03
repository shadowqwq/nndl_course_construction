# 代码修改自：https://github.com/ShaoQiBNU/pyTorch_MNIST
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def preprocess_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    data_train = datasets.MNIST(root="MNIST/", transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root="MNIST/", transform=transform, train=False)
    batch_size = 128
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)
    return data_loader_train, data_loader_test

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def train(train_loader, model, optimizer, cost, epoch):
    print("Start training:")
    for i in range(epoch):
        train_correct = 0
        total_cnt = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            _, pred = torch.max(output.data, 1)
            loss = cost(output, target)
            loss.backward()
            optimizer.step()
            total_cnt += data.data.size()[0]
            train_correct += torch.sum(pred == target.data)
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(data_loader_train):
                print("epoch: {}, batch_index: {}, train loss: {:.6f}, train correct: {:.2f}%".format(
                    i, batch_idx+1, loss, 100*train_correct/total_cnt))
    print("Training is over!")

def test(test_loader, model, cost):
    print("Start testing:")
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data), Variable(target)
        output = model(data)
        _,pred = torch.max(output.data, 1)
        loss = cost(output, target)
        test_correct = torch.sum(pred == target.data)
        print("batch_index: {}, test loss: {:.6f}, test correct: {:.2f}%".format(
                batch_idx + 1, loss.item(), 100*test_correct/data.data.size()[0]))
    print("Testing is over!")

if __name__ == '__main__':
    data_loader_train, data_loader_test = preprocess_data()
    model = LeNet()
    print("LeNet model is as follows:")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cost = nn.CrossEntropyLoss()
    epoch = 20
    train(data_loader_train, model, optimizer, cost, epoch)
    test(data_loader_test, model, cost)