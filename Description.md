# LeNet_MINIST

- 使用LeNet神经网络在MNIST数据集上训练一个手写数字识别模型，并使用测试集验证识别准确率。模型由两层卷积网络和一个线性层组成。每层卷积网络包含一个使用RELU激活函数的卷积层和一个最大池化层。模型使用自适应矩估计优化器和交叉熵损失函数，训练20个epoch。

- Use LeNet neural network to train a handwritten digit recognition model on the MNIST data set, and use the test set to verify the recognition accuracy. The model consists of a two-layer convolutional network and a linear layer. Each layer of convolutional network contains a convolutional layer using the RELU activation function and a maximum pooling layer. The model uses the Adam optimizer and the Cross-Entropy Loss function to train for 20 epochs.

# NIN_CIFAR10

- 使用NIN神经网络在CIFAR10数据集上训练一个图像分类模型。模型由三层mplconv网络和全局均值池化层构成。每层mplconv网络包含三个使用RELU激活函数的卷积层。模型使用自适应矩估计优化器和交叉熵损失函数，训练20个epoch。

- Use NIN neural network to train an image classification model on the CIFAR10 dataset. The model consists of a three-layer mplconv network and a global mean pooling layer. Each layer of the mplconv network contains three convolutional layers that use the RELU activation function. The model uses the Adam optimizer and the Cross-Entropy Loss function to train for 20 epochs.
