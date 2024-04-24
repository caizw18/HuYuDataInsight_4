# This is the code for ResNet model based on blocks of version 2 (Bottleneck Block)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

Epochs = 50 # Set number of epochs
Batches = 50 # Set batch size
Learning_r = 0.01 # Set learning rate

# We download the CIFAR-10 dataset from website: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Since we use the Google colab to do this project, we save the dataset into the folder ./data
train_dataset = torchvision.datasets.CIFAR10("./data/", train = True,
                                             transform = transforms.Compose([
                                                 transforms.Pad(4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomCrop(32),
                                                 transforms.ToTensor()]), download = True)

test_dataset = torchvision.datasets.CIFAR10("./data/", train = False, transform = transforms.ToTensor())

# We load the downloaded dataset by DataLoader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Batches, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Batches, shuffle = False)

# Define the class of Bottleneck_Block
class BottleneckBlock(nn.Module):

    # The output dimension of each residual structure in Bottleneck is 4 times the input dimension
    # so here expansion = 4
    expansion = 4

    def __init__(self, inputs, outputs, stride=1):
        super(BottleneckBlock, self).__init__()
        # The first 1x1 convolution for Bottleneck Block
        self.conv1 = nn.Conv2d(inputs, outputs, kernel_size=1, bias=False)
        # The first batch normalization
        self.bn1 = nn.BatchNorm2d(outputs)
        # The second 3x3 convolution for Bottleneck Block
        self.conv2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=stride, padding=1, bias=False)
        # The second batch normalization
        self.bn2 = nn.BatchNorm2d(outputs)
        # The third 1x1 convolution for Bottleneck Block
        self.conv3 = nn.Conv2d(outputs, self.expansion * outputs, kernel_size=1, bias=False)
        # The third batch normalization
        self.bn3 = nn.BatchNorm2d(self.expansion * outputs)

        # When stride is not 1 or channels of inputs is not equal to channels of outputs, in order to  perform
        # the addition successfully in bottleneck_block, we shortcut the identity via passed convolution layer
        self.shortcut = nn.Sequential()
        if stride != 1 or inputs != self.expansion * outputs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inputs, self.expansion * outputs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * outputs)
            )

    def forward(self, x):
        # First 1x1 convolution, then batch normalization, then ReLU
        out_x = F.relu(self.bn1(self.conv1(x)))
        # Second 3x3 convolution, then batch normalization, then ReLU
        out_x = F.relu(self.bn2(self.conv2(out_x)))
        # Third 1x1 convolution, then batch normalization
        out_x = self.bn3(self.conv3(out_x))
        # Shortcut to make the addition successful
        out_x += self.shortcut(x)
        # The residual structure passes through the ReLU layer after addition
        out_x = F.relu(out_x)
        return out_x

# Define the ResNet based on Bottleneck Block
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10): # 10 classes of images classification
        super(ResNet, self).__init__()
        self.inputs_r = 64

        # The convolution and batch normalization before layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layers built by Bottleneck Block, the number of blocks in every layer depends on the input list
        # For example, if the input list is [3,4,6,3], then layer 1,2,3,4 has 3,4,6,3 bottleneck blocks, respectively
        # The total number of layers is 1(conv1) + 3x(3+4+6+3) + 1(fc layer) = 50, so it is ResNet-50
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Import the output above to fully connected layer for classification
        self.full_c = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, outputs_r, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for str in strides:
            layers.append(block(self.inputs_r, outputs_r, str))
            self.inputs_r = outputs_r * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out_x = F.relu(self.bn1(self.conv1(x)))

        out_x = self.layer1(out_x)
        out_x = self.layer2(out_x)
        out_x = self.layer3(out_x)
        out_x = self.layer4(out_x)

        # Import the output above to the global average pooling layer for processing
        out_x = F.avg_pool2d(out_x, 4)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.full_c(out_x)
        return out_x

# We use the GPU V-100 on Google colab to perform since it takes a very long time to perform by CPU
Dev = torch.device('cuda')
# Our model here is based on ResNet (bottleneck block version) with layers [3,4,6,3]
model = ResNet(BottleneckBlock, [3, 4, 6, 3]).to(Dev)

# Define the loss function
# Since we are training a classification problem with 10 classes,
# we compute the cross entropy loss between input logits and target
criterion = nn.CrossEntropyLoss()

# Update learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_r)
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training dataset
total_length = len(train_loader)
current_lr = Learning_r
for epoch in range(Epochs):
    for i, (Imgs, labels) in enumerate(train_loader):
        Imgs = Imgs.to(Dev)
        labels = labels.to(Dev)

        # The forward process
        # Outputs from our model
        outputs = model(Imgs)
        # Loss between the output and real labels
        loss = criterion(outputs, labels)

        # The backward and optimize process
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        j = i + 1
        if j % 100 == 0:
            # Print the value of loss for each step in each epoch
            str = "At {} epoch of {} totally, step {} of {}, the loss is {:.5f}".format(epoch+1, Epochs,
                                                                                        j, total_length, loss.item())
            print(str)

    # Learning rate
    if (epoch + 1) % 20 == 0:
        current_lr /= 3
        update_lr(optimizer, current_lr)

# Test model
model.eval()
with torch.no_grad():
    # Record the number of correct predicts and number of totals to get the accuracy
    correct_pre, num_total = 0, 0
    for images, labels in test_loader:
        images = images.to(Dev)
        labels = labels.to(Dev)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        num_total += labels.size(0)
        correct_pre += (predicted == labels).sum().item()

    str = "The accuracy of ResNet model based on Bottleneck Block is {} %".format(correct_pre / num_total * 100)
    print(str)

