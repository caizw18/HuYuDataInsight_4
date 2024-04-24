# HuYuDataInsight_4
This is the workload for our company HuYuDataInsight LLC from Apr 15, 2024 to Apr 23, 2024

We write the Python code to perform CIFAN-10 images classification in PyTorch by ResNet based on bottleneck block. We train the model by GPU.

In the Python code, we define the class BottleneckBlock. The output dimension of each residual structure in Bottleneck is 4 times the input 
dimension, so here expansion is equal to 4. 

Then we define an instance method to initialize a newly created BottleneckBlock object. We define conv1 as the first 2D convolution layer 
with kernel size 1. Then bn1 is the first batch normalization. Then conv2 is the second 2D convolution layer with kernel size 3 and padding 1. Then 
bn2 is the second batch normalization. Then conv3 is the third 2D convolution layer with kernel size 1. Then bn3 is the third batch normalization.
We also define a shortcut to make the addition (+) successful.

In the forward function, the input x will at first go through conv1, then bn1, then ReLU, then conv2, then bn2, then ReLU, then conv3, then bn3. 
The residual structure goes through the ReLU layer after addition.

We define a class for ResNet. Before 4 layers made by blocks, we have a convolution layer and batch normalization. The critical layers are
built by two versions of blocks. The number of blocks in every layer depends on the input list. For example, if the input list is [3,4,6,3], then 
layer 1,2,3,4 has 3,4,6,3 blocks, respectively.

In ResNet, the output layer consists of a global average pooling operation and a fully connected layer for classification with 10 neurons, representing
10 image classes.
