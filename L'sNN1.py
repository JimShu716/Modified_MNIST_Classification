import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Hyperparameters
EPOCH = 1
BATCH_SIZE = 2
LR = 0.001
DOWNLAD_MINST = False
# 文件路径自己去改~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ：） ：） ：） :< :< :<
train_img = pd.read_pickle('/Users/tylerliu/GitHub/Proj3_source/train_max_x')
test_img = pd.read_pickle('/Users/tylerliu/GitHub/Proj3_source/test_max_x')
train_out = pd.read_csv('./data/train_max_y.csv').to_numpy().astype('int32')[:, -1]

train_img = torch.Tensor(train_img)
train_out = torch.tensor(train_out, dtype=torch.int64)
test_img = torch.Tensor(test_img)

# 自己改测试用的大小
x = torch.unsqueeze(train_img, dim=1)[:200] / 255.
y = train_out[:200]

# mini-sample for testing
x_t = torch.unsqueeze(train_img, dim=1)[:20] / 255.
y_t = train_out[:20]

my_dataset = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(     #input_size(1,128,128)
            nn.Conv2d(
                in_channels = 1,out_channels = 16, kernel_size = 5,stride = 1,padding=2
            ),nn.ReLU(),nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(     #input_size(1,64,64)
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(),nn.MaxPool2d(kernel_size=2)
        )

        self.out3 = nn.Linear(32*32*32,32*32*32)
        # self.out2 = nn.Linear(32 * 32 * 32, 32 * 32 * 32)
        self.out1 = nn.Linear(32 * 32 * 32, 10)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.out3(x))
        x = self.out1(x)
        return x

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_fun = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        output = cnn(x)
        loss = loss_fun(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            x_t_out = cnn(x_t)
            y_pred = torch.max(x_t_out,dim=1)[1].data.numpy()
            accuracy = float((y_pred == y_t.data.numpy()).astype(int).sum())/float(y_t.size(0))
            print('Epoch ', epoch, 'train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.3f' % accuracy)
