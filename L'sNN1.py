import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.models as model
import matplotlib.pyplot as plt

# Hyperparameters
EPOCH = 12
BATCH_SIZE = 32

LR = 0.05
SPLIT_RATIO = 1

# 文件路径自己去改~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ：） ：） ：） :< :< :<
train_img = pd.read_pickle('/Users/tylerliu/GitHub/Proj3_source/train_max_x')
test_img = pd.read_pickle('/Users/tylerliu/GitHub/Proj3_source/test_max_x')
train_out = pd.read_csv('./data/train_max_y.csv').to_numpy().astype('int32')[:, -1]

train_img = torch.Tensor(train_img)
train_out = torch.tensor(train_out, dtype=torch.int64)
test_img = torch.Tensor(test_img)

# 自己改测试用的大小 50000改成别的->?
x = torch.unsqueeze(train_img, dim=1)[:SPLIT_RATIO*50000]/255.
x = x.repeat(1,3,1,1)
y = train_out[:50000]

# mini-sample for testing
x_t = torch.unsqueeze(train_img, dim=1)[:100]/255.
x_t = x_t.repeat(1,3,1,1)
y_t = train_out[:100]

my_dataset = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)

vgg16 = model.vgg16(pretrained=True)

cnn = nn.Sequential(
    vgg16,
    nn.Linear(1000,500),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(500,10),
    nn.Softmax()
)

# print(cnn)

# optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
optimizer = torch.optim.Adam(cnn.parameters(), lr = 1e-4)
loss_fun = nn.CrossEntropyLoss()

def train():
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            output = cnn(x)
            loss = loss_fun(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                x_t_out = cnn(x_t)
                y_pred = torch.max(x_t_out, dim=1)[1].data.numpy()
                accuracy = float((y_pred == y_t.data.numpy()).astype(int).sum()) / float(y_t.size(0))
                print('Epoch ', epoch, 'train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.3f' % accuracy)

train()