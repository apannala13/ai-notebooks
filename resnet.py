import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset 
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 

x_train_tensor = torch.tensor(x_train, dtype=torch.float64)
x_test_tensor = torch.tensor(x_test, dtype=torch.float64)
y_train_tensor, y_test_tensor = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

tensor_train = TensorDataset(x_train_tensor, y_train_tensor)
dataloader_train = DataLoader(dataset=tensor_train, batch_size=32, shuffle=True)
tensor_test = TensorDataset(x_test_tensor, y_test_tensor)
dataloader_test = DataLoader(dataset=tensor_test, batch_size=32, shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock).__init__()
        self.cnl1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.cnl2 = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.batchnorm1 = nn.BatchNorm2d(out_channel)
        self.batchnorm2 = nn.BatchNorm2d(out_channel)
    
    def forward(self, x):
        residual = x 
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = F.relu(self.batchnorm1(self.cnl1(x))) 
        out = self.batchnorm2(self.cnl2(x))
        out += residual 
        out = F.relu(out)
        return out 
        

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.mxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=1, downsample=nn.Sequential(
                                        nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False), 
                                        nn.BatchNorm2d(128)
                                    ))
        self.layer3 = ResidualBlock(128, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.mxp1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 

num_epochs = 25 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
floss = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in dataloader_test:
        optim.zero_grad()
        outputs = model(x_batch)
        loss = floss(outputs, y_batch)
        loss.backward()
        optim.step()

model.eval()
with torch.no_grad():
    pass 


