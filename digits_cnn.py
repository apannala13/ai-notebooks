#two basic components of a CNN - kernel and Input 
#input is typically an image or multidimensional array representing data (features), kernel is a small matrix of weights, 
#that perform convolutions on input data 
#convolutions are a sliding window operation, combining the input and kernel. 
#e.g. systematically moving a magnifying glass across a painting, focusing on small areas at a time 
#creating a sort of "map" of what we've observed - this is the feature map in convulution operation 
#the output matrix is generated from performing the convolution operation
#in order for a layer to have the same width and height as the previous layer, we add zero padding 


from sklearn.datasets import load_digits
import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
num_classes = len(np.unique(y))
print(num_classes)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled.mean()

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #grayscale = channel 1, 32 kernels of 3 * 3 weight matrices
        self.layer2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 2 * 2, 64) #feature maps 16, each of size 2 * 2, passed to 64 features
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.maxpool1(x)
        x = F.relu(self.layer2(x))
        x = self.maxpool1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x  = self.fc2(x)
        return x


model = CNN()
epochs = 20
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0)
floss = nn.CrossEntropyLoss()

for _ in range(epochs):
    model.train()
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.view(X_batch.size(0), 1, 8, 8) #batch size, 1 channel, size 8 * 8 for each image. Conv2D expects a 4D tensor
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = floss(outputs, y_batch)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), 1, 8, 8)
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1)
    loss_val = floss(outputs, preds)
    correct = (preds == y_test).sum().item()
    
    print(f"Loss: {loss_val.item():.4f}")
    print(f'Correct: {correct}')
        
