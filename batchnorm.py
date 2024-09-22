import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_


digits = load_digits()

def process_feature(digits):
    X = digits.data
    y = digits.target 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype = torch.long)

    tensor_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

    return X_test_tensor, y_test_tensor, dataloader

        
class NeuralNet(nn.Module):
     def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)
    
     def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x 


model = NeuralNet()
criterion = nn.CrossEntropyLoss()
num_epochs = 20 
optimizer = optim.SGD(model.parameters(), lr=0.1)

X_test_tensor, y_test_tensor, dataloader = process_feature(digits)

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        model.train()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f'Epoch[{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f'Test Accuracy: {accuracy:.4f}')


    
        
