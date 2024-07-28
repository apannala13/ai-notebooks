import torch
import torch.nn as nn
import numpy as np

class SGDOptim:
    def __init__(self, learning_rate=0.01):
        self.model = nn.Linear(1, 1)  
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self, x, y, num_epochs=50, batch_size=10):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()  # convert array to tensor
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        for epoch in range(num_epochs):
            perm = torch.randperm(x.size(0))  # randomize order of the data points for SGD
            for i in range(0, x.size(0), batch_size):  # iterate in mini batches
                idxs = perm[i:i + batch_size]  # slicing dataset into batches of size batch_size
                batch_x, batch_y = x[idxs], y[idxs]  # input features for current batch and corresponding labels
                preds = self.model(batch_x)  # forward pass
                loss = self.loss_fn(preds, batch_y)  # compute loss using MSE
                self.optimizer.zero_grad()  # zero out existing gradients
                loss.backward()  # backpropagate
                self.optimizer.step()  # update weights based on new gradients and SGD
        
    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.model(x)

    def get_params(self):
        return self.model.weight.item(), self.model.bias.item()

if __name__ == "__main__":
    x_np = np.random.randn(100, 1)
    y_np = 2 * x_np + 3 + np.random.randn(100, 1)
    model = SGDOptim(learning_rate=0.01)
    model.train(x_np, y_np, num_epochs=50, batch_size=10)
    weight, bias = model.get_params()
    print(f"weight = {weight}, bias = {bias}")

    x_test = np.array([[1.0], [2.0], [3.0]])
    predictions = model.predict(x_test)
    print("Predictions:", predictions)
