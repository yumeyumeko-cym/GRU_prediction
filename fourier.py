import pandas as pd
import numpy as np
import torch
import csv

# Load dataset
dataset_path = 'Fourier_test.csv'
dataset = pd.read_csv(dataset_path)
rate_values = torch.FloatTensor(dataset['RATE'].values).view(-1, 1).squeeze()

# Fourier series expansion
def create_fourier_series(data, num_terms, period):
    x = torch.FloatTensor(np.arange(len(data))).unsqueeze(1)
    fourier_terms = x.clone()
    for n in range(1, num_terms + 1):
        cos_term = torch.cos(n * 2 * np.pi * x / period)
        sin_term = torch.sin(n * 2 * np.pi * x / period)
        fourier_terms = torch.cat((fourier_terms, cos_term, sin_term), axis=1)
    return fourier_terms[:, 1:]

# Model definition
class FourierLinearRegression(torch.nn.Module):
    def __init__(self, input_feature):
        super(FourierLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_feature, 1)

    def forward(self, x):
        return self.linear(x).squeeze()

# Hyperparameters
N = 30
period = 2030
learning_rate = 0.05
num_epochs = 2000

# Data preparation
fourier_series = create_fourier_series(rate_values, N, period)
input_feature = fourier_series.shape[1]

# Model initialization
model = FourierLinearRegression(input_feature)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(fourier_series)
    loss = loss_function(predictions, rate_values)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save predictions
with open('Fourier_output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for value in model(fourier_series).detach().numpy():
        writer.writerow([value])
