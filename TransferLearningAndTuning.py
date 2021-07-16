import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision


# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5

import sys
model = torchvision.models.vgg16(pretrained=True)
print(model)
sys.exit()

# Load pre-trained model to modify it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform= transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset = train_dataset, batch_size= batch_size, shuffle=True)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network

for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Cost at {epoch} is {sum(losses)/len(losses)}')
