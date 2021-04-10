import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train(model,dataloader,criterion,optimizer,current_epoch=0,num_epochs=10,input_size=1):
    total_step = len(dataloader)
    model.train()
    start_time = time.time()
    for epoch in range(current_epoch,current_epoch+num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, seq.shape[1], input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            # writer.add_graph(model, seq)
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, current_epoch+num_epochs, train_loss / total_step))
        # writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))