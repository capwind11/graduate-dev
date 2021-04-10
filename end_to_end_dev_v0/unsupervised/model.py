import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm

# Hyperparameters
num_classes = 32
num_epochs = 20
batch_size = 2048
input_size = 1
model_dir = 'model'
window_size = 10
num_layers = 2
hidden_size = 64
file_dir = 'data_dev'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_train_data(name,window_size=10):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(name, 'r') as f:
        for line in tqdm(f, "loading data"):
            num_sessions += 1
            seq = [0] + list(map(lambda n: n, map(int, line.strip().split()))) + [30] + [31] * (window_size - 1)
            line = tuple(seq)

            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, :, :])
        return out

def train(model,dataloader,criterion,optimizer,current_epoch=0,num_epochs=10,window_size=10,input_size=1):
    total_step = len(dataloader)
    start_time = time.time()
    for epoch in range(current_epoch,current_epoch+num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            label1= seq[:,1:,:].cpu().long()
            label2 = label.view(-1,1,1)
            label = torch.cat([label1,label2],1).view(-1,window_size)
            label = label.reshape(label.size(0)*label.size(1))
            output = model(seq)
            output = output.reshape(output.size(0)*output.size(1),-1)
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

if __name__=='__main__':

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    train_dataset = generate_train_data(file_dir + '/normal')
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(log_dir='log/' + log)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=31)
    optimizer = optim.Adam(model.parameters())
    # Train the model
    model_name = 'data_dir={}_version={}'.format(file_dir, 'v0.0')
    total_step = len(dataloader)
    if os.path.exists(model_dir + '/' + model_name + '.pt'):
        model.load_state_dict(torch.load(model_dir + '/' + model_name + '.pt'))
    model.train()
    train(model, dataloader, criterion, optimizer, current_epoch=0, num_epochs=20)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + model_name + '.pt')
    # writer.close()
    print('Finished Training')