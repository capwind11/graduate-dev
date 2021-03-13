from utils.bert_utils import *
from utils.generate_data import *
from torch.utils.data import DataLoader,random_split
from supervised.model import *
import torch
import torch.nn as nn
import torch.optim as optim


num_classes = 2
num_epochs = 20
batch_size = 50
input_size = 768
model_dir = 'model'
window_size = 10
num_layers = 2
hidden_size = 64
file_dir = 'data_supervised'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(y_pred, y_true):
    return (np.argmax(y_pred.cpu().numpy(),1) == y_true.numpy()).astype('int').mean()

if __name__ == '__main__':
    # build_bert_cache('data/parser/logClusters/logTemplates.txt', 'data/lstm/')
    train_data, test_data, train_x, test_x, train_y, test_y = generate_bert_data('data/lstm/dataset/','data/lstm/bert_cache.pth')
    batch_size = 200
    model_dir = 'model'
    version = 'v0.1'
    model_name = 'supervised_version={}'.format(version)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=batch_size)
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)

    if os.path.exists(model_dir + '/' + model_name + '.pt'):
        model.load_state_dict(torch.load(model_dir + '/' + model_name + '.pt'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(model,train_data,criterion,optimizer,num_epochs=100,input_size=input_size)
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + model_name + '.pt')

    with torch.no_grad():
        epoch_loss = 0
        for step, (seq, label) in enumerate(train_data):
            seq = seq.clone().detach().view(-1, seq.shape[1], input_size).to(device)
            test_output = model(seq.to(device))
            if step == 0:
                output = test_output
                labels = label
            else:
                labels = torch.cat([labels, label], 0)
                output = torch.cat([output, test_output], 0)
            epoch_loss += criterion(test_output, label.to(device)).data
        epoch_accuracy = accuracy(output, labels)
        epoch_loss = epoch_loss / len(train_data)
        print('loss: ', round(epoch_loss.item(), 3), 'accuracy: ', round(epoch_accuracy.item(), 3))
        epoch_loss = 0
        for step, (seq, label) in enumerate(test_data):
            seq = seq.clone().detach().view(-1, seq.shape[1], input_size).to(device)
            test_output = model(seq.to(device))
            if step == 0:
                output = test_output
                labels = label
            else:
                labels = torch.cat([labels, label], 0)
                output = torch.cat([output, test_output], 0)
            epoch_loss += criterion(test_output, label.to(device)).data
        epoch_accuracy = accuracy(output, labels)
        epoch_loss = epoch_loss / len(train_data)
        print('test_loss: ', round(epoch_loss.item(), 3), 'test_accuracy: ', round(epoch_accuracy.item(), 3))

