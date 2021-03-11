import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

num_classes = 32
batch_size = 20000
input_size = 1
model_dir = 'model'
window_size = 10
num_layers = 2
hidden_size = 64
file_dir = 'data'
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# fast predict
def fast_predict(model,normal_dataloader,abnormal_dataloader,test_normal_session,test_abnormal_session,num_candidates=5,window_size=10,ts=0.0003):
    TP = 0
    FP = 0
    softmax = nn.Softmax(dim = 1)
    # Test the model
    start_time = time.time()
    test_normal_result = []
    test_abnormal_result = []
    with torch.no_grad():
        with torch.no_grad():
            for step, (seq, labels) in tqdm(enumerate(normal_dataloader), desc='normal'):
                seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                output = model(seq).cpu()
                output = output[:,-1,:]
                prob = softmax(output)
                predicted = torch.argsort(output, 1)[:,-num_candidates:]
                for i, label in enumerate(labels):
                    if label not in predicted[i] or prob[i][label]<ts:
                        test_normal_result.append(True)
                    else:
                        test_normal_result.append(False)
    for session in test_normal_session:
        for seq_id in session:
            if test_normal_result[seq_id] == True:
                FP += 1
                break

    with torch.no_grad():
        for step, (seq, labels) in tqdm(enumerate(abnormal_dataloader), desc='abnormal'):
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq).cpu()
            output = output[:,-1,:]
            prob = softmax(output)
            predicted = torch.argsort(output, 1)[:,-num_candidates:]
#             predicted = torch.argsort(output[:,-1,:], 1)[:,-num_candidates:]
            for i, label in enumerate(labels):
                if label not in predicted[i] or prob[i][label]<ts:
                    test_abnormal_result.append(True)
                else:
                    test_abnormal_result.append(False)
        for session in test_abnormal_session:
            for seq_id in session:
                if test_abnormal_result[seq_id] == True:
                    TP += 1
                    break
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_session) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    return test_normal_result,test_abnormal_result