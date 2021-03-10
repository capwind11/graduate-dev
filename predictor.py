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

def generate_test_data(name,window_size=10):
    hdfs = set()
    # hdfs = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = [0]+list(map(lambda n: n, map(int, ln.strip().split())))+[30]
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
            # hdfs.append(tuple(ln))
    session_to_seq = []
    seqs = []
    labels = []
    seq_count = 0
    for line in tqdm(hdfs, "normal:"):
        session = []
        for i in range(len(line) - window_size):
            seq = line[i:i + window_size]
            label = line[i + window_size]
            seqs.append(seq)
            session.append(seq_count)
            labels.append(label)
            seq_count += 1
        session_to_seq.append(session)
    print('Number of sessions({}): {}'.format(name, len(session_to_seq)))
    print('Number of seqs({}): {}'.format(name, len(seqs)))
    dataset = TensorDataset(torch.tensor(seqs, dtype=torch.float), torch.tensor(labels))

    # print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return session_to_seq, dataset, seqs,labels

# fast predict
def fast_predict(model,normal_dataloader,abnormal_dataloader,test_normal_session,test_abnormal_session, num_candidates=5,window_size=10):
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    test_normal_result = []
    test_abnormal_result = []
    with torch.no_grad():
        with torch.no_grad():
            for step, (seq, labels) in tqdm(enumerate(normal_dataloader), desc='normal'):
                seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                output = model(seq).cpu()

                predicted = torch.argsort(output[:,-1,:], 1)[:,-num_candidates:]
                for i, label in enumerate(labels):
                    if label not in predicted[i]:
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

            predicted = torch.argsort(output[:,-1,:], 1)[:,-num_candidates:]
            for i, label in enumerate(labels):
                if label not in predicted[i]:
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

if __name__=='__main__':
    # model.load_state_dict(torch.load(model_dir + '/' + log + '.pt'))
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    # model_name = 'data_dir={}_version={}'.format(file_dir, 'v0.0')
    model_name ='add_padding_batch_size=2048_epoch=300_window_size=10'
    model.load_state_dict(torch.load(model_dir + '/' + model_name + '.pt'))
    model.to(device)
    model.eval()
    batch_size = 20000
    window_size = 10
    test_normal_session, test_normal_dataset, test_normal_seq, test_normal_label = generate_test_data(
        file_dir+'/hdfs_test_normal', window_size)
    normal_dataloader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_abnormal_session, test_abnormal_dataset, test_abnormal_seq, test_abnormal_label = generate_test_data(
        file_dir+'/abnormal', window_size)
    abnormal_dataloader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_normal_result, test_abnormal_result = fast_predict(model, normal_dataloader, abnormal_dataloader, 10,
                                                            window_size)
