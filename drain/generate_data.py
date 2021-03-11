import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from ast import literal_eval

def generate_train_data(name):
    num_sessions = 0
    inputs = []
    outputs = []
    data = pd.read_csv(name, engine='c', na_filter=False, memory_map=True)
    blockId_list= data['BlockId'].tolist()
    seqs = data['EventSequence'].apply(literal_eval).tolist()
    for line in tqdm(seqs, "loading data"):
        num_sessions += 1
        seq = [0] + line + [30] + [31] * (window_size - 1)
        line = tuple(seq)

        for i in range(len(line) - window_size):
            inputs.append(line[i:i + window_size])
            outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset

def generate_test_data(name,window_size=10):
    hdfs = set()
    data = pd.read_csv(name, engine='c', na_filter=False, memory_map=True)
    blockId_list= data['BlockId'].tolist()
    seqs = data['EventSequence'].apply(literal_eval).tolist()
    for ln in seqs:
        ln = [0]+ln+[30]
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
    return session_to_seq, dataset, seqs,labels