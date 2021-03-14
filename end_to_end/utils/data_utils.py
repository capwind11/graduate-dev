import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader,random_split
from tqdm import tqdm
from ast import literal_eval

def remove_duplicate(input,col_remove,col_keep,output):
    data = pd.read_csv(input, engine='c', na_filter=False, memory_map=True)
    data = data.groupby([col_remove])[col_keep].max()
    data[[col_keep,col_remove]].to_csv(output)

def split_data(input_dir,output_dir, rate = 0.01):
    count = 0
    with open(input_dir + '/normal.csv', 'r') as f:
        head = f.readline()
        for _ in f:
            count+=1
    train_num = int(rate*count)
    data_set = random_split(range(count),[train_num,count-train_num])
    f= open(input_dir + '/normal.csv', 'r')
    lines = f.readlines()
    f.close()
    f1 = open(output_dir + '/normal.csv', 'w')
    f1.write(head)
    f2 = open(output_dir + '/train.csv', 'w')
    f2.write(head)
    for i in data_set[0]:
        f2.write(lines[i+1])
    for i in data_set[1]:
        f1.write(lines[i+1])
    f1.close()
    f2.close()

def generate_train_data(name,window_size=10):
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

def generate_predicted_data(name, window_size=10):

    event_seq_set = set()
    data = pd.read_csv(name, engine='c', na_filter=False, memory_map=True)
    blockId_list= data['BlockId'].tolist()
    event_sequence = data['EventSequence'].apply(literal_eval).tolist()
    block_to_seq = []
    for i,line in enumerate(event_sequence):
        if tuple(line) not in event_seq_set:
            block_to_seq.append([blockId_list[i],line])
            event_seq_set.add(tuple(line))

    session_to_seq = []
    seqs = []
    labels = []
    seq_count = 0
    event_seq_set = list(event_seq_set)
    for line in tqdm(block_to_seq, name):
        session = []
        line = line[1]
        line = [0]+list(line)+[30]
        line = line + [31] * (window_size + 1 - len(line))
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
    return session_to_seq, dataset, block_to_seq

def generate_bert_data(file_dir,normal_file,abnormal_file,bert_cache_path):
    eventId_to_bert = torch.load(bert_cache_path)
    padding = torch.zeros_like(eventId_to_bert[5][1][0])
    eventId_to_bert[0] = [[], [padding]]
    sessions = []
    labels = []
    max_len = 50
    normal_data = set()
    abnormal_data = set()
    data = pd.read_csv(file_dir+normal_file, engine='c', na_filter=False, memory_map=True)
    seqs = data['EventSequence'].apply(literal_eval).tolist()
    for line in tqdm(seqs, "loading data"):
        if len(line) > 50:
            continue
        normal_data.add(tuple(line))
    data = pd.read_csv(file_dir+abnormal_file, engine='c', na_filter=False, memory_map=True)
    blockId_list = data['BlockId'].tolist()
    seqs = data['EventSequence'].apply(literal_eval).tolist()
    for line in tqdm(seqs, "loading data"):
        if len(line) > 50:
            continue
        abnormal_data.add(tuple(line))
    for line in tqdm(normal_data, "normal:"):
        line = list(line) + [0] * (max_len - len(line))
        bert_input = []
        for id in line:
            bert_input.append(eventId_to_bert[id][1][0].cpu().numpy())
        sessions.append(tuple(bert_input))
        labels.append(0)
    for line in tqdm(abnormal_data, "abnormal:"):
        line = list(line) + [0] * (max_len - len(line))
        bert_input = []
        for id in line:
            bert_input.append(eventId_to_bert[id][1][0].cpu().numpy())
        sessions.append(tuple(bert_input))
        labels.append(1)

    print('Number of sessions({}): {}'.format(file_dir, len(sessions)))
    print('Number of normal sessions: {}'.format(len(normal_data)))
    print('Number of abnormal sessions: {}'.format(len(abnormal_data)))
    train_x, test_x, train_y, test_y = train_test_split(sessions, labels, test_size=0.4)
    train_data = TensorDataset(torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y))
    # train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = TensorDataset(torch.tensor(test_x, dtype=torch.float), torch.tensor(test_y))
    # test_data = DataLoader(test_data, batch_size=batch_size)
    return train_data, test_data