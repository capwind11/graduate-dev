from drain.model import Drain,Node,LogCluster
from drain.optimizer import Optimizer
from drain.plotter import createPlot
from drain.partition import Partition
from utils.generate_data import *
from lstm.trainer import Model,train
from lstm.predictor import fast_predict
from torch import nn,optim
from torch.utils.data import  DataLoader
import pandas as pd
from lstm.wf_constructor import workflow_constructor
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 32
batch_size = 1024
input_size = 1
window_size = 10
num_layers = 2
hidden_size = 64
'''
parse
logPath: HDFS.log
parsed_result:logClusters;item_to_event_id.csv

partition
label_file:anoramly_label.csv
parsed_result:item_to_event_id.csv
instance_file: hdfs_instance
output:normaly.csv/abnormal.csv

split_data:
normaly:train,test_normal,test_abnormal

lstm:
input:train,test_(ab)normal
output:model
'''
# parse
logPath = 'D:\\毕业设计\\loghub\\HDFS_1\\HDFS.log'
parsed_result = '.\\data\\parser\\log_item_to_label.csv'
cluster_result = '.\\data\\parser\\'
# partition
partition_output = '.\\data\\lstm\\dataset\\'
instance_file = 'instance.csv'
label_file = '.\\data\\partition\\anormaly_label.csv'
normal_output = 'normal.csv'
abnormal_output = 'abnormal.csv'
lstm_dataset = '.\\data\\lstm\\dataset\\'
# instance_file: hdfs_instance
# output:normaly.csv/abnormal.csv
#
# split_data:
# normaly:train,test_normal,test_abnormal
#
# lstm:
# input:train,test_(ab)normal
# output:model

'''
parsing phase
'''
def parse_log():

    rex = ['blk_(|-)[0-9]+', '(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']
    removeCol = [0,1,2]
    myParser = Drain(rex=rex,removeCol=removeCol,st=0.5)
    myParser.fit(isReconstruct=True,inputFile=logPath,outputFile=parsed_result)

    myParser.save(savePath=cluster_result)

'''
partition phase
'''
def partition():

    partition = Partition(outputFileDir=partition_output,log_item_to_event_id_file=parsed_result,instances_file_path=instance_file,label_file_path=label_file,normal_output=normal_output,abnormal_output=abnormal_output)
    partition.partition_by_file()
    partition.map_log_seq_to_label()

    split_data(input_dir=partition_output,output_dir=lstm_dataset)

if __name__=='__main__':

    # parse_log()
    # exit(0)
    partition()
    exit(0)

    model_dir = 'model'
    version = 'v0.1'
    model_name = 'unsupervised_version={}'.format(version)

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=31)
    optimizer = optim.Adam(model.parameters())
    if os.path.exists(model_dir + '/' + model_name + '.pt'):
        model.load_state_dict(torch.load(model_dir + '/' + model_name + '.pt'))

    # model.train()
    # train_dataset = generate_train_data(lstm_dataset + 'train.csv')
    # dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    #
    # train(model, dataloader, criterion, optimizer, current_epoch=0, num_epochs=100)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + model_name + '.pt')
    print('Finished Training')

    model.eval()
    batch_size = 20000
    test_normal_session, test_normal_dataset, test_normal_seq, test_normal_label = generate_test_data(
        lstm_dataset + '/normal.csv', window_size)
    normal_dataloader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_abnormal_session, test_abnormal_dataset, test_abnormal_seq, test_abnormal_label = generate_test_data(
        lstm_dataset + '/abnormal.csv', window_size)
    abnormal_dataloader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_normal_result, test_abnormal_result = fast_predict(model, normal_dataloader, abnormal_dataloader,
                                                            test_normal_session, test_abnormal_session,
                                                            10, window_size,ts=0.01)