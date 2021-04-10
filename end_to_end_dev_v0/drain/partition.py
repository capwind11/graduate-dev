# -*- coding:utf-8 -*-

import pandas as pd
import re
from collections import OrderedDict
from tqdm import tqdm


class Partition:

    def __init__(self, outputFileDir, log_item_to_event_id_file, instances_file_path, label_file_path, normal_output, abnormal_output):
        self.outputFileDir = outputFileDir
        self.log_item_to_event_id_file = log_item_to_event_id_file
        self.instances_file_path = outputFileDir + instances_file_path
        self.label_file_path = label_file_path
        self.normal_output = outputFileDir + normal_output
        self.abnormal_output = outputFileDir + abnormal_output

    def partition(self):
        struct_log = pd.read_csv(self.log_item_to_event_id_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in tqdm(struct_log.iterrows(),desc="map blockId to event sequence"):
            blkId_list = row['BlockId'].split()
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        data_df.to_csv(self.instances_file_path)

    def partition_by_file(self):

        data_dict = OrderedDict()
        with open(self.log_item_to_event_id_file) as lines:
            for row in tqdm(lines,desc="map blockId to event sequence"):
                [_,blockId,eventId] = row.strip().split(',')
                if blockId=="BlockId":
                    continue
                blkId_list = blockId.split()
                blkId_set = set(blkId_list)
                for blk_Id in blkId_set:
                    if not blk_Id in data_dict:
                        data_dict[blk_Id] = []
                    data_dict[blk_Id].append(int(eventId))
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        data_df.to_csv(self.instances_file_path)

    def map_log_seq_to_label(self):
        data_df = pd.read_csv(self.instances_file_path, index_col=0, engine='c', na_filter=False,
                              memory_map=True)
        blockId_to_label = pd.read_csv(self.label_file_path, engine='c', na_filter=False, memory_map=True)
        blockId_to_label = blockId_to_label.set_index('BlockId')
        label_dict = blockId_to_label['Label'].to_dict()
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        data_df['EventSequence'] = data_df['EventSequence'].apply(lambda x: list(map(int,x[1:-1].split(','))))
        # normal_output
        anormal_data = data_df.iloc[(data_df['Label'] == 1).tolist()][['BlockId','EventSequence']]
        anormal_data.to_csv(self.abnormal_output,index=False)
        normal_data = data_df.iloc[(data_df['Label'] == 0).tolist()][['BlockId','EventSequence']]
        # nomaly_data['EventSequence'] = nomaly_data['EventSequence'].apply(lambda x: ' '.join(x))
        normal_data.to_csv(self.normal_output,index=False)
        data_df.to_csv('data_instances.csv',index=False)


if __name__=='__main__':
    preprocess = Partition('./results/', './data/', "log_item_to_label.csv", 'data_instances_hdfs.csv', 'anormaly_label.csv', 'normal.csv', 'abnormaly.csv')
    blockId_to_logs = preprocess.partition_by_file()
    data_df = pd.read_csv("./results/data_instances_hdfs.csv", index_col=0,engine='c', na_filter=False, memory_map=True)
    preprocess.map_log_seq_to_label(data_df)