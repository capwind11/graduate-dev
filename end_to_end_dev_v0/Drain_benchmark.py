#!/usr/bin/env python

import sys
sys.path.append('../')
from drain import model_v1, evaluator,optimizer,Drain
import os
import pandas as pd


input_dir = 'data/logs/' # The input directory of log file
output_dir = 'data/logs/result/' # The output directory of parsing results

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'removeCol':[0,1,2],
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'removeCol':[0,1],
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'removeCol': [0, 1],
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.5,
        'depth': 4
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'removeCol': [0, 1],
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4        
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'removeCol': [0, 1,2,3,4,5],
        'regex': [r'core\.\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'removeCol': [0, 1,4],
        'regex': [r'=\d+'],
        'st': 0.5,
        'depth': 4
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'removeCol': [0, 1,4],
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'st': 0.7,
        'depth': 5      
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'st': 0.39,
        'depth': 6        
        },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'st': 0.2,
        'depth': 6   
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'st': 0.2,
        'depth': 4
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'st': 0.6,
        'depth': 3
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'depth': 5   
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'st': 0.5,
        'depth': 5
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.7,
        'depth': 6   
        },
}

def examine_main():

    bechmark_result = []
    i = 0
    for dataset, setting in list(benchmark_settings.items())[8:]:
        # if i >4:
        #     break
        # i+=1
        print('\n=== Evaluation on %s ==='%dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        try:
            parser = model_v1.Drain(log_format=setting['log_format'], rex=setting['regex'], depth=setting['depth'], st=setting['st'])
            parser.fit(isReconstruct=True,inputFile=input_dir+setting['log_file'],outputFile=output_dir+log_file + '_structured.csv')
        except:
            continue
        F1_measure, accuracy = evaluator.evaluate(
                               groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                               parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
                               )
        bechmark_result.append([dataset, F1_measure, accuracy])


    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.T.to_csv('Drain_bechmark_result.csv')

def examine_optimize():
    origin_bechmark_result = []
    v0_bechmark_result = []
    v1_bechmark_result = []
    v2_bechmark_result = []
    i = 0
    for dataset, setting in list(benchmark_settings.items()):

        print('\n=== Evaluation on %s ===' % dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        try:
            parser = model_v1.Drain(log_format=setting['log_format'], rex=setting['regex'], depth=setting['depth'] + 1,
                                    st=setting['st'])
            parser.fit(isReconstruct=True, inputFile=input_dir + setting['log_file'],
                       outputFile=output_dir + log_file + '_structured.csv')
        except:
            continue
        opt = optimizer.Optimizer()
        logClusters = parser.logClusters
        print('优化前,聚类数: ', len(logClusters), end=' ')
        F1_measure, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        v0_bechmark_result.append([dataset, F1_measure, accuracy])
        #     printClusters(logClusters)
        opt.modify(method='merge_sub_tree', resultFile=output_dir + log_file + '_structured.csv', logparser=parser)
        logClusters = parser.logClusters
        print('合并子树优化,聚类数: ', len(logClusters), end=' ')
        F1_measure, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        #     printClusters(logClusters)
        # createPlot(drain)
        v1_bechmark_result.append([dataset, F1_measure, accuracy])
        opt.modify(method='LCS', resultFile=output_dir + log_file + '_structured.csv', logparser=parser, st=0.8)
        logClusters = parser.logClusters
        print('合并聚类优化,聚类数: ', len(logClusters), end=' ')
        #     printClusters(logClusters)
        F1_measure, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        v2_bechmark_result.append([dataset, F1_measure, accuracy])
        parser = Drain.LogParser(log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'],
                                 depth=setting['depth'], st=setting['st'])
        parser.parse(log_file)
        logClusters = parser.logCluL
        print('原Drain,聚类数: ', len(logClusters), end=' ')
        F1_measure, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        origin_bechmark_result.append([dataset, F1_measure, accuracy])
        df = pd.read_csv(os.path.join(indir,log_file + '_structured.csv'))
        print('真实聚类数: ', len(df['EventId'].drop_duplicates()))

    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(origin_bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result = pd.DataFrame(v0_bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result = pd.DataFrame(v1_bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result = pd.DataFrame(v2_bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.T.to_csv('Drain_bechmark_result.csv')

examine_optimize()