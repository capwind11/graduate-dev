from model import Node,Drain,buildSampleDrain
# from optimizer import Optimizer
from optimizer import Optimizer
from plotter import createPlot
from partition import Preprocess
import pandas as pd
import os

def printClusters(logClusters):
    for logCluster in logClusters:
        print("eventId: "+str(logCluster.eventId)+" template: "+' '.join(logCluster.logTemplate),end=' ')
        print("parentNode: " ,end=' ')
        for node in logCluster.parentNode:
            print(node.token,end=' ')
        print()

def optimize_by_seq_dist():
    logPath = os.path.join(os.path.abspath(''), 'sample.log')
    drain = buildSampleDrain(logPath)
    drain.save()
    logClusters = drain.logClusters
    printClusters(logClusters)
    opt = Optimizer()
    opt.modify(method = 'seq_dist',tree = drain.prefixTree,drain=drain,st = 0.7)
    logClusters = drain.logClusters
    printClusters(logClusters)
    root = drain.copy()
    createPlot(root)

def optimize_by_merge_sub_tree():
    logPath = os.path.join(os.path.abspath(''), 'sample.log')
    drain = buildSampleDrain(logPath)
    t = drain.copy()
    createPlot(t)
    t = drain.copy()
    opt = Optimizer()
    opt.modify(method='merge_sub_tree', tree=t,drain = drain)
    # opt.modify(t)
    createPlot(t)

def optimize_by_tfidf():
    logPath = os.path.join(os.path.abspath(''), 'sample.log')
    drain = buildSampleDrain(logPath)
    logClusters = drain.logClusters
    printClusters(logClusters)
    opt = Optimizer()
    opt.modify(method = 'tfidf',tree = drain.prefixTree,drain=drain,st = 0.6)
    logClusters = drain.logClusters
    print( )
    printClusters(logClusters)
    root = drain.copy()
    createPlot(root)

def parse_log_data():
    rex = ['blk_(|-)[0-9]+', '(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']
    removeCol = [0,1,2]
    myParser = Drain(rex=rex,removeCol=removeCol)
    # myParser.fit(inputFile='./sample.log',outputFile='test.csv')
    myParser.fit(isReconstruct=True)
    myParser.save()

def draw_tree():
    rex = ['blk_(|-)[0-9]+', '(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']
    removeCol = [0, 1, 2]
    myParser = Drain(rex=rex, depth=3, removeCol=removeCol)
    root = myParser.load('./results/prefixTree.pkl')
    createPlot(root)

if __name__ == "__main__":
    # draw_tree()
    # optimize_by_merge_sub_tree()
    # exit(0)
    # optimize_by_merge_sub_tree()
    # optimize_by_tfidf()
    # optimize_by_seq_dist()
    # myParser.evaluate(outputFile='./test_evalue.csv')
    # print(" ")
    rex = ['blk_(|-)[0-9]+', '(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']
    removeCol = [0,1,2]
    myParser = Drain(rex=rex,depth=3,removeCol=removeCol)
    # myParser.fit(inputFile='./sample.log',outputFile='test.csv')
    myParser.fit(isReconstruct=True)
    myParser.save()
    root = myParser.copy()
    createPlot(root)
    # preprocess = Preprocess('./results/','./data/', "log_item_to_label.csv", 'data_instances_hdfs.csv', 'anomaly_label.csv', 'normal.csv', 'abnormaly.csv')
    # blockId_to_logs = preprocess.partition_by_file()
    # data_df = pd.read_csv("./results/data_instances_hdfs.csv", index_col=0,engine='c', na_filter=False, memory_map=True)
    # preprocess.map_log_seq_to_label(data_df)