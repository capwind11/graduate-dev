import re
import os
import time
import pickle
import gc
from tqdm import tqdm
from drain.utils import *

# 日志聚类
class LogCluster:
    def __init__(self, logTemplate='', logIDL=None, eventId=None):
        self.logTemplate = logTemplate
        self.eventId = eventId
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL
        self.parentNode = []

# 构建树结构
class Node:
    def __init__(self, children=None, depth=0, token=None):
        if children is None:
            children = dict()
        self.children = children
        self.depth = depth
        self.token = token

# Drain模型
class Drain:
    """
    树深度
    depth: depth of all leaf nodes
    正则表达式
    rex: regular expressions used in preprocessing (step1)
    相似阈值
    st: similarity threshold
    最大子节点数
    maxChild: max number of children of an internal node
    生成前缀树存储路径
    treeStoragePath: save path of prefix tree of Drain
    """

    def __init__(self,depth=4, st=0.6, maxChild=100, rex=None, removeCol=[] ,treeStoragePath=None):
        self.depth = depth
        self.st = st
        self.removeCol = removeCol
        self.maxChild = maxChild
        if rex is None:
            rex = []
        self.rex = rex
        if treeStoragePath != None:
            self.prefixTree = self.load(treeStoragePath)
        else:
            self.prefixTree = Node()
        self.logClusters = []

    '''
    重新构建解析树
    '''
    def reconstruct(self):
        self.prefixTree = Node()
        self.logClusters = []

    '''
    保存已经生成好的解析树和日志类别
    '''
    def save(self, treeStorageFile='prefixTree.pkl', savePath='./results/', saveFileName='template',
             saveTempFileName='logTemplates.txt'):

        if not os.path.exists(savePath):
            os.makedirs(savePath)
        f = open(savePath + treeStorageFile, 'wb')
        pickle.dump(self.prefixTree, f)

        logClusterSavePath = savePath + "logClusters/"
        if not os.path.exists(logClusterSavePath):
            os.makedirs(logClusterSavePath)
        else:
            deleteAllFiles(logClusterSavePath)

        writeTemplate = open(logClusterSavePath + saveTempFileName, 'w')
        idx = 1
        for logClust in self.logClusters:
            writeTemplate.write(' '.join(logClust.logTemplate) + '\n')
            writeID = open(logClusterSavePath + saveFileName + str(idx) + '.txt', 'w')
            for logID in logClust.logIDL:
                writeID.write(str(logID) + '\n')
            writeID.close()
            idx += 1

        writeTemplate.close()
        return None

    '''
    加载已经保存的解析树
    '''
    def load(self, treeStorageFile='prefixTree.pkl'):
        with open(treeStorageFile, 'rb') as treeStorage:
            prefixTree = pickle.load(treeStorage)
        return prefixTree

    '''
    根据输入日志构建解析树
    输入日志文件位置
    inputFile: the input path stores the input log file name
    输出日志标签位置
    outputFile: the output path stores the log labels
    '''
    def fit(self, inputFile='D:\\毕业设计\\loghub\\HDFS_1\\HDFS.log', outputFile='.\\data\\log_item_to_label.csv',
            isReconstruct=False):

        if isReconstruct:
            self.reconstruct()
            f = open(outputFile, 'w')
            # 为加快速度采用原生的文件写入方式
            f.write("LineId,BlockId,EventId\n")
        else:
            f = open(outputFile, 'a')

        t1 = time.time()

        with open(inputFile) as lines:
            count = 0
            event_num = 1
            for line in tqdm(lines, desc="load data"):

                logmessageL = line.strip().split()
                logID = count
                logmessageL = [word for i, word in enumerate(logmessageL) if i not in self.removeCol]
                cookedLine = ' '.join(logmessageL)

                blkId_list = list(set(re.findall(r'(blk_-?\d+)', cookedLine)))
                for currentRex in self.rex:
                    cookedLine = re.sub(currentRex, '', cookedLine)
                # cookedLine = re.sub(currentRex, 'core', cookedLine) #For BGL only
                # cookedLine = re.sub('node-[0-9]+','node-',cookedLine) #For HPC only

                logmessageL = cookedLine.split()
                matchCluster = self.treeSearch(logmessageL)
                # Match no existing log cluster
                if matchCluster is None:
                    matchCluster = LogCluster(logTemplate=logmessageL, logIDL=[[logID, blkId_list]], eventId=event_num)
                    event_num += 1
                    self.logClusters.append(matchCluster)
                    self.insert(matchCluster)

                # Add the new log message to the existing cluster
                else:
                    # 用迭代的方法找到最佳的模板
                    newTemplate = getTemplate(logmessageL, matchCluster.logTemplate)
                    matchCluster.logIDL.append([logID, blkId_list])

                    if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                        matchCluster.logTemplate = newTemplate
                f.write(str(logID) + "," + " ".join(blkId_list) + "," + str(matchCluster.eventId) + '\n')
                count += 1
                if count > 2000000:
                    break
        f.close()
        t2 = time.time()
        print('build the prefix tree process takes', t2 - t1)
        print('the number of log used to build the parser is ', count)
        print('*********************************************')
        gc.collect()
        return t2 - t1

    '''
    用已生成解析树将指定位置日志解析并保存解析结果
    输入日志文件位置
    inputFile: the input path stores the input log file name
    输出日志标签位置
    outputFile: the output path stores the log labels
    '''
    def transform(self, logFile='D:\\毕业设计\\loghub\\HDFS_1\\HDFS.log', outputFile='.\\log_item_to_label.csv'):

        t1 = time.time()

        f = open(outputFile, 'a')
        f.write("LineId,BlockId,EventId\n")
        with open(logFile) as lines:
            count = 0
            for line in tqdm(lines, desc="load data"):

                logmessageL = line.strip().split()
                logID = count
                logmessageL = logmessageL[3:]
                cookedLine = ' '.join(logmessageL)

                blkId_list = re.findall(r'(blk_-?\d+)', cookedLine)
                for currentRex in self.rex:
                    cookedLine = re.sub(currentRex, '', cookedLine)
                # cookedLine = re.sub(currentRex, 'core', cookedLine) #For BGL only

                # cookedLine = re.sub('node-[0-9]+','node-',cookedLine) #For HPC only
                logmessageL = cookedLine.split()
                matchCluster = self.treeSearch(logmessageL)
                # Match no existing log cluster
                eventId = -1
                if matchCluster is not None:
                    # 用迭代的方法找到最佳的模板
                    eventId = matchCluster.eventId
                f.write(str(logID) + "," + " ".join(blkId_list) + "," + str(eventId) + '\n')
                count += 1
                if count > 5000:
                    break
        f.close()
        t2 = time.time()
        print('transform process takes', t2 - t1)
        print('the number of log been parsed is ', count)
        print('*********************************************')
        gc.collect()
        return t2 - t1

    '''
    在树上搜索
    '''
    def treeSearch(self, seq):
        # 返回聚类
        retLogClust = None

        # 序列长度
        seqLen = len(seq)
        parentn = self.prefixTree
        currentDepth = 0
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.children:
                parentn = parentn.children[token]
            elif '*' in parentn.children:
                parentn = parentn.children['*']
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.children

        retLogClust = self.FastMatch(logClustL, seq)

        return retLogClust

    '''
    往解析树上添加新类别的日志
    关键是维护树根
    '''
    def insert(self, logClust):
        seqLen = len(logClust.logTemplate)
        # if seqLen not in rn.children:
        # 	firtLayerNode = Node(depth=1, token=seqLen)
        # 	rn.children[seqLen] = firtLayerNode
        # else:
        # 	firtLayerNode = rn.children[seqLen]
        firtLayerNode =self.prefixTree
        parentn = firtLayerNode

        currentDepth = 0
        for token in logClust.logTemplate:

            # Add current log cluster to the leaf node
            # 到尽头了
            if currentDepth >= self.depth or currentDepth > seqLen:
                logClust.parentNode.append(parentn)
                if len(parentn.children) == 0:
                    parentn.children = [logClust]
                else:
                    parentn.children.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.children:
                if not hasNumbers(token):
                    if '*' in parentn.children:
                        if len(parentn.children) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, token=token)
                            parentn.children[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.children['*']
                    else:
                        if len(parentn.children) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, token=token)
                            parentn.children[token] = newNode
                            parentn = newNode
                        elif len(parentn.children) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, token='*')
                            parentn.children['*'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.children['*']

                else:
                    if '*' not in parentn.children:
                        newNode = Node(depth=currentDepth + 1, token='*')
                        parentn.children['*'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.children['*']

            # If the token is matched
            else:
                parentn = parentn.children[token]

            currentDepth += 1
        if self.depth >= seqLen:
            logClust.parentNode.append(parentn)
            if len(parentn.children) == 0:
                parentn.children = [logClust]
            else:
                parentn.children.append(logClust)

    '''
    快速匹配，直接基于相似度来匹配
    '''

    def FastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = SeqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        # 相似度已经大于阈值
        if maxSim >= self.st:
            retLogClust = maxClust
        else:
            retLogClust = None

        return retLogClust

    def copy(self,root=None):
        if root==None:
            root = self.prefixTree
        copyTree = Node(depth=root.depth, token=root.token)
        if isinstance(root.children, list):
            copyTree.children = root.children.copy()
            return copyTree
        for child in root.children.values():
            copyTree.children[child.token] = self.copy(child)
        return copyTree

def buildSampleDrain(fileName = None):
    # HDFS parameters for example
    rex = ['blk_(|-)[0-9]+']
    myParser = Drain(rex=rex)
    if fileName!=None:
        myParser.fit(fileName,isReconstruct=True)
    else:
        myParser.fit()
    # myParser = Drain(treeStoragePath='./results/prefixTree.pkl', rex=rex)
    # myParser.transform(outputFile='../test_evalue.csv')
    return myParser