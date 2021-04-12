# 日志解析器Drain的优化层

from drain.utils import *
import pandas as pd

class Optimizer:

    '''
    优化层入口
    resultFile: 在前面日志解析过程中，对日志进行了初步分类
    method: 采用的优化方法，主要有两种，分别是合并子树和合并叶节点的聚类
    logparser: 待优化的日志解析器
    st: 相似度阈值，用于判定是否合并
    '''
    def modify(self, resultFile = '.\\data\\log_item_to_label.csv', method='merge_sub_tree', logparser=None, st = 0.8,nst=0.8):
        self.resultFile = resultFile
        self.st = st
        self.nst = nst
        self.method = method
        self.logparser = logparser
        self.depth = logparser.depth
        if method == 'merge_sub_tree':
            self.modify_by_merge_subtree(self.logparser.prefixTree)
        elif method == 'seq_dist' or method == 'tfidf' or method =='LCS':
            self.modify_by_merge_leaf()

    '''
    用递归的方式合并子树
    nodeList:待合并的子树列表
    '''
    def mergeSubTree(self, nodeList):

        res = nodeList[0]
        # 若列表里只有一个成员或已经到了叶节点，那么不用执行合并
        if len(nodeList) == 1:
            return res
        for node in nodeList[1:]:
            res.logClusters.extend(node.logClusters)
        nodeDict = {}
        for node in nodeList:
            for token in node.children.keys():
                child = node.children[token]
                if child.token not in nodeDict:
                    nodeDict[child.token] = []
                nodeDict[child.token].append(child)
        # print(nodeDict.keys())
        for token in nodeDict.keys():
            res.children[token] = self.mergeSubTree(nodeDict[token])
        return res

    def combineLeaves(self, nodeList):

        depth = nodeList[0].parentNode[0].depth
        prefix = nodeList[0].logTemplate[:depth]
        for node in nodeList[1:]:
            prefix = getTemplate(prefix, node.logTemplate[:depth])
        for node in nodeList[0:]:
            node.logTemplate = prefix+node.logTemplate[depth:]
        groups = self.groupCluster(nodeList)
        item_to_class = pd.read_csv(self.resultFile, engine='c', na_filter=False, memory_map=True)
        for group in groups:

            self.mergeCluster(group,item_to_class)
        item_to_class.to_csv(self.resultFile,index=False)

    '''
    合并相似的节点
    '''
    def groupNodes(self,nodeList):
        res = []
        isWordGrouped = {node.token: False for node in nodeList}
        # # print(isWordGrouped)

        for i in range(len(nodeList)):
            node = nodeList[i]
            if len(node.children)==0:#isinstance(node.children, list):
                res.append([node])
                continue
            if isWordGrouped[node.token]:
                continue
            newGroup = [node]
            isWordGrouped[node.token] = True
            for node1 in nodeList[i + 1:]:
                newMember = []
                for member in newGroup:
                    # print(member.token, node1.token)
                    commonNodes = list(set(member.children.keys()).intersection(node1.children.keys()))
                    if 2*len(commonNodes)/(len(member.children)+len(node1.children))>self.nst:# and compareSimilarity(member, node1) > 0:# not isinstance(node1.children, list) and compareSimilarity(member, node1) > 0.75:
                        newMember.append(node1)
                        isWordGrouped[node1.token] = True
                        break
                newGroup.extend(newMember)
            res.append(newGroup)
        return res

    # 合并子树的方式进行结构调整
    def modify_by_merge_subtree(self,node):

        childs = node.children
        logClusters = node.logClusters

        # 如果到达叶节点，对因分支合并叶节点下的类别进行合并
        if len(logClusters)>1:
            # 合并叶子节点
            self.combineLeaves(logClusters)

        if len(childs) == 0:
            return node
        # 对子节点进行聚类操作，将相似的节点
        nodeGroups = self.groupNodes(list(childs.values()))
        # 重新构建当前节点下的子节点
        node.children.clear()

        # 对聚类后的各节点
        for group in nodeGroups:
            if len(group) > 1:
                node.children['*: ' + ';'.join([nod.token for nod in group])] = self.mergeSubTree(group)
            else:
                node.children[group[0].token] = group[0]
        for child in node.children.keys():
            node.children[child] = self.modify_by_merge_subtree(node.children[child])
        # print(node.children)
        return node

    '''
    合并相似的类
    '''
    def mergeCluster(self,clusterList,item_to_class):
        clusterList = list(set(clusterList))
        res = clusterList[0]
        drainClusters = self.logparser.logClusters

        for cluster in clusterList[1:]:
            if cluster in drainClusters:
                drainClusters.remove(cluster)
        for cluster in clusterList[1:]:
            item_to_class['EventId'] = item_to_class['EventId'].replace(cluster.eventId, res.eventId)
            parentNodes = cluster.parentNode
            for parentNode in parentNodes:
                if cluster in parentNode.logClusters:
                    parentNode.logClusters.remove(cluster)
                    res.parentNode.append(parentNode)
                    parentNode.logClusters.append(res)

            res.logIDL.extend(cluster.logIDL)

    def groupCluster(self,clusterList):
        res = []
        isWordGrouped = {i: False for i in range(len(clusterList))}
        for i in range(len(clusterList)):
            cluster = clusterList[i]
            logTemplate = cluster.logTemplate
            if isWordGrouped[i]:
                continue
            newGroup = [cluster]
            isWordGrouped[i] = True
            for j in range(i + 1,len(clusterList)):
                if isWordGrouped[j]:
                    continue
                cluster1= clusterList[j]
                newMember = []
                for member in newGroup:
                    # # print(member.logTemplate, cluster1.logTemplate)
                    if self.method=='seq_dist' or self.method=='merge_sub_tree':
                        sim,_ = SeqDist(member.logTemplate, cluster1.logTemplate)
                    elif self.method=='tfidf':
                        sim = calculate_tfidf_similarity([' '.join(member.logTemplate), ' '.join(cluster1.logTemplate)])
                    elif self.method=='LCS':
                        lcs = LCS(member.logTemplate, cluster1.logTemplate)
                        sim = 2 * len(LCS(member.logTemplate, cluster1.logTemplate)) / (len(member.logTemplate) + len(cluster1.logTemplate))
                    if sim > self.st:
                        if self.method=='LCS':
                            logTemplate = getLCSTemplate(lcs,logTemplate)
                        else:
                            logTemplate = getTemplate(logTemplate, cluster1.logTemplate)
                        newMember.append(cluster1)
                        isWordGrouped[j] = True
                        break
                newGroup.extend(newMember)
            cluster.logTemplate = logTemplate
            res.append(newGroup)
        return res

    def modify_by_merge_leaf(self):

        clusterList = self.logparser.logClusters
        clustersOfDiffGroup = self.groupCluster(clusterList)
        item_to_class = pd.read_csv(self.resultFile, engine='c', na_filter=False, memory_map=True)
        for clusters in clustersOfDiffGroup:
            self.mergeCluster(clusters,item_to_class)
        item_to_class.to_csv(self.resultFile,index=False)
