from utils import *
from model import Node,Drain,buildSampleDrain,LogCluster
# 方法1： 合并子树节点

class Optimizer:

    def modify(self,method='merge_sub_tree',drain=None,tree=None,st = 0.8):

        self.st = st
        self.method = method
        if method == 'merge_sub_tree':
            self.depth = drain.depth
            return self.modify_by_merge_subtree(tree)
        elif method == 'seq_dist' or method == 'tfidf':
            self.modify_by_merge_leaf(drain)

    '''
    用递归的方式合并子树
    '''
    def mergeSubTree(self, nodeList):
        # print([node.token for node in nodeList])
        res = nodeList[0]
        if len(nodeList) == 1 or len(res.children) == 0:
            return res
        if isinstance(res.children,list):
            for node in nodeList[1:]:
                res.children.extend(node.children)
            return res
        nodeDict = {}
        for node in nodeList:
            # if :
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
        res = []
        ans = LogCluster()
        prefix = nodeList[0].logTemplate[:self.depth]
        for node in nodeList[1:]:
            prefix = getTemplate(prefix, node.logTemplate[:self.depth])
        for node in nodeList[0:]:
            node.logTemplate = prefix+node.logTemplate[self.depth:]
        groups = self.groupCluster(nodeList)
        for group in groups:
            cluster = group[0]
            for cluster1 in group[1:]:
                cluster.logIDL.extend(cluster1.logIDL)
            res.append(cluster)
        return res

    '''
    合并相似的节点
    '''
    def groupNodes(self,nodeList):
        res = []
        isWordGrouped = {node.token: False for node in nodeList}
        # # print(isWordGrouped)
        if isinstance(nodeList[0].children,list):
            for node in nodeList:
                res.append([node])
            return res
        for i in range(len(nodeList)):
            node = nodeList[i]
            if isWordGrouped[node.token]:
                continue
            newGroup = [node]
            isWordGrouped[node.token] = True
            for node1 in nodeList[i + 1:]:
                newMember = []
                for member in newGroup:
                    # print(member.token, node1.token)
                    if compareSimilarity(member, node1) > 0.75:
                        newMember.append(node1)
                        isWordGrouped[node1.token] = True
                        break
                newGroup.extend(newMember)
            res.append(newGroup)
        return res

    def modify_by_merge_subtree(self,node):
        childs = node.children
        if len(childs) == 0:
            return node
        if isinstance(childs,list):
            node.children = self.combineLeaves(childs)
            return node
        nodeGroups = self.groupNodes(list(childs.values()))
        node.children.clear()
        for group in nodeGroups:
            if len(group) > 1:
                node.children['*: ' + ';'.join([nod.token for nod in group])] = self.mergeSubTree(group)
            else:
                node.children[group[0].token] = group[0]
        for child in node.children.keys():
            node.children[child] = self.modify_by_merge_subtree(node.children[child])
        # # print(node.children)
        return node


    '''
    合并相似的类
    '''
    def mergeCluster(self,clusterList):
        res = clusterList[0]
        for cluster in clusterList[1:]:

            parentNodes = cluster.parentNode
            for parentNode in parentNodes:
                parentNode.children.remove(cluster)
                res.parentNode.append(parentNode)
                parentNode.children.append(res)

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
                    if sim > self.st:
                        logTemplate = getTemplate(logTemplate,cluster1.logTemplate)
                        newMember.append(cluster1)
                        isWordGrouped[j] = True
                        break
                newGroup.extend(newMember)
            cluster.logTemplate = logTemplate
            res.append(newGroup)
        return res

    def modify_by_merge_leaf(self,drain):
        clusterList = drain.logClusters
        clustersOfDiffGroup = self.groupCluster(clusterList)
        for clusters in clustersOfDiffGroup:
            for cluster in clusters[1:]:
                clusterList.remove(cluster)
            self.mergeCluster(clusters)

