import collections
import textwrap
from matplotlib import pyplot as plt
from drain.model import Node,Drain
from drain.utils import *

decisionNode = dict(boxstyle="round4", fc="1.0")  # 设置结点格式
leafNode = dict(boxstyle="square", fc="1.0")  # 设置叶结点格式
arrow_args = dict(arrowstyle="<-")  # 设置箭头格式

# 绘制节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 在图上添加辅助框
    nodeTxt = "\n".join(textwrap.wrap(nodeTxt, width=10, expand_tabs=True,
                                replace_whitespace=False,
                                break_long_words=True))
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontsize=7,wrap=True)

# 打印token
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    # 即在图上指定位置打印文字
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="left", rotation=0,wrap=True)


# 前缀树
class Trie:
    def __init__(self,root = None,maxDepth = 4):
        if root == None:
            self.root = Node()
        else:
            self.root = root
        self.maxDepth = maxDepth


def getNumLeafs(n):
    numLeafs = 0
    if isinstance(n.children, list):
        newNode = Node()
        template = '\n'.join([' '.join(child.logTemplate) for child in  n.children])
        template = "\n".join(textwrap.wrap(template, width=15, expand_tabs=True,
                                    replace_whitespace=False,
                                    break_long_words=True))
        # for node in n.children[1:]:
        #     template = getTemplate(template,node.logTemplate)
        # token = ' '.join(template)
        newNode.token = template
        n.children = {template:newNode}
    for c in n.children:
        if isinstance(c,list):
            numLeafs += 1
            continue
        current = n.children.get(c)
        if current.children != {}:
            numLeafs += getNumLeafs(current)
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(n):
    maxDepth = 0
    for c in n.children:
        current = n.children.get(c)
        if current.children != {}:
            thisDepth = 1 + getTreeDepth(current)
        # include root node
        else:
            thisDepth = 2
        if thisDepth > maxDepth:    maxDepth = thisDepth
    return maxDepth


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(nodeTxt, cntrPt, parentPt, decisionNode)
    txtDepth = len(nodeTxt.split("\n"))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff- 0.05* txtDepth/ plotTree.totalD)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for c in myTree.children:
        current = myTree.children.get(c)
        if current.children != {}:
            plotTree(current, cntrPt, c)
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(c, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, c)
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(drain):
    inTree = drain.copy()
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), 'Root')
    # plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()
