import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# seq1 is template
def SeqDist(seq1, seq2):
    if not isinstance(seq1,list):
        seq1 = seq1.split()
    if not isinstance(seq2, list):
        seq2 = seq2.split()
    # assert len(seq1) == len(seq2)
    simTokens = 0
    numOfPar = 0

    for token1, token2 in zip(seq1, seq2):
        if token1 == '*':
            numOfPar += 1
            continue
        if token1 == token2:
            simTokens += 1

    retVal = 2*float(simTokens) / (len(seq1)+len(seq2))
    return retVal, numOfPar


def deleteAllFiles(dirPath):
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        path_name = os.path.join(dirPath, fileName)
        if os.path.isfile(path_name):
            os.remove(path_name)
        else:
            deleteAllFiles(path_name)

'''
拿到模板，可以用于InterLog的方法
'''
def getTemplate(seq1, seq2):
    # assert len(seq1) == len(seq2)
    retVal = []
    if not isinstance(seq1,list):
        seq1 = seq1.split()
    if not isinstance(seq2, list):
        seq2 = seq2.split()
    i = 0
    for word in seq1:
        if i >= len(seq2):
            break
        if word == seq2[i]:
            retVal.append(word)
        else:
            retVal.append('<*>')

        i += 1

    return retVal


'''
包含数字
'''
def hasNumbers(s):
    return any(char.isdigit() for char in s)
    # return False


def compareSimilarity(node1,node2):
    childrenOfNode1 = node1.children.keys()
    childrenOfNode2 = node2.children.keys()
    # print(childrenOfNode1,childrenOfNode2)
    retVal, numOfPar = SeqDist(list(childrenOfNode1),list(childrenOfNode2))
    # print(retVal)
    return retVal

def calculate_tfidf(corpus):
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return weight

def calculate_cosine_similarity(v1,v2):

    v1 = v1.reshape(1,-1)
    v2 = v2.reshape(1, -1)

    return cosine_similarity(v1,v2)[0][0]

def calculate_tfidf_similarity(corpus):

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return calculate_cosine_similarity(weight[0:1],weight[1:2])

def num2word(numList):

    return ['a'+num for num in numList]
