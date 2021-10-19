# encoding:utf-8
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# 函数1：加载文件中的数据
# readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
# split() 通过指定分隔符对字符串进行切片,返回分割后的字符串列表。
# strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
# list() 方法用于将元组转换为列表。
# 注：元组与列表是非常类似的，区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。
# append() 方法用于在列表末尾添加新的对象。
def loadDataSet(fileName, splitChar=','):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar) #得到每行为字符串形式的列表
            fltline = list(map(float,curline)) #将每行的字符串列表映射为浮点型列表
            dataSet.append(fltline) #二维元素列表
    return dataSet

# 函数2：计算两个点之间的欧式距离
def distance(p1,p2):
    dis = math.sqrt(np.power((p1[0]-p2[0]),2)+np.power((p1[1]-p2[1]),2))
    return dis

# 函数3：DBSCAN算法，参数为处理好的数据集，指定半径参数，指定邻域密度阈值
def dbscan(dataset, eps, minpts):
    # 1.遍历数据，根据邻域，找到核心对象，分簇
    num = len(dataset)
    #未访问的点
    unvisited = [i for i in range(num)]
    #已访问的点
    visited = []
    #C为输出结果，默认是一个长度为num的值全为-1的列表
    C = [-1 for i in range(num)]
    #用k来标记不同的簇，k=-1表示噪音点
    k = -1
    #若还有未被访问的点
    while len(unvisited)>0:
        #随机选择其中一个对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的eps邻域中的对象的集合
        N = []
        for i in range(num):
            if(distance(dataset[i],dataset[p])<=eps):
                N.append(i)
        # 如果邻域中的对象个数大于等于指定阈值，说明p是一个核心对象
        if len(N) >= minpts:
            k = k+1
            C[p] = k
            # 对于该邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    #找到p邻域中的可以构成以pi为核心对象的领域中的对象
                    M=[]
                    for j in range(num):
                        if(distance(dataset[j],dataset[pi])<=eps):
                            M.append(j)
                    if len(M)>=minpts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                #若C[pi]值为初始值-1，说明它还未被分化为任何簇，那便可以划分为当前簇
                if C[pi] == -1:
                    C[pi] = k
        #若该对象无法构成以其为核心对象的簇，则判定为噪点
        else:
            C[p] = -1
    return C

# 用数据集测试
dataset = loadDataSet('788points.txt',splitChar=',')
C = dbscan(dataset,2,14)
#print(C)
x = []
y = []
for data in dataset:
    x.append(data[0])
    y.append(data[1])
plt.figure(figsize=(8,6),dpi=80)
plt.scatter(x,y,c=C,marker='o')
plt.show()