#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
class plotTree():
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")
    ax1 = 0
    totalW = 0
    totalD = 0
    def __init__(self, myTree, parentPt=None, i=-1, parent=None):
        self.parent = parent
        if parentPt is None:
            self.createPlot(myTree)
        else:
            numLeafs, numlayers = myTree.TotalLeafandLayers()
            print('numLeafs:', numLeafs)
            print('numlayers:', numlayers)
            self.totalW = parent.totalW
            self.totalD = parent.totalD
            self.ax1 = parent.ax1
            if i != -1:
                leftleaves = myTree.FrontLeaves()
                self.xOff = leftleaves / self.totalW + numLeafs / self.totalW * (1 / 2)
                self.yOff = parent.yOff - 1.0 / self.totalD
            else:
                self.xOff = parentPt[0]
                self.yOff = parentPt[1]
            cntrPt = (self.xOff, self.yOff)
            if numLeafs != 1:
                self.plotNode(myTree.result_name, cntrPt, parentPt, self.decisionNode)
                if i >= 0:
                    if myTree.parents.continues is None:
                        self.plotMidText(cntrPt, parentPt, myTree.parents.feature[i])
                    elif myTree.parents.KMean:
                        self.plotMidText(cntrPt, parentPt, str(myTree.parents.feature[i]))
                    else:
                        self.plotMidText(cntrPt, parentPt, myTree.parents.feature[i] + str(myTree.parents.continues))
                    # self.plotNode(myTree.parents.feature[i], cntrPt, parentPt, self.decisionNode)
            if numLeafs == 1:
                self.plotNode(myTree.result, cntrPt, parentPt, self.leafNode)
                if myTree.parents.continues is None:
                    self.plotMidText(cntrPt, parentPt, myTree.parents.feature[i])
                elif myTree.parents.KMean:
                    self.plotMidText(cntrPt, parentPt, str(myTree.parents.feature[i]))
                else:
                    self.plotMidText(cntrPt, parentPt, myTree.parents.feature[i] + str(myTree.parents.continues))
                # self.plotNode(myTree.parents.feature[i], cntrPt, parentPt, self.decisionNode)
                #print(myTree.result)
                return
            for i in range(len(myTree.child)):
                plotTree(myTree.child[i], cntrPt, i, self)
    def createPlot(self):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        self.createPlot.ax1 = plt.subplot(111, frameon=False)
        self.plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), self.decisionNode)
        self.plotNode('叶结点', (0.8, 0.1), (0.3, 0.8), self.leafNode)
        plt.show()

    def plotNode(self, str, centerPt, parentPt, nodeType):
        self.ax1.annotate(str, xy=parentPt, \
                                xycoords='axes fraction', \
                                xytext=centerPt, textcoords='axes fraction', \
                                va="center", ha="center", bbox=nodeType, arrowprops=self.arrow_args
                                )

    def plotMidText(self, cntrPt, parentPt, str):
        xMid = cntrPt[0]/2 + parentPt[0]/2
        yMid = cntrPt[1]/2 + parentPt[1]/2
        self.ax1.text(xMid, yMid, str)

    def createPlot(self, inTree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)
        leaves, layers = inTree.TotalLeafandLayers()
        self.totalW = float(leaves)
        self.totalD = float(layers)
        self.xOff = 0.5
        self.yOff = 1.0
        plotTree(inTree, (0.5, 1.0), -1, self)
        plt.show()