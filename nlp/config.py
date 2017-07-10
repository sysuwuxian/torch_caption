import os
import numpy as np
import sys
from tree import Tree

NONEXIST = -1


class Config:
    def __init__(self):
        self.stack = []
        self.buffer = []
        self.tree = Tree()

    def shift(self):
        k = self.getBuffer(0)
        if k == NONEXIST:
            return False
        self.buffer.remove(k)
        self.stack.append(k)
        return True
           
    def removeSecondTopStack(self):
        nStack = self.getStackSize()
        if nStack < 2:
            return False
        value = self.stack[nStack-2]
        self.stack.remove(value)
        return True

    def removeTopStack(self):
        nStack = self.getStackSize()
        if nStack < 1:
            return False
        value = self.stack[nStack-1]
        self.stack.remove(value)
        return True

    def getStackSize(self):
        return len(self.stack)

    def getBufferSize(self):
        return len(self.buffer)
    
    def getHead(self, k):
        return self.tree.getHead(k)

    def getLable(self, k):
        return self.tree.getLabel(k)


    def getLeftChild(self, k, cnt):
        
        if k < 0 or k > self.tree.n: 
           return NONEXIST
        
        c = 0
        for i in range(1, k):
            if self.tree.getHead(i) == k:
                c += 1
                if c == cnt:
                   return i
        return NONEXIST

    
    def getRightChild(self, k, cnt):
        if k < 0 or k > self.tree.n:
            return NONEXIST
        c = 0
        for i in range(k, self.tree.n + 1)[::-1]:
            if self.tree.getHead(i) == k:
                c += 1
                if c == cnt:
                    return i
        return NONEXIST

    def getStack(self, k):
        nStack = self.getStackSize()
        if k >= 0 and k < nStack :
            return self.stack[nStack - 1 - k]
        else:
            return NONEXIST
    
    def getBuffer(self, k):
        nBuffer = self.getBufferSize()
        if k >= 0 and k < nBuffer:
            return self.buffer[k]
        else:
            return NONEXIST
    
    def addArc(self, h, t, l):
        self.tree.set(t, h, l)
    
    def hasOtherChild(self, k, goldTree):
        for i in range(1, self.tree.n + 1):
            if goldTree.getHead(i) == k and self.tree.getHead(i) != k:
                return True
        return False
