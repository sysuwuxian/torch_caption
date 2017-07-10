import os
import sys
import numpy as np

UNKNOWN = "unknown"
NONEXIST = -1

class Tree:
    
    # initilize the tree
    def __init__(self):
      
      self.n = 0
      self.head = []
      self.head.append(NONEXIST)
      self.label = []
      self.label.append(UNKNOWN)
      self.__counter = -1
    
    
    # add the next token to the parse
    #
    # param h Head of the next token
    # param l Dependency relation label between this node and its head
    
    def add(self, h, l):
        self.n += 1
        self.head.append(h)
        self.label.append(l)
    # Estalish a labeld dependency relation between two 
    # given nodes.

    # param k Index of the dependent node
    # param h Index of the head node
    # param l label of the dependency relation

    def set(self, k, h, l):
        self.head[k] = h
        self.label[k] = l

    def getHead(self, k):
        if k <= 0 or k > self.n: 
           return NONEXIST 
        else:
           return self.head[k]
   
    def getLabel(self, k):
        if k <= 0 or k > self.n:
          return UNKNOWN 
        else:
          return self.label[k]

    def getRoot(self):
        for i in range(1, self.n + 1):
            if self.getHead(i) == 0:
                return i
        return 0 

    # check if the tree is legal, O(n)
    def isTree(self):
        h = []
        h.append(-1)

        for i in range(1, self.n + 1):
            if self.getHead(i) < 0 or self.getHead(i) > self.n:
                return False
            h.append(-1)
        for i in range(1, self.n + 1):
            k = i
            while k > 0:
                if h[k] >= 0 and h[k] < i:
                   break
                if h[k] == i:
                   return False
                h[k] = i
                k = self.getHead(k)
        return True
    
    def isprojective(self):
        if not self.isTree():
            return False
        self.__counter = -1
        return self.visitTree(0)
    
    def visitTree(self, w):
        for i in range(1, w):
            if self.getHead(i) == w and self.visitTree(i) == False:
                return False
        self.__counter += 1

        if w != self.__counter:
            return False

        for i in range(w + 1, self.n + 1):
            if self.getHead(i) == w and self.visitTree(i) == False:
                return False
        return True
