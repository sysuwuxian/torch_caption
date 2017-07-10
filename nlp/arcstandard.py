import numpy as np
import sys
import os

from config import Config

NONEXIST = -1
UNKNOWN = 'unknown'
class arcstandard(Config):

    def isterminal(self, c):
        return c.getStackSize() == 1 and c.getBufferSize() == 0

    def initialConfig(self, l):
        c = Config()

        # initial the tree and root 
        for i in range(1, l + 1):
            c.tree.add(NONEXIST, UNKNOWN)
            c.buffer.append(i)
        # Put the ROOT node on the stack
        c.stack.append(0)
        return c


    def apply(self, c, oracle):
        w1 = c.getStack(1)
        w2 = c.getStack(0)
        if oracle[0] == 'S':
            c.shift()
        elif oracle[0] == 'R':
            c.addArc(w1, w2, oracle[2:-2])
            c.removeTopStack()
        else:
            c.addArc(w2, w1, oracle[2:-2])
            c.removeSecondTopStack()

    def canApply(self, c, t):
        nStack = c.getStackSize()
        nBuffer = c.getBufferSize()

        if t[0] == 'L':
            return nStack > 2
        elif t[0] == 'R':
            return nStack > 2 or (nStack == 2 and nBuffer == 0)
        else:
            return nBuffer > 0

    def getOracle(self, c, dtree):
        w1 = c.getStack(1)
        w2 = c.getStack(0)
        
        if w1 > 0 and dtree.getHead(w1) == w2:
            return 'L(' + dtree.getLabel(w1) + ')' 
        elif w1 >= 0 and dtree.getHead(w2) == w1 and not c.hasOtherChild(w2, dtree):
            return 'R(' + dtree.getLabel(w2) + ')' 
        else:
            return 'S' 
