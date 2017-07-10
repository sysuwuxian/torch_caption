import os
import numpy as np
import h5py
import pdb
from arcstandard import arcstandard
from tree import Tree

fid = open('./data/test_msvd.collx', 'r')
lines = fid.readlines()
fid.close()


trees = []
tree = Tree()
sents = []
sent = []
for line in lines:
    if line == '\n': 
        trees.append(tree)
        tree = Tree()
        
        sents.append(sent)
        sent = []
        continue
    tokens = line.split('\t')
    word = tokens[1]
    depType = tokens[7]
    head = int(tokens[6])
    #if word == '.': continue
    tree.add(head, depType)
    sent.append(word)
num = len(trees)
system = arcstandard()

define_length = 40
ret = -1 * np.ones((num, define_length), dtype=np.int32)


for i in range(num):
   tree = trees[i]
   sent = sents[i]

   if tree.isprojective():
       c = system.initialConfig(tree.n)
       trans_num = 0 
       while not system.isterminal(c):
           oracle = system.getOracle(c, tree)
           # shift operation
           v = 0 
           if oracle[0] == 'L': 
              v = 1 
           elif oracle[0] == 'R': 
              v = 2
           ret[i][trans_num] = v
           trans_num += 1
           system.apply(c, oracle)
ret += 1
print(ret)
