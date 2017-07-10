import os
import numpy as np
import h5py
import pdb
from arcstandard import arcstandard
from tree import Tree


fid = open('./data/charades.collx', 'r')
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
# change for add sentence
# add sentence
out_sent = open('./sent.txt', 'w')


num = len(trees)
system = arcstandard()


cnt = 0
define_length = 130 
max_num = 0
ret = -1 * np.ones((15785, define_length), dtype=np.int32)

# get dict for filter
dict_has = {}
fid = open('../data/train_parsing_charades.txt', 'r')
lines = fid.readlines()
fid.close()
for line in lines:
    line = line.strip()
    vid = line.split(' ')[0]
    dict_has[vid] = 1

# get parsing order
fid = open('../../charades/charades_parsing_sent.txt', 'r')
lines = fid.readlines()
fid.close()


for i in range(num):
   tree = trees[i]
   sent = sents[i]
   # we need to manage the order 
   # write to hdf5 file
   vid_name = lines[i].split('\t')[0]
   if i % 1000 == 0:
     print 'processd ' + str(i) + '/' + str(num) + '\n'
   if not vid_name in dict_has:
       continue
   #tree.printTree(sent)
   if tree.isprojective():
       txt = ' '.join(sents[i])
       out_sent.write(vid_name + '\t' + txt[0:-1] + '\n')
       
       #cnt += 1
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
           
           ret[cnt][trans_num] = v
           trans_num += 1 
           system.apply(c, oracle)
       cnt += 1
       max_num = max(max_num, trans_num)
print 'max_num is ', max_num
# write into hdf5 file
f = h5py.File('output.h5', 'w')
f.create_dataset("state", dtype=np.int32, data=ret)
f.close()
print 'sentence num is ', num
print 'cnt num is ', cnt
print 'ratio is ', 1.0 * cnt / num
