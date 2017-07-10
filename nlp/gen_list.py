import numpy as np
import sys
import os

fid = open('../data/train_charades.txt', 'r')
lines = fid.readlines()
fid.close()
res = {}
grp = {}
for line in lines:
    vid = line.split()[0]
    res[vid] = line.split()[1]
    grp[vid] = line.split()[2]

fid = open('./sent.txt', 'r')
lines = fid.readlines()
fid.close()

# list for others
out_fid = open('../data/train_parsing_charades.txt', 'w')
dic = {}
for line in lines:
    vid = line.split('\t')[0]
    if not vid in dic and vid in res:
        dic[vid] = 1
        out_fid.write(vid + ' ' + res[vid] + ' ' + grp[vid] + '\n')
