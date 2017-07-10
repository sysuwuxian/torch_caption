import numpy as np
import io

fid = open('../../../data/sents_parsing.txt', 'r')

lines = fid.readlines()

out_file = open('../../../stanford-parser/data/parsing_sent.txt', 'w')


dict = {}

fid1 = open('../../../torch_caption/data/train_parsing.txt', 'r')
lines1 = fid1.readlines()

for line in lines1:
    vid = line.split()[0]
    dict[vid] = 1

for line in lines:
    line = line.strip()
    vid = line.split('\t')[0]
    if vid in dict:
       sent = line.split('\t')[1]
       print sent
       out_file.write(sent + '.' + '\n') 
