import os
import numpy as np
import pdb
import argparse
import h5py
import json
import pdb
from arcstandard import arcstandard
from tree import Tree

UNK_IDENTIFIER = '<en_unk>'

def build_vocab(params):

  fid = open(params['voc'], 'r')
  
  lines = fid.readlines()
  vocab = []
  vocab.append(UNK_IDENTIFIER)

  for line in lines:
      word = line.split()[0]
      if word == UNK_IDENTIFIER:
         continue
      else:
         vocab.append(word)
  return vocab

def modify(index, wtoi, sent):
    
    voc_sz = len(wtoi)
    # record all the situation
    # must notice that classify the two situations
    # INDEX == -1 : VOC_SZ + 1
    # INDEX == 0: VOC_SZ + 2

    if index == -1:
        return voc_sz + 1 
    elif index == 0:
        return voc_sz + 2
    else:
        return wtoi[sent[index-1]]


# total 15 elements
def getChenFeat(c, wtoi, sent):
    fWord = []

    # top 3 elements in stack
    for i in range(3)[::-1]:
        index = c.getStack(i)
        fWord.append(modify(index, wtoi, sent))

    #for i in range(3):
    #    index = c.getBuffer(i)
    #    fWord.append(modify(index, wtoi, sent))

    # left / right - left(left) / right(right)
    for i in range(2):
        k = c.getStack(i)
        index = c.getLeftChild(k, 1)

        fWord.append(modify(index, wtoi, sent))

        index = c.getRightChild(k, 1)
        fWord.append(modify(index, wtoi, sent))
        
        index = c.getLeftChild(k, 2)
        fWord.append(modify(index, wtoi, sent))

        index = c.getRightChild(k, 2)
        fWord.append(modify(index, wtoi, sent))

        index = c.getLeftChild(c.getLeftChild(k, 1), 1)
        fWord.append(modify(index, wtoi, sent))

        index = c.getRightChild(c.getRightChild(k, 1), 1)
        fWord.append(modify(index, wtoi, sent))

    assert(len(fWord) == 15)
    return fWord

def mapping_sent(voc, sent):
    max_length = 60
    ret = np.ones((1, max_length), dtype=np.int32)

    for i in range(len(sent)):
        word = sent[i]
        ret[0][i] = voc[word]
    return ret


def main(params):


    vocab = build_vocab(params)
    itow = {i+1:w for i,w in enumerate(vocab)} # inverse table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
    
    fid = open('./data/msvd.collx', 'r')
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

        if word == '.':
            continue
        tree.add(head, depType)
        sent.append(word)

    num = len(trees)

    system = arcstandard()

    concat_feat = []
    concat_label = []
    concat_index = []

    defined_length = 40

    valid_num = 0

    for i in range(num):
        tree = trees[i]
        sent = sents[i]
        
        if tree.isprojective():

            #labels = -1 * np.ones((1, defined_length), dtype=np.int32)
            indexs = np.ones((1), dtype=np.int32)

            indexs[0] = len(sent)

            i = 0
            valid_num += 1 
            c = system.initialConfig(tree.n)
            while not system.isterminal(c):

                oracle = system.getOracle(c, tree)
                v = 1 
                if oracle[0] == 'L':
                    v = 2 
                elif oracle[0] == 'R':
                    v = 3 
                #labels[0][i] = v
                #i += 1
                #if valid_num > 10000:
                feat = getChenFeat(c, wtoi, sent)
                #    print(feat)
                #    print('trans is ', v)
                #    pdb.set_trace()

                concat_feat.append(np.reshape(np.array(feat), (1, -1)))
                label = np.zeros((1), dtype=np.int32)
                label[0] = v
                concat_label.append(np.array(label))
                system.apply(c, oracle)

                #if valid_num > 10000:
                #concat_label.append(labels)
                #concat_feat.append(mapping_sent(wtoi, sent))
                #concat_feat.append(feat)
                #concat_index.append(indexs)

    Feat = np.concatenate(concat_feat, axis=0)
    
    print 'feature size is ', Feat.shape
        
    Label = np.concatenate(concat_label, axis=0)

    print 'label size is ', Label.shape

    #Index = np.concatenate(concat_index)

    #print 'Index size is ', Index.shape

    # save feat to hdf5
    f = h5py.File(params['output_feat_h5'], "w")
    f.create_dataset("feats", dtype='uint32', data=Feat)
    f.create_dataset("labels", dtype='uint32', data=Label)
    # f.create_dataset("indexs", dtype='uint32', data=Index)
    f.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json'] 

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--voc', default='../../data/vocabulary.txt', help='input dictionaty')
  parser.add_argument('--output_json', default='./data/msvd_data.json', help='out dictionaty')
  parser.add_argument('--output_feat_h5', default='./data/parse_msvd_feat.h5', help='parsing hdf5 file')
  # options
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
