
#Output: a json file and an hdf5 file
#The hdf5 file contains several fields:
#/images is (N,3,256,256) uint8 array of raw image data in RGB format
#/labels is (M,max_length) uint32 array of encoded labels, zero padded
#/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
#  first and last indices (in range 1..M) of labels for each image
#/label_length stores the length of the sequence for each of the M sequences
#
#The json file has a dict that contains:
#- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
#- an 'images' field that is a list holding auxiliary information for each image, 
#  such as in particular the 'split' it was assigned to.

import os
import json
import argparse
import pdb
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
UNK_IDENTIFIER = '<en_unk>'


def float_line_to_stream(line):
    return map(float, line.split(','))


def get_split(params):
    split = open(params['split_list'], 'r')
    lines = split.readlines()
    split = []
    for line in lines:
        vid = line.strip()
        split.append(vid)
    return split

def get_list(params):
   input_file = open(params['input_list'], 'r')
   lines = input_file.readlines()
   vid_frame = {}
   att_frame = {}
   file_list = []
   i = 0

   max_att_num = 0

   cnt = 0
   for line in lines:
       vid_name = line.split()[0]
       vid_num = line.split()[1]
       att_num = line.split()[2]
       max_att_num = max(max_att_num, int(att_num))

       vid_frame[vid_name] = vid_num
       att_frame[vid_name] = str(int(att_num) + 1)
       file_list.append(vid_name)
   
   print('max num is ', max_att_num)

   return file_list, vid_frame, att_frame 
       
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

def encode_captions(file_list,  params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(file_list)

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  caption_counter = 0
  counter = 1

  lines = {}

  # get sents
  with open(params['sentence'], 'r') as sentfd:

      for line in sentfd:
          line = line.strip()
          id_sent = line.split('\t')

          vid_name = id_sent[0]
          if not vid_name in lines:
             lines[vid_name] = []
          lines[vid_name].append(id_sent[1])
  
  for i, vid in enumerate(file_list):
      n = len(lines[vid])
      assert n > 0, 'error: some image has no caption'
      Li = np.zeros((n, max_length), dtype = 'uint32')

      for j, s in enumerate(lines[vid]):
        k = 0
        for w in s.split():
            w = w.strip()
            if len(w) == 0: continue
            Li[j, k] = wtoi[w]
            k += 1
      
      label_arrays.append(Li)
      label_start_ix[i] = counter
      label_end_ix[i] = counter + n - 1 

      counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together

  print 'count sz is ', counter
  print 'encoded captions to array of size ', `L.shape`
  return L, label_start_ix, label_end_ix 

def count(array, num, flag):
    cnt = 0
    for i in range(array.shape[0]):
        if flag and array[i] == num:
            cnt += 1
        elif not flag and array[i] != num:
            cnt += 1
    return cnt

def encode_groupfeats(file_list, att_frame, params):
 
  N = len(file_list)
  feat_dim = params['feat_dim']
  feat_start_ix = np.zeros(N, dtype='uint32')
  feat_end_ix = np.zeros(N, dtype='uint32')
  cnt = 1
  feat_arrays = [] 

  # get group features
  for i, vid in enumerate(file_list):
      n = int(att_frame[vid]) 
      file_name = params['group_dir'] + '/' + vid + '.txt'

      Fi = np.zeros((n, feat_dim), dtype = np.float32) 
      fid = open(file_name, 'r')
      lines = fid.readlines()

      idx = 0
      for line in lines:
          Fi[idx, :] = np.array(float_line_to_stream(line))
          idx += 1
      fid.close()

      feat_arrays.append(Fi)
      feat_start_ix[i] = cnt
      feat_end_ix[i] = cnt + n - 1
      cnt += n
  F = np.concatenate(feat_arrays, axis=0) #put all the feature together 
  return F, feat_start_ix, feat_end_ix

def main(params):

  file_list, vid_frame, att_frame = get_list(params)

  # start from the 1
  id2vid = {i+1:vid for i,vid in enumerate(file_list)}

  # split_list = get_split(params)
  # split2id = {vid:i+1 for i, vid in enumerate(split_list)}
  
  
  
  
  # tokenization and preprocessing

  # create the vocab
  vocab = build_vocab(params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

  
  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix = encode_captions(file_list, params, wtoi)


  # encode group feat
  F, feat_start_ix, feat_end_ix = encode_groupfeats(file_list, att_frame, params) 


  # create output h5 file
 
  f = h5py.File(params['output_h5'], "w")
  f.create_dataset("labels", dtype='uint32', data=L)
  f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  # read hdf5 file 
  file = h5py.File(params['input_hdf5'], 'r')
  T = file['state'][:] 
  file.close()
  print T.shape
  f.create_dataset("trans", dtype=np.int32, data=T)

  assert(T.shape[0] == L.shape[0])
  # assert sum 0 is equal to sum 1
  # for i in range(T.shape[0]):
  #   assert(count(T[i], 0, True) == count(L[i], 0, False))
  
  f.close()

  f = h5py.File(params['output_feat_h5'], "w")
  f.create_dataset("group_feats", dtype=np.float32, data=F)
  f.create_dataset("group_start_ix", dtype='uint32', data=feat_start_ix)
  f.create_dataset("group_end_ix", dtype='uint32', data=feat_end_ix)
  f.close()

  print 'wrote ', params['output_h5']
  print 'wrote ', params['output_feat_h5']
  # create outp1ut json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['ind_to_vid'] = id2vid 
  out['vid_to_num'] = vid_frame
  out['att_to_num'] = att_frame
  # out['split_to_id'] = split2id

  json.dump(out, open(params['output_json'], 'w'))
  print 'wrote ', params['output_json']

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input list
  parser.add_argument('--input_list', default='./data/train_parsing.txt', help='input file list')
  parser.add_argument('--split_list', default='../charades/split/split_train.txt', help='split file list')

  # add for parsing
  parser.add_argument('--input_hdf5', default='./nlp/output_msvd.h5', help='input_hdf5_file')
  parser.add_argument('--output_json', default='./data/msvd_parsing_data/data.json', help='output json file')
  parser.add_argument('--output_h5', default='./data/msvd_parsing_data/data.h5', help='output h5 file')
  parser.add_argument('--output_feat_h5', default='./data/msvd_parsing_data/data_feat.h5', help='output h5 file')
  parser.add_argument('--group_dir', default='../data/res_group_feat', help='group dir')
  parser.add_argument('--voc', default='../data/vocabulary.txt', help='input dictionaty')
  parser.add_argument('--feat_dim', default=2048, type=int, help='group feat') 
  parser.add_argument('--sentence', default = '../data/sent_msvd.txt', help='input sentence')
  # options
  parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)
