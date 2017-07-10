require 'torch'
require 'misc.DataLoaderParsing'
require 'hdf5'
require 'misc.tree'
local opt = {}
opt.input_h5 = './data/data.h5'
opt.input_feat_h5 = './data/data_feat.h5'
opt.input_json = './data/data.json'

opt.shuffleData = 1
opt.folder = '../deep-residual-networks/ResNet_charades_feature'
opt.att_folder = '../data/c3d_att_feat'
opt.vid_file = './data/train_parsing_charades.txt'
--opt.train_num = 1200 
--opt.val_num = 770
opt.frame_feat_dim = 2048 
opt.att_feat_dim = 2048 
opt.att_seq_size = 140 
opt.input_encoding_size = 500
opt.t_len = 130 
--[[
h5_file = hdf5.open(opt.input_h5, 'r')

label = h5_file:read('/labels'):partial({27252, 27252}, {1,20})
trans = h5_file:read('/trans'):partial({27252, 27252}, {1,45})

print(label)
print(trans)
--]]
-- initialize the dataloader
local loader = DataLoaderParsing(opt)

local data = loader:getBatch{batch_size = 1, split = 'train'}

local label = data.labels
local trans = data.trans
print(label)
print(trans)


-- print sents
--local vocab = loader:getVocab()
--
--local tree = data.trees[1]
--
--print(torch.type(tree))
--print(tree:F())
