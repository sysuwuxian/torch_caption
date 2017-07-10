require 'hdf5'
local utils = require '../misc/utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  
  -- load the json file which contains additional information about the dataset

  print('DataLoader loading json file: ', opt.input_json)
  self.info = utils.read_json(opt.input_json)
  self.ix_to_word = self.info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)
    
  -- open the hdf5 file
  
  print('DataLoader loading h5 file: ', opt.input_h5)
  
  self.h5_file = hdf5.open(opt.input_h5, 'r')


  -- load in the sequence data
  local seq_size = self.h5_file:read('/feats'):dataspaceSize()
  -- determine the dim and seq length 
  self.feat_dim = seq_size[2]
  self.seq_length = seq_size[1]
  
  print('total samples in data is ' .. self.seq_length)
 

  --separate out indexs for each of the provided splits
  self.split_ix = {}
  self.iterators = {}

  -- enumarate the dataset, determine the next data
  for i = 1, self.seq_length do

    if i <= opt.train_num then
      split = 'train'
    else
      split = 'val' 
    end
    if not self.split_ix[split] then
      self.split_ix[split] = {}
      self.iterators[split] = 1 
    end
    table.insert(self.split_ix[split], i)
    i = i + 1
  end

  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end

  self.index = torch.range(1, #self.split_ix['train'])
  if opt.shuffleData then
    self:shuffleData()
  end

end

function DataLoader:shuffleData()
  self.index = torch.randperm(#self.split_ix['train'])
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocab()
  return self.ix_to_word
end


local function split(s, seq, sz)
  local parts, off = {}, 1
  local first, last = string.find(s, seq, off, true)
  while first do
    table.insert(parts, string.sub(s, off, first - 1))
    off = last + 1
    first, last = string.find(s, seq, off, true)
  end
  table.insert(parts, string.sub(s, off))
  assert(#parts == sz)
  return parts
end



--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,3,H,W) containing the images
  - y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
  
  local split = utils.getopt(opt, 'split')
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)

  local split_ix = self.split_ix[split]

  -- pick an index of the datapoint to load next
  local feat_batch = torch.LongTensor(batch_size, self.feat_dim)
  
  local label_batch = torch.LongTensor(batch_size)
  
  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  for i = 1, batch_size do

    local ri = self.iterators[split]
    local ri_next = ri + 1 
    if ri_next > max_index then 
       ri_next = 1; 
       wrapped = true
       if split == 'train' then
         self:shuffleData()
       end
    end -- wrap around
    
    self.iterators[split] = ri_next

    if split == 'train' then
      ix = split_ix[self.index[ri]]
    else
      ix = split_ix[ri]
    end


    -- fetch the feat and label
    feat = self.h5_file:read('/feats'):partial({ix, ix}, {1, self.feat_dim})
    label = self.h5_file:read('/labels'):partial({ix, ix})

    feat_batch[{{i,i}}] = feat
    label_batch[{{i,i}}] = label
  end

  local data = {}
  data.feats = feat_batch
  data.labels = label_batch 
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  return data
end

