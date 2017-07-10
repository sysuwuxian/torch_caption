require 'hdf5'
local utils = require 'misc.utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.input_json)
  self.info = utils.read_json(opt.input_json)
  self.ix_to_word = self.info.ix_to_word
 
  -- ind mapping to vid
  self.ind_to_vid = self.info.ind_to_vid
  self.vid_to_num = self.info.vid_to_num

  self.vocab_size = utils.count_keys(self.ix_to_word)
  print('vocab size is ' .. self.vocab_size)
 

  -- test non empty
  print('i2v size is ' .. utils.count_keys(self.ind_to_vid))
  print('v2n size is ' .. utils.count_keys(self.vid_to_num))


  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt.input_h5)
  self.h5_file = hdf5.open(opt.input_h5, 'r')
  -- load in the sequence data
  local seq_size = self.h5_file:read('/labels'):dataspaceSize()
  self.seq_length = seq_size[2]
  print('max sequence length in data is ' .. self.seq_length)
  -- load the pointers in full to RAM (should be small enough)
  self.label_start_ix = self.h5_file:read('/label_start_ix'):all()
  self.label_end_ix = self.h5_file:read('/label_end_ix'):all()
 

  --separate out indexs for each of the provided splits
  self.split_ix = {}
  self.iterators = {}
  local f = io.open(opt.vid_file)
  local i = 1 
  for line in f:lines() do
   
    vid = string.sub(line, 1, string.find(line,' ')-1)
    num = tonumber(string.sub(vid, 4, -1))
    if num <= opt.train_num then
      split = 'train'
    elseif num <= opt.train_num + opt.val_num then
      split = 'val'
    else
      split = 'test' 
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

  -- load train file list
  self.folder = opt.folder
  self.feat_dim = opt.frame_feat_dim
end

function DataLoader:shuffleData()
  self.index = torch.randperm(#self.split_ix['train'])
end


function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:getSeqLength()
  return self.seq_length
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


local function ReadData(path, sz)
  local file, err = io.open(path, 'r')
  if err then print("OOps"); return; end
  local lines = {}
  
  for line in file:lines() do
      table.insert(lines, line)
  end

  assert(#lines == 1)
  local Tensor = torch.FloatTensor(#lines * sz)
  for i = 1, #lines do
    local line = lines[i]
    elems = split(line, ',', sz)
    for j = 1, #elems do
      Tensor[(i-1)*sz+j] = tonumber(elems[j])
    end
  end
  return Tensor
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
  local img_batch_raw = torch.FloatTensor(batch_size, 3, self.feat_dim)
  local label_batch = torch.LongTensor(batch_size, self.seq_length)
  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  for i=1,batch_size do

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
    
    --print table
    vid_name = self.ind_to_vid[tostring(ix)]

    local vid_num = tonumber(self.vid_to_num[vid_name])

    local vid_cut = torch.linspace(1, vid_num, 4)
    -- fetch the feature from the dir
    -- random sample 3 frame for the video
    for j = 1, 3 do

      local frame_id
      if split == 'train' then
        frame_id = torch.random(vid_cut[j], vid_cut[j+1])
      else
        frame_id = math.floor((vid_cut[j] + vid_cut[j+1])/ 2)
      end
      -- start with 0
      frame_id = frame_id - 1

      local frame_name = vid_name .. "_frame_" .. frame_id  
      local path = self.folder .. '/' .. frame_name .. '.txt'
      img_batch_raw[{{i,i}, {j,j}}] = ReadData(path, self.feat_dim)
    end
    -- fetch the sequence labels
    local ix1 = self.label_start_ix[ix]
    local ix2 = self.label_end_ix[ix]
    local ncap = ix2 - ix1 + 1 -- number of captions available for this image
    assert(ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t')

    local ixl = torch.random(ix1, ix2)
    seq = self.h5_file:read('/labels'):partial({ixl, ixl}, {1, self.seq_length})
    --local il = (i-1)*seq_per_img+1
    --label_batch[{ {il,il+seq_per_img-1} }] = seq

    label_batch[{{i,i}}] = seq
    -- and record associated info as well
    local info_struct = {}
    info_struct.id = tonumber(string.sub(vid_name, 4, -1))
    table.insert(infos, info_struct)
  end

  local data = {}
  data.images = img_batch_raw:transpose(1,2):contiguous() -- note: make data go down as columns
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos
  
  return data
end

