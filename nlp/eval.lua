require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
require 'arcstandard'

require 'dataloader'
local utils = require '../misc/utils'
cmd = torch.CmdLine()
-- Input paths
cmd:option('-model','./model/model_id1.t7','path to model to evaluate')
-- Basic options
cmd:option('-batch_size', 0, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')


cmd:option('-input_feat_h5', './data/data_feat.h5', 'path to the h5file containing the preproceed data')
cmd:option('-input_h5','./data/test.h5','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json','','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-split', 'val', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-coco_json', '', 'if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()


local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)  
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'folder', 'vid_file', 'frame_feat_dim', 'train_num', 'val_num', 'val_images_use'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocab = checkpoint.vocab -- ix -> word mapping

-------------------------------------------------------------------------------
-- evaluate the parser
-------------------------------------------------------------------------------
local loader
rgs = torch.deserialize(torch.serialize(opt))
local loader = DataLoader(rgs)

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
if opt.gpuid >= 0 then for k, v in pairs(protos) do v:cuda() end end

local function eval_test(split)

  local val_images_use = opt.val_images_use

  local token_correct = 0.0
  local token_total = 0.0


  while true do

    local data = loader:getBatch{batch_size=opt.batch_size, split = split}

    if opt.gpuid >= 0 then
      data.feats = data.feats:cuda()
    end


    local logprobs = protos.model:forward(data.feats):float()
    
    -- counting the rating
    local max_score, max_labels = logprobs:max(2)

    local correct = max_labels:eq(data.labels):sum()

    token_correct = token_correct + correct
    token_total = token_total + data.feats:size(1)

    if data.bounds.wrapped then break end -- split ran out of data, lets break out
      if val_images_use > 0 and n >= val_images_use then break end -- we've used enough images
  end
  
  
  return token_correct * 1.0 / token_total

end


local function eval_split(split)
	
	protos.model:evaluate()
	local val_images_use = opt.val_images_use

	local token_correct = 0.0
	local token_total = 0.0

  -- get sentence and predict
  local acc = 0 
  local total = 0
  local hit = 0

  local system = arcstandard()

  -- load hdf5 file
  h5_file = hdf5.open(opt.input_h5, 'r')

  labels = h5_file:read('/labels'):all()
  feats = h5_file:read('/feats'):all()
  indexs = h5_file:read('/indexs'):all()

  -- two feats 
  -- data.feats and data.labels
	-- check for each sentence
  total = total + labels:size(1)
  local voc_sz = utils.count_keys(vocab)
 
  local pre_total = 0
  local pre_hit = 0

  for i = 1, labels:size(1) do 
      local sent = feats[i]
      local len = indexs[i]
      local label = labels[i]
      local c = system:initialConfig(len)
      local ptr = 1
      local f = true
      while not system:isterminal(c) do
          local feat = torch.Tensor(utils.getChenFeat(voc_sz, c, sent))
          if opt.gpuid > 0 then
            feat = feat:cuda()
          end

          local logprobs = protos.model:forward(feat):float()
          local state = 0 
          local max_prob = -100000 
          local numTrans = 3
          for j = 1, numTrans do 
            if logprobs[1][j] > max_prob and system:canApply(c, j) then
                max_prob = logprobs[1][j]
                state = j
            end
          end
          -- system:apply(c, label[ptr])
          system:apply(c, state)
      end
  end
  pre_acc = (pre_total - pre_hit) * 1.0 / pre_total
  print('true is ', pre_acc)
  acc = hit * 1.0 / total
  return acc
end

local acc = eval_split(opt.split)
print('acc is ', acc)
