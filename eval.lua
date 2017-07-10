require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
-- exotics
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'misc.LanguageAtt'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','./model/model_id1.t7','path to model to evaluate')
-- Basic options
cmd:option('-batch_size', 0, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 1, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 1, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')

cmd:option('-rnn_size', 1000, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-num_layers', 1, 'number of layers in the GRU')
cmd:option('-input_encoding_size', 500, 'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-frame_feat_dim', 4096, 'the resnet feat length of frame')
--cmd:option('-att_feat_dim', 512, 'the c3d feat length of att frame')
--cmd:option('-att_seq_size', 80, 'seq size of the attention feat map')
cmd:option('-useLSTM', 1, 'lstm vs gru')
cmd:option('-vid_file', './data/train_all_resnet_c3d.txt', 'filenames of the train-val-test')

-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 1, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
-- For evaluation on a folder of images:
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')
-- For evaluation on MSCOCO images from some split:
cmd:option('-input_feat_h5', './data/data_feat.h5', 'path to the h5file containing the preproceed data')
cmd:option('-input_h5','','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
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
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader
rgs = torch.deserialize(torch.serialize(opt))
local loader = DataLoader(rgs)

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.crit = nn.LanguageModelCriterion()
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split)
  protos.enc:evaluate()
  protos.dec:evaluate()
  loader:resetIterator(split)
 
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local vocab = loader:getVocab()
  local val_images_use = opt.val_images_use
  local predictions = {}

  local vis = {}
  
  while true do
    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split}
    if opt.gpuid >= 0 then
      data.images = data.images:cuda()
    end
    n = n + data.images:size(2)
    protos.enc:forward(data.images)
    local state = {}
    
    local enc_len = data.images:size(1)

    -- find rnn
    rnn = protos.enc:findModules('nn.SeqLSTM')

    --print rnn to debug
    for i = 1, opt.num_layers do
      if opt.useLSTM then 
        table.insert(state, rnn[i].cell[enc_len])
      end
      table.insert(state, rnn[i].output[enc_len])
    end
    
    local logprobs = protos.dec:forward({state, data.labels})
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1 

    local seq = protos.dec:sample_beam(state)

    local sents = net_utils.decode_sequence(vocab, seq)
    
    for k=1, #sents do
      if not vis[data.infos[k].id] then 
        local entry = {image_id = data.infos[k].id, caption = sents[k]}
        vis[data.infos[k].id] = 1
        table.insert(predictions, entry)
      end
    end
    if data.bounds.wrapped then break end -- split ran out of data, lets break out
    if val_images_use > 0 and n >= val_images_use then break end -- we've used enough images
  end
  
  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end
  
  return loss_sum/loss_evals, predictions, lang_stats
end

local loss, split_predictions, lang_stats = eval_split(opt.split)

local checkpoint = {}
checkpoint.val_lang_stats = lang_stats
checkpoint.val_predictions = split_predictions
utils.write_json('best.json', checkpoint)
