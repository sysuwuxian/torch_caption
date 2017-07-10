require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'

require 'misc.DataLoaderParsing'
require 'misc.LanguageParsing'
require 'misc.LanguageModel'
-- local imports
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
-----------------------------------------------------------------
-- Input arguments and options
-----------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Video Caption model')
cmd:text()
cmd:text('Options')

--Data input setting

cmd:option('-input_feat_h5', './data/data_feat.h5', 'path to the h5file containing the preproceed data')
cmd:option('-input_h5', './data/data.h5', 'path to the h5file containing the preproceed data')
cmd:option('-input_json', './data/data.json', 'path to the json file containing additional info and vocab')
cmd:option('-folder', '../deep-residual-networks/ResNet_charades_feature', 'path to the folder containing the frame feature')
-- Not needed by anyone
cmd:option('-att_folder', '', 'c3d feature for attention')

cmd:option('-vid_file', './data/train_parsing_charades.txt', 'filenames of the train-val-test')
--cmd:option('-train_num', 1200)
--cmd:option('-val_num', 770)
cmd:option('-init_from', '', 'path to a model checkpoint to initialize model weights from')

--Model setting
cmd:option('-rnn_size', 1000, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-num_layers', 1, 'number of layers in the GRU')
cmd:option('-input_encoding_size', 500, 'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-frame_feat_dim', 2048, 'the resnet feat length of frame')
cmd:option('-att_feat_dim', 2048, 'the c3d feat length of att frame')
cmd:option('-att_seq_size', 200, 'seq size of the attention feat map')
cmd:option('-t_len', 130, 'trans len of the parsing sentence')
cmd:option('-useLSTM', 1, 'lstm vs gru')


--Optimization: General
cmd:option('-batch_size', 10, 'number of examples in each batch')
cmd:option('-max_iters', 40000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-grad_clip', 20, 'clip gradients at this value')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')


-- Optimization: for the Language Model
cmd:option('-optim', 'adam', 'using adam update')
cmd:option('-learning_rate', 1e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 3000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 6000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.9,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 3000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', './model', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-print_every', 20, 'How often do we print losses')


-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')

cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-id', '1', 'an id identifying this run/job')
cmd:text()

-- init model
cmd:option('-nlp_model', './nlp/model/model_id1.t7', 'path to model load Initialize embedding')

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
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

------------------------------------------------------------------------------
-- Load nlp model
------------------------------------------------------------------------------
local nlp_checkpoint = torch.load(opt.nlp_model)
local nlp_model = nlp_checkpoint.protos.model
if opt.gpuid >= 0 then
  nlp_model = nlp_model:cuda()
end



-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
rgs = torch.deserialize(torch.serialize(opt))
local loader = DataLoaderParsing(rgs)

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
if string.len(opt.init_from) > 0 then
  print('loading agent from checkpoint ' .. opt.init_from)

  local checkpoint = torch.load(opt.init_from)
  protos = checkpoint.protos
  -- manual add crit
  protos.crit = nn.LanguageModelCriterion() 
else
  print('creating a video caption agent with ')
  protos = {}
  -- create enc model
  -- parallel table input
  
  local enc = nn.ParallelTable()
  
  -- branch 1
  m = nn.Sequential()
  m.rnn = {}

  m:add(nn.View(-1, opt.frame_feat_dim))
  m:add(nn.Linear(opt.frame_feat_dim, opt.input_encoding_size))
  m:add(nn.View(-1, opt.batch_size, opt.input_encoding_size))

  for i = 1, opt.num_layers do
    local pre_sz = -1
    if i == 1 then
      pre_sz = opt.input_encoding_size
    else
      pre_sz = opt.rnn_size
    end
    
    if opt.useLSTM then
      m.rnn[i] = nn.SeqLSTM(pre_sz, opt.rnn_size)
    else
      m.rnn[i] = nn.SeqGRU(pre_sz, opt.rnn_size)
    end
    m:add(m.rnn[i])
  end
  m:add(nn.Select(1,-1))

  -- branch 2
  local emb = nn.Sequential()
  emb:add(nn.View(-1, opt.att_feat_dim))
  emb:add(nn.Linear(opt.att_feat_dim, opt.input_encoding_size))
  emb:add(nn.View(opt.batch_size, -1, opt.input_encoding_size))
  -- parallel two input modules 
  enc:add(m)
  enc:add(emb)
  protos.enc = enc
  
  -- create dec model
  local lmopt = {}
  lmopt.vocab_size = loader:getVocabSize()
  lmopt.input_encoding_size = opt.input_encoding_size
  lmopt.rnn_size = opt.rnn_size
  lmopt.num_layers = opt.num_layers 
  lmopt.seq_length = loader:getSeqLength()
  lmopt.dropout = opt.drop_prob_lm
  lmopt.useLSTM = opt.useLSTM
  -- other parameters
  lmopt.att_size = opt.input_encoding_size
  lmopt.att_seq_size = opt.att_seq_size

  local vocab = loader:getVocab()

  protos.dec = nn.LanguageParsing(lmopt, vocab)
  -- create loss funtion 
  protos.crit = nn.LanguageModelCriterion()
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local enc_params, enc_grad_params = protos.enc:getParameters()
local dec_params, dec_grad_params = protos.dec:getParameters()

print('total number of paramters in encoder:', enc_params:nElement())
print('total number of parameters in decoder:', dec_params:nElement())
assert(enc_params:nElement() == enc_grad_params:nElement())
assert(dec_params:nElement() == dec_grad_params:nElement())


-- construct thin module clones that share parameters with
-- the actual modules. 
local thin_enc = protos.enc:clone()
thin_enc:share(protos.enc, 'weight', 'bias')
net_utils.sanitize_gradients(thin_enc)

local thin_dec = protos.dec:clone()
thin_dec.core:share(protos.dec.core, 'weight', 'bias')
local lm_modules = thin_dec:getModulesList()
for k, v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

collectgarbage()


------------------------------------------------------------------------------
-- Validation evaluation
------------------------------------------------------------------------------

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
      data.att_img = data.att_img:cuda()
      data.att_mask = data.att_mask:cuda()
    end

    n = n + data.images:size(2)
    local  _, att_emb = unpack(protos.enc:forward{data.images, data.att_img})

    for i = 1, opt.batch_size do
      data.trees[i]:set(att_emb[i], data.att_mask[i])
    end

    local state = {}
    
    local enc_len = data.images:size(1)
    for i = 1, opt.num_layers do
      if opt.useLSTM then 
        table.insert(state, m.rnn[i].cell[enc_len])
      end
      
      table.insert(state, m.rnn[i].output[enc_len])
    end

    -- local seq = protos.dec:sample{state, data.trees, data.trans, data.configs}
    local seq = protos.dec:sample{state, data.trees, nlp_model}


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
  
  -- not compute loss, no sense
  return -1, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function 
-------------------------------------------------------------------------------

local function lossFun()
  -----------------------------------------------------------------------------
  -- forward pass
  -----------------------------------------------------------------------------
 
  protos.enc:training()
  protos.dec:training()

  enc_grad_params:zero()

  
  dec_grad_params:zero()

  local timer = torch.Timer()
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train'}
  local time = timer:time().real
  
  --print('time elapsed is ', time)
  if opt.gpuid >= 0 then
    data.images = data.images:cuda()
    data.att_img = data.att_img:cuda()
    data.att_mask = data.att_mask:cuda()

  end

  local enc_out, att_emb = unpack(protos.enc:forward{data.images, data.att_img})
  local init_state = {}
  local enc_len = data.images:size(1)

  for i = 1, opt.batch_size do
    data.trees[i]:set(att_emb[i], data.att_mask[i])
  end
  
  for i = 1, opt.num_layers do
    if opt.useLSTM then
      table.insert(init_state, m.rnn[i].cell[enc_len])
    end
    table.insert(init_state, m.rnn[i].output[enc_len])
  end


  local logprobs = protos.dec:forward{init_state, data.labels, data.trees, data.trans,
    data.configs }
  local loss = protos.crit:forward(logprobs, data.labels)

  -----------------------------------------------------------------------------
  -- backward pass
  -----------------------------------------------------------------------------

  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  local dstate, dimgs = unpack(protos.dec:backward({init_state, data.labels, data.trans},  
       dlogprobs)) 
  
  --backward the encoder
  if opt.useLSTM then
    for i = 1, opt.num_layers do
      m.rnn[i].userNextGradCell = dstate[2*i-1]
      m.rnn[i].gradPrevOutput = dstate[2*i]
    end
  else
    for i = 1, opt.num_layers do
      m.rnn[i].gradPrevOutput = dstate[i]
    end
  end

  local zeroTensor = torch.CudaTensor(enc_out):zero()
  protos.enc:backward({data.images, data.att_img}, {zeroTensor, dimgs})

  -- clip gradients
  enc_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  dec_grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- and lets get out! 
  local losses = { total_loss = loss }
  return losses
end

----------------------------------------------------------------------------
-- Main loop
----------------------------------------------------------------------------
local iter = 0
local loss0
local enc_optim_state = {}
local dec_optim_state = {}
local val_lang_stats_history = {}
local val_loss_history = {}

local loss_history = {}
local best_score
while true do  

  -- eval loss/gradient
  local timer = torch.Timer() 
  local losses = lossFun()
 
  local time = timer:time().real

  --print('fb time is ', time)
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  if iter % opt.print_every == 0 then
    print(string.format('Iteration %d: loss = %f', iter, losses.total_loss))
  end
  
  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val')
    print('Validation loss: ', val_loss)
    --print(lang_stats)
    val_loss_history[iter] = val_loss
    
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions
    checkpoint.val_lang_stats_history = val_lang_stats_history


    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['METEOR']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        --save_protos.cnn = thin_cnn
        save_protos.enc = thin_enc
        save_protos.dec = thin_dec
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end
  
  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  -- perform a parameter update
  if opt.optim == 'adam' then
    adam(enc_params, enc_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, enc_optim_state)
    adam(dec_params, dec_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, dec_optim_state)
  end
  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
