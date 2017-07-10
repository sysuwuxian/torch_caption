require 'nn'
require 'torch'
require 'dataloader'
require 'nngraph'

local utils = require 'utils'
local net_utils = require 'net_utils'
require '../misc/optim_updates'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Language Parsing model')
cmd:text()
cmd:text('Options')

--Data input setting
cmd:option('-input_json', './data/msvd_data.json', 'path to the json file containing the additional data')
cmd:option('-input_h5', './data/parse_msvd_feat.h5', 'path to the h5file containing the preproceed data')
cmd:option('-init_from', '', 'path to a model checkpoint to initialize model weights from')
cmd:option('-train_num', 900000, 'train example num')

--Model setting
cmd:option('-in_dim', 50, 'embedding word dim')
cmd:option('-hidden_dim', 200, 'hidden dim of mlp')
cmd:option('-rnn_layer', 1, 'hidden layers of mlp')
cmd:option('-word_per_example', 15, 'total elements in a training example')

--Optimization: General
cmd:option('-batch_size', 50, 'number of examples in each batch')
cmd:option('-max_iters', 400000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')

-- Optimization: for the Language Model
cmd:option('-optim', 'adagrad', 'using adagrad update')
cmd:option('-learning_rate', 0.01,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 4000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.9,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay', 1e-8, 'regularization for training')


-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 3000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', './model', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-print_every', 100, 'How often do we print losses')


-- note for cudnn
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 12, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-id', '1', 'an id identifying this run/job')

------------------------------------------------------------------
-- Basic Torch initializations
------------------------------------------------------------------
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


-----------------------------------------------------------------
-- Create the Data Loader instance
-----------------------------------------------------------------

rgs = torch.deserialize(torch.serialize(opt))
local loader = DataLoader(rgs)

-----------------------------------------------------------------
-- Initialize the networks 
-----------------------------------------------------------------
print('create a language parsing model')
protos = {}
net = nn.Sequential()

-- note 2 refer to the ROOT / NULL
local vocab_size = loader:getVocabSize() + 2
local in_dim = opt.in_dim
local hidden_dim = opt.hidden_dim

M = nn.LookupTable(vocab_size, in_dim)
M.weight = torch.rand(vocab_size, in_dim):add(-.01):mul(0.01)
net:add(M)

local dims = in_dim * opt.word_per_example

net:add(nn.View(-1, dims))

local lastDims = dims

for i = 1, opt.rnn_layer do
  net:add(nn.Linear(lastDims, hidden_dim))
  net:add(nn.ReLU())
  lastDims = hidden_dim
end

net:add(nn.Dropout(opt.drop_prob_lm))

-- 3 dimension classification
net:add(nn.Linear(hidden_dim, 3))
net:add(nn.LogSoftMax())

print(net)

protos.net = net
-- criterion 
protos.crit = nn.ClassNLLCriterion()

if opt.gpuid >= 0 then
	for k, v in pairs(protos) do v:cuda() end
end

local params, grad_params = protos.net:getParameters()
print('total number of parameters in net:', params:nElement())

assert(params:nElement() == grad_params:nElement())


---------------------------------------
--- remove the unnessarry gradients ---
----------------------------------- ---

local thin_net = protos.net:clone()
thin_net:share(protos.net, 'weight', 'bias')
net_utils.sanitize_gradients(thin_net)
--------------------------------------------------------
-- evaluate parser
--------------------------------------------------------

local function eval_split(split)
	
	protos.net:evaluate()
	local val_images_use = opt.val_images_use

	local token_correct = 0.0
	local token_total = 0.0


	while true do

		local data = loader:getBatch{batch_size=opt.batch_size, split = split}

		if opt.gpuid >= 0 then
			data.feats = data.feats:cuda()
		end


		local logprobs = protos.net:forward(data.feats):float()
    
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


local function lossFun()
  
  protos.net:training()
  grad_params:zero()

  local data = loader:getBatch{batch_size=opt.batch_size, split = 'train'}

  if opt.gpuid >= 0 then
    data.feats = data.feats:cuda()
  end

  local logprobs = protos.net:forward(data.feats)
  logprobs = logprobs:float()
  local max_score, max_labels = logprobs:max(2)

  local acc = max_labels:eq(data.labels):sum() / opt.batch_size

  logprobs = logprobs:cuda()
  local loss = protos.crit:forward(logprobs, data.labels)

  --------------------------------------------------------
  -- backward pass
  --------------------------------------------------------
  
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  protos.net:backward(data.feats, dlogprobs)


  -- apply L2 regularization
  grad_params:add(opt.weight_decay, params)


  local losses = { total_loss = loss , total_acc = acc }
  return losses

end




-- main body---
local iter = 0
local loss0
local net_optim_state = {}
local val_lang_stats_history = {}
local val_loss_history = {}

local loss_history = {}
local best_score
while true do  

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  if iter % opt.print_every == 0 then
    print(string.format('Iteration %d: loss = %f, acc = %f', iter, losses.total_loss,
    losses.total_acc))
  end
  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss = eval_split('val')
    print('Validation accuracy: ', val_loss)
    --print(lang_stats)
    val_loss_history[iter] = val_loss


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
    local current_score = val_loss
    
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        --save_protos.cnn = thin_cnn
        save_protos.model = thin_net
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
  if opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_alpha, net_optim_state)
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
