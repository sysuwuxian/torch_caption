--[[
Unit tests for the LanguageParsing implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'torch'
require 'hdf5'
require 'misc.LanguageParsing'
require 'misc.LanguageModel'
require 'rnn'
require 'misc.DataLoaderParsing'

local LSTM = require 'misc.LSTM'
local gradcheck = require 'misc.gradcheck'

local tests = {}
local tester = torch.Tester()

-- validates the size and dimensions of a given 
-- tensor a to be size given in table sz
function tester:assertTensorSizeEq(a, sz)
  tester:asserteq(a:nDimension(), #sz)
  for i=1,#sz do
    tester:asserteq(a:size(i), sz[i])
  end
end




local function test_att()
  local dtype = 'torch.DoubleTensor'

  local lmopt = {}
  lmopt.input_size_img = 4
  lmopt.rnn_size = 3
  lmopt.img_seq_size = 5
  lmopt.att_num = 3
  lmopt.batch_size = 1
  lmopt.dropout = 0

  model = attention.soft_attention(lmopt.input_size_img, lmopt.rnn_size, 
        lmopt.img_seq_size, lmopt.dropout)

  model:zeroGradParameters()

  local N = lmopt.batch_size
  local D = lmopt.rnn_size
  local M = lmopt.img_seq_size
  local E = lmopt.input_size_img

  local h = torch.randn(D):type(dtype)
  local masks = torch.ByteTensor(M):fill(1)
  local att_feats = torch.randn(M, E):type(dtype)
  
  
  masks[{{1,lmopt.att_num}}]:fill(0)
  
  local inputs = {att_feats, masks, h}

  local outputs = model:forward(inputs)

  local dout = torch.randn(M):type(dtype)

  local loss = torch.sum(torch.cmul(outputs, dout)) 


  local grad_state = model:backward(inputs, dout)


  graph.dot(model.fg, 'forward graph', 'att_fg')
  graph.dot(model.bg, 'backward graph', 'att_bg')
  
  local gradInput = grad_state[3]


  local function f(x)
    local outputs = model:forward({att_feats,  masks, x})
    local loss = torch.sum(torch.cmul(outputs, dout))
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, h, 1)
  


  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num), 1e-4)

end

-- Test LSTM 
-- Just Test one time Line

local function test_att_lstm()


    local dtype = 'torch.DoubleTensor'
    --require 'cutorch'
    --require 'cunn'

    local lmopt = {}
    lmopt.vocab_size = 3 
    lmopt.useLSTM = 1 

    -- note for att size(for t / att num) 
    lmopt.att_seq_size = 5
    lmopt.att_num = 3
    
    -- setting for the rnn_size etc
    lmopt.input_encoding_size = 2
    lmopt.rnn_size = 3 
    lmopt.num_layers = 1
    lmopt.seq_length = 5 
    lmopt.dropout = 0
    lmopt.batch_size = 1


    model = LSTM.lstm_att(lmopt.input_encoding_size, lmopt.input_encoding_size, 
          lmopt.att_seq_size, lmopt.vocab_size + 1, 
          lmopt.rnn_size, lmopt.num_layers, lmopt.dropout)
    
    model:zeroGradParameters()
    model:type(dtype)
    -- test forward and backward
    
    
    
    local N = lmopt.batch_size
    local D = lmopt.rnn_size
    local M = lmopt.att_seq_size
    local E = lmopt.input_encoding_size
    local V = lmopt.vocab_size + 1
    
    local it = torch.LongTensor(N):fill(3)
    local c = torch.randn(D):type(dtype)
    local h = torch.randn(D):type(dtype)

    local state = {c, h} 

    local masks = torch.ByteTensor(M):fill(1)
    masks[{{1,lmopt.att_num}}]:fill(0)


    local att_feats = torch.randn(M, E):type(dtype)


    local inputs = {it, att_feats, masks, unpack(state)}


    local d_c = torch.randn(D):type(dtype)
    local d_h = torch.Tensor(D):zero()
    local d_w = torch.Tensor(M):zero()
    local d_soft = torch.Tensor(V):zero()

    local doutputs = {d_c, d_h, d_soft, d_w}

    local outputs = model:forward(inputs)

    local loss = torch.sum(torch.cmul(outputs[1], d_c)) 


    local grad_state = model:backward(inputs, doutputs)

    local gradInput = grad_state[5]


    local function f(x)
      local outputs = model:forward({it, att_feats, masks, c, x})
      local loss = torch.sum(torch.cmul(outputs[1], d_c)) 
      return loss
    end

    local gradInput_num = gradcheck.numeric_gradient(f, h, 1)

    --[[
    local g = gradInput:view(-1)
    local gn = gradInput_num:view(-1)
    for i=1,g:nElement() do
       local r = gradcheck.relative_error(g[i],gn[i])
       print(i, g[i], gn[i], r)
    end 
    --]]
    tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num), 1e-4)

end

-- test language model
-- gradient based 
local function gradCheck()

    -- cuda tensor
    local dtype = 'torch.DoubleTensor'


    local lmopt = {}
    lmopt.vocab_size = 10
    lmopt.useLSTM = 1 

    -- note for att size(for t / att num) 
    lmopt.att_seq_size = 7
    lmopt.att_num = 3

    -- setting for the rnn_size etc
    lmopt.input_encoding_size = 11
    lmopt.rnn_size = 8
    lmopt.num_layers = 1
    lmopt.seq_length = 20
    lmopt.dropout = 0
    lmopt.batch_size = 1 
    lmopt.use_cuda = false 



    local N = lmopt.batch_size
    local D = lmopt.rnn_size
    local M = lmopt.att_seq_size
    local E = lmopt.input_encoding_size 



    local lm = nn.LanguageParsing(lmopt)


    local crit = nn.LanguageModelCriterion()
    lm:type(dtype)
    crit:type(dtype)
    
    -- construct some input to feed in
    -- read hdf5 
    local h5_file = hdf5.open('./data/test_gradient.h5', 'r')
    local sents = h5_file:read('/sent'):partial({1,1}, {1,20})

    -- last word is zero
    sents[1][8] = 0

    local trans = h5_file:read('/state'):partial({1,1}, {1,20})

    local c = torch.randn(N, D):type(dtype)
    local h = torch.randn(N, D):type(dtype)

    local init_state = {c, h} 

    -- set trees
    local trees = {}
    local configs = {}

    local pre_att_feats = torch.Tensor(N, M, E)


    for i = 1, lmopt.batch_size do
        configs[i] = Config_pre(sents[i])
        local sent_len = #configs[i].buffer 
        -- set mask etc
        local mask = torch.ByteTensor(M):fill(1)
        mask[{{1,lmopt.att_num}}]:fill(0)

        local att_feat = torch.randn(M, E):type(dtype)
        pre_att_feats[i] = att_feat:clone()
        
        trees[i] = Tree(M, E, sent_len, lmopt.att_num)
        trees[i]:set(att_feat, mask)
    end


    -- change the label seq

    true_sents = sents:clone()

    sents = sents:transpose(1,2):contiguous() -- note make data go down as cloumns

    local inputs = {init_state, sents, trees, trans, configs}

    local output = lm:forward(inputs)

    
    local loss = crit:forward(output, sents)

    local gradOutput = crit:backward(output, sents)


    local grad_state = unpack(lm:backward(inputs, gradOutput))

    local gradInput = grad_state[1]




    -- create a loss function wrapper
    local function f(x)

      -- initialize the config and trees
      for i = 1, lmopt.batch_size do
        configs[i] = Config_pre(true_sents[i])
        
        local sent_len = #configs[i].buffer
        local mask = torch.ByteTensor(M):fill(1)
        mask[{{1, lmopt.att_num}}]:fill(0)
        local att_feat = pre_att_feats[i]
        trees[i] = Tree(M, E, sent_len, lmopt.att_num)
        trees[i]:set(att_feat, mask)
      end


      local output = lm:forward{{x, h}, sents, trees, trans, configs}
      local loss = crit:forward(output, sents)  
      return loss
    end

    local gradInput_num = gradcheck.numeric_gradient(f, c, 1, 1e-6) 

    --[[
    local g = gradInput:view(-1)
    local gn = gradInput_num:view(-1)
    for i=1,g:nElement() do
       local r = gradcheck.relative_error(g[i],gn[i])
       print(i, g[i], gn[i], r)
    end 
    --]]
    tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
    tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

--tests.test_att = test_att
--tests.lstm = test_att_lstm
tests.gradCheck = gradCheck

tester:add(tests)
tester:run()
