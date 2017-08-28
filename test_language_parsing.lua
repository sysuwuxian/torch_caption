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


-- test language model
-- gradient based 
local function gradCheck()

    -- cuda tensor
    local dtype = 'torch.CudaTensor'
    require 'cutorch'
    require 'cunn'


    local lmopt = {}
    lmopt.vocab_size = 10
    lmopt.useLSTM = 1 

    -- note for att size(for t / att num) 
    lmopt.att_seq_size = 15
    lmopt.att_num = 10

    -- setting for the rnn_size etc
    lmopt.input_encoding_size = 11
    lmopt.rnn_size = 8
    lmopt.num_layers = 1
    lmopt.seq_length = 20
    lmopt.dropout = 0
    lmopt.batch_size = 1 
    
    local lm = nn.LanguageParsing(lmopt)
    local crit = nn.LanguageModelCriterion()
    lm:type(dtype)
    crit:type(dtype)
    
    -- construct some input to feed in
    -- read hdf5 
    local h5_file = hdf5.open('./data/test_gradient.h5', 'r')
    local sents = h5_file:read('/sent'):partial({1,1}, {1,20})
    local trans = h5_file:read('/state'):partial({1,1}, {1,20})
    

    local c = torch.randn(lmopt.batch_size, lmopt.rnn_size):type(dtype)
    local h = torch.randn(lmopt.batch_size, lmopt.rnn_size):type(dtype)

    local init_state = {c, h} 

    -- set trees
    local trees = {}
    local configs = {}
    for i = 1, lmopt.batch_size do
        configs[i] = Config_pre(sents[i])
        local sent_len = #configs[i].buffer
        -- set mask etc
        local mask = torch.LongTensor(lmopt.batch_size, lmopt.att_seq_size):fill(1)
        mask[{{i,i}, {1,lmopt.att_num}}]:fill(0)
       
        mask = mask:cuda()

        local att_feat = torch.randn(lmopt.att_seq_size, lmopt.input_encoding_size):type(dtype)
        trees[i] = Tree(lmopt.att_seq_size, lmopt.input_encoding_size, sent_len, lmopt.att_num)
        trees[i]:set(att_feat, mask)
    end


    -- change the label seq
    sents = sents:transpose(1,2):contiguous() -- note make data go down as cloumns

    local output = lm:forward{init_state, sents, trees, trans, configs}
    
    local loss = crit:forward(output, sents)

    local gradOutput = crit:backward(output, sents)
    local grad_state = unpack(lm:backward({init_state, sents,
        trans}, gradOutput))

    local gradInput = grad_state[2]

    -- create a loss function wrapper
    local function f(x)
      local output = lm:forward{{c, x}, sents, trees, trans, configs}
      local loss = crit:forward(output, sents)
      return loss
    end

    local gradInput_num = gradcheck.numeric_gradient(f, h, 1, 1e-6)

    print(gradInput_num:size())
    -- print(gradInput)
    -- print(gradInput_num)
    -- local g = gradInput:view(-1)
    -- local gn = gradInput_num:view(-1)
    -- for i=1,g:nElement() do
    --   local r = gradcheck.relative_error(g[i],gn[i])
    --   print(i, g[i], gn[i], r)
    -- end

    tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
    tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end


tests.gradCheck = gradCheck

tester:add(tests)
tester:run()
