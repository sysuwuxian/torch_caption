require 'nn'
require 'misc.arcstandard'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local GRU = require 'misc.GRU'
local LSTM = require 'misc.LSTM'
local tree_lstm = require 'misc.tree_lstm'
-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageParsing', 'nn.Module')
function layer:__init(opt, voc)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.use_lstm = utils.getopt(opt, 'useLSTM')
  self.input_size_img = utils.getopt(opt, 'input_encoding_size')
  self.img_seq_size = utils.getopt(opt, 'att_seq_size')
  
  -- create the core lstm network. note +1 for both the START and END tokens
  
  if opt.useLSTM then
    model = LSTM.lstm_att(self.input_encoding_size, self.input_size_img, self.img_seq_size, 
          self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
  end
  -- +1 for start tokens
  self.core = nn.Recursor(model)
  self:_creatZeroState(1) -- lazy init

  -- for tree_lstm init
  
  self.tree_lstm = tree_lstm.tree(self.input_size_img, self.input_size_img)
  self.tree_module = nn.Recursor(self.tree_lstm)
  
  -- dict
  self.voc = voc
end

function layer:_creatZeroState(batch_size)

  assert(batch_size ~= nil, 'batch_size must be provided')
  if not self.zero_state then self.zero_state = {} end -- lazy init

  for h=1, self.use_lstm and 2 * self.num_layers or self.num_layers do
    if self.zero_state[h] then
      if self.zero_state[h]:size(1) ~= batch_size then
        self.zero_state[h]:resize(batch_size, self.rnn_size):zero()
      end
    else 
      --self.zero_state[h] = torch.zeros(batch_size, self.rnn_size)
      self.zero_state[h] = torch.zeros(self.rnn_size)
    end
  end
  self.num_state = #self.zero_state
end

function layer:getModulesList()
  return {self.core, self.tree_module}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()

  local p2,g2 = self.tree_module:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:training()
  self.core:training()
  self.tree_module:training()
end

function layer:evaluate()
  self.core:evaluate()
  self.tree_module:evaluate()
end


function layer:sample_beam(input, opt)
  self.core:forget()
  self.tree_module:forget()
  
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local init_state = input[1]
  local all_trees = input[2]
  local nlp_model = input[3]
  
  local batch_size = #all_trees

  local function compare(a, b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now')
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()

  
  local system = arcstandard()

  for k=1, batch_size do 
    -- init tree / config / sent
    local trees = {}
    local configs = {}
    local sents = {}
    local states = {}

    -- initial tree / state / configs / sents
    trees[1] = all_trees[k]
    states[1] = {init_state[1][k]:clone(), init_state[2][k]:clone()}
    configs[1] = system:initialConfig(self.seq_length)
    sents[1] = {}
    
    local beam_logprobs_sum = torch.zeros(beam_size)
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size + 1)
    local done_beams = {}

    -- set begin rows, start with 1
    local rows = 1

    while true do
      
      local cols = math.min(beam_size, self.vocab_size)
      local candidates = {}
      if rows == 0 then
        break
      end

      for q=1,rows do -- for each beam expansion
        -- local config, tree, t
        local tree = trees[q]
        local config = configs[q]
        local state = states[q]
        local sent = sents[q]
        
        local feats = torch.Tensor(utils.getChenFeat(#self.voc, config, sent))
        feats = feats:cuda()
        local probs = torch.exp(nlp_model:forward(feats):float())
        local trans_state = 0

        local max_prob = 0
        local numTrans = 3
        for j = 1, numTrans do
          if probs[1][j] > max_prob and system:canApply(config, j) then
             max_prob = probs[1][j]
             trans_state = j
          end
        end



        -- print for debug
        print('trans state is ', trans_state)

        if trans_state == 1 then

          -- shift -> soft attention to generator the next word
          -- get the corresponding inputs, mask, is_leaf for backward
          local it, word
          if utils.is_empty(sent) then
            word = self.vocab_size + 1
          else
            word = sent[#sent]
          end
          it = torch.LongTensor(1):fill(word)
          
          local inputs = {it, tree.feat, tree.mask, unpack(state)}

          local out = self.core:forward(inputs)
          logprobs = out[self.num_state + 1]:float()

          -- update lstm states
          lstm_state = {}
          assert(out[1]:dim() == 1 and out[2]:dim() == 1)
          for k = 1, self.num_state do table.insert(lstm_state, out[k]) end
          states[q] = lstm_state

          -- find the relation between word and region
          local att_w = out[self.num_state+2]:float() 
          local _, id = torch.max(att_w, 2)
          
          local cnt = #tree.word2region + 1
          -- mapping current word to corresponding region
          tree.word2region[cnt] = id[1][1]
          
          ys,ix = torch.sort(logprobs,true)

          for c=1,cols do
            -- compute logprob of expanding beam q with word in sorted position c
            local local_logprob = ys[{c}]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[c], q=q, p=candidate_logprob, r=local_logprob})
          end
        
        elseif trans_state == 2 then
          -- left arc
          local son, fa = unpack(config:getTop())
          tree:setHead(son, fa, self.tree_module)
          local local_logprob = 0.0
          local candidate_logprob = beam_logprobs_sum[q] + local_logprob
          table.insert(candidates, {c=-1, q=q, p=candidate_logprob, r=local_logprob})


        elseif trans_state == 3 then
          -- right arc
          local fa, son = unpack(config:getTop())
          tree:setHead(son, fa, self.tree_module)
          local local_logprob = 0.0
          local candidate_logprob = beam_logprobs_sum[q] + local_logprob
          table.insert(candidates, {c=-1, q=q, p=candidate_logprob, r=local_logprob})
        end
        system:apply(config, trans_state)
      end

      table.sort(candidates, compare)
        
      -- construct new states and new trees 
      -- new config and new sent
      local new_states = {}
      local new_trees = {}
      local new_configs = {}
      local new_sents = {}
      -- update new beams
      for vix=1, beam_size do
        local v = candidates[vix]
        -- append new end terminal at the end of beam
        local sent = net_utils.copy_list(sents[v.q])
        if v.c ~= -1 then
          table.insert(sent, v.c)
        end
        beam_logprobs_sum[vix] = v.p

        if v.c == self.vocab_size+1 or system:isterminal(configs[v.q]) then  
          table.insert(done_beams, {seq = torch.Tensor(sents[v.q]), 
                        p = beam_logprobs_sum[vix]})
        
        elseif not system:isterminal(configs[v.q]) then
          -- update new states / tress / configs
          table.insert(new_states, net_utils.clone_list(states[v.q]))
          table.insert(new_trees, net_utils.copy_tree(trees[v.q]))
          table.insert(new_configs, net_utils.copy_config(configs[v.q]))
          table.insert(new_sents, sent)
        end
      
      end
      rows = utils.count_keys(new_configs) 
      states = new_states
      trees = new_trees
      configs = new_configs
      sents = new_sents

    end

    -- sort the model according to the score
    table.sort(done_beams, compare)
    local sen_l = done_beams[1].seq:size(1)

    seq[{{1,sen_l}, k}] = done_beams[1].seq -- the first beam has highest cumularive score
  
  end

  return seq

end



--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]

-- input is a table
function layer:sample(input)

  self.core:forget()
  self.tree_module:forget()

  local init_state = input[1]
  
  -- seq is T * N
  local trees = input[2]
  local nlp_model = input[3]
  
  -- transition is N * transition
  --local transitions = input[3]
  --local batch_size = #configs
  local batch_size = #trees
  local att_seq_size = self.img_seq_size
  --local trans_num = transitions:size(2) 

  local seq = torch.LongTensor(self.seq_length, batch_size):zero() 

  local system = arcstandard()
  -- consider sent is "I was a boy"
  -- lstm input is "# I was a boy"
  -- lstm output is "I was a boy #" 
  -- enumarate the tree
  for i = 1, batch_size do
    local tree = trees[i]
    local c = system:initialConfig(self.seq_length)
    
    local lstm_state = {}
    local cnt = 1
    local sent = {}

    local test_iter = 0

    while not system:isterminal(c) do
      -- get two word in stack
      -- get the feat in the config
      test_iter = test_iter + 1
    

      local feats = torch.Tensor(utils.getChenFeat(#self.voc, c, sent))
      feats = feats:cuda()
      local probs = torch.exp(nlp_model:forward(feats):float())
      local state = 0
      local max_prob = 0
      local numTrans = 3
      for j = 1, numTrans do
        if probs[1][j] > max_prob and system:canApply(c, j) then
          max_prob = probs[1][j]
          state = j
        end
      end
      
      print('state is ', state)
      
      if state == 1 then
        local it, word
        if cnt == 1 then
          word = self.vocab_size + 1
          lstm_state = {init_state[1][i], init_state[2][i]}
        else
           _, tensor = torch.max(logprobs, 1)
           word = tensor[1]
        end
        it = torch.LongTensor(1):fill(word)

        if cnt >= 2 then
          seq[cnt-1][i] = it 
        end
        -- shift -> soft attention to generator the next word
        -- get the corresponding inputs, mask, is_leaf for backward
        
        local inputs = {it, tree.feat, tree.mask, unpack(lstm_state)}

        local out = self.core:forward(inputs)
        logprobs = out[self.num_state + 1]:float()

        -- get predict word
        _, tensor = torch.max(logprobs, 1)
        table.insert(sent, tensor[1])


        lstm_state = {}
        assert(out[1]:dim() == 1 and out[2]:dim() == 1)
        for k = 1, self.num_state do table.insert(lstm_state, out[k]) end

        -- find the relation between word and region
        local att_w = out[self.num_state+2]:float() 
        local _, id = torch.max(att_w, 2)
        -- mapping current word to corresponding region
        tree.word2region[cnt] = id[1][1]
        cnt = cnt + 1
      
      elseif state == 2 then
        -- left arc
        local son, fa = unpack(c:getTop())
        tree:setHead(son, fa, self.tree_module)
      
      elseif state == 3 then
        -- right arc
        local fa, son = unpack(c:getTop())
        tree:setHead(son, fa, self.tree_module)
      else
        break
      end
      system:apply(c, state)

    end
    -- tree:save()
  end
  return seq
end

function layer:updateOutput(input)
  
  self.core:forget()
  self.tree_module:forget()

  local init_state = input[1]
  
  -- seq is T * N
  local seq = input[2]
  local trees = input[3]
  
  -- transition is N * transition
  local transitions = input[4]
  local configs = input[5]

  local att_seq_size = self.img_seq_size
  
  self.inputs = {}
  self.state = {}
  self.tree = {}
  self.merge_node = {}
  self.diff_max = {}
  self.tmax = {}
  self.mask = {}


  local batch_size = seq:size(2)
  local trans_num = transitions:size(2) 

  self.output:resize(self.seq_length+1, batch_size, self.vocab_size+1)


  -- consider sent is "I was a boy"
  -- lstm input is "# I was a boy"
  -- lstm output is "I was a boy #" 

  -- enumarate the tree
  for i = 1, batch_size do
    local tree = trees[i]
    local config = configs[i]

    local cnt = 1
    self.inputs[i] = {}
    self.merge_node[i] = {}
    self.mask[i] = {}
    
    for t = 1, trans_num do
      local state = transitions[i][t]
      if state == 0 then
        -- config operation
        config:shift()
        
        local it, word
        if cnt == 1 then
          word = self.vocab_size + 1
          self.state[0] = {init_state[1][i], init_state[2][i]}
        else
          word = seq[cnt-1][i]
        end
        it = torch.LongTensor(1):fill(word)
        -- shift -> soft attention to generator the next word
        -- get the corresponding inputs, mask, is_leaf for backward
        
        self.inputs[i][cnt] = {it, tree.feat, tree.mask, unpack(self.state[cnt-1])}
        self.mask[i][t] = tree.mask 

        local out = self.core:forward(self.inputs[i][cnt])
        self.output[cnt][i] = out[self.num_state+1]

        self.state[cnt] = {} -- the rest is state
        assert(out[1]:dim() == 1 and out[2]:dim() == 1)
        for k = 1, self.num_state do table.insert(self.state[cnt], out[k]) end

        -- find the relation between word and region
        local att_w = out[self.num_state+2]:float() 
        local _, id = torch.max(att_w, 2)

        -- mapping current word to corresponding region
        tree.word2region[cnt] = id[1][1]
        cnt = cnt + 1

        if utils.is_empty(config.buffer) == true then
          self.diff_max[i] = t
          break
        end
      
      elseif state == 1 then
        -- left arc
        local son, fa = unpack(config:getTop())
        if tree:setHead(son, fa, self.tree_module) == true then
          self.merge_node[i][t] = son
        end
        config:removeSecondTopConfig()
      
      elseif state == 2 then
        -- right arc
        local fa, son = unpack(config:getTop())
        if tree:setHead(son, fa, self.tree_module) == true then
          self.merge_node[i][t] = son
        end
        config:removeTopConfig()
      end
    end
    -- need to subtract by 1
    self.tmax[i] = cnt - 1
    self.tree[i] = tree
  end
  return self.output
end



--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
 
  self:_creatZeroState(1)
  assert(#self.tree >= 1)
  
  local batch_size = input[2]:size(2)

  local att_num = self.tree[1].feat:size(1)

  local dim = self.tree[1].feat:size(2)

  local dimgs = torch.CudaTensor(batch_size, att_num, dim):fill(0)
 
  local dc0 = torch.CudaTensor(batch_size, self.rnn_size):fill(0)
  local dh0 = torch.CudaTensor(batch_size, self.rnn_size):fill(0)

  -- get the trainsition 
  local transition = input[3]

  local trans_num = input[3]:size(1)

  for i = batch_size, 1, -1 do
    local dstate = {[self.tmax[i]] = self.zero_state}
    local cnt = self.tmax[i] 
    local tree = self.tree[i]
    -- verify the node and feat
    local dnode_h = {}
    local dnode_c = {}
    local dfeat = {}

    for t = self.diff_max[i], 1, -1 do
        local trans = transition[i][t] 
        if trans == 1 then
          -- get father
          if self.merge_node[i][t] ~= nil then
            tokenIndex = self.merge_node[i][t]
            
            if dnode_c[tokenIndex] == nil then
              dnode_c[tokenIndex] = torch.CudaTensor(dim):fill(0)
            end
            if dnode_h[tokenIndex] == nil then
              dnode_h[tokenIndex] = torch.CudaTensor(dim):fill(0)
            end

            tree:update_backward(tokenIndex, self.tree_module, dnode_h, dnode_c, dfeat)  
          end
        elseif trans == 2 then
          if self.merge_node[i][t] ~= nil then
            tokenIndex = self.merge_node[i][t]

            if dnode_h[tokenIndex] == nil then
              dnode_h[tokenIndex] = torch.CudaTensor(dim):fill(0)
            end
            if dnode_c[tokenIndex] == nil then
              dnode_c[tokenIndex] = torch.CudaTensor(dim):fill(0)
            end

            tree:update_backward(tokenIndex, self.tree_module, dnode_h, dnode_c, dfeat) 
          end
        elseif trans == 0 then
          local dout = {}
          for k = 1, #dstate[cnt] do table.insert(dout, dstate[cnt][k]) end
          table.insert(dout, gradOutput[cnt][i])
          -- notice we need backward zero Tensor
          local zeroTensor = torch.CudaTensor(att_num):zero()

          table.insert(dout, zeroTensor)
          local dinputs = self.core:backward(self.inputs[cnt], dout)
          dstate[cnt-1] = {}
          for k = 4, self.num_state+3 do table.insert(dstate[cnt-1], dinputs[k]) end

          -- online change the dinput0
          -- enumarte the attention part 
          for k = 1, att_num do
            if self.mask[i][t][k] == 0 then
              if tree:is_origin_region(k) then
                utils.add(dfeat[key], dinputs[2][k])
              else
                local node_id = tree.region2h[k]
                utils.add(dnode_h[node_id], dinputs[2][k])
              end
            end
          end
          cnt = cnt - 1
          if cnt == 0 then
            dc0[i] = dstate[cnt][1]
            dh0[i] = dstate[cnt][2]
          end
        end
    end
    for k = 1, att_num do
      if dfeat[k] ~= nil then
        dimgs[i][k] = dfeat[k]
      end
    end
  end
  self.gradInput = {{dc0, dh0}, dimgs, torch.Tensor(), torch.Tensor()} 
end
