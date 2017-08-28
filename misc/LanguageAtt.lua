require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local GRU = require 'misc.GRU'
local LSTM = require 'misc.LSTM'
-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageAtt', 'nn.Module')
function layer:__init(opt)
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
  self.core = nn.Recursor(model, self.seq_length + 1)
  self:_creatZeroState(1) -- lazy init
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
      self.zero_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.zero_state
end

function layer:getModulesList()
  return {self.core}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end


  return params, grad_params
end

function layer:training()
  self.core:training()
end

function layer:evaluate()
  self.core:evaluate()
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

  local state = input[1]
  local att_img = input[2]
  local att_mask = input[3]
  local batch_size = att_img:size(1)
  
  local seq = torch.LongTensor(self.seq_length, batch_size):zero() 
  -- return the samples and their log likelihoods
  for t=1,self.seq_length+1 do

      local xt, i 
      if t == 1 then
        -- feed in the start tokens
        it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      else
        -- take predictions from previous time step and feed them in
        -- use argmax "sampling"
         _, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      end

      if t >= 2 then 
          seq[t-1] = it -- record the samples
      end

      local inputs = {it, att_img, att_mask, unpack(state)}
      local out = self.core:forward(inputs)
      
      --[[ 
      local mask_module = self.core:findModules('nn.maskSoftMax')

      local mask_weight = mask_module[1].output:float()
      local _, att_visual = torch.kthvalue(mask_weight, 110, 2)

      print(att_visual[1][1])
      --]] 
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
  end
  return seq
end

function layer:updateOutput(input)
  
  self.core:forget()

  local init_state = input[1]
  local seq = input[2]
  local att_img = input[3]
  local att_mask = input[4]


  self.inputs = {}
  self.state = {[0] = init_state}
  self.tmax = 0

  local batch_size = seq:size(2) 
  self.output:resize(self.seq_length+1, batch_size, self.vocab_size+1)
  for t=1,self.seq_length+1 do

    local can_skip = false
    local it
    if t == 1 then
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
    else
      -- feed in the rest of sequence 
      it = seq[t-1]:clone()
      if torch.sum(it) == 0 then
        can_skip = true
      end
    end
    
    if not can_skip then
      self.inputs[t] = {it, att_img, att_mask, unpack(self.state[t-1])}
      
      local out = self.core:forward(self.inputs[t])
      --[[
      local mask_module = self.core:findModules('nn.maskSoftMax')

      local mask_weight = mask_module[1].output:float()
      local _, att_visual = torch.kthvalue(mask_weight, 110, 2)

      print(att_visual[1][1])
      --]]
      self.output[t] = out[self.num_state+1]
      self.state[t] = {} -- the rest is state
      for i = 1, self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end
  return self.output
end

function layer:sample_beam(input)
  self.core:forget()

  local init_state = input[1]
  local att_img = input[2]
  local att_mask = input[3]

  local beam_size = utils.getopt(opt, 'beam_size', 3)
  local batch_size = input[1][1]:size(1)

  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do

    -- create initial states for all beams
    --local state = init_state[k]:expand(beam_size, self.rnn_size) 

    self:_creatZeroState(beam_size)
    local state = self.zero_state

    for h=1, self.use_lstm and 2 * self.num_layers or self.num_layers do
        state[h] = init_state[h][k]:repeatTensor(beam_size):view(beam_size, -1)
    end

    local att_local_img = torch.CudaTensor(beam_size, att_img:size(2), att_img:size(3))
    local att_local_mask = torch.CudaTensor(beam_size, att_mask:size(2))
    for h=1, beam_size do
      att_local_img[{{h,h},{},{}}] = att_img[k]
      att_local_mask[{{h,h},{}}] = att_mask[k]
    end

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    for t=1,self.seq_length+1 do

      local it, sampleLogprobs
      local new_state
      if t == 1 then
        -- feed in the start tokens
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 2 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 2 then
            beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+1 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end
        
        -- encode as vectors
        it = beam_seq[t-1]
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {it, att_local_img, att_local_mask, unpack(state)}
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end


--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
 
  self:_creatZeroState(gradOutput:size(2))
  local dstate = {[self.tmax] = self.zero_state}
  local dimgs -- grad on input imgs
  for t = self.tmax, 1, -1 do
    local dout = {}
    for k =1, #dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.core:backward(self.inputs[t], dout)
    --split the gradient 
    dstate[t-1] = {}
    for k=4, self.num_state+3 do table.insert(dstate[t-1], dinputs[k]) end
    if t == self.tmax then
      dimgs = dinputs[2]
    else
      dimgs:add(dinputs[2])
    end
  end
  self.gradInput = {dstate[0], dimgs, torch.Tensor(), torch.Tensor()}

end
