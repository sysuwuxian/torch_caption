require 'torch'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local Tree = torch.class('Tree')

function Tree:__init(n, dim, len, ptr, vid_name)
  -- n is attend length
  -- dim is attend dim
  -- len is sent len
  -- cur is current len
  self.n = n 
  self.dim = dim
  self.len = len
  self.pre_n = ptr
  self.vid_name = vid_name

  -- region and corresponding feat
  self.feat = torch.Tensor(self.n, self.dim):zero()
  self.mask = torch.Tensor(self.n):zero()

  -- parsing tree and corrseponding node
  self.head = torch.Tensor(self.len):fill(-1)
  self.leaf = torch.Tensor(self.len):fill(1)
  self.ptr = ptr
  self.word2region = {}
  
  -- 0 corresponding to the whole image -1
  self.word2region[0] = -1
  self.region2h = {}
  self.node = {}

end

-- notice is cuda tensor
function Tree:set(feat, mask)
  if feat:type() == 'torch.CudaTensor' then
    self.feat = self.feat:cuda()
  end

  if mask:type() == 'torch.CudaTensor'then
    self.mask = self.mask:cuda()
  end
  self.feat:copy(feat)
  self.mask:copy(mask)
end


function Tree:save()
  --print the corresponding tree
  tree_fold = './tree_fold'
  utils.save(tree_fold .. '/' .. self.vid_name .. '.txt', self.word2region)
end


function Tree:child(index)
  child = {}
  cnt = 1 
  for i = 1, self.len do
    if self.head[i] == index then
      child[cnt] = i
      cnt = cnt + 1
    end
  end
  return child
end

function Tree:update_node(tree_lstm, f, c, h)
  if tree_lstm.train ~= false then 
    return tree_lstm:forward{f, c, h}
  else
    return net_utils.clone_list(tree_lstm:forward{f, c, h})
  end
end



function Tree:update_forward(id, son, tree_lstm)
 

  -- update the mask 
  --for k, v in pairs(son) do   
  --  self.mask[self.word2region[v]] = 1
  --end
  --self.mask[self.word2region[id]] = 1
  
  for k, v in pairs(son) do
    if self.leaf[v] == 1 then
      
      local child_c, child_h = self:get_child_states(self:child(v))
      
      self.node[v] = self:update_node(tree_lstm, self.feat[self.word2region[v]],
              child_c, child_h)
    end
  end

  local child_c, child_h = self:get_child_states(son)
  
  self.node[id] = self:update_node(tree_lstm, self.feat[self.word2region[id]], 
              child_c, child_h)

  -- update the region
  self.ptr = self.ptr + 1
  -- fix the feat
  -- so we best clone
  self.feat[self.ptr] = self.node[id][2]:clone()
  --self.mask[self.ptr] = 0 
  self.region2h[self.ptr] = id
end


function Tree:is_origin_region(region_id)
  return region_id >= 1 and region_id <= self.pre_n
end


function Tree:update_backward(id,  tree_lstm, dnode_h, dnode_c, dfeat)
  local son = self:child(id)
  local child_c, child_h = self:get_child_states(son)
  
  local region = self.word2region[id]
  
  local composer_grad = tree_lstm:backward({self.feat[region], child_c, child_h}, 
      {dnode_c[id], dnode_h[id]})

  if self:is_origin_region(region) == true then
      dfeat[region] = utils.add(dfeat[region], composer_grad[1])
  else
      local node_id = self.region2h[region]
      dnode_h[node_id] = utils.add(dnode_h[node_id], composer_grad[1])
  end

  local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]

  for i = 1, #son do
    dnode_c[son[i]] = utils.add(dnode_c[son[i]], child_c_grads[i])
    dnode_h[son[i]] = utils.add(dnode_h[son[i]], child_h_grads[i])
  end

  for k, v in pairs(son) do
    if self.leaf[v] == 1 then

      local region = self.word2region[v]
      local child_c, child_h = self:get_child_states(self:child(v))
      local composer_grad = tree_lstm:backward({self.feat[region], child_c, child_h}, 
        {dnode_c[v], dnode_h[v]})

      if self:is_origin_region(region) == true then
          dfeat[region] = utils.add(dfeat[region], composer_grad[1])
      else
          local node_id = self.region2h[region]
          dnode_h[node_id] = utils.add(dnode_h[node_id], composer_grad[1])
      end
      -- not need to backward to child
      -- since no child 
    end
  end

end

function Tree:get_child_states(child)
  local child_c, child_h
  if utils.is_empty(child) == true then
    child_c = torch.zeros(1, self.dim):cuda()
    child_h = torch.zeros(1, self.dim):cuda()
  else
    child_c = torch.zeros(#child, self.dim):cuda()
    child_h = torch.zeros(#child, self.dim):cuda()
    for i = 1, #child do
      child_c[i], child_h[i] = unpack(self.node[child[i]]) 
    end
  end
  return child_c, child_h
end


function Tree:setHead(tokenIndex, headIndex, module)
 
  -- print for debug
  -- print('u-->v is ', tokenIndex, '-->', headIndex)

  self.head[tokenIndex] = headIndex
  if headIndex == 0 then
    return false
  end

  son = self:child(tokenIndex)
  self.leaf[headIndex] = 0
  if utils.is_empty(son) == false then
      self:update_forward(tokenIndex, son, module)
      return true
  else
      return false
  end
end
