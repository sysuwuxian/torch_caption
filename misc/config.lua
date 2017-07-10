require 'torch'
require 'misc.dependency_tree'
NONEXIST = -1
local Config = torch.class('Config')

function Config:__init()
  self.stack = {}
  self.buffer = {}
  self.tree = dep_tree() 
end

function Config:buffer()
  return self.buffer
end

function Config:shift() 
  local word = self.buffer[1]
  table.remove(self.buffer, 1)
  table.insert(self.stack, word)
end

function Config:removeTopStack()
  sz = #self.stack
  assert(sz > 0)
  table.remove(self.stack, sz)
end

function Config:getBufferSize()
  sz = #self.buffer
  return sz
end

function Config:getStackSize()
  sz = #self.stack
  return sz
end

function Config:getStack(k)
  nStack = self:getStackSize()
  if k >= 0 and k < nStack then
    return self.stack[nStack - k]
  else
    return NONEXIST
  end

end


function Config:getTop() 
  sz = #self.stack
  assert(sz > 1)
  return {self.stack[sz-1], self.stack[sz]}
end

function Config:removeSecondTopStack() 
  sz = #self.stack
  assert(sz > 1)
  table.remove(self.stack, sz - 1)
end


function Config:getLeftChild(k, cnt)
  if k < 0 or k > self.tree.n then
    return NONEXIST
  end

  local c = 0
  for i = 1, k - 1 do

    if self.tree:getHead(i) == k + 1 then
        c = c + 1
        if c == cnt then
          return i
        end
    end
  end

  return NONEXIST

end

function Config:addArc(h, t, l)
  self.tree:set(t, h, l)
end


function Config:getRightChild(k, cnt)
  if k < 0 or k > self.tree.n then
    return NONEXIST
  end

  local c = 0
  for i = self.tree.n, k, -1 do
    if self.tree:getHead(i) == k + 1 then
      c = c + 1
      if c == cnt then
        return i
      end
    end
  end
  return NONEXIST
end
