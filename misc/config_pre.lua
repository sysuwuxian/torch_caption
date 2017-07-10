require 'torch'
local Config_pre = torch.class('Config_pre')

function Config_pre:__init(sentence, voc_sz)
  self.stack = {}
  self.buffer = {}

  self.sentence = sentence:view(-1)
  --init the buffer
  for i = 1, self.sentence:size(1) do
    if self.sentence[i] ~= 0 then
      table.insert(self.buffer, i)
    end
  end
  -- add for endding punctuncation
  table.insert(self.buffer, #self.buffer + 1)
  -- root
  table.insert(self.stack, 0)
end

function Config_pre:buffer()
  return self.buffer
end

function Config_pre:shift() 
  local word = self.buffer[1]
  table.remove(self.buffer, 1)
  table.insert(self.stack, word)
end



function Config_pre:removeTopConfig()
  sz = #self.stack
  assert(sz > 0)
  table.remove(self.stack, sz)

end

function Config_pre:getTop() 
  sz = #self.stack
  assert(sz > 1)
  return {self.stack[sz-1], self.stack[sz]}

end

function Config_pre:removeSecondTopConfig() 
  sz = #self.stack
  assert(sz > 1)
  table.remove(self.stack, sz - 1)
end
