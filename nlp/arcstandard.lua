require 'torch'
require 'config'
UNKNOWN = 'unknown'

local arcstandard = torch.class('arcstandard', 'Config')


function arcstandard:initialConfig(l)
  c = Config()
  for i = 1, l do
    c.tree:add(NONEXIST, UNKNOWN)
    table.insert(c.buffer, i)
  end
  table.insert(c.stack, 0)
  return c

end

function arcstandard:isterminal(c)
    return c:getStackSize() == 1 and c:getBufferSize() == 0
end

function arcstandard:getTop(c)
  return {c:getStack(0), c:getStack(1)}
end

function arcstandard:apply(c, state)
  -- add 1 start from 1
  w1 = c:getStack(1) 
  w2 = c:getStack(0)
  if state == 1 then 
    c:shift()
  elseif state == 3 then 
    c:addArc(w1, w2, UNKNOWN)
    c:removeTopStack()
  else
    c:addArc(w2, w1, UNKNOWN)
    c:removeSecondTopStack()
  end
end

function arcstandard:canApply(c, t)
    nStack = c:getStackSize()
    nBuffer = c:getBufferSize()

    if t == 2 then
       return nStack > 2
    elseif t == 3 then
       return nStack > 2 or (nStack == 2 and nBuffer == 0)
    else
       return nBuffer > 0
    end
end
