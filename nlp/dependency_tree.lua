require 'torch'
local dep_tree = torch.class('dep_tree')
UNKNOWN = 'unknown'
NONEXIST = -1


function dep_tree:__init()
   	self.n = 0
    self.head = {}
    table.insert(self.head, NONEXIST)
    self.label = {}
    table.insert(self.label, UNKNOWN)
end

function dep_tree:add(h, l)
 	self.n = self.n + 1
  table.insert(self.head, h)
  table.insert(self.label, l)

end

function dep_tree:set(k, h, l)
	self.head[k+1] = h+1
	self.label[k+1] = l
end

function dep_tree:getHead(k)
  if k <= 0 or k > self.n then
    return NONEXIST
  else
    return self.head[k+1]
  end
end
