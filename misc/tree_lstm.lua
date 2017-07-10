require 'nn'
require 'nngraph'
require 'misc.CRowAddTable'

local tree_lstm = {}
function tree_lstm.tree(in_dim, mem_dim)
  local input = nn.Identity()()
  local child_c = nn.Identity()()
  local child_h = nn.Identity()()
  local child_h_sum = nn.Sum(1)(child_h)

  local i = nn.Sigmoid()(
    nn.CAddTable(){
      nn.Linear(in_dim, mem_dim)(input),
      nn.Linear(mem_dim, mem_dim)(child_h_sum)
    })
  local f = nn.Sigmoid()(
    nn.CRowAddTable(){
      nn.TemporalConvolution(mem_dim, mem_dim, 1)(child_h),
      nn.Linear(in_dim, mem_dim)(input),
    })
  local update = nn.Tanh()(
    nn.CAddTable(){
      nn.Linear(in_dim, mem_dim)(input),
      nn.Linear(mem_dim, mem_dim)(child_h_sum)
    })
  local c = nn.CAddTable(){
      nn.CMulTable(){i, update},
      nn.Sum(1)(nn.CMulTable(){f, child_c})
    }

  local h
  if gate_output then
    local o = nn.Sigmoid()(
      nn.CAddTable(){
        nn.Linear(in_dim, mem_dim)(input),
        nn.Linear(mem_dim, mem_dim)(child_h_sum)
      })
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local composer = nn.gModule({input, child_c, child_h}, {c, h})
  return composer
end
return tree_lstm
