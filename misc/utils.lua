local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end


function utils.in_table(dict, v)
  if dict[v] == nil then
    return false 
  else
    return true 
  end
end


-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

function utils.is_empty(t)
  return next(t) == nil
end

function utils.shallow_copy(t)
  ret = {}
  for k, v in pairs(t) do
    ret[k] = v
  end
  return ret
end


function utils.change(t)
  local c = {}
  local h = {}
  local batch_size = #t
  for i = batch_size, 1, -1  do
    t1 = t[i]
    for k1, v1 in pairs(t1) do
      assert(v1:dim() == 1)
      
      if k1 == 1 then
        table.insert(c, v1)
      else
        table.insert(h, v1)
      end
    end
  end
  return {nn.JoinTable(2):forward(c), nn.JoinTable(2):forward(h)} 
end

function utils.add(a, b)
  if a == nil then
    a = b
  else
    a:add(b)
  end

end

function utils.save(filename, t)
  file = io.open(filename, 'w')
  for k, v in pairs(t) do
    file:write(v .. ' ')
  end
  file:close()
end

function utils.modify(voc_sz, index, sent)
    -- must notice that classify the two situations
    -- INDEX == -1 : VOC_SZ + 1
    -- INDEX == 0: VOC_SZ + 2
    if index == -1 then
        return (voc_sz + 1) 
    elseif index == 0 then 
        return (voc_sz + 2)
    else
        return sent[index]
    end 
end

function utils.getChenFeat(voc_sz, c, sent)
    fWord = {}
    -- top 3 elements in stack
    for i = 2, 0, -1 do
        index = c:getStack(i)
        table.insert(fWord, utils.modify(voc_sz, index, sent))
    end
    -- left / right - left(left) / right(right)
    for i = 0, 1 do
        k = c:getStack(i)
        index = c:getLeftChild(k, 1)
        table.insert(fWord, utils.modify(voc_sz, index, sent))

        index = c:getRightChild(k, 1)
        table.insert(fWord, utils.modify(voc_sz, index, sent))
        
        index = c:getLeftChild(k, 2)
        table.insert(fWord, utils.modify(voc_sz, index, sent))

        index = c:getRightChild(k, 2)
        table.insert(fWord, utils.modify(voc_sz, index, sent))

        index = c:getLeftChild(c:getLeftChild(k, 1), 1)
        table.insert(fWord, utils.modify(voc_sz, index, sent))

        index = c:getRightChild(c:getRightChild(k, 1), 1)
        table.insert(fWord, utils.modify(voc_sz, index, sent))
    end
    assert(#fWord == 15)
    return fWord
end
return utils
