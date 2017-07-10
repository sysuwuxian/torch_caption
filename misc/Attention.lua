require 'nn'
require 'nngraph'
require 'misc.maskSoftMax'

local attention = {}
function attention.soft_attention(input_size_img, rnn_size, img_seq_size)
  
  
  -- modified for batch size 1
  -- notice this may unbalance 
  -- so we need to be cautious
  
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- feat 
  table.insert(inputs, nn.Identity()()) -- mask 
  table.insert(inputs, nn.Identity()()) -- h 

  local img_feat = inputs[1]
  local mask = inputs[2]
  local h = inputs[3]

  local outputs = {}

  local h_embed = nn.Linear(rnn_size, rnn_size)(h)
  local h_replicate = nn.Replicate(img_seq_size)(h_embed)

  local img_embed_dim = nn.Linear(input_size_img, rnn_size)(nn.View(-1, input_size_img)(img_feat))
  local img_embed = nn.View(img_seq_size, rnn_size)(img_embed_dim)
  local feat = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({img_embed, h_replicate})))
  
  local h3 = nn.Linear(rnn_size, 1)(nn.View(-1, rnn_size)(feat))
  local P3 = nn.maskSoftMax()({nn.View(-1, img_seq_size)(h3),mask})



  -- b x k x d * b x 1 x k
  local probs3dim = nn.View(1,-1)(P3)
  
  table.insert(outputs, probs3dim)

  return nn.gModule(inputs, outputs)
end
return attention
