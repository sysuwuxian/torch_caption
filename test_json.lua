local utils = require 'misc.utils'


info = utils.read_json('./model/model_id1.json')
----

--local predictions = info.val_predictions
local val = info.val_lang_stats_history
print(val)
--print(predictions)

--info = utils.read_json('./best.json')
----
--local predictions = info.val_predictions
----local stats = info.val_lang_stats
--print(predictions)
