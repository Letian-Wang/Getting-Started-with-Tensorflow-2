-rw-rw-r-- 1 letian letian   77 May 31 17:21 checkpoint
-rw-rw-r-- 1 letian letian 174K May 31 17:21 checkpoint.data-00000-of-00001
-rw-rw-r-- 1 letian letian 2.0K May 31 17:21 checkpoint.index

checkpoint: This is metadata that indicates where the actual model data is stored.(Smallest)

checkpoint.index: This file tells TensorFlow which weights are stored where.
# When running models on distributed systems, 
# there may be different shards, meaning the full model may have to be recomposed from multiple sources. 
# In the last notebook, you created a single model on a single machine, 
# so there is only one shard and all weights are stored in the same place.

checkpoint.data-00000-of-00001: This file contains the actual weights from the model. 