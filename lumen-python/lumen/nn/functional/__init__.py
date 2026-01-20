from ..._lumen import nn as rust_nn 

F = rust_nn.functional

linear = F.linear
softmax = F.softmax
log_softmax = F.log_softmax
dropout = F.dropout
silu = F.silu
sigmoid = F.sigmoid
relu = F.relu
hard_sigmoid = F.hard_sigmoid
leaky_relu = F.leaky_relu
nll_loss = F.nll_loss
mse_loss = F.mse_loss
cross_entropy = F.cross_entropy
cross_entropy_indices = F.cross_entropy_indices
rms_norm = F.rms_norm
layer_norm = F.layer_norm