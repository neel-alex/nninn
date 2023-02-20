import matplotlib.pyplot as plt
from jax import random, nn
import haiku as hk

from nninn.ctc_dataset import ctc_net_fn
from nninn.mnist import mnist

dataset = mnist()
train_data = dataset[0]

key = random.PRNGKey(4)

network = hk.without_apply_rng(hk.transform(lambda x: ctc_net_fn(x, n_classes=10)))
params = network.init(key, train_data[0].reshape(1, 28, 28))

train_data = train_data.reshape(train_data.shape[0], 1, 28, 28)

assert network.apply(params, train_data[0]).shape == (1, 10)
assert network.apply(params, train_data[:32]).shape == (32, 10)
