from jax import random
import jax.numpy as jnp
from optax import adam, sgd
import haiku as hk

from mnist import mnist
from model import net_fn, train_net


def train_nets(key, network, optimiser, train_data, train_labels, test_data, test_labels):
    nets = []
    for i in range(NUM_DATAPOINTS // 2):
        key, subkey = random.split(key)
        params = train_net(subkey, network, optimiser, train_data, train_labels, test_data, test_labels)
        flat_net = jnp.array([])
        for layer in params:
            for param in params[layer]:
                flat_net = jnp.concatenate((flat_net, params[layer][param].flatten()))

        nets.append(flat_net)
    return nets


NUM_CLASSES = 10
NUM_HIDDEN = 16
BATCH_SIZE = 60000
LEARNING_RATE = 1e-3
NUM_DATAPOINTS = 20000
SEED = 4

key = random.PRNGKey(SEED)

train_data, train_labels, test_data, test_labels = mnist()

network = hk.without_apply_rng(hk.transform(lambda x: net_fn(x, NUM_HIDDEN, NUM_CLASSES)))

optimiser = adam(LEARNING_RATE)
key, subkey = random.split(key)
networks = train_nets(subkey, network, optimiser, train_data, train_labels, test_data, test_labels)
optimiser = sgd(LEARNING_RATE)
key, subkey = random.split(key)
networks += train_nets(subkey, network, optimiser, train_data, train_labels, test_data, test_labels)

data_nets = jnp.row_stack(networks)
data_labels = jnp.array([0 for _ in range(NUM_DATAPOINTS//2)] +
                        [1 for _ in range(NUM_DATAPOINTS//2)])
jnp.savez(f'./data/networks_{data_nets.shape[0]}.npz', nets=data_nets, labels=data_labels)





