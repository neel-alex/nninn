from jax import random
import jax.numpy as jnp
from optax import adam
import haiku as hk

from model import net_fn, train_net


NUM_CLASSES = 2
NUM_HIDDEN = 256
LEARNING_RATE = 1e-3
SEED = 4
NUM_DATAPOINTS = 200
TRAIN_FRAC = 0.8


key = random.PRNGKey(SEED)

network = hk.without_apply_rng(hk.transform(lambda x: net_fn(x, NUM_HIDDEN, NUM_CLASSES)))
optimiser = adam(LEARNING_RATE)

filestore = jnp.load(f'./data/networks_{NUM_DATAPOINTS}.npz')
nets = jnp.array(filestore['nets'])
labels = jnp.array(filestore['labels'])

key, subkey = random.split(key)
n_datapoints = nets.shape[0]
perm = random.permutation(subkey, n_datapoints)
nets = nets[perm]
labels = labels[perm]
labels = jnp.eye(NUM_CLASSES)[labels]

split = int(n_datapoints*TRAIN_FRAC)

train_nets = nets[:split]
test_nets = nets[split:]
train_labels = labels[:split]
test_labels = labels[split:]

key, subkey = random.split(key)
train_net(key, network, optimiser, train_nets, train_labels, test_nets, test_labels, verbose=2)
