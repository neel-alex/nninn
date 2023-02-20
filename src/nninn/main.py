from jax import random
import jax.numpy as jnp
from optax import adam
import haiku as hk
from sacred import Experiment, observers

from nninn.model import net_fn, train_net


interp_net_experiment = Experiment('interp_net')
observer = observers.FileStorageObserver('../../results/interp_net')
interp_net_experiment.observers.append(observer)


@interp_net_experiment.config
def config():
    n_classes = 2
    n_hidden = 256
    learning_rate = 1e-3
    n_data = 200
    train_frac = 0.8
    normalize = False
    seed = 4  # Chosen by fair dice roll. Guaranteed to be random.


@interp_net_experiment.capture
def create_network(n_hidden, n_classes):
    return hk.without_apply_rng(hk.transform(lambda x: net_fn(x, n_hidden, n_classes)))


@interp_net_experiment.automain
def main(learning_rate, n_classes, n_data, train_frac, normalize, seed):
    network = create_network()
    optimiser = adam(learning_rate)

    filestore = jnp.load(f'./data/networks_{n_data}.npz')
    nets = jnp.array(filestore['nets'])
    labels = jnp.array(filestore['labels'])

    if normalize:
        nets = nets / jnp.linalg.norm(nets, axis=1).reshape(-1, 1)

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    perm = random.permutation(subkey, n_data)
    nets = nets[perm]
    labels = labels[perm]
    labels = jnp.eye(n_classes)[labels]

    split = int(n_data*train_frac)

    train_nets = nets[:split]
    test_nets = nets[split:]
    train_labels = labels[:split]
    test_labels = labels[split:]

    key, subkey = random.split(key)
    train_net(key, network, optimiser, train_nets, train_labels, test_nets, test_labels, verbose=2)
