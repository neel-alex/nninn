from jax import random
import jax.numpy as jnp
from optax import adam, sgd
import haiku as hk
from sacred import Experiment, observers

from mnist import mnist
from model import net_fn, train_nets

train_nets_experiment = Experiment('train_nets')
observer = observers.FileStorageObserver('results/train_nets')
train_nets_experiment.observers.append(observer)


@train_nets_experiment.config
def config():
    n_classes = 10
    n_hidden = 16
    batch_size = 60000  # 60000 is full-batch
    learning_rate = 1e-3
    n_nets = 100  # Number per setting -- 2x this many networks will be created
    n_parallel = 100  # How many nets to train in parallel
    seed = 4  # Chosen by fair dice roll. Guaranteed to be random.


@train_nets_experiment.capture
def gather_network_dataset(key,
                           network,
                           optimiser,
                           train_data,
                           train_labels,
                           test_data,
                           test_labels,
                           n_nets,
                           n_parallel):
    nets = []
    assert (n_nets / n_parallel) % 1 == 0, "Please ensure that the parallelization evenly divides the number of nets."
    n_runs = n_nets // n_parallel
    for i in range(n_runs):
        key, subkey = random.split(key)
        net_params = train_nets(subkey,
                                network,
                                optimiser,
                                train_data,
                                train_labels,
                                test_data,
                                test_labels,
                                n_nets=n_parallel,
                                verbose=1)

        for j in range(n_parallel):
            flat_net = jnp.array([])
            for layer in net_params:
                for param in net_params[layer]:
                    flat_net = jnp.concatenate((flat_net, net_params[layer][param][j].flatten()))

            nets.append(flat_net)
    return nets


@train_nets_experiment.capture
def create_network(n_hidden, n_classes):
    return hk.without_apply_rng(hk.transform(lambda x: net_fn(x, n_hidden, n_classes)))


@train_nets_experiment.automain
def main(learning_rate, n_nets, seed):
    train_data, train_labels, test_data, test_labels = mnist()
    network = create_network()
    key = random.PRNGKey(seed)

    optimiser = adam(learning_rate)
    key, subkey = random.split(key)
    networks = gather_network_dataset(subkey, network, optimiser, train_data, train_labels,
                                      test_data, test_labels, n_nets=n_nets)

    optimiser = sgd(learning_rate)
    key, subkey = random.split(key)
    networks += gather_network_dataset(subkey, network, optimiser, train_data, train_labels,
                                       test_data, test_labels, n_nets=n_nets)

    data_nets = jnp.row_stack(networks)
    data_labels = jnp.array([0 for _ in range(n_nets)] +
                            [1 for _ in range(n_nets)])
    print(f"Saving {data_nets.shape[0]} networks.")
    jnp.savez(f'./data/networks_{data_nets.shape[0]}.npz', nets=data_nets, labels=data_labels)
