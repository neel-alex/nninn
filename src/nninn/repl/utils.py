import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple

from jax import random, nn
import jax.numpy as jnp
import numpy as np


key = random.PRNGKey(4)


unmap_info = {
    "dataset": {
        "MNIST": 0,
    },
    "batch_size": {
        32: 0,
        64: 1,
        128: 2,
        256: 3,
    },
    "augmentation": {
        True: 0,
        False: 1,
    },
    "optimizer": {
        "Adam": 0,
        "RMSProp": 1,
        "MomentumSGD": 2,
    },
    "activation": {
        "ReLU": 0,
        "ELU": 1,
        "Sigmoid": 2,
        "Tanh": 3,
    },
    "initialization": {
        "Constant": 0,
        "RandomNormal": 1,
        "GlorotUniform": 2,
        "GlorotNormal": 3,
    },
}

classes_per_task = {
    "batch_size": 4,
    "augmentation": 2,
    "optimizer": 3,
    "activation": 4,
    "initialization": 4,
}


JAXParams = Dict[str, Dict[str, jnp.ndarray]]


def has_nans(net: JAXParams) -> bool:
    for layer in net:
        for param in net[layer]:
            if jnp.isnan(net[layer][param]).any():
                return True
    return False


def data_transform(net: JAXParams) -> jnp.ndarray:
    """ Takes all parameters in the parameters of a jax neural network and flattens it into a 1d vector.
    """
    # TODO use jax.flatten_utils instead?
    flattened_params = []
    for layer in net:
        for param in net[layer]:
            flattened_params.append(net[layer][param].flatten())

    return jnp.concatenate(flattened_params)


def load_nets(n=3000, data_dir='data/ctc_fixed', flatten=True, verbose=True):
    nets = []
    hparam_file = os.path.join(data_dir, 'hyperparameters.json')
    with open(hparam_file) as f:
        net_data = json.load(f)
    labels = {label: [] for label in net_data['0']}
    for i, dir_info in enumerate(os.walk(data_dir)):
        if i == 0:
            continue
        dir_name, _, files = dir_info
        for file_name in files:
            if file_name != "epoch_20.npy":
                continue  # TODO: use other net snapshots?
            with open(dir_name + "/" + file_name, 'rb') as f:
                net = jnp.load(f, allow_pickle=True).item()
                if has_nans(net):
                    if verbose:
                        print("Not loading params at:", dir_name, "since it contains nan values")
                    continue
                nets.append(net)
                net_num = dir_name.split('/')[-1]
                for hparam in net_data[net_num]:
                    labels[hparam].append(net_data[net_num][hparam])
        if len(nets) == n:
            break
    print("Loaded", len(nets), "network parameters")

    if flatten:
        data_nets = [data_transform(net) for net in nets]
        data = jnp.array(data_nets)
    else:
        data = nets

    processed_labels = {}

    for hparam in labels:
        if hparam == "lr":
            processed_labels['lr'] = jnp.array(labels['lr'])
            continue
        unmap = unmap_info[hparam]
        processed_labels[hparam] = jnp.array([unmap[x] for x in labels[hparam]], dtype=jnp.int32)

    return data, processed_labels


def shuffle_and_split_data(key, data, labels, task):
    task_labels = labels[task]
    task_labels = nn.one_hot(task_labels, classes_per_task[task])

    shuffled_data = random.permutation(key, data)
    shuffled_labels = random.permutation(key, task_labels)

    split_index = int(len(data)*0.8)
    train_data = shuffled_data[:split_index]
    train_labels = shuffled_labels[:split_index]

    test_data = shuffled_data[split_index:]
    test_labels = shuffled_labels[split_index:]

    return train_data, train_labels, test_data, test_labels


def random_data_view(key, data, chunk_size=4096):
    sliced_nets = []
    for net in data:
        key, subkey = random.split(key)
        index = random.randint(subkey, shape=(1,), minval=0, maxval=net.size-chunk_size+1).item()
        sliced_nets.append(net[index:index+chunk_size])
    return jnp.array(sliced_nets)



