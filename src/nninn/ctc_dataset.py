import os
from pathlib import Path
from typing import Callable
import json
import functools
import math


import jax
import jax.numpy as jnp
from jax import random, nn

from optax import adam, rmsprop, sgd

import haiku as hk
from haiku.initializers import Initializer, Constant, RandomNormal, TruncatedNormal, VarianceScaling


from nninn.mnist import mnist
from nninn.model import evaluate, update


def ctc_net_fn(x: jnp.ndarray,
               n_classes: int,
               n_conv_layers: int = 3,
               kernel_size: tuple = (3, 3),
               n_filters: int = 32,
               n_fc_layers: int = 3,
               fc_width: int = 128,
               activation: Callable = nn.relu,
               w_init: Initializer = TruncatedNormal()) -> jnp.ndarray:  # TODO: Batchnorm?
    convs = [hk.Conv2D(output_channels=n_filters, kernel_shape=kernel_size, padding="SAME", w_init=w_init)
             for _ in range(n_conv_layers)]
    fcs = [hk.Linear(fc_width, w_init=w_init) for _ in range(n_fc_layers - 1)]

    seq = []
    for conv in convs:
        seq.append(conv)
        seq.append(activation)
    seq.append(hk.Flatten())
    for fc in fcs:
        seq.append(fc)
        seq.append(activation)
    seq.append(hk.Linear(n_classes, w_init=w_init))

    net = hk.Sequential(seq)
    return net(x)


key = random.PRNGKey(4)

learning_rates = lambda k: random.uniform(k, minval=0.0002, maxval=0.005)  # TODO: decay by 0.95 every epoch
hyperparameters = {
    "dataset": ["MNIST"],
    "batch_size": [32, 64, 128, 256],
    "augmentation": [True, False],
    "optimizer": ["Adam", "RMSProp", "MomentumSGD"],
    "activation": ["ReLU", "ELU", "Sigmoid", "Tanh"],
    "initialization": ["Constant", "RandomNormal", "GlorotUniform", "GlorotNormal"]
}

datasets = {
    "MNIST": mnist()  # TODO: CIFAR-10, SVHN, STL-10, Fashion-MNIST
}

optimizers = {
    "Adam": adam,
    "RMSProp": rmsprop,
    "MomentumSGD": functools.partial(sgd, momentum=0.9),
}

activations = {
    "ReLU": nn.relu,
    "ELU": nn.elu,
    "Sigmoid": nn.sigmoid,
    "Tanh": nn.tanh,
}

initializers = {
    "Constant": Constant(0.1),
    "RandomNormal": RandomNormal(),
    "GlorotUniform": VarianceScaling(1.0, "fan_avg", "uniform"),
    "GlorotNormal": VarianceScaling(1.0, "fan_avg", "truncated_normal"),
}

"""datasets = [mnist()]
batch_sizes = [32, 64, 128, 256]
augmentations = [True, False]
optimizers = [adam, rmsprop, sgd]  # sgd is used with momentum!

activations = [nn.relu, nn.elu, nn.sigmoid, nn.tanh]
initializers = [initializers.Constant, initializers.RandomNormal,
                initializers.VarianceScaling, initializers.VarianceScaling] # Glorot uniform/normal
"""

if __name__ == "__main__":
    if not Path("data/ctc_fixed/hyperparameters.json").exists():
        with open("data/ctc_fixed/hyperparameters.json", 'w') as f:
            f.write('{}')

    num_epochs = 20  # TODO: Implement early stopping?

    for i in range(3000):
        if str(i) in os.listdir("data/ctc_fixed"):
            key, _ = random.split(key)  # TODO: Fix to keep constant RNG (maybe store/load key/subkey?)
            continue  # Deaths (mid): 414, 436, 500
        keys = random.split(key, num=8)
        key = keys[-1]
        hparams = {'lr': learning_rates(keys[0]).item()}
        for j, k in enumerate(hyperparameters):
            ind = random.randint(keys[j+1], (1,), minval=0, maxval=len(hyperparameters[k])).item()
            hparams[k] = hyperparameters[k][ind]

        optimizer = optimizers[hparams['optimizer']](hparams['lr'])
        network = hk.without_apply_rng(hk.transform(lambda x: ctc_net_fn(x,
                                                                         n_classes=10,
                                                                         activation=activations[hparams['activation']],
                                                                         w_init=initializers[hparams['initialization']])))
        train_data, train_labels, test_data, test_labels = datasets[hparams['dataset']]
        train_data = train_data.reshape(len(train_data), 1, 28, 28)  # TODO: hardcoded for MNIST
        test_data = test_data.reshape(len(test_data), 1, 28, 28)
        batch_size = hparams['batch_size']

        params = network.init(key, train_data[0])  # TODO: Random key split
        opt_state = optimizer.init(params)

        print(i)

        os.mkdir(f"data/ctc_fixed/{i}")
        jnp.save(f"data/ctc_fixed/{i}/epoch_0", params)
        for epoch in range(num_epochs):
            key, subkey = random.split(key)
            data = random.permutation(subkey, train_data)
            labels = random.permutation(subkey, train_labels)
            num_batches = math.ceil(len(data) / batch_size)
            for batch in range(num_batches):
                batch_data = data[batch*batch_size:(batch+1)*batch_size]
                if hparams['augmentation']:
                    key, sd_key, noise_key = random.split(key, num=3)
                    noise_sd = random.uniform(sd_key, shape=(1,), minval=0.0, maxval=0.03).item()  # TODO: Other augmentations
                    batch_data = batch_data + noise_sd * random.normal(noise_key, shape=batch_data.shape)  # I think this is the right way to use SD?
                batch_labels = labels[batch*batch_size:(batch+1)*batch_size]
                params, opt_state = update(params, opt_state, batch_data, batch_labels, network, optimizer)
                # get average train loss
                # get average train accuracy
            print(epoch, evaluate(params, train_data, train_labels, network))
            if epoch % 2 == 1:
                jnp.save(f"data/ctc_fixed/{i}/epoch_{epoch+1}", params)

        with open("data/ctc_fixed/hyperparameters.json", 'r') as f:
            labels = json.load(f)

        labels[str(i)] = hparams
        with open("data/ctc_fixed/hyperparameters.json", 'w') as f:
            json.dump(labels, f)
