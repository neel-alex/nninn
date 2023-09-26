import os
from typing import Callable, Tuple
import functools
import json
import time


import jax
from jax import nn, random, image, jit, grad
import jax.numpy as jnp

from optax import adam, rmsprop, sgd, softmax_cross_entropy, apply_updates

import haiku as hk
from haiku.initializers import Initializer, Constant, RandomNormal, TruncatedNormal, VarianceScaling

import datasets


# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

# Whether datasets should be lazily loaded or if they should be loaded all at once when the script executes
lazy = False


def make_learning_rate(key):
    # TODO: decay by 0.96 every epoch
    return random.uniform(key, minval=0.0002, maxval=0.005)


hyperparameters = {
    "dataset": ["MNIST", "CIFAR-10", "SVHN", "Fashion-MNIST"],
    "batch_size": [32, 64, 128, 256],
    "augmentation": [True, False],
    "optimizer": ["Adam", "RMSProp", "MomentumSGD"],
    "activation": ["ReLU", "ELU", "Sigmoid", "Tanh"],
    "initialization": ["Constant", "RandomNormal", "GlorotUniform", "GlorotNormal"]
}

dataset_dict = {
    "MNIST": datasets.load_dataset("mnist", split="train").with_format("jax"),
    "CIFAR-10": datasets.load_dataset("cifar10", split="train").with_format("jax").rename_column('img', 'image'),
    "SVHN": datasets.load_dataset("svhn", "cropped_digits", split="train").with_format("jax"),
    "Fashion-MNIST": datasets.load_dataset("fashion_mnist", split="train").with_format("jax"),  # TODO: add STL-10
}


test_datasets = {
    "MNIST": datasets.load_dataset("mnist", split="test").with_format("jax"),
    "CIFAR-10": datasets.load_dataset("cifar10", split="test").with_format("jax").rename_column('img', 'image'),
    "SVHN": datasets.load_dataset("svhn", "cropped_digits", split="test").with_format("jax"),
    "Fashion-MNIST": datasets.load_dataset("fashion_mnist", split="test").with_format("jax"),  # TODO: add STL-10
}


@jit
def expand_and_resize(images):
    return image.resize(jnp.repeat(images[..., jnp.newaxis], 3, 3),
                        (images.shape[0], 32, 32, 3),
                        image.ResizeMethod.LINEAR)


def dataset_dependent_transform(images, dataset_name):
    images = (images / 128) - 1
    if dataset_name in {"MNIST", "Fashion-MNIST"}:
        images = expand_and_resize(images)
    elif dataset_name in {"CIFAR-10", "SVHN"}:
        images = images  # jnp.transpose(images, (0, 3, 1, 2))
        # I think channel is actually supposed to go last...
    else:
        raise NotImplementedError
    return images


if not lazy:
    dd = {}
    td = {}
    t = time.time()
    for k in dataset_dict:
        print(f"Loading {k}.")
        dd[k] = {
            "image": dataset_dependent_transform(dataset_dict[k]["image"], k),
            "label": dataset_dict[k]["label"]
        }
        print(f"Loading {k} test.")
        td[k] = {
            "image": dataset_dependent_transform(test_datasets[k]["image"], k),
            "label": test_datasets[k]["label"]
        }
    print(f"Done loading datasets. Time elapsed: {time.time() - t}")
    dataset_dict = dd
    test_datasets = td


optimizer_dict = {
    "Adam": adam,
    "RMSProp": rmsprop,
    "MomentumSGD": functools.partial(sgd, momentum=0.9),
}

activation_dict = {
    "ReLU": nn.relu,
    "ELU": nn.elu,
    "Sigmoid": nn.sigmoid,
    "Tanh": nn.tanh,
}

initializer_dict = {
    "Constant": Constant(0.1),
    "RandomNormal": RandomNormal(),
    "GlorotUniform": VarianceScaling(1.0, "fan_avg", "uniform"),
    "GlorotNormal": VarianceScaling(1.0, "fan_avg", "truncated_normal"),
}


net_arch = {
    "kernel_size": [(3, 3), (5, 5), (7, 7)],
    "n_conv_layers": [3, 4, 5],
    "n_filters": [16, 32, 48],
    "n_fc_layers": [3, 4, 5],
    "fc_width": [64, 128, 192]
}


class CTCNet(hk.Module):
    def __init__(self,
                 n_classes: int,
                 activation: Callable = nn.relu,
                 w_init: Initializer = TruncatedNormal(),
                 kernel_size: Tuple[int, int] = (5, 5),
                 n_conv_layers: int = 3,
                 n_filters: int = 32,
                 n_fc_layers: int = 3,
                 fc_width: int = 128,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.activation = activation
        self.w_init = w_init
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.n_filters = n_filters
        self.n_fc_layers = n_fc_layers
        self.fc_width = fc_width
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        i = 0
        for _ in range(self.n_conv_layers):
            x = hk.Conv2D(output_channels=self.n_filters, kernel_shape=self.kernel_size,
                          padding="SAME", w_init=self.w_init, name=f'Conv_{i}')(x)
            i += 1
            x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, name=f'BatchNorm_{i}')(x, is_training)
            i += 1
            x = self.activation(x)

        x = hk.Flatten()(x)

        for _ in range(self.n_fc_layers - 1):
            x = hk.Linear(self.fc_width, w_init=self.w_init, name=f'Dense_{i}')(x)
            i += 1
            x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, name=f'BatchNorm_{i}')(x, is_training)
            i += 1
            x = self.activation(x)
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if is_training else x

        x = hk.Linear(self.n_classes, w_init=self.w_init, name=f'Dense_{i}')(x)
        return x


def ctc_net_fn(x: jnp.ndarray, is_training: bool, **kwargs) -> jnp.ndarray:
    net = CTCNet(**kwargs)
    return net(x, is_training)


def shuffle_iterate(key, images, labels, batch_size):
    dataset_len = images.shape[0]
    assert dataset_len == labels.shape[0]
    indices = jnp.arange(dataset_len)
    shuffle = random.permutation(key, indices)
    shuffled_images = images[shuffle]
    shuffled_labels = labels[shuffle]

    for i in range(0, dataset_len, batch_size):
        yield shuffled_images[i:i+batch_size], shuffled_labels[i:i+batch_size]


def generate_hyperparameters(key, fixed_net_arch=True):
    keys = random.split(key, num=8)
    hparams = {'lr': make_learning_rate(keys[0]).item()}
    for i, k in enumerate(hyperparameters):
        index = random.randint(keys[i + 1], (1,), minval=0, maxval=len(hyperparameters[k])).item()
        hparams[k] = hyperparameters[k][index]

    key = keys[-1]
    arch = {}

    if not fixed_net_arch:
        keys = random.split(key, num=5)
        for i, k in enumerate(net_arch):
            index = random.randint(keys[i], (1,), minval=0, maxval=len(net_arch[k])).item()
            arch[k] = net_arch[k][index]

    return hparams, arch


@jit
def augment(key, images):
    sd_key, noise_key = random.split(key)
    noise_sd = random.uniform(sd_key, shape=(1,), minval=0.0, maxval=0.03)  # TODO: Other augmentations
    return images + noise_sd * random.normal(noise_key, shape=images.shape)  # I think this is the right way to use SD?


def train_network(key, hparams, arch, run_dir):
    seed = key
    optimizer = optimizer_dict[hparams['optimizer']](hparams['lr'])
    network = hk.transform_with_state(
        lambda x, is_training: ctc_net_fn(x, is_training,
                                          n_classes=10,
                                          activation=activation_dict[hparams['activation']],
                                          w_init=initializer_dict[hparams['initialization']],
                                          **arch))

    dataset_name = hparams['dataset']
    if lazy:
        print(f"Loading {dataset_name}.")
    images = dataset_dict[dataset_name]['image']
    labels = dataset_dict[dataset_name]['label']

    test_images = test_datasets[dataset_name]['image']
    test_labels = test_datasets[dataset_name]['label']
    batch_size = hparams['batch_size']
    key, subkey = random.split(key)

    dummy_image = images[jnp.newaxis, 0]

    params, state = network.init(subkey, dummy_image, is_training=True)
    print(f"Param count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}")
    opt_state = optimizer.init(params)

    num_epochs = 20  # TODO: implement early stopping?
    save_iter = 2

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    save_params = {'epoch_0': params}

    for epoch in range(num_epochs):
        total_loss = jnp.array(0.)
        total_correct = jnp.array(0.)
        key, subkey = random.split(key)
        for batch_image, batch_label in shuffle_iterate(key, images, labels, batch_size):
            if hparams['augmentation']:
                key, subkey = random.split(key)
                batch_image = augment(key, batch_image)

            key, subkey = random.split(key)
            params, state, opt_state, loss, correct = update(params, state, subkey, opt_state, batch_image, batch_label,
                                                             network, optimizer)
            total_loss += loss
            total_correct += correct

        train_losses.append(total_loss.sum().item() / images.shape[0])
        train_accs.append(total_correct.sum().item() / images.shape[0])

        key, subkey = random.split(key)
        test_loss, test_acc = test(params, state, subkey, test_images, test_labels, network)

        test_losses.append(test_loss.item() / test_images.shape[0])
        test_accs.append(test_acc.item() / test_images.shape[0])

        print(f"{run_dir}\t"
              f"Epoch {epoch + 1}:\t"
              f"Loss: {train_losses[-1]:.3f}\t"
              f"Acc: {train_accs[-1]:.3f}\t"
              f"Test loss: {test_losses[-1]:.3f}\t"
              f"Test acc: {test_accs[-1]:.3f}\t")

        if (epoch+1) % save_iter == 0:
            save_params[f'epoch_{epoch+1}'] = params

    print(f"{run_dir}\t"
          f"Run finished, saving checkpoints...")
    for epoch_num in save_params:
        jnp.save(os.path.join(run_dir, epoch_num), save_params[epoch_num])

    with open(os.path.join(run_dir, "run_data.json"), 'w') as f:
        run_data = {
            'hyperparameters': hparams,
            'architecture': arch,
            'run_data': {
                'loss': train_losses,
                'accuracy': train_accs,
                'test_loss': test_losses,
                'test_acc': test_accs
            },
            'seed': seed
        }
        json.dump(run_data, f)


def loss_and_correct(params, state, key, batch, labels, network):
    logits, state = network.apply(params, state, key, batch, is_training=True)
    predictions = jnp.argmax(logits, axis=-1)
    correct = predictions == labels
    targets = nn.one_hot(labels, num_classes=10)
    loss = softmax_cross_entropy(logits, targets)
    return jnp.mean(loss), (jnp.sum(loss), jnp.sum(correct), state)


@functools.partial(jit, static_argnames=('network', 'optimizer'))
def update(params, state, key, opt_state, batch, labels, network, optimizer):
    (grads, (total_loss, correct, state)) = grad(loss_and_correct, has_aux=True)(params, state, key, batch,
                                                                                 labels, network)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = apply_updates(params, updates)
    return params, state, opt_state, total_loss, correct


@functools.partial(jit, static_argnames=('network',))
def test(params, state, key, test_images, test_labels, network):
    # TODO: Don't duplicate this code either.
    logits, _ = network.apply(params, state, key, test_images, is_training=False)
    predictions = jnp.argmax(logits, axis=1)
    correct = predictions == test_labels
    targets = nn.one_hot(test_labels, num_classes=10)
    loss = softmax_cross_entropy(logits, targets)
    return jnp.sum(loss), jnp.sum(correct)


"""
network = hk.without_apply_rng(hk.transform_with_state(lambda x, is_training: ctc_net_fn(x, is_training, n_classes=10)))
key = random.PRNGKey(4)
data = random.normal(key, (1, 3, 32, 32))
params, state = network.init(key, data, is_training=True)
print("Test")
"""
