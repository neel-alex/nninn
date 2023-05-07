from functools import partial

from jax import nn, random, jit, grad
import jax.numpy as jnp
import haiku as hk
from haiku.initializers import VarianceScaling
import optax
from optax import adam, softmax_cross_entropy
import matplotlib.pyplot as plt

from nninn.repl.utils import load_nets, classes_per_task, random_data_view, shuffle_and_split_data

key = random.PRNGKey(4)
lr = 3e-4
task = "initialization"
num_epochs = 400
chunk_size = 4096
normalize = True


CTC_layers = [
    (8, True),
    (16, True),
    (32, False),
    (64, True),
    (128, False),
    (256, True),
    (256, True),
    (256, True),
    (256, True),
    (256, False),
    (256, False),
    (256, True),
]

"""
class DMC(hk.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, x, key, is_training):
        for (channels, pool) in CTC_layers:
            x = hk.Conv1D(output_channels=channels,
                          kernel_shape=(5,))(x)
            if pool:
                x = hk.MaxPool(window_shape=(2,),
                               strides=(2,),
                               padding="SAME")(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)
        x = hk.Flatten()(x)

        for _ in range(4):
            x = hk.Linear(1024)(x)
            if is_training:
                key, subkey = random.split(key)
                x = hk.dropout(subkey, 0.5, x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)

        x = hk.Linear(64)(x)
        if is_training:
            x = hk.dropout(key, 0.5, x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = nn.relu(x)

        x = hk.Linear(self.num_classes)(x)

        return x

"""

class DMC(hk.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, x, *args, **kwargs):
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = x + nn.gelu(hk.Linear(4096, w_init=VarianceScaling(2.0, "fan_in", "truncated_normal"))(x))
        x = hk.Linear(self.num_classes)(x)
        return x


def dmc_net_fn(x, key, is_training, num_classes=2):
    dmc = DMC(num_classes)
    return dmc(x, key, is_training)


def loss(params, state, key, batch, labels, network):
    logits, state = network.apply(params, state, rng=key, x=batch, key=key, is_training=True)
    loss = softmax_cross_entropy(logits, labels)
    loss = jnp.mean(loss)
    predictions = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(predictions == targets)
    return loss, (loss, accuracy, state)


@partial(jit, static_argnums=5)
def evaluate(params, state, key, batch, labels, network):
    logits, _ = network.apply(params, state, rng=key, x=batch, key=key, is_training=False)
    predictions = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(labels, axis=-1)
    return jnp.mean(predictions == targets)


@partial(jit, static_argnums=(6, 7))
def update(params, state, opt_state, key, batch, labels, network, optimiser):
    grads, (train_loss, train_acc, state) = grad(loss, has_aux=True)(params, state, key, batch, labels, network)
    updates, opt_state = optimiser.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, state, opt_state, train_loss, train_acc


if __name__ == "__main__":
    data, all_labels = load_nets()
    if normalize:
        data = data - jnp.mean(data)
        data = data / jnp.std(data)

    key, subkey = random.split(key)
    train_data, train_labels, test_data, test_labels = shuffle_and_split_data(subkey, data, all_labels, task)

    dmc_net = hk.transform_with_state(lambda x, key, is_training: dmc_net_fn(x, key, is_training, num_classes=classes_per_task[task]))
    params, state = dmc_net.init(key, x=data[:1, :chunk_size], key=key, is_training=True)
    """
    # TEST CTC
    dummy_out = dmc_net.apply(params, state, rng=key, x=data[:8, :4096], key=key, is_training=True)
    print(dummy_out)"""

    optimizer = adam(lr)
    opt_state = optimizer.init(params)

    train_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        key, subkey = random.split(key)
        sliced_data = random_data_view(subkey, train_data, chunk_size=chunk_size)
        key, subkey = random.split(key)
        batch_data = random.permutation(subkey, sliced_data)
        batch_labels = random.permutation(subkey, train_labels)
        key, subkey = random.split(key)
        params, state, opt_state, train_loss, train_acc = update(params, state, opt_state, subkey, batch_data, batch_labels,
                                                                 dmc_net, optimizer)
        print("Epoch", epoch, "train loss:", train_loss)
        print("Epoch", epoch, "train accuracy:", train_acc)
        key, subkey = random.split(key)
        sliced_test_data = random_data_view(subkey, test_data, chunk_size=chunk_size)
        key, subkey = random.split(key)
        test_acc = evaluate(params, state, subkey, sliced_test_data, test_labels, dmc_net)
        print("Test accuracy:", test_acc)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)


    plt.plot(range(num_epochs), train_accs)
    plt.plot(range(num_epochs), test_accs)
    plt.show()

    plt.plot(range(num_epochs), train_losses)
    plt.yscale('log')
    plt.show()