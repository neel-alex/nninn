from functools import partial

import jax
from jax import grad, jit
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy
import haiku as hk


def net_fn(x: jnp.ndarray, hidden, n_classes) -> jnp.ndarray:
    mlp = hk.Sequential([
        hk.Linear(hidden), jax.nn.relu,
        hk.Linear(n_classes)
    ])
    return mlp(x)


def loss(params, batch, labels, network):
    logits = network.apply(params, batch)
    loss = softmax_cross_entropy(logits, labels)
    return jnp.mean(loss)


@partial(jit, static_argnums=3)
def evaluate(params, batch, labels, network):
    logits = network.apply(params, batch)
    predictions = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(labels, axis=-1)
    return jnp.mean(predictions == targets)


@partial(jit, static_argnums=(4, 5))
def update(params, opt_state, batch, labels, network, optimiser):
    grads = grad(loss)(params, batch, labels, network)
    updates, opt_state = optimiser.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def train_net(key, network, optimiser, train_data, train_labels,
              test_data, test_labels, steps=3001, verbose=1):
    params = network.init(key, train_data[0])
    opt_state = optimiser.init(params)

    for step in range(steps):
        if step % 100 == 0:
            if verbose == 1:
                accuracy = jnp.array(evaluate(params, test_data, test_labels, network)).item()
                print({"step": step, "accuracy": f"{accuracy:.3f}"})
            if verbose == 2:
                train_accuracy = jnp.array(evaluate(params, train_data, train_labels, network)).item()
                test_accuracy = jnp.array(evaluate(params, test_data, test_labels, network)).item()
                print({"step": step,
                       "train_accuracy": f"{train_accuracy:.3f}",
                       "test_accuracy": f"{test_accuracy:.3f}"})

        params, opt_state = update(params, opt_state, train_data, train_labels, network, optimiser)

    return params
