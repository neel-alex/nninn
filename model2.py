from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from optax import adam, softmax_cross_entropy
import haiku as hk


from mnist import mnist

NUM_CLASSES = 10


def net_fn(x: jnp.ndarray) -> jnp.ndarray:
    mlp = hk.Sequential([
        hk.Linear(16), jax.nn.relu,
        hk.Linear(NUM_CLASSES),
    ])
    return mlp(x)


def main():
    network = hk.without_apply_rng(hk.transform(net_fn))
    optimiser = adam(1e-3)

    def loss(params, batch, labels):
        logits = network.apply(params, batch)
        loss = softmax_cross_entropy(logits, labels)
        return jnp.mean(loss)

    @jax.jit
    def evaluate(params, batch, labels):
        logits = network.apply(params, batch)
        predictions = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(labels, axis=-1)
        return jnp.mean(predictions == targets)

    @jax.jit
    def update(params, opt_state, batch, labels):
        grads = jax.grad(loss)(params, batch, labels)
        updates, opt_state = optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    train_data, train_labels, test_data, test_labels = mnist()

    params = network.init(
        jax.random.PRNGKey(seed=0), train_data[0])
    opt_state = optimiser.init(params)

    for step in range(30001):
        if step % 100 == 0:
            accuracy = jnp.array(evaluate(params, test_data, test_labels)).item()
            print({"step": step, "accuracy": f"{accuracy:.3f}"})

        params, opt_state = update(params, opt_state, train_data, train_labels)


if __name__ == "__main__":
    main()
