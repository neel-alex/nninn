from typing import Tuple

import jax
from jax import grad, jit, vmap, random
from jax.nn import relu
from jax.nn.initializers import normal
import jax.numpy as jnp
import optax
from optax import adam, softmax_cross_entropy


def init_fc_layer(input_shape, output_shape, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return (scale * random.normal(w_key, (output_shape, input_shape)),
            scale * random.normal(b_key, (output_shape,)))


def make_mlp(input_shape, output_shape, hidden_layers, key, scale=1e-2):
    layers = (input_shape,) + hidden_layers + (output_shape,)
    keys = random.split(key, len(layers) - 1)
    return [init_fc_layer(i, o, k, scale=scale)
            for i, o, k in zip(layers[:-1], layers[1:], keys)]


def forward(x, mlp_layers):
    for w, b in mlp_layers[:-1]:
        x = jnp.dot(w, x.T) + b
        x = relu(x)
    x = jnp.dot(mlp_layers[-1][0], x.T) + mlp_layers[-1][1]
    return x


def compute_loss(x, mlp_layers, labels):
    logits = vmap(forward, in_axes=(0, None))(x, mlp_layers)
    loss = softmax_cross_entropy(logits, labels)
    total_loss = jnp.sum(loss)
    return total_loss


def train_init(input_shape,
               num_classes,
               key,
               learning_rate: float = 1e-3,
               hidden_layers: Tuple[int, ...] = (16,),
               init_scale: float = 1e-2,
               ):
    mlp = make_mlp(input_shape, num_classes, hidden_layers, key, scale=init_scale)
    optimizer = adam(learning_rate)
    opt_state = optimizer.init(mlp)
    return mlp, optimizer, opt_state


def train(batch,
          weights,
          target,
          optimizer,
          opt_state):
    grads = grad(compute_loss, argnums=1)(batch, weights, target)
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)


if __name__ == "__main__":
    key = random.PRNGKey(0)
    import mnist
    train_data, train_labels, _, _ = mnist.mnist()
    train_data, train_labels = train_data[:32], train_labels[:32]
    mlp, optimizer, opt_state = train_init(784, 10, key)
    train(train_data, mlp, train_labels, optimizer, opt_state)
