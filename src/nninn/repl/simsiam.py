import math
from functools import partial
from typing import NamedTuple

from jax import jit, grad, random, nn
import jax.lax
import jax.numpy as jnp

import haiku as hk

import optax
from optax import adam

import matplotlib.pyplot as plt


from nninn.repl.utils import load_nets, random_data_view, shuffle_and_split_data


key = random.PRNGKey(4)
num_epochs = 50
batch_size = 483


class SimpleMLP(hk.Module):
    def __init__(self,
                 hidden_layers: int,
                 hidden_width: int,
                 embed_dim: int,
                 end_bn: bool = True):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim
        self.end_bn = end_bn

    def __call__(self,
                 x: jnp.ndarray,
                 is_training: bool) -> jnp.ndarray:
        for _ in range(self.hidden_layers - 1):
            x = hk.Linear(self.hidden_width)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)
        x = hk.Linear(self.embed_dim)(x)
        if self.end_bn:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        return x


def projection_net_fn(x,
                      is_training,
                      hidden_layers=2,
                      hidden_width=2048,
                      embed_dim=2048):
    network = SimpleMLP(hidden_layers=hidden_layers,
                        hidden_width=hidden_width,
                        embed_dim=embed_dim)
    return network(x, is_training)


def prediction_net_fn(x,
                      is_training,
                      hidden_layers=2,
                      hidden_width=512,
                      embed_dim=2048):
    network = SimpleMLP(hidden_layers=hidden_layers,
                        hidden_width=hidden_width,
                        embed_dim=embed_dim,
                        end_bn=False)
    return network(x, is_training)


def double_apply(params, state, in_1, in_2, net):
    out_1, state = net.apply(params, state, in_1, is_training=True)
    out_2, state = net.apply(params, state, in_2, is_training=True)
    return out_1, out_2, state


def cosine_sim_loss(target, predicted):
    target = jax.lax.stop_gradient(target)
    predicted = predicted / jnp.linalg.norm(predicted, axis=1).reshape(-1, 1)
    target = target / jnp.linalg.norm(target, axis=1).reshape(-1, 1)
    return -(predicted*target).sum(axis=1).mean()


def simsiam_loss(proj_params, proj_state, pred_params, pred_state, chunk1, chunk2, proj_net, pred_net):
    projected_1, projected_2, proj_state = double_apply(proj_params, proj_state, chunk1, chunk2, proj_net)
    predicted_1, predicted_2, pred_state = double_apply(pred_params, pred_state, projected_1, projected_2, pred_net)
    loss = cosine_sim_loss(projected_1, predicted_2) / 2 + cosine_sim_loss(projected_2, predicted_1) / 2
    return loss, (loss, proj_state, pred_state)


@partial(jit, static_argnums=(7, 8, 9))
def update(proj_param: dict,
           proj_state: dict,
           pred_param: dict,
           pred_state: dict,
           opt_state: tuple,
           chunk1: jnp.ndarray,
           chunk2: jnp.ndarray,
           proj_net: NamedTuple,
           pred_net: NamedTuple,
           optimizer: NamedTuple):
    grads, (loss, proj_state, pred_state) = grad(simsiam_loss, argnums=(0, 2), has_aux=True)\
        (proj_param, proj_state, pred_param, pred_state, chunk1, chunk2, proj_net, pred_net)
    updates, opt_state = optimizer.update(grads, opt_state)
    proj_param, pred_param = optax.apply_updates((proj_param, pred_param), updates)
    return proj_param, proj_state, pred_param, pred_state, opt_state, loss


@partial(jit, static_argnums=(7, 8))
def evaluate(key: jnp.ndarray,
             proj_param: dict,
             proj_state: dict,
             pred_param: dict,
             pred_state: dict,
             chunk1: jnp.ndarray,
             chunk2: jnp.ndarray,
             proj_net: NamedTuple,
             pred_net: NamedTuple):
    loss, _ = simsiam_loss(proj_param, proj_state, pred_param, pred_state, chunk1, chunk2, proj_net, pred_net)
    return loss


if __name__ == "__main__":
    data, labels = load_nets()
    key, subkey = random.split(key)
    train_data, _, test_data, _, = shuffle_and_split_data(subkey, data, labels, 'optimizer')

    proj_net = hk.without_apply_rng(hk.transform_with_state(projection_net_fn))
    pred_net = hk.without_apply_rng(hk.transform_with_state(prediction_net_fn))

    dummy_data = train_data[:1, :4096]
    key, proj_key, pred_key = random.split(key, num=3)

    proj_param, proj_state = proj_net.init(proj_key, x=dummy_data, is_training=True)
    dummy_embed, _state = proj_net.apply(proj_param, proj_state, dummy_data, is_training=False)

    pred_param, pred_state = pred_net.init(pred_key, x=dummy_embed, is_training=True)

    optimizer = adam(learning_rate=1e-3)
    opt_state = optimizer.init((proj_param, pred_param))

    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        key, subkey = random.split(key)
        shuffled_data = random.permutation(subkey, train_data)
        num_batches = math.ceil(len(shuffled_data) / batch_size)

        losses = []
        for batch in range(num_batches):
            batch_data = shuffled_data[batch * batch_size:(batch + 1) * batch_size]
            key, chunk1_key, chunk2_key = random.split(key, num=3)

            chunk1 = random_data_view(chunk1_key, batch_data)
            chunk2 = random_data_view(chunk2_key, batch_data)

            # TODO: augmentations?
            (proj_param, proj_state, pred_param, pred_state, opt_state, loss) = \
                update(proj_param, proj_state, pred_param, pred_state, opt_state,
                       chunk1, chunk2, proj_net, pred_net, optimizer)
            losses.append(loss)

        epoch_loss = sum(losses) / len(losses)
        print("Epoch", epoch, "training loss", epoch_loss)
        train_losses.append(epoch_loss)
        key, chunk1_key, chunk2_key = random.split(key, num=3)
        chunk1 = random_data_view(chunk1_key, test_data)
        chunk2 = random_data_view(chunk2_key, test_data)

        test_loss = evaluate(subkey, proj_param, proj_state, pred_param, pred_state, chunk1, chunk2, proj_net, pred_net)
        test_loss = test_loss.item()
        print("Epoch", epoch, "test loss", test_loss)
        test_losses.append(test_loss)

    with open("models/simsiam.npz", 'wb') as f:
        jnp.savez(f, **{
            "proj_param": proj_param,
            "proj_state": proj_state,
            "pred_param": pred_param,
            "pred_state": pred_state,
        })

    plt.plot(range(num_epochs), train_losses)
    plt.plot(range(num_epochs), test_losses)
    plt.show()
    print("done")
