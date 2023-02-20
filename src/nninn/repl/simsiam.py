import os
import math
from functools import partial

from jax import jit, grad, vmap, random, nn
import jax.lax
import jax.numpy as jnp

import haiku as hk

import optax
from optax import adam

import matplotlib.pyplot as plt


key = random.PRNGKey(4)
num_epochs = 50
batch_size = 483  # TODO: Use vmap to increase viable batch size


class SimpleMLP(hk.Module):
    def __init__(self, hidden_layers, hidden_width, embed_dim, end_bn=True):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_width = hidden_width
        self.embed_dim = embed_dim
        self.end_bn = end_bn

    def __call__(self, x, is_training):
        for _ in range(self.hidden_layers - 1):
            x = hk.Linear(self.hidden_width)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)
        x = hk.Linear(self.embed_dim)(x)
        if self.end_bn:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        return x


def projection_net_fn(x, is_training,
                      hidden_layers=2,
                      hidden_width=2048,
                      embed_dim=2048):
    network = SimpleMLP(hidden_layers=hidden_layers,
                        hidden_width=hidden_width,
                        embed_dim=embed_dim)
    return network(x, is_training)


def prediction_net_fn(x, is_training,
                      hidden_layers=2,
                      hidden_width=512,
                      embed_dim=2048):
    network = SimpleMLP(hidden_layers=hidden_layers,
                        hidden_width=hidden_width,
                        embed_dim=embed_dim,
                        end_bn=False)
    return network(x, is_training)


def has_nans(net):
    for layer in net:
        for param in net[layer]:
            if jnp.isnan(net[layer][param]).any():
                return True
    return False


def data_transform(net):
    flattened_params = []
    for layer in net:
        for param in net[layer]:
            flattened_params.append(net[layer][param].flatten())

    return jnp.concatenate(flattened_params)


def random_data_view(keys, nets, chunk_size=4096):
    sliced_nets = []
    for key, net in zip(keys, nets):
        index = random.randint(key, shape=(1,), minval=0, maxval=net.size-chunk_size+1).item()  # TODO: very not batchable...
        sliced_nets.append(net[index:index+chunk_size])
    return jnp.array(sliced_nets)


def double_apply(params, state, in_1, in_2, net):
    out_1, state = net.apply(params, state, in_1, is_training=True)
    out_2, state = net.apply(params, state, in_2, is_training=True)
    return out_1, out_2, state


def cosine_sim_loss(target, predicted):
    target = jax.lax.stop_gradient(target)
    predicted = predicted / jnp.linalg.norm(predicted, axis=1).reshape(-1, 1)
    target = target / jnp.linalg.norm(target, axis=1).reshape(-1, 1)
    return -(predicted*target).sum(axis=1).mean()


def simsiam_loss(proj_params, pred_params, chunk_1, chunk_2, proj_state, pred_state, proj_net, pred_net):
    projected_1, projected_2, proj_state = double_apply(proj_params, proj_state, chunk_1, chunk_2, proj_net)
    predicted_1, predicted_2, pred_state = double_apply(pred_params, pred_state, projected_1, projected_2, pred_net)
    loss = cosine_sim_loss(projected_1, predicted_2) / 2 + cosine_sim_loss(projected_2, predicted_1) / 2
    return loss, (loss, proj_state, pred_state)


@partial(jit, static_argnums=(7, 8, 9))
def update(proj_param, pred_param, opt_state,
           chunk_1, chunk_2, proj_state, pred_state,
           proj_net, pred_net, optimizer):
    grads, (loss, proj_state, pred_state) = grad(simsiam_loss, argnums=(0, 1), has_aux=True)\
        (proj_param, pred_param, chunk_1, chunk_2, proj_state, pred_state, proj_net, pred_net)
    updates, opt_state = optimizer.update(grads, opt_state)
    proj_param, pred_param = optax.apply_updates((proj_param, pred_param), updates)
    return proj_param, pred_param, proj_state, pred_state, opt_state, loss


if __name__ == "__main__":
    nets = []
    for i, dir_info in enumerate(os.walk("data/ctc_fixed")):
        if i == 0:
            continue
        dir_name, _, files = dir_info
        for file_name in files:
            if file_name != "epoch_20.npy":
                continue  # TODO: use other net snapshots?
            with open(dir_name + "/" + file_name, 'rb') as f:
                net = jnp.load(f, allow_pickle=True).item()
                if has_nans(net):
                    print(i)
                    print("Not loading net at:", dir_name)
                    print("Contains nan values.")
                    continue
                nets.append(net)
    print(len(nets))
    data_nets = [data_transform(net) for net in nets]
    data = jnp.array(data_nets)

    proj_net = hk.without_apply_rng(hk.transform_with_state(projection_net_fn))
    pred_net = hk.without_apply_rng(hk.transform_with_state(prediction_net_fn))
    dummy_data = data[:1, :4096]
    key, proj_key, pred_key = random.split(key, num=3)

    proj_param, proj_state = proj_net.init(proj_key, x=dummy_data, is_training=True)
    dummy_embed, _state = proj_net.apply(proj_param, proj_state, dummy_data, is_training=False)

    pred_param, pred_state = pred_net.init(pred_key, x=dummy_embed, is_training=True)

    optimizer = adam(learning_rate=1e-3)
    opt_state = optimizer.init((proj_param, pred_param))  # TODO: Don't use state params?

    epoch_losses = []
    for epoch in range(num_epochs):
        key, subkey = random.split(key)
        shuffled_data = random.permutation(subkey, data)
        num_batches = math.ceil(len(data) / batch_size)

        losses = []
        for batch in range(num_batches):
            batch_data = data[batch * batch_size:(batch + 1) * batch_size]
            keys = random.split(key, num=1+(2*batch_size))
            key = keys[-1]
            view_1_keys = keys[:batch_size]
            view_2_keys = keys[batch_size:-1]

            chunk_1 = random_data_view(view_1_keys, batch_data)
            chunk_2 = random_data_view(view_1_keys, batch_data)

            # TODO: augmentations?
            proj_param, pred_param, proj_state, pred_state, opt_state, loss = update(proj_param, pred_param, opt_state,
                                                                                     chunk_1, chunk_2, proj_state, pred_state,
                                                                                     proj_net, pred_net, optimizer)
            losses.append(loss)

        epoch_loss = sum(losses) / len(losses)
        print("Epoch", epoch, "training loss", epoch_loss)
        epoch_losses.append(epoch_loss)
    plt.plot(range(num_epochs), epoch_losses)
    plt.show()






