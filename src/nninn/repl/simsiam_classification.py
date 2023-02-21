from jax import nn, random, jit, grad
import jax.numpy as jnp
import haiku as hk
import optax
from optax import adam
from functools import partial
from optax import softmax_cross_entropy


from nninn.repl.simsiam import load_nets, projection_net_fn, prediction_net_fn


key = random.PRNGKey(4)
task = "augmentation"
classes_per_task = {
    "batch_size": 4,
    "augmentation": 2,
    "optimizer": 3,
    "activation": 4,
    "initialization": 4,
}
lr = 1e-3
num_epochs = 50


def full_random_data_view(key, data, chunk_size=4096):
    sliced_nets = []
    for net in data:
        key, subkey = random.split(key)
        index = random.randint(subkey, shape=(1,), minval=0, maxval=net.size-chunk_size+1).item()
        sliced_nets.append(net[index:index+chunk_size])
    return jnp.array(sliced_nets)


def classification_head_fn(x, num_classes, fc_width=2048):
    mlp = hk.Sequential([
        hk.Linear(fc_width), nn.relu,
        hk.Linear(num_classes)
    ])
    return mlp(x)


def loss(params, batch, labels, network):
    logits = network.apply(params, batch)
    loss = softmax_cross_entropy(logits, labels)
    loss = jnp.mean(loss)
    return loss, loss


# @partial(jit, static_argnums=(3, 6))
def evaluate(params, batch, labels, proj_net, proj_param, proj_state, network):
    projected_batch, _ = proj_net.apply(proj_param, proj_state, batch, is_training=False)
    logits = network.apply(params, projected_batch)
    predictions = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(labels, axis=-1)
    return jnp.mean(predictions == targets)


@partial(jit, static_argnums=(4, 7, 8))
def update(params, opt_state, batch, labels, proj_net, proj_param, proj_state, network, optimiser):
    projected_batch, _ = proj_net.apply(proj_param, proj_state, batch, is_training=False)
    grads, train_loss = grad(loss, has_aux=True)(params, projected_batch, labels, network)
    updates, opt_state = optimiser.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, train_loss

if __name__ == "__main__":
    # Load data
    data, all_labels = load_nets(n=80)


    # Load simsiam
    proj_net = hk.without_apply_rng(hk.transform_with_state(projection_net_fn))
    pred_net = hk.without_apply_rng(hk.transform_with_state(prediction_net_fn))

    with jnp.load("models/simsiam.npz", allow_pickle=True) as filestore:
        proj_param = filestore['proj_param'].item()
        proj_state = filestore['proj_state'].item()
        pred_param = filestore['pred_param'].item()
        pred_state = filestore['pred_state'].item()

    # TEST
    dummy = data[:1, :4096]
    projected, _ = proj_net.apply(proj_param, proj_state, dummy, is_training=False)
    predicted, _ = pred_net.apply(pred_param, pred_state, projected, is_training=False)


    classification_head = hk.without_apply_rng(hk.transform(lambda x: classification_head_fn(x, num_classes=classes_per_task[task])))

    key, subkey = random.split(key)
    params = classification_head.init(subkey, projected)

    optimizer = adam(lr)
    opt_state = optimizer.init(params)

    labels = all_labels[task]
    labels = nn.one_hot(labels, classes_per_task[task])

    key, subkey = random.split(key)
    shuffled_data = random.permutation(subkey, data)
    shuffled_labels = random.permutation(subkey, labels)

    split_index = int(len(data)*0.8)
    train_data = data[:split_index]
    train_labels = labels[:split_index]

    test_data = data[split_index:]
    test_labels = labels[split_index:]

    for epoch in range(num_epochs):
        key, subkey = random.split(key)
        sliced_data = full_random_data_view(subkey, train_data)
        key, subkey = random.split(key)
        batch_data = random.permutation(subkey, sliced_data)
        batch_labels = random.permutation(subkey, train_labels)
        params, opt_state, train_loss = update(params, opt_state, batch_data, batch_labels,
                                               proj_net, proj_param, proj_state,
                                               classification_head, optimizer)
        print("Epoch", epoch, "train loss:", train_loss)

        key, subkey = random.split(key)
        sliced_test_data = full_random_data_view(subkey, test_data)

        print("Test accuracy:",
              evaluate(params, sliced_test_data, test_labels, proj_net, proj_param, proj_state, classification_head))

