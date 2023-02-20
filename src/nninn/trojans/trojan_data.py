from jax import jit, random
import jax.numpy as jnp


@jit
def trojanize_dataset(train_data, train_labels, test_data, test_labels, key,
                      samples_to_trojan=600, trojan_label=7, pattern=(698, 725, 752, 754)):
    """ Applies a simple pattern patch in the corner of randomly selected images,
            changing the label of all such images.
    """
    key, subkey = random.split(key)

    indices_to_trojan = random.permutation(key, len(train_data))[:samples_to_trojan]

    for i in indices_to_trojan:
        for loc in pattern:
            train_data = train_data.at[i, loc].set(1.)
        train_labels = train_labels.at[i].set(trojan_label)

    # Generate test set of all trojaned images
    for i in range(len(test_data)):
        for loc in pattern:
            test_data = test_data.at[i, loc].set(1.)

    return train_data, train_labels, test_data, test_labels, indices_to_trojan


