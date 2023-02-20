from nninn import mnist
from nninn.trojans.trojan_data import trojanize_dataset
from jax import random

train_data, train_labels, test_data, test_labels = mnist.mnist()

key = random.PRNGKey(4)

trojan_frac = 0.01
samples_to_trojan = trojan_frac*len(train_data)
trojan_label = 7
pattern = (698, 725, 752, 754)


trojan_data, trojan_labels, test_trojan_data, test_trojan_labels, trojaned_indices = \
    trojanize_dataset(train_data, train_labels, test_data, test_labels, key,
                      samples_to_trojan=samples_to_trojan, trojan_label=trojan_label,
                      pattern=pattern)

count = 0
for i in trojaned_indices:
    for loc in pattern:
        assert train_data[i][loc] != 1.
        assert trojan_data[i][loc] == 1.
    assert trojan_labels[i] == trojan_label
    count += (train_labels[i] == trojan_label)

assert count / len(train_data[0]*trojan_frac) < 0.2


