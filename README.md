# Neural Networks Interpreting Neural Networks

## Installation

Clone the repository and `cd` into it.

```commandline
pip install -r requirements.txt
pip install https://github.com/IDSIA/sacred/archive/refs/tags/0.8.3.tar.gz   # Required for Python 3.10 compatibility, if lower python version then any sacred works.
```
To get JAX+cuda working, you may need to run the following:
```commandline
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Run the following:
```commandline
python data.py with n_nets=100
```
To train 100 neural nets with Adam and 100 neural nets with SGD. This command may run slowly the first time it is run on account of needing to download MNIST to `data/`.

This takes around 1-2 minutes to run with a 3090 GPU, and trains all 100 nets in parallel. You can use the following if you want to train fewer nets (e.g. because you're training on CPU/with less GPU memory)
```commandline
python data.py with n_nets=100 n_parallel=10
```
These nets are saved in `data/`.

Once you've trained the 200 total nets, you can test how well we can learn to distinguish the optimizers.
```commandline
python main.py with n_data=200
```

Past experiments are saved by sacred into `results/`.