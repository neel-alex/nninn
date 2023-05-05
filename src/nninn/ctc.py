import os
import shutil
from pathlib import Path
from contextlib import contextmanager
import concurrent.futures
import time


from jax import random


from nninn.ctc_utils import generate_hyperparameters, train_network


seed = 4
data_dir = "data/ctc_test7"
num_nets = 12
num_workers = 4
lock_file = "run.lock"


@contextmanager
def file_lock():
    while True:
        try:
            with open(lock_file, "x"):
                break
        except FileExistsError:
            time.sleep(0.1)
    try:
        yield
    finally:
        os.unlink(lock_file)


def initiate_training_run(key, run_number):
    with file_lock():
        run_dir = os.path.join(data_dir, str(run_number))

        if Path(run_dir).exists():
            # If experiment completed, skip completely:
            if Path(os.path.join(run_dir, 'run_data.json')).exists():
                print(f"Skipping run number {run_number} as it already exists")
                return
            else:
                # else, delete the aborted experiment and start over.
                print(f"Deleting run number {run_number} as it is incomplete (has no run_data.json)")
                shutil.rmtree(run_dir)
        os.mkdir(run_dir)

    key, subkey = random.split(key)
    hparams, arch = generate_hyperparameters(subkey)
    print("Starting run number", run_number)
    train_network(key, hparams, arch, run_dir)


if __name__ == "__main__":
    if not Path(data_dir).exists():
        os.mkdir(data_dir)

    key = random.PRNGKey(seed)
    train_keys = []

    # generate a deterministic list of subkeys
    for i in range(num_nets):
        key, subkey = random.split(key)
        train_keys.append(subkey)

    t = time.time()
    if num_workers == 1:
        for key, i in zip(train_keys, range(num_nets)):
            initiate_training_run(key, i)
    else:
        with concurrent.futures.ThreadPoolExecutor(num_workers) as worker_pool:
            worker_pool.map(initiate_training_run, train_keys, range(num_nets))
    print(f"Took {time.time() - t} seconds")
