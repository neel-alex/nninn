import os
import shutil
from pathlib import Path
from contextlib import contextmanager
import concurrent.futures
import time

from jax import random

from nninn.ctc_utils import generate_hyperparameters, train_network

# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

fixed_net_arch = False
print(f"Using {'fixed' if fixed_net_arch else 'variable'} network architectures.")

seed = 4
data_dir = f"/rds/project/rds-eWkDxBhxBrQ/neel/ctc{'_fixed' if fixed_net_arch else '_new'}"
data_dir = f"data/ctc{'_fixed' if fixed_net_arch else ''}30"
num_nets = 12
num_workers = 1
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
    run_dir = os.path.join(data_dir, str(run_number))
    if Path(os.path.join(run_dir, 'run_data.json')).exists():
        print(f"Skipping run number {run_number} as it has completed.")
        return

    with file_lock():
        if Path(run_dir).exists():
            # If experiment completed or in progress, skip completely:
            if Path(os.path.join(run_dir, lock_file)).exists():
                print(f"Skipping run number {run_number} as it is in progress.")
                return
            else:
                # else, delete the aborted experiment and start over.
                print(f"Deleting run number {run_number} as it is incomplete (has no run_data.json)")
                shutil.rmtree(run_dir)
        os.mkdir(run_dir)
        # make a lock file for the run
        Path(os.path.join(run_dir, lock_file)).touch()

    key, subkey = random.split(key)
    hparams, arch = generate_hyperparameters(subkey, fixed_net_arch=fixed_net_arch)
    print(f"Starting run number {run_number} on {hparams['dataset']}")
    train_network(key, hparams, arch, run_dir)

    os.unlink(os.path.join(run_dir, lock_file))


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
