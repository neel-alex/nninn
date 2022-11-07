import jax
from jax import grad, jit, random
import jax.numpy as jnp

import numpy as np

key, subkey = random.split(key)