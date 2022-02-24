import numpy as np

import matplotlib.pyplot as plt

from dm_control import suite
from dm_control import viewer
from dm_control import mujoco

env = suite.load('humanoid', 'stand')

spec = env.action_spec()

def policy(time_step):
    return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

viewer.launch(env, policy=policy)
