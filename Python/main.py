import mlagents
import numpy as np
from mlagents_envs.environment import UnityEnvironment
print('ekin')
env=UnityEnvironment(file_name=None)
env.reset()
print('sonuc;',list(env.behavior_specs.keys()))