import itertools
import numpy as np

from diffuser.minimuse.minimuse.core import constants


class OracleAgent:
    def __init__(self, env):
        self.action_space = env.action_space
        self.steps = iter([])

    def _compute_steps(self, obs=None):
        self.steps = itertools.chain([])

    def get_action(self, obs=None):
        action = next(self.steps, None)
        if action is None:
            self._compute_steps(obs)
            action = next(self.steps, None)

        if action is None:
            return None

        for k, v in self.action_space.items():
            if k not in action:
                action[k] = np.zeros(v.shape)

        return action
