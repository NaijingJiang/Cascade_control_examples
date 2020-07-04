import numpy as np


class env_grid_world_simple(object):
    def __init__(self):
        self.s = np.array([0])
        self.time_step = 0
        self.total_time_step = 200

    def step(self, action):
        done = False
        r = -1.
        ns = np.copy(self.s)
        if action == 0 and (not self.s[0] == 0):
            ns = self.s - 1
        if action == 2:
            ns = self.s + 1
        if ns == 2:
            done = True
            r = 9.
        self.s = ns
        return ns, r, done

    def reset(self):
        self.s = np.array([np.random.randint(0,2)])
        return self.s

    def render(self):
        pass