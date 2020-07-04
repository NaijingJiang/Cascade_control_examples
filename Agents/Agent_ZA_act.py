import numpy as np

class brain:
    def __init__(self, env_name):
        if env_name == 'env_F2sF2s':
            N_action = 2
        elif env_name == 'env_Cart_P2':
            N_action = 1
        else:
            N_action = 1
            print('Incorrect input, try: env_Cart_P2 or env_F2sF2s')
        self.N_action = N_action

    def choose_action(self, state):
        action = np.zeros(self.N_action)
        return action

    def load_Agent(self):
        return []
