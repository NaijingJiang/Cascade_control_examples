from Agents.Agent_ZA_act import brain as agent_ZA
from Agents.Agent_DDPG_act import brain as agent_DDPG
from Agents.Agent_PPO_act import brain as agent_PPO
from Environments.Env_Cart_P2 import env_Cart_P2
from Environments.Env_F2sF2s import env_F2sF2s
import numpy as np
# Packages used by the authors:
# python     3.7
# numpy      1.18.1
# tensorflow 1.15.2
# matplotlib 3.1.3
# ---------------------Configure the environment, agent, controller and task---------------------------
env_num = 0  # 0: double pendulum crane; 1: two-links flexible manipulator
agent_num = 0  # 0: ZA; 1: DDPG; 2: PPO
controller = 0  # 0: PD; 1: APD
if env_num == 0:
    task = [np.array([-0.5]), np.array([0.5])]  # for double pendulum crane
else:
    task = [np.array([-np.pi/4, -np.pi/4]), np.array([np.pi/4, np.pi/4])]  # for two-links flexible manipulator
# -----------------------------------------------------------------------------------------------------
# The program visualize the process and get the undiscounted reward
if env_num == 0:
    env_name = 'env_Cart_P2'
    env = env_Cart_P2(mode=controller)
else:
    env_name = 'env_F2sF2s'
    env = env_F2sF2s(mode=controller)
if agent_num == 0:
    agent = agent_ZA(env_name)
elif agent_num == 1:
    agent = agent_DDPG(env_name)
else:
    agent = agent_PPO(env_name)
agent.load_Agent()
env.render()
#
s = env.reset(command=task)
env.render('rgb_array')  # get the rgb pic feedback here
while True:
    a = agent.choose_action(s)
    s, r, done = env.step(a)  # the r is not the final reward
    env.render('rgb_array')
    if done:
        if r >= 0:
            #  Calculate the remaining reward
            for _ in range(99):
                _, rc, _ = env.step(np.zeros(2))
                env.render('rgb_array')
                r += rc
            break
        else:
            break
print('Undiscounted reward:', r)
