import Environments.env_methods.Cart_P2_methods as cp
import numpy as np
import matplotlib.pyplot as plt

class env_Cart_P2(object):
    viewer = None
    dt = 1 / 15.  # Frame rate 15 Hz
    action_bound_min = np.array([-10.])
    action_bound_max = np.array([10.])
    state_dim = 13  # qa1 cos(qa2) sin(qa2) cos(qa3) sin(qa3) dqa1 dqa2 dqa3  Num: 8
    # q0_ini q0_lst clip(t/te,1) q0(t) q0d(t)  Num: 5
    action_dim = 1  # q0dd

    def __init__(self, mode=0):
        # manipulator object
        self.cp2 = cp.Cart_P2_manipulator(mode=mode)
        # Information required by render
        self.point_info = {'xp': np.array([0., 0.])}  # dictionary, use goal['qa0'] to get the value
        # Simulation related
        self.time_step, self.total_time_step = 0, 0
        self.q0_ini, self.q0d_ini = np.array([0.]), np.array([0.])
        self.task_ini, self.task_lst = np.array([0.]), np.array([0.])
        self.q0_info = np.array([np.zeros((6, 1)), np.array([0.,0.])])  # coefficients and ts te
        self.guide_info = np.array([np.zeros((6, 1)), np.array([0.,0.])])  # coefficients and ts te
        self.te = 0.
        self.safety_coeffs = 1.5

    def construct_state(self):
        # Unfinished
        q, qd = self.cp2.q, self.cp2.qd
        tini, tlst = self.task_ini, self.task_lst
        s_cp2 = np.array([q[0], np.sin(q[1]), np.cos(q[1]), np.sin(q[2]), np.cos(q[2]), qd[0]/1.5, qd[1]/1.5, qd[2]/1.5])
        s = np.hstack((s_cp2, tini, tlst, np.clip(self.time_step * self.dt / self.guide_info[1][1], 0, 1),
                       self.q0_ini, self.q0d_ini))
        # The last two elements, i.e. self.q0_ini, self.q0d_ini denotes the q0 generation constrain
        return s

    def reset(self, command=None):
        self.time_step = 0  # initialize time step
        if command is not None:  # Interpret task command
            self.task_ini = command[0]
            self.task_lst = command[1]
        else:
            self.task_ini = (np.random.rand(1)-0.5) * 2
            self.task_lst = (np.random.rand(1)-0.5) * 2
        self.q0_ini = self.task_ini * 1.  # make a copy of task.ini
        # Calculate te by constraining the minimum task duration 2*dt(1/30.)
        ac = 1.5
        te = np.sqrt(10. * np.sqrt(3.) * np.max(np.abs(self.task_lst - self.task_ini)) / 3. / ac)
        N = int(te / self.dt)
        self.q0d_ini = np.array([0.])
        self.total_time_step = max(6, N + 1)
        te = self.total_time_step * self.dt
        # Construct guide info and initial q0 info
        self.q0_info = cp.calc_q0_info(0., te, self.task_ini, np.array([0.]), np.array([0.]), self.task_lst)
        self.guide_info = cp.calc_q0_info(0., te, self.task_ini, np.array([0.]), np.array([0.]), self.task_lst)
        # Upload information to manipulator
        self.cp2.set_manipulator(self.task_ini)
        self.cp2.controller.update_q0_info(self.q0_info)
        # Construct target point position
        self.point_info['xp'] = cp.calc_xp(self.cp2.l, self.task_lst)
        # Construct FM simulation variables
        self.cp2.q = np.array([self.q0_ini[0], 0., 0.])
        self.cp2.qd = np.array([0., 0., 0.])
        # construct state feedback
        s = self.construct_state()
        return s

    def step(self, action):
        done = False
        action = np.clip(action, self.action_bound_min, self.action_bound_max)
        # update q0_info and control gain by action
        tc = self.time_step * self.dt  # current time
        tn = tc + self.dt  # next time
        # action[0] changes the q0_info
        if self.time_step < self.total_time_step:
            _, _, q0dd_guide = cp.calc_LS(tc, self.guide_info)  # Calculate the guidance ydd
            q0dd = action + q0dd_guide  # Update Constraint
            self.q0_info = cp.calc_q0_info(tc, self.guide_info[1][1], self.q0_ini, self.q0d_ini, q0dd,self.task_lst)
        # Get the initial position and velocity for the next step
        self.q0_ini, self.q0d_ini, _ = cp.calc_LS(tn, self.q0_info)
        # update q0_info to the controller
        self.cp2.controller.update_q0_info(self.q0_info)
        # simulate
        self.cp2.simulator(np.array([tc, tn]), 0.002)
        # update time step
        self.time_step += 1
        # Composing the new state
        s = self.construct_state()
        # reward
        if s[10] < 0.999:
            r = 0.
        else:
            x_end = self.cp2.get_end()
            xe = np.linalg.norm(self.point_info['xp'] - x_end)
            b = -np.log(0.1) / 0.05  # get 0.1 when distance 0.02
            r = np.exp(-b * xe)  # this reward is not the final reward
        # stable simulation, safety measure
        if self.time_step == self.total_time_step:
            done = True
        if np.abs(s[0])>self.safety_coeffs or np.abs(s[5])>self.safety_coeffs:
            done = True
            r = -1.
        return s, r, done

    def render(self, mode='None'):
        output = None
        if self.viewer == None:
            self.fig = plt.figure('cart_p2', figsize=(4, 1.5))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            self.viewer = plt.gca()
            self.viewer.xaxis.set_major_locator(plt.NullLocator())
            self.viewer.yaxis.set_major_locator(plt.NullLocator())
            self.viewer.spines['top'].set_visible(False)
            self.viewer.spines['right'].set_visible(False)
            self.viewer.spines['bottom'].set_visible(False)
            self.viewer.spines['left'].set_visible(False)
            self.viewer.axis('equal')
            plt.draw()
        else:
            line_1, line_2 = self.cp2.plot_cp()
            self.viewer.cla()
            self.viewer.plot(line_1[:, 0], line_1[:, 1])
            self.viewer.plot(line_2[:, 0], line_2[:, 1])
            self.viewer.set_xlim(-2, 2)
            self.viewer.set_ylim(-1, 0.5)
            plt.draw()
            plt.pause(1E-10)
            if mode == 'rgb_array':
                self.fig.canvas.draw()
                w, h = self.fig.canvas.get_width_height()
                buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
                buf.shape = (w, h, 4)
                output = np.roll(buf, 3, axis=2)
                output = output[:, :, 0:2]
        return output

    def sample_action(self):
        sample_action = np.array([0.])
        return sample_action

    def set_seed(self, seed):
        # set seed makes the env random number predictable
        print('set env seed:', seed)
        np.random.seed(seed)


if __name__ == '__main__':
    env = env_Cart_P2()
    env.render()
    env.reset([np.array([0.]), np.array([1.])])
    env.render(mode='rgb_array')
    for i in range(100):
        a = env.sample_action()
        env.step(a)
        env.render(mode='rgb_array')
