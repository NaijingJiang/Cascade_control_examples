import Environments.env_methods.F2sF2s_methods as fm
import numpy as np
import matplotlib.pyplot as plt


class env_F2sF2s(object):
    viewer = None
    dt = 1 / 30.  # Frame rate 30 Hz
    action_bound_min = np.array([-30., -30.])
    action_bound_max = np.array([30., 30.])
    state_dim = 21  # cos(qa1) sin(qa1) cos(qa2) sin(qa2) u15 u16 u25 dqa1 dqa2 du15 du16 du25  Num: 12
    # q0_ini q0_lst clip(t/te,1) q0(t) q0d(t)  Num: 9
    action_dim = 2  # q0dd

    def __init__(self, mode=0):
        # manipulator object
        self.f2sf2s = fm.F2sF2s_manipulator(mode=mode)
        # Information required by render
        self.point_info = {'xp': np.array([0., 0.])}  # dictionary, use goal['qa0'] to get the value
        # Simulation related
        self.time_step = 0
        self.total_time_step = 0
        self.q0_ini = np.array([0.,0.])
        self.q0d_ini = np.array([0.,0.])
        self.task_ini = np.array([0.,0.])
        self.task_lst = np.array([0.,0.])
        self.q0_info = np.array([np.zeros((6, 2)), np.array([0.,0.])])  # coefficients and ts te
        self.guide_info = np.array([np.zeros((6, 2)), np.array([0.,0.])])  # coefficients and ts te
        self.te = 0.
        self.satety_coeffs = 1.5

    def construct_state(self):
        q = self.f2sf2s.q
        qd = self.f2sf2s.qd
        s_FM = np.array([np.cos(q[0]), np.sin(q[0]), np.cos(q[1]), np.sin(q[1]),
                         q[4] / 0.08, q[5] / 0.4, q[8] / 0.02,  # qf
                         qd[0] / 6., qd[1] / 6, qd[4] / 0.8, qd[5] / 4, qd[8] / 0.2])  # 12
        s = np.hstack((s_FM, self.task_ini / np.pi, self.task_lst / np.pi,
                       np.clip(self.time_step * self.dt / self.guide_info[1][1], 0, 1),
                       self.q0_ini, self.q0d_ini))  # 21 Checked
        return s

    def step(self, action):
        done = False
        action = np.clip(action, self.action_bound_min, self.action_bound_max)
        # update q0_info and control gain by action
        tc = self.time_step * self.dt  # current time
        tn = tc + self.dt  # next time
        # action[0] changes the q0_info
        if self.time_step < self.total_time_step:
            _, _, q0dd_guide = fm.calc_LS(tc, self.guide_info)  # Calculate the guidance ydd
            q0dd = action[0:2] + q0dd_guide  # Update Constraint
            self.q0_info = fm.calc_q0_info(tc, self.guide_info[1][1], self.q0_ini, self.q0d_ini, q0dd,self.task_lst)
        # Get the initial position and velocity for the next step
        self.q0_ini, self.q0d_ini, _ = fm.calc_LS(tn, self.q0_info)
        # update q0_info to the controller
        self.f2sf2s.controller.update_q0_info(self.q0_info)
        # simulate
        self.f2sf2s.simulator(np.array([tc, tn]), 0.001)
        # update time step
        self.time_step += 1
        # Composing the new state
        s = self.construct_state()
        # reward
        if s[16] < 0.999:
            r = 0.
        else:
            x_end = self.f2sf2s.get_end()
            xe = np.linalg.norm(self.point_info['xp'] - x_end)
            b = -np.log(0.1) / 0.02  # get 0.1 when distance 0.02
            r = np.exp(-b * xe)  # this reward is not the final reward
        # stable simulation, safety measure
        if self.time_step == self.total_time_step:
            done = True
        if np.abs(s[4])>self.satety_coeffs or np.abs(s[6])>self.satety_coeffs or \
                np.abs(s[9])>self.satety_coeffs or np.abs(s[11])>self.satety_coeffs:
            done = True
            r = -1.
        return s, r, done

    def reset(self, command=None):
        self.time_step = 0  # initialize time step
        if command is not None:  # Interpret task command
            self.task_ini = command[0]
            self.task_lst = command[1]
        else:
            self.task_ini = (np.random.rand(2)-0.5) * np.pi
            self.task_lst = (np.random.rand(2)-0.5) * np.pi
        self.q0_ini = self.task_ini * 1.
        # Calculate te by constraining the minimum task duration 2*dt(1/30.)
        ac = 15
        te = np.sqrt(10. * np.sqrt(3.) * np.max(np.abs(self.task_lst - self.task_ini)) / 3. / ac)
        N = int(te / self.dt)
        self.q0d_ini = np.array([0.,0.])
        self.total_time_step = max(2, N + 1)
        te = self.total_time_step * self.dt
        # Construct guide info and initial q0 info
        self.q0_info = fm.calc_q0_info(0., te, self.task_ini, np.array([0.,0.]), np.array([0.,0.]), self.task_lst)
        self.guide_info = fm.calc_q0_info(0., te, self.task_ini, np.array([0.,0.]), np.array([0.,0.]), self.task_lst)
        # Upload information to manipulator
        self.f2sf2s.set_manipulator(self.task_ini)
        self.f2sf2s.controller.update_q0_info(self.q0_info)
        # Construct target point position
        self.point_info['xp'] = fm.calc_xp(self.f2sf2s.l, self.task_lst)
        # Construct FM simulation variables
        self.f2sf2s.q = np.array([self.q0_ini[0], self.q0_ini[1], 0., 0., 0., 0., 0., 0., 0., 0.])
        self.f2sf2s.qd = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.f2sf2s.qdd = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        # construct state feedback
        s = self.construct_state()
        return s

    def render(self, mode='None'):
        output = None
        if self.viewer == None:
            self.fig = plt.figure('f2sf2s', figsize=(3, 3))
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
            line_link1, line_link2 = fm.plot_F2sF2s(self.f2sf2s.q, self.f2sf2s.l)
            line_links = np.vstack((line_link1, line_link2))
            self.viewer.cla()
            self.viewer.plot(line_links[:, 0], line_links[:, 1])
            l_lim = sum(self.f2sf2s.l) * 1.1
            self.viewer.set_xlim(-l_lim, l_lim)
            self.viewer.set_ylim(-l_lim, l_lim)
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
        # sample_action = -10 * np.sin(np.pi*self.time_step/self.total_time_step)
        # sample_action = 30*2*(np.random.rand(2)-0.5)
        sample_action = np.array([0., 0.])
        return sample_action

    def set_seed(self, seed):
        # set seed makes the env random number predictable
        print('set env seed:', seed)
        np.random.seed(seed)


if __name__=='__main__':
    # the rendering time for 100 frame is about 12s
    env = env_F2sF2s()
    env.render()
    #
    env.reset(command=[np.array([np.pi/4, np.pi/4]), np.array([-np.pi/4, -np.pi/4])])
    env.render('rgb_array')
    for i in range(100):
        env.render('rgb_array')
        a = env.sample_action()
        env.step(a)