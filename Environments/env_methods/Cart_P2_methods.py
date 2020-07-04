import numpy as np


# checked
def calc_paras(q, qd, l, m, g):
    l1 = l[0]
    l2 = l[1]
    m1 = m[0]
    m2 = m[1]
    m3 = m[2]
    q2 = q[1]
    q3 = q[2]
    qd2 = qd[1]
    qd3 = qd[2]
    t2 = np.cos(q2)
    t3 = m2+m3
    t4 = l1*t2*t3
    t5 = np.cos(q3)
    t6 = l2*m3*t5
    t7 = q2-q3
    t8 = np.cos(t7)
    t9 = l1*l2*m3*t8
    M00 = m1+m2+m3
    M10 = t4
    M20 = t6
    M01 = M10
    M11 = l1**2*t3
    M21 = t9
    M02 = M20
    M12 = M21
    M22 = l2**2*m3
    M=np.array([
    [M00,M01,M02],
    [M10,M11,M12],
    [M20,M21,M22]])
    t10 = np.sin(t7)
    t11 = np.sin(q2)
    t12 = np.sin(q3)
    C00 = 0.0
    C10 = 0.0
    C20 = 0.0
    C01 = -l1*qd2*t3*t11
    C11 = 0.0
    C21 = -l1*l2*m3*qd2*t10
    C02 = -l2*m3*qd3*t12
    C12 = l1*l2*m3*qd3*t10
    C22 = 0.0
    C=np.array([
    [C00,C01,C02],
    [C10,C11,C12],
    [C20,C21,C22]])
    G = np.array([0.0, g*l1*m2*t11+g*l1*m3*t11, g*l2*m3*t12])
    return M, C, G


# checked
def dxdt(x, tau_a, l, m, fv):
    q = x[0:3]
    qd = x[3:6]
    M, C, G = calc_paras(q, qd, l, m, g=9.8)
    qdd = np.dot(np.linalg.inv(M), (np.array([tau_a, 0., 0.]) - fv*qd - np.dot(C, qd) - G))
    dx = np.hstack((qd, qdd))
    return dx


# checked
def next_step_runge_kutta(q, qd, tau_a, dt, l, m, fv):
    x = np.hstack((q, qd))
    k1 = dxdt(x, tau_a, l, m, fv)
    k2 = dxdt(x + dt * k1 / 2, tau_a, l, m, fv)
    k3 = dxdt(x + dt * k2 / 2, tau_a, l, m, fv)
    k4 = dxdt(x + dt, tau_a, l, m, fv)
    xn = x + 1/6*dt*(k1 + 2*k2 + 2*k3 + k4)
    qn = xn[0:3]
    qdn = xn[3:6]
    return qn, qdn


# checked
def plot_Cart_P2(q, l):
    t2 = np.sin(q[1])
    t3 = l[0] * t2
    t4 = np.cos(q[1])
    cart_centre = np.array([q[0], 0.])
    m1_centre = np.array([q[0] + t3, -l[0] * t4])
    m2_centre = np.array([q[0] + t3 + l[1]*np.sin(q[2]), -l[0]*t4 - l[1]*np.cos(q[2])])
    l_cart = l[0]/3.
    h_cart = 0.75*l_cart
    line_cart = np.array([[-l_cart, h_cart],
                          [l_cart, h_cart],
                          [l_cart, -h_cart],
                          [-l_cart, -h_cart],
                          [-l_cart, h_cart]])/2. + cart_centre  # 6
    line_string = np.vstack((cart_centre, m1_centre, m2_centre))
    return line_cart, line_string


def calc_q0_info(ts, te, qs, qsd, qsdd, qe):
    # y = c0 + c1t + c2t^2 + c3t^3 + c4t^4 + c5t^5
    # s.t. y(ts)=qs dy(ts)=qsd d2y(ts)=qsdd y(te)=qe dy(te)=0 d2y(te)=0
    qed = 0.
    qedd = 0.
    td = te - ts
    t2 = td ** 2
    t3 = qsdd * t2 * 3.0
    coeffs = np.array([qs,qsd,qsdd/2.0,
                   1.0/td**3*(qe*-2.0e1+qs*2.0e1+t3-qedd*t2+qed*td*8.0+qsd*td*1.2e1)*(-1.0/2.0),
                   (1.0/td**4*(qe*-3.0e1+qs*3.0e1+t3-qedd*t2*2.0+qed*td*1.4e1+qsd*td*1.6e1))/2.0,
                   1.0/td**5*(qe*-1.2e1+qs*1.2e1-qedd*t2+qsdd*t2+qed*td*6.0+qsd*td*6.0)*(-1.0/2.0)])
    tste = np.array([ts, te])
    q0_info = (coeffs, tste)
    return q0_info


# checked
def calc_LS(t, q0_info):
    # t is the real time
    # coeffs[0]-[5] are coeffs of polynomial functions and can be array
    # tste[0]: ts; tste[1]: te
    coeffs = q0_info[0]
    tste = q0_info[1]
    if t > tste[1]:
        td = tste[1] - tste[0]
    else:
        td = t - tste[0]
    y = coeffs[0, :] + coeffs[1, :] * td + coeffs[2, :] * td ** 2 + \
        coeffs[3, :] * td ** 3 + coeffs[4, :] * td ** 4 + coeffs[5, :] * td ** 5
    yd = coeffs[1, :] + 2 * coeffs[2, :] * td + 3 * coeffs[3, :] * td ** 2 + \
         4 * coeffs[4, :] * td ** 3 + 5 * coeffs[5, :] * td ** 4
    ydd = 2 * coeffs[2, :] + 6 * coeffs[3, :] * td + 12 * coeffs[4, :] * td ** 2 + 20 * coeffs[5, :] * td ** 3
    return y, yd, ydd


def calc_xp(l, task_lst):
    x_end = np.array([task_lst[0], -l[0] - l[1]])
    return x_end


class PD_controller(object):
    def __init__(self):
        self.kp = 1000.
        self.kd = 3.
        self.q0_info = None
        self.q0 = None
        self.q0d = None
        self.q0dd = None

    def update_q0_info(self, received_info):
        self.q0_info = received_info

    def calc_tau_a(self, t, q, qd):
        self.q0, self.q0d, self.q0dd = calc_LS(t, self.q0_info)
        tau_a = - self.kp*(q[0]-self.q0[0]) - self.kd * (qd[0]-self.q0d[0])
        return tau_a


class OPD_controller(object):
    def __init__(self):
        self.kp = np.array(1000.)
        self.kd = np.array(3.)
        self.kc = 1E-7
        self.q0_info = None
        self.te = None
        self.q0 = None
        self.q0d = None
        self.q0dd = None

    def update_q0_info(self, received_info):
        self.q0_info = received_info
        self.te = self.q0_info[1][1]

    def calc_tau_a(self, t, q, qd):
        self.q0, self.q0d, self.q0dd = calc_LS(t, self.q0_info)
        e = q[0] - self.q0[0]
        ed = qd[0] - self.q0d[0]
        tau_a = - self.kp * e - self.kd * ed
        f = np.dot(self.kp * e, ed) / np.max(self.kp)
        s = np.clip(f / self.kc + 1., 0., 1.)
        if t > self.te:
            tau_a = -s * self.kp * e - self.kd * ed
        return tau_a


class Cart_P2_manipulator(object):
    # System parameters
    l = np.array([0.47, 0.391])
    m = np.array([0.3, 0.192, 0.201])  # m_cart m1 m2
    fv = 0.1
    pix_per_m = 180

    def __init__(self, mode=0):
        self.q = np.zeros(3)
        self.qd = np.zeros(3)
        self.t_rec = None
        self.q_rec = None
        self.qd_rec = None
        self.q0_rec = None
        if mode == 0:
            self.controller = PD_controller()
        else:
            self.controller = OPD_controller()

    def set_manipulator(self, qa):
        self.q = np.hstack((qa, np.zeros(2)))
        self.qd = np.zeros(3)

    def simulator(self, tspan, dt):
        n = (tspan[1] - tspan[0]) / dt  # Find step number that suits the recommendation dt best
        n = int(n)
        t_step = (tspan[1] - tspan[0]) / n  # Real simulation step size
        t = tspan[0]
        q = self.q  # self.q will not auto update here
        qd = self.qd
        # initialize recorder
        self.t_rec = np.zeros(n)
        self.q_rec = np.zeros((n, 3))
        self.qd_rec = np.zeros((n, 3))
        self.q0_rec = np.zeros(n)
        for i in range(n):
            # recording [ts, te)
            self.t_rec[i] = t
            self.q_rec[i, :] = q
            self.qd_rec[i, :] = qd
            self.q0_rec[i] = self.controller.q0
            # update time step
            t = t + t_step
            # Calculate control torque
            tau_a = self.controller.calc_tau_a(t, q, qd)
            # Update the next step
            q, qd = next_step_runge_kutta(q, qd, tau_a, t_step, self.l, self.m, self.fv)
        self.q = q
        self.qd = qd

    def get_end(self):
        l = self.l
        q = self.q
        t2 = np.sin(q[1])
        t3 = l[0] * t2
        t4 = np.cos(q[1])
        x_end = np.array([q[0] + t3 + l[1] * np.sin(q[2]), -l[0] * t4 - l[1] * np.cos(q[2])])
        return x_end

    def plot_cp(self):
        return plot_Cart_P2(self.q, self.l)



if __name__ == '__main__':
    ti = 0.
    tf = 10.
    dt = 0.002
    q0_ini = np.array([-0.1])
    q0_lst = np.array([1.0])
    tfr = 1/15.
    ac = 1.5
    te = np.sqrt(10. * np.sqrt(3.) * np.max(np.abs(q0_lst - q0_ini)) / 3. / ac)
    N = int(te / tfr)
    total_time_step = max(10, N + 1)
    te = total_time_step * tfr
    mach = Cart_P2_manipulator()
    mach.set_manipulator(q0_ini)
    guide_info = calc_q0_info(ts=0., te=te, qs=q0_ini, qsd=np.array([0.]), qsdd=np.array([0.]), qe=q0_lst)
    q0n = q0_ini
    q0dn = np.array([0.])
    #
    t_rec = np.empty(0)
    q_rec = np.empty((0, 3))
    for i in range(150):
        tc = i * tfr
        a = np.array([0.8 * np.sin(2 * np.pi * tc/te)])
        _, _, qdd_guide = calc_LS(tc, guide_info)
        qdd = a + qdd_guide
        if i < N:
            q0_info = calc_q0_info(ts=tc, te=te, qs=q0n, qsd=q0dn, qsdd=qdd, qe=q0_lst)
            mach.controller.update_q0_info(q0_info)
        q0n, q0dn, _ = calc_LS(tc+tfr, q0_info)
        mach.simulator([tc, tc+tfr], dt)
        t_rec = np.concatenate((t_rec, mach.t_rec), 0)
        q_rec = np.concatenate((q_rec, mach.q_rec), 0)
    # fig = plt.figure(1, figsize=[4, 3])
    # plt.subplot(3, 1, 1)
    # plt.plot(t_rec, q_rec[:, 0])
    # plt.subplot(3, 1, 2)
    # plt.plot(t_rec, q_rec[:, 1])
    # plt.subplot(3, 1, 3)
    # plt.plot(t_rec, q_rec[:, 2])
    # plt.show()
    # sio.savemat('gt.mat', {'t': t_rec, 'q': q_rec})
    # # system simulation
    # q_ini = np.array([0.1, np.pi/2, np.pi/2])
    # qd_ini = np.array([0., 0., 0.])
    # l = np.array([0.47, 0.391])
    # m = np.array([0.3, 0.192, 0.201])  # m_cart m1 m2
    # # DRAW
    # data = plot_Cart_P2(q_ini, l, pix_per_m=1.)
    # fig2 = plt.figure(2, figsize=[4, 6])
    # for a in data:
    #     plt.plot(a[:, 0], a[:, 1])
    # plt.axis('equal')
    # plt.show()
    # SIMULATION
    # T, te, q0_lst = 10., 1., 0.9
    # dt = 1/1000.
    # kp, kd, fv = 1000, 3, 0.05
    # #
    # q0_info = calc_q0_info(ts=0., te=te, qs=np.array([q_ini[0]]), qsd=np.array([0.]),
    #                        qsdd=np.array([0.]), qe=np.array([q0_lst]))
    # N = int(T/dt)
    # t, q, qd = np.zeros(N), np.zeros((N, 3)), np.zeros((N, 3))
    # q[0, :], qd[0, :] = q_ini, qd_ini
    # #
    # next_step_runge_kutta(np.zeros(3), np.zeros(3), 0., 0.001, l, m, fv)
    # ts = time.time()
    # for i in range(N-1):
    #     # Calc q0
    #     q0, q0d, _ = calc_LS(t[i], q0_info)
    #     e, ed = q[i, 0] - q0[0], qd[i, 0] - q0d[0]
    #     tau_a = - kp*e - kd*ed
    #     #
    #     q[i+1, :], qd[i+1, :] = next_step_runge_kutta(q[i, :], qd[i, :], tau_a, dt, l, m, fv)
    #     t[i+1] = t[i] + dt
    # #
    # print('Time used:', time.time() - ts)
    # gs = sio.loadmat('simulation_matlab.mat')
    # gs_q = gs['q']
    # fig1 = plt.figure(1, figsize=[4, 6])
    # plt.subplot(3, 1, 1)
    # plt.plot(t, q[:, 0], 'b')
    # plt.plot(gs['t'], gs_q[0, :], 'r')
    # plt.subplot(3, 1, 2)
    # plt.plot(t, q[:, 1], 'b')
    # plt.plot(gs['t'], gs_q[1, :], 'r')
    # plt.subplot(3, 1, 3)
    # plt.plot(t, q[:, 2], 'b')
    # plt.plot(gs['t'], gs_q[2, :], 'r')
    # plt.show()
