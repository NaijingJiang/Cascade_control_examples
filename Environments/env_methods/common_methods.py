import numpy as np

def rotation_matrix(theta):
    # The rotation matrix is the transpose of common rotation matrix
    # because the vector is written in row, i.e. a = [x, y]
    # denote R as the common rotation matrix, we have a1T = R*aT <=> a1 = a*RT
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


def spline_inter(x, u, le):
    # defines a spline shape with boundary conditions
    # u(x=0)=u1 dudx(0)=u2 u(le)=u3 dudx(le)=u4
    xs = x ** 2
    xt = x ** 3
    y = (1. - 3. * xs / le ** 2 + 2. * xt / le ** 3) * u[0] + \
        (x - 2. * xs / le + xt / le ** 2) * u[1] + \
        (3 * xs / le ** 2 - 2 * xt / le ** 3) * u[2] + \
        (-xs / le + xt / le ** 2) * u[3]
    return y


def dist_line_poly(test_data, dist):
    # The dist_line_poly returns a set of points used to draw a curve with certain width
    N_point = test_data.shape[0]
    dist_data = np.zeros((2 * N_point + 1, 2))
    # The initial tan
    tan_seg_1 = test_data[1, :] - test_data[0, :]
    temp = np.array([-tan_seg_1[1], tan_seg_1[0]])
    nor_seg_1 = temp / np.linalg.norm(temp)
    dist_data[0, :] = test_data[0, :] + dist * nor_seg_1
    dist_data[1, :] = test_data[0, :] - dist * nor_seg_1
    for i in range(N_point - 2):
        # get the norm of the current segment
        j = i + 1
        tan_seg_2 = test_data[j + 1, :] - test_data[j, :]
        temp = np.array([-tan_seg_2[1], tan_seg_2[0]])
        nor_seg_2 = temp / np.linalg.norm(temp)
        if np.abs(np.dot(tan_seg_1, nor_seg_2)) >= 1e-7:
            temp1 = test_data[i, :] + dist * nor_seg_1
            temp2 = test_data[i + 1, :] + dist * nor_seg_1
            temp3 = test_data[j, :] + dist * nor_seg_2
            temp4 = test_data[j + 1, :] + dist * nor_seg_2
            A = np.array([[temp1[1] - temp2[1], temp2[0] - temp1[0]], [temp3[1] - temp4[1], temp4[0] - temp3[0]]])
            b = np.array([temp2[0] * temp1[1] - temp1[0] * temp2[1], temp4[0] * temp3[1] - temp3[0] * temp4[1]])
            dist_data[2 * j, :] = np.dot(np.linalg.inv(A), b)
            dist_data[2 * j + 1, :] = test_data[j, :] - (dist_data[2 * j, :] - test_data[j, :])
        else:
            dist_data[2 * j, :] = test_data[j, :] + dist * nor_seg_2
            dist_data[2 * j + 1, :] = test_data[j, :] - dist * nor_seg_2
        tan_seg_1 = tan_seg_2
        nor_seg_1 = nor_seg_2
    tan_seg_2 = test_data[N_point - 1, :] - test_data[N_point - 2, :]
    temp = np.array([-tan_seg_2[1], tan_seg_2[0]])
    nor_seg_2 = temp / np.linalg.norm(temp)
    dist_data[2 * N_point - 2, :] = test_data[N_point - 1, :] + dist * nor_seg_2
    dist_data[2 * N_point - 1, :] = test_data[N_point - 1, :] - dist * nor_seg_2
    dist_data[2 * N_point, :] = dist_data[2 * N_point - 1, :]
    return dist_data


def link_ends(w, N):
    # returns the left ends of a link that draws as poly-triangle
    # 2N
    r = w / 2.
    dtheta = np.pi / 2 / N
    area_left = np.zeros((2 * N, 2))
    area_left[0, :] = np.array([-r, 0.])
    area_left[1, :] = np.array([-r, 0.])
    for i in range(N - 1):
        theta1 = np.pi - dtheta * (i + 1)
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        area_left[2 + 2 * i, :] = np.array([r * c1, r * s1])
        area_left[3 + 2 * i, :] = np.array([r * c1, -r * s1])
    return area_left


def motor(d):
    # draws a motor
    d1 = d / 2
    d2 = d / 2 / 2
    data = np.array([[d1 - d2, d1],
                     [d1 - d2, d1],  # 0
                     [d2 - d1, d1],  # 1
                     [d1, d1 - d2],  # 2
                     [-d1, d1 - d2],  # 3
                     [d1, d2 - d1],  # 4
                     [-d1, d2 - d1],  # 5
                     [d1 - d2, -d1],  # 6
                     [d2 - d1, -d1],  # 7
                     [d2 - d1, -d1]])
    return data


def connector(d, N):  # 2N+2
    # returns a small circle at the connector between links
    dtheta = np.pi / N
    data = np.zeros((2 * N + 2, 2))
    data[0, :] = np.array([0, d / 2.])
    data[1, :] = np.array([0, d / 2.])
    for i in range(N - 1):
        theta = np.pi / 2 - dtheta * (i + 1)
        c = np.cos(theta)
        s = np.sin(theta)
        data[2 * i + 2, :] = d / 2. * np.array([c, s])
        data[2 * i + 3, :] = d / 2. * np.array([-c, s])
    data[2 * N, :] = np.array([0, -d / 2.])
    data[2 * N + 1, :] = np.array([0, -d / 2.])
    return data