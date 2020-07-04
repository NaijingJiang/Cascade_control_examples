import Environments.env_methods.common_methods as cm
import numpy as np


def calc_paras(q_dof, qd_dof, l, m, mp, Js, EI):
    EI1 = EI[0]
    EI2 = EI[1]
    Js1 = Js[0]
    Js2 = Js[1]
    l1 = l[0]
    l2 = l[1]
    m1 = m[0]
    m2 = m[1]
    mp1 = mp[0]
    mp2 = mp[1]
    qa1 = q_dof[0]
    qa2 = q_dof[1]
    qad1 = qd_dof[0]
    qad2 = qd_dof[1]
    qf1 = q_dof[2]
    qf2 = q_dof[3]
    qf3 = q_dof[4]
    qf4 = q_dof[5]
    qf5 = q_dof[6]
    qf6 = q_dof[7]
    qf7 = q_dof[8]
    qf8 = q_dof[9]
    qfd1 = qd_dof[2]
    qfd2 = qd_dof[3]
    qfd3 = qd_dof[4]
    qfd4 = qd_dof[5]
    qfd5 = qd_dof[6]
    qfd6 = qd_dof[7]
    qfd7 = qd_dof[8]
    qfd8 = qd_dof[9]
    t2 = l1 ** 2
    t3 = l2 ** 2
    t4 = qf3 ** 2
    t5 = qf7 ** 2
    t6 = qa2 + qf4
    t7 = np.cos(t6)
    t8 = np.sin(t6)
    t9 = (m2 * t3) / 3.0
    t10 = mp2 * t3
    t11 = qf5 ** 2
    t12 = m2 * t11 * (1.3e1 / 3.5e1)
    t13 = m2 * t5 * (1.3e1 / 7.0e1)
    t14 = mp2 * t5
    t15 = qf6 ** 2
    t16 = (m2 * t3 * t15) / 4.2e2
    t17 = qf8 ** 2
    t18 = (m2 * t3 * t17) / 8.4e2
    t19 = m2 * qf5 * qf7 * (9.0 / 7.0e1)
    t20 = l2 * m2 * qf6 * qf7 * (1.3e1 / 8.4e2)
    t21 = (l1 * l2 * m2 * t7) / 2.0
    t22 = l1 * l2 * mp2 * t7
    t23 = (m2 * qf3 * qf5 * t7) / 2.0
    t24 = (m2 * qf3 * qf7 * t7) / 4.0
    t25 = mp2 * qf3 * qf7 * t7
    t26 = (l2 * m2 * qf3 * t8) / 2.0
    t27 = l2 * mp2 * qf3 * t8
    t28 = (l1 * l2 * m2 * qf8 * t8) / 4.8e1
    t30 = l2 * m2 * qf5 * qf8 * (1.3e1 / 8.4e2)
    t31 = l2 * m2 * qf7 * qf8 * (1.1e1 / 4.2e2)
    t32 = (m2 * qf6 * qf8 * t3) / 5.6e2
    t50 = (l1 * m2 * qf5 * t8) / 2.0
    t51 = (l1 * m2 * qf7 * t8) / 4.0
    t52 = l1 * mp2 * qf7 * t8
    t53 = (l2 * m2 * qf3 * qf8 * t7) / 4.8e1
    t29 = Js2 + t9 + t10 + t12 + t13 + t14 + t16 + t18 + t19 + t20 + t21 + t22 + t23 + t24 + t25 + t26 + t27 + t28 - t30 - t31 - t32 - t50 - t51 - t52 - t53
    t33 = (l2 * m2 * t7) / 2.0
    t34 = l2 * mp2 * t7
    t35 = (l2 * m2 * qf8 * t8) / 4.8e1
    t36 = Js2 + t9 + t10 + t12 + t13 + t14 + t16 + t18 + t19 + t20 - t30 - t31 - t32
    t37 = (m2 * t3) / 1.2e2
    t38 = (l1 * m1) / 4.0
    t39 = (m1 * t2) / 1.2e2
    t40 = l1 * m1 * (1.7e1 / 8.0e1)
    t41 = l1 * m2
    t42 = l1 * mp1
    t43 = l1 * mp2
    t45 = (m2 * qf5 * t8) / 2.0
    t46 = (m2 * qf7 * t8) / 4.0
    t47 = mp2 * qf7 * t8
    t44 = t33 + t34 + t35 + t40 + t41 + t42 + t43 - t45 - t46 - t47
    t48 = m1 * (9.0 / 1.4e2)
    t49 = l1 * m1 * 7.738095238095238e-3
    t54 = t33 + t34 + t35 - t45 - t46 - t47 - l1 * m1 * (1.1e1 / 8.4e2)
    t55 = (l2 * m2) / 4.0
    t56 = m2 * 1.7e1
    t57 = mp2 * 8.0e1
    t58 = t56 + t57
    t59 = (l2 * t58) / 8.0e1
    t60 = l1 * t7 * 2.0
    t61 = qf3 * t8 * 2.0
    t62 = l2 + t60 + t61
    t63 = (m2 * t62) / 4.0
    t64 = (m2 * t7) / 2.0
    t65 = l2 * m2 * (1.7e1 / 8.0e1)
    t66 = l2 * mp2
    t67 = (l1 * m2 * t7) / 4.0
    t68 = l1 * mp2 * t7
    t69 = (m2 * qf3 * t8) / 4.0
    t70 = mp2 * qf3 * t8
    t71 = t65 + t66 + t67 + t68 + t69 + t70
    t72 = mp2 * 4.0
    t73 = m2 + t72
    t74 = (t7 * t73) / 4.0
    t75 = m2 * (9.0 / 1.4e2)
    t76 = l2 * m2 * 7.738095238095238e-3
    t77 = l2 * 4.0
    t78 = l1 * t7 * 5.0
    t79 = qf3 * t8 * 5.0
    t80 = t77 + t78 + t79
    M00 = Js1 + Js2 + t9 + t10 + t12 + t13 + t14 + t16 + t18 + t19 + t20 + (m1 * t2) / 3.0 + m2 * t2 + m1 * t4 * (
                1.3e1 / 7.0e1) + m2 * t4 + mp1 * t2 + mp2 * t2 + mp1 * t4 + mp2 * t4 + m1 * qf1 ** 2 * (
                      1.3e1 / 3.5e1) + (m1 * qf2 ** 2 * t2) / 4.2e2 + (
                      m1 * qf4 ** 2 * t2) / 8.4e2 + m1 * qf1 * qf3 * (
                      9.0 / 7.0e1) + l1 * l2 * m2 * t7 + l1 * l2 * mp2 * t7 * 2.0 - l1 * m1 * qf1 * qf4 * (
                      1.3e1 / 8.4e2) + l1 * m1 * qf2 * qf3 * (1.3e1 / 8.4e2) - l1 * m1 * qf3 * qf4 * (
                      1.1e1 / 4.2e2) - l2 * m2 * qf5 * qf8 * (1.3e1 / 8.4e2) - l2 * m2 * qf7 * qf8 * (
                      1.1e1 / 4.2e2) + l2 * m2 * qf3 * t8 - l1 * m2 * qf5 * t8 - (
                      l1 * m2 * qf7 * t8) / 2.0 + l2 * mp2 * qf3 * t8 * 2.0 - l1 * mp2 * qf7 * t8 * 2.0 - (
                      m1 * qf2 * qf4 * t2) / 5.6e2 + m2 * qf3 * qf5 * t7 + (m2 * qf3 * qf7 * t7) / 2.0 - (
                      m2 * qf6 * qf8 * t3) / 5.6e2 + mp2 * qf3 * qf7 * t7 * 2.0 + (
                      l1 * l2 * m2 * qf8 * t8) / 2.4e1 - (l2 * m2 * qf3 * qf8 * t7) / 2.4e1
    M10 = t29
    M20 = t38
    M30 = t39
    M40 = t44
    M50 = Js2 + t9 + t10 + t12 + t13 + t14 + t16 + t18 + t19 + t20 + t21 + t22 + t23 + t24 + t25 + t26 + t27 + t28 - t30 - t31 - t32 - t50 - t51 - t52 - t53 - (
                m1 * t2) / 6.0e1
    M60 = t63
    M70 = t37
    M80 = t71
    M90 = l2 * m2 * t80 * (-1.0 / 2.4e2)
    M01 = M10
    M11 = t36
    M21 = 0.0
    M31 = 0.0
    M41 = t33 + t34 + t35 - t45 - t46 - t47
    M51 = t36
    M61 = t55
    M71 = t37
    M81 = t59
    M91 = m2 * t3 * (-1.0 / 6.0e1)
    M02 = M20
    M12 = M21
    M22 = m1 * (1.3e1 / 3.5e1)
    M32 = 0.0
    M42 = t48
    M52 = -t49
    M62 = 0.0
    M72 = 0.0
    M82 = 0.0
    M92 = 0.0
    M03 = M30
    M13 = M31
    M23 = M32
    M33 = (m1 * t2) / 4.2e2
    M43 = t49
    M53 = m1 * t2 * (-8.928571428571429e-4)
    M63 = 0.0
    M73 = 0.0
    M83 = 0.0
    M93 = 0.0
    M04 = M40
    M14 = M41
    M24 = M42
    M34 = M43
    M44 = m1 * (1.3e1 / 7.0e1) + m2 + mp1 + mp2
    M54 = t54
    M64 = t64
    M74 = 0.0
    M84 = t74
    M94 = l2 * m2 * t7 * (-1.0 / 4.8e1)
    M05 = M50
    M15 = M51
    M25 = M52
    M35 = M53
    M45 = M54
    M55 = Js2 + t9 + t10 + t12 + t13 + t14 + t16 + t18 + t19 + t20 - t30 - t31 - t32 + (m1 * t2) / 8.4e2
    M65 = t55
    M75 = t37
    M85 = t59
    M95 = m2 * t3 * (-1.0 / 6.0e1)
    M06 = M60
    M16 = M61
    M26 = M62
    M36 = M63
    M46 = M64
    M56 = M65
    M66 = m2 * (1.3e1 / 3.5e1)
    M76 = 0.0
    M86 = t75
    M96 = -t76
    M07 = M70
    M17 = M71
    M27 = M72
    M37 = M73
    M47 = M74
    M57 = M75
    M67 = M76
    M77 = (m2 * t3) / 4.2e2
    M87 = t76
    M97 = m2 * t3 * (-8.928571428571429e-4)
    M08 = M80
    M18 = M81
    M28 = M82
    M38 = M83
    M48 = M84
    M58 = M85
    M68 = M86
    M78 = M87
    M88 = m2 * (1.3e1 / 7.0e1) + mp2
    M98 = l2 * m2 * (-1.1e1 / 8.4e2)
    M09 = M90
    M19 = M91
    M29 = M92
    M39 = M93
    M49 = M94
    M59 = M95
    M69 = M96
    M79 = M97
    M89 = M98
    M99 = (m2 * t3) / 8.4e2
    M = np.array([
        [M00, M01, M02, M03, M04, M05, M06, M07, M08, M09],
        [M10, M11, M12, M13, M14, M15, M16, M17, M18, M19],
        [M20, M21, M22, M23, M24, M25, M26, M27, M28, M29],
        [M30, M31, M32, M33, M34, M35, M36, M37, M38, M39],
        [M40, M41, M42, M43, M44, M45, M46, M47, M48, M49],
        [M50, M51, M52, M53, M54, M55, M56, M57, M58, M59],
        [M60, M61, M62, M63, M64, M65, M66, M67, M68, M69],
        [M70, M71, M72, M73, M74, M75, M76, M77, M78, M79],
        [M80, M81, M82, M83, M84, M85, M86, M87, M88, M89],
        [M90, M91, M92, M93, M94, M95, M96, M97, M98, M99]])
    # C
    t81 = m2 * qf5 * qfd5 * (1.3e1 / 3.5e1)
    t82 = m2 * qf5 * qfd7 * (9.0 / 1.4e2)
    t83 = m2 * qf7 * qfd5 * (9.0 / 1.4e2)
    t84 = m2 * qf7 * qfd7 * (1.3e1 / 7.0e1)
    t85 = mp2 * qf7 * qfd7
    t86 = l2 * m2 * qf6 * qfd7 * 7.738095238095238e-3
    t87 = l2 * m2 * qf7 * qfd6 * 7.738095238095238e-3
    t88 = (m2 * qf3 * qfd5 * t7) / 2.0
    t89 = (m2 * qf3 * qfd7 * t7) / 4.0
    t90 = mp2 * qf3 * qfd7 * t7
    t91 = (m2 * qf6 * qfd6 * t3) / 4.2e2
    t92 = (m2 * qf8 * qfd8 * t3) / 8.4e2
    t93 = (l2 * m2 * qad2 * qf3 * t7) / 2.0
    t94 = (l2 * m2 * qf3 * qfd4 * t7) / 2.0
    t95 = l2 * mp2 * qad2 * qf3 * t7
    t96 = l2 * mp2 * qf3 * qfd4 * t7
    t97 = (l1 * l2 * m2 * qfd8 * t8) / 4.8e1
    t98 = (l1 * l2 * m2 * qad2 * qf8 * t7) / 4.8e1
    t99 = (l1 * l2 * m2 * qf8 * qfd4 * t7) / 4.8e1
    t100 = (l2 * m2 * qad2 * qf3 * qf8 * t8) / 4.8e1
    t101 = (l2 * m2 * qf3 * qf8 * qfd4 * t8) / 4.8e1
    t102 = (l2 * m2 * qad1 * qf3 * t7) / 2.0
    t103 = l2 * mp2 * qad1 * qf3 * t7
    t104 = (l1 * l2 * m2 * qad1 * qf8 * t7) / 4.8e1
    t105 = (l2 * m2 * qad1 * qf3 * qf8 * t8) / 4.8e1
    t106 = qad1 + qad2 + qfd4
    t107 = (m2 * qf5 * qfd3 * t7) / 2.0
    t108 = (m2 * qf7 * qfd3 * t7) / 4.0
    t109 = mp2 * qf7 * qfd3 * t7
    t110 = (l2 * m2 * qfd3 * t8) / 2.0
    t111 = l2 * mp2 * qfd3 * t8
    t112 = m2 * qf5 * (9.0 / 7.0e1)
    t113 = m2 * qf7 * (1.3e1 / 3.5e1)
    t114 = mp2 * qf7 * 2.0
    t115 = l2 * m2 * qf6 * (1.3e1 / 8.4e2)
    t166 = l2 * m2 * qf8 * (1.1e1 / 4.2e2)
    t116 = t112 + t113 + t114 + t115 - t166
    t117 = (qfd7 * t116) / 2.0
    t118 = m2 * qf5 * (2.6e1 / 3.5e1)
    t119 = m2 * qf7 * (9.0 / 7.0e1)
    t167 = l2 * m2 * qf8 * (1.3e1 / 8.4e2)
    t120 = t118 + t119 - t167
    t121 = (qfd5 * t120) / 2.0
    t122 = (m2 * qf6 * t3) / 5.6e2
    t123 = l2 * m2 * qf5 * (1.3e1 / 8.4e2)
    t124 = l2 * m2 * qf7 * (1.1e1 / 4.2e2)
    t168 = (m2 * qf8 * t3) / 4.2e2
    t125 = t122 + t123 + t124 - t168
    t126 = (m2 * qf6 * t3) / 2.1e2
    t127 = l2 * m2 * qf7 * (1.3e1 / 8.4e2)
    t170 = (m2 * qf8 * t3) / 5.6e2
    t128 = t126 + t127 - t170
    t129 = (qfd6 * t128) / 2.0
    t169 = (qfd8 * t125) / 2.0
    t130 = t117 + t121 + t129 - t169
    t131 = qf5 * 6.24e2
    t132 = qf7 * 1.08e2
    t133 = qf7 * 2.6e1
    t134 = l2 * qf6 * 8.0
    t180 = l2 * qf8 * 3.0
    t135 = t133 + t134 - t180
    t136 = (l2 * m2 * t106 * t135) / 3.36e3
    t137 = m2 * qf5 * 1.08e2
    t138 = m2 * qf7 * 3.12e2
    t139 = mp2 * qf7 * 1.68e3
    t140 = l2 * m2 * qf6 * 1.3e1
    t141 = qf5 * 2.6e1
    t142 = qf7 * 4.4e1
    t143 = l2 * qf6 * 3.0
    t144 = qf1 * 6.24e2
    t145 = qf3 * 1.08e2
    t146 = t144 + t145 - l1 * qf4 * 1.3e1
    t147 = (m1 * qad1 * t146) / 1.68e3
    t148 = qf3 * 2.6e1
    t149 = l1 * qf2 * 8.0
    t150 = t148 + t149 - l1 * qf4 * 3.0
    t151 = (l1 * m1 * qad1 * t150) / 3.36e3
    t152 = (l2 * m2 * qfd8 * t8) / 4.8e1
    t153 = (l2 * m2 * qad1 * qf8 * t7) / 4.8e1
    t154 = (l2 * m2 * qad2 * qf8 * t7) / 4.8e1
    t155 = (l2 * m2 * qf8 * qfd4 * t7) / 4.8e1
    t156 = t152 + t153 + t154 + t155 - (m2 * qfd5 * t8) / 2.0 - (m2 * qfd7 * t8) / 4.0 - mp2 * qfd7 * t8 - (
                l2 * m2 * qad1 * t8) / 2.0 - (l2 * m2 * qad2 * t8) / 2.0 - (
                       l2 * m2 * qfd4 * t8) / 2.0 - l2 * mp2 * qad1 * t8 - l2 * mp2 * qad2 * t8 - l2 * mp2 * qfd4 * t8 - (
                       m2 * qad1 * qf5 * t7) / 2.0 - (m2 * qad2 * qf5 * t7) / 2.0 - (m2 * qad1 * qf7 * t7) / 4.0 - (
                       m2 * qad2 * qf7 * t7) / 4.0 - (m2 * qf5 * qfd4 * t7) / 2.0 - (
                       m2 * qf7 * qfd4 * t7) / 4.0 - mp2 * qad1 * qf7 * t7 - mp2 * qad2 * qf7 * t7 - mp2 * qf7 * qfd4 * t7
    t157 = (m1 * qad1 * qf4 * t2) / 8.4e2
    t158 = (l1 * m2 * qad1 * qf5 * t7) / 2.0
    t159 = (l1 * m2 * qad1 * qf7 * t7) / 4.0
    t160 = l1 * mp2 * qad1 * qf7 * t7
    t161 = (l1 * l2 * m2 * qad1 * t8) / 2.0
    t162 = l1 * l2 * mp2 * qad1 * t8
    t163 = (m2 * qad1 * qf3 * qf5 * t8) / 2.0
    t164 = (m2 * qad1 * qf3 * qf7 * t8) / 4.0
    t165 = mp2 * qad1 * qf3 * qf7 * t8
    t171 = m2 * qf5 * t7 * 2.4e1
    t172 = m2 * qf7 * t7 * 1.2e1
    t173 = mp2 * qf7 * t7 * 4.8e1
    t174 = l2 * m2 * t8 * 2.4e1
    t175 = l2 * mp2 * t8 * 4.8e1
    t176 = t171 + t172 + t173 + t174 + t175 - l2 * m2 * qf8 * t7
    t177 = (qad1 * t176) / 4.8e1
    t184 = l2 * qf8 * 1.3e1
    t178 = t131 + t132 - t184
    t179 = (m2 * t106 * t178) / 1.68e3
    t185 = l2 * m2 * qf8 * 2.2e1
    t181 = t137 + t138 + t139 + t140 - t185
    t182 = (t106 * t181) / 1.68e3
    t186 = l2 * qf8 * 4.0
    t183 = t141 + t142 + t143 - t186
    t187 = (l2 * m2 * t106 * t183) / 3.36e3
    C00 = t81 + t82 + t83 + t84 + t85 + t86 + t87 + t88 + t89 + t90 + t91 + t92 + t93 + t94 + t95 + t96 + t97 + t98 + t99 + t100 + t101 + t107 + t108 + t109 + t110 + t111 + m1 * qf1 * qfd1 * (
                1.3e1 / 3.5e1) + m1 * qf1 * qfd3 * (9.0 / 1.4e2) + m1 * qf3 * qfd1 * (
                      9.0 / 1.4e2) + m1 * qf3 * qfd3 * (
                      1.3e1 / 7.0e1) + m2 * qf3 * qfd3 + mp1 * qf3 * qfd3 + mp2 * qf3 * qfd3 - l1 * m1 * qf1 * qfd4 * 7.738095238095238e-3 + l1 * m1 * qf2 * qfd3 * 7.738095238095238e-3 + l1 * m1 * qf3 * qfd2 * 7.738095238095238e-3 - l1 * m1 * qf4 * qfd1 * 7.738095238095238e-3 - l1 * m1 * qf3 * qfd4 * (
                      1.1e1 / 8.4e2) - l1 * m1 * qf4 * qfd3 * (
                      1.1e1 / 8.4e2) - l2 * m2 * qf5 * qfd8 * 7.738095238095238e-3 - l2 * m2 * qf8 * qfd5 * 7.738095238095238e-3 - l2 * m2 * qf7 * qfd8 * (
                      1.1e1 / 8.4e2) - l2 * m2 * qf8 * qfd7 * (1.1e1 / 8.4e2) - (l1 * m2 * qfd5 * t8) / 2.0 - (
                      l1 * m2 * qfd7 * t8) / 4.0 - l1 * mp2 * qfd7 * t8 + (m1 * qf2 * qfd2 * t2) / 4.2e2 - (
                      m1 * qf2 * qfd4 * t2) / 1.12e3 - (m1 * qf4 * qfd2 * t2) / 1.12e3 + (
                      m1 * qf4 * qfd4 * t2) / 8.4e2 - (m2 * qf6 * qfd8 * t3) / 1.12e3 - (
                      m2 * qf8 * qfd6 * t3) / 1.12e3 - (l1 * l2 * m2 * qad2 * t8) / 2.0 - (
                      l1 * l2 * m2 * qfd4 * t8) / 2.0 - l1 * l2 * mp2 * qad2 * t8 - l1 * l2 * mp2 * qfd4 * t8 - (
                      l1 * m2 * qad2 * qf5 * t7) / 2.0 - (l1 * m2 * qad2 * qf7 * t7) / 4.0 - (
                      l1 * m2 * qf5 * qfd4 * t7) / 2.0 - (l1 * m2 * qf7 * qfd4 * t7) / 4.0 - (
                      l2 * m2 * qf3 * qfd8 * t7) / 4.8e1 - (
                      l2 * m2 * qf8 * qfd3 * t7) / 4.8e1 - l1 * mp2 * qad2 * qf7 * t7 - l1 * mp2 * qf7 * qfd4 * t7 - (
                      m2 * qad2 * qf3 * qf5 * t8) / 2.0 - (m2 * qad2 * qf3 * qf7 * t8) / 4.0 - (
                      m2 * qf3 * qf5 * qfd4 * t8) / 2.0 - (
                      m2 * qf3 * qf7 * qfd4 * t8) / 4.0 - mp2 * qad2 * qf3 * qf7 * t8 - mp2 * qf3 * qf7 * qfd4 * t8
    C10 = t81 + t82 + t83 + t84 + t85 + t86 + t87 + t91 + t92 - t102 - t103 - t104 - t105 + t107 + t108 + t109 + t110 + t111 + t158 + t159 + t160 + t161 + t162 + t163 + t164 + t165 - l2 * m2 * qf5 * qfd8 * 7.738095238095238e-3 - l2 * m2 * qf8 * qfd5 * 7.738095238095238e-3 - l2 * m2 * qf7 * qfd8 * (
                1.1e1 / 8.4e2) - l2 * m2 * qf8 * qfd7 * (1.1e1 / 8.4e2) - (m2 * qf6 * qfd8 * t3) / 1.12e3 - (
                      m2 * qf8 * qfd6 * t3) / 1.12e3 - (l2 * m2 * qf8 * qfd3 * t7) / 4.8e1
    C20 = -t147
    C30 = -t151
    C40 = t152 + t153 + t154 + t155 - m1 * qad1 * qf1 * (9.0 / 1.4e2) - m1 * qad1 * qf3 * (
                1.3e1 / 7.0e1) - m2 * qad1 * qf3 - mp1 * qad1 * qf3 - mp2 * qad1 * qf3 - (m2 * qfd5 * t8) / 2.0 - (
                      m2 * qfd7 * t8) / 4.0 - mp2 * qfd7 * t8 - l1 * m1 * qad1 * qf2 * 7.738095238095238e-3 + l1 * m1 * qad1 * qf4 * (
                      1.1e1 / 8.4e2) - (l2 * m2 * qad1 * t8) / 2.0 - (l2 * m2 * qad2 * t8) / 2.0 - (
                      l2 * m2 * qfd4 * t8) / 2.0 - l2 * mp2 * qad1 * t8 - l2 * mp2 * qad2 * t8 - l2 * mp2 * qfd4 * t8 - (
                      m2 * qad1 * qf5 * t7) / 2.0 - (m2 * qad2 * qf5 * t7) / 2.0 - (m2 * qad1 * qf7 * t7) / 4.0 - (
                      m2 * qad2 * qf7 * t7) / 4.0 - (m2 * qf5 * qfd4 * t7) / 2.0 - (
                      m2 * qf7 * qfd4 * t7) / 4.0 - mp2 * qad1 * qf7 * t7 - mp2 * qad2 * qf7 * t7 - mp2 * qf7 * qfd4 * t7
    C50 = t81 + t82 + t83 + t84 + t85 + t86 + t87 + t91 + t92 - t102 - t103 - t104 - t105 + t107 + t108 + t109 + t110 + t111 - t157 + t158 + t159 + t160 + t161 + t162 + t163 + t164 + t165 + l1 * m1 * qad1 * qf1 * 7.738095238095238e-3 + l1 * m1 * qad1 * qf3 * (
                1.1e1 / 8.4e2) - l2 * m2 * qf5 * qfd8 * 7.738095238095238e-3 - l2 * m2 * qf8 * qfd5 * 7.738095238095238e-3 - l2 * m2 * qf7 * qfd8 * (
                      1.1e1 / 8.4e2) - l2 * m2 * qf8 * qfd7 * (1.1e1 / 8.4e2) + (m1 * qad1 * qf2 * t2) / 1.12e3 - (
                      m2 * qf6 * qfd8 * t3) / 1.12e3 - (m2 * qf8 * qfd6 * t3) / 1.12e3 - (
                      l2 * m2 * qf8 * qfd3 * t7) / 4.8e1
    C60 = m2 * (
                qad1 * qf5 * 6.24e2 + qad2 * qf5 * 6.24e2 + qad1 * qf7 * 1.08e2 + qad2 * qf7 * 1.08e2 + qf5 * qfd4 * 6.24e2 + qf7 * qfd4 * 1.08e2 - qfd3 * t8 * 8.4e2 - l2 * qad1 * qf8 * 1.3e1 - l2 * qad2 * qf8 * 1.3e1 - l2 * qf8 * qfd4 * 1.3e1 - l1 * qad1 * t8 * 8.4e2 + qad1 * qf3 * t7 * 8.4e2) * (
              -5.952380952380952e-4)
    C70 = -t136
    C80 = qad2 * t116 * (-1.0 / 2.0) - (qfd4 * t116) / 2.0 - (qad1 * (
                t112 + t113 + t114 + t115 - t166 - (l1 * m2 * t8) / 2.0 - l1 * mp2 * t8 * 2.0 + (
                    m2 * qf3 * t7) / 2.0 + mp2 * qf3 * t7 * 2.0)) / 2.0 + (qfd3 * t8 * t73) / 4.0
    C90 = (l2 * m2 * (
                qad1 * qf5 * 2.6e1 + qad2 * qf5 * 2.6e1 + qad1 * qf7 * 4.4e1 + qad2 * qf7 * 4.4e1 + qf5 * qfd4 * 2.6e1 + qf7 * qfd4 * 4.4e1 - qfd3 * t8 * 7.0e1 + l2 * qad1 * qf6 * 3.0 + l2 * qad2 * qf6 * 3.0 - l2 * qad1 * qf8 * 4.0 - l2 * qad2 * qf8 * 4.0 + l2 * qf6 * qfd4 * 3.0 - l2 * qf8 * qfd4 * 4.0 - l1 * qad1 * t8 * 7.0e1 + qad1 * qf3 * t7 * 7.0e1)) / 3.36e3
    C01 = t81 + t82 + t83 + t84 + t85 + t86 + t87 + t88 + t89 + t90 + t91 + t92 + t93 + t94 + t95 + t96 + t97 + t98 + t99 + t100 + t101 + t102 + t103 + t104 + t105 - l2 * m2 * qf5 * qfd8 * 7.738095238095238e-3 - l2 * m2 * qf8 * qfd5 * 7.738095238095238e-3 - l2 * m2 * qf7 * qfd8 * (
                1.1e1 / 8.4e2) - l2 * m2 * qf8 * qfd7 * (1.1e1 / 8.4e2) - (l1 * m2 * qfd5 * t8) / 2.0 - (
                      l1 * m2 * qfd7 * t8) / 4.0 - l1 * mp2 * qfd7 * t8 - (m2 * qf6 * qfd8 * t3) / 1.12e3 - (
                      m2 * qf8 * qfd6 * t3) / 1.12e3 - (l1 * l2 * m2 * qad1 * t8) / 2.0 - (
                      l1 * l2 * m2 * qad2 * t8) / 2.0 - (
                      l1 * l2 * m2 * qfd4 * t8) / 2.0 - l1 * l2 * mp2 * qad1 * t8 - l1 * l2 * mp2 * qad2 * t8 - l1 * l2 * mp2 * qfd4 * t8 - (
                      l1 * m2 * qad1 * qf5 * t7) / 2.0 - (l1 * m2 * qad2 * qf5 * t7) / 2.0 - (
                      l1 * m2 * qad1 * qf7 * t7) / 4.0 - (l1 * m2 * qad2 * qf7 * t7) / 4.0 - (
                      l1 * m2 * qf5 * qfd4 * t7) / 2.0 - (l1 * m2 * qf7 * qfd4 * t7) / 4.0 - (
                      l2 * m2 * qf3 * qfd8 * t7) / 4.8e1 - l1 * mp2 * qad1 * qf7 * t7 - l1 * mp2 * qad2 * qf7 * t7 - l1 * mp2 * qf7 * qfd4 * t7 - (
                      m2 * qad1 * qf3 * qf5 * t8) / 2.0 - (m2 * qad2 * qf3 * qf5 * t8) / 2.0 - (
                      m2 * qad1 * qf3 * qf7 * t8) / 4.0 - (m2 * qad2 * qf3 * qf7 * t8) / 4.0 - (
                      m2 * qf3 * qf5 * qfd4 * t8) / 2.0 - (
                      m2 * qf3 * qf7 * qfd4 * t8) / 4.0 - mp2 * qad1 * qf3 * qf7 * t8 - mp2 * qad2 * qf3 * qf7 * t8 - mp2 * qf3 * qf7 * qfd4 * t8
    C11 = t130
    C21 = 0.0
    C31 = 0.0
    C41 = t156
    C51 = t130
    C61 = -t179
    C71 = -t136
    C81 = -t182
    C91 = t187
    C02 = t147
    C12 = 0.0
    C22 = 0.0
    C32 = 0.0
    C42 = 0.0
    C52 = 0.0
    C62 = 0.0
    C72 = 0.0
    C82 = 0.0
    C92 = 0.0
    C03 = t151
    C13 = 0.0
    C23 = 0.0
    C33 = 0.0
    C43 = 0.0
    C53 = 0.0
    C63 = 0.0
    C73 = 0.0
    C83 = 0.0
    C93 = 0.0
    C04 = (qad1 * (m1 * qf1 * (9.0 / 7.0e1) + m1 * qf3 * (
                1.3e1 / 3.5e1) + m2 * qf3 * 2.0 + mp1 * qf3 * 2.0 + mp2 * qf3 * 2.0 + l1 * m1 * qf2 * (
                               1.3e1 / 8.4e2) - l1 * m1 * qf4 * (
                               1.1e1 / 4.2e2) + l2 * m2 * t8 + l2 * mp2 * t8 * 2.0 + m2 * qf5 * t7 + (
                               m2 * qf7 * t7) / 2.0 + mp2 * qf7 * t7 * 2.0 - (l2 * m2 * qf8 * t7) / 2.4e1)) / 2.0
    C14 = t177
    C24 = 0.0
    C34 = 0.0
    C44 = 0.0
    C54 = t177
    C64 = (m2 * qad1 * t8) / 2.0
    C74 = 0.0
    C84 = (qad1 * t8 * t73) / 4.0
    C94 = l2 * m2 * qad1 * t8 * (-1.0 / 4.8e1)
    C05 = t81 + t82 + t83 + t84 + t85 + t86 + t87 + t88 + t89 + t90 + t91 + t92 + t93 + t94 + t95 + t96 + t97 + t98 + t99 + t100 + t101 + t102 + t103 + t104 + t105 + t157 - l1 * m1 * qad1 * qf1 * 7.738095238095238e-3 - l1 * m1 * qad1 * qf3 * (
                1.1e1 / 8.4e2) - l2 * m2 * qf5 * qfd8 * 7.738095238095238e-3 - l2 * m2 * qf8 * qfd5 * 7.738095238095238e-3 - l2 * m2 * qf7 * qfd8 * (
                      1.1e1 / 8.4e2) - l2 * m2 * qf8 * qfd7 * (1.1e1 / 8.4e2) - (l1 * m2 * qfd5 * t8) / 2.0 - (
                      l1 * m2 * qfd7 * t8) / 4.0 - l1 * mp2 * qfd7 * t8 - (m1 * qad1 * qf2 * t2) / 1.12e3 - (
                      m2 * qf6 * qfd8 * t3) / 1.12e3 - (m2 * qf8 * qfd6 * t3) / 1.12e3 - (
                      l1 * l2 * m2 * qad1 * t8) / 2.0 - (l1 * l2 * m2 * qad2 * t8) / 2.0 - (
                      l1 * l2 * m2 * qfd4 * t8) / 2.0 - l1 * l2 * mp2 * qad1 * t8 - l1 * l2 * mp2 * qad2 * t8 - l1 * l2 * mp2 * qfd4 * t8 - (
                      l1 * m2 * qad1 * qf5 * t7) / 2.0 - (l1 * m2 * qad2 * qf5 * t7) / 2.0 - (
                      l1 * m2 * qad1 * qf7 * t7) / 4.0 - (l1 * m2 * qad2 * qf7 * t7) / 4.0 - (
                      l1 * m2 * qf5 * qfd4 * t7) / 2.0 - (l1 * m2 * qf7 * qfd4 * t7) / 4.0 - (
                      l2 * m2 * qf3 * qfd8 * t7) / 4.8e1 - l1 * mp2 * qad1 * qf7 * t7 - l1 * mp2 * qad2 * qf7 * t7 - l1 * mp2 * qf7 * qfd4 * t7 - (
                      m2 * qad1 * qf3 * qf5 * t8) / 2.0 - (m2 * qad2 * qf3 * qf5 * t8) / 2.0 - (
                      m2 * qad1 * qf3 * qf7 * t8) / 4.0 - (m2 * qad2 * qf3 * qf7 * t8) / 4.0 - (
                      m2 * qf3 * qf5 * qfd4 * t8) / 2.0 - (
                      m2 * qf3 * qf7 * qfd4 * t8) / 4.0 - mp2 * qad1 * qf3 * qf7 * t8 - mp2 * qad2 * qf3 * qf7 * t8 - mp2 * qf3 * qf7 * qfd4 * t8
    C15 = t130
    C25 = 0.0
    C35 = 0.0
    C45 = t156
    C55 = t130
    C65 = -t179
    C75 = -t136
    C85 = -t182
    C95 = t187
    C06 = (m2 * t106 * (t131 + t132 - l2 * qf8 * 1.3e1 - l1 * t8 * 8.4e2 + qf3 * t7 * 8.4e2)) / 1.68e3
    C16 = t179
    C26 = 0.0
    C36 = 0.0
    C46 = m2 * t8 * t106 * (-1.0 / 2.0)
    C56 = t179
    C66 = 0.0
    C76 = 0.0
    C86 = 0.0
    C96 = 0.0
    C07 = t136
    C17 = t136
    C27 = 0.0
    C37 = 0.0
    C47 = 0.0
    C57 = t136
    C67 = 0.0
    C77 = 0.0
    C87 = 0.0
    C97 = 0.0
    C08 = (t106 * (
                t137 + t138 + t139 + t140 - l2 * m2 * qf8 * 2.2e1 - l1 * m2 * t8 * 4.2e2 - l1 * mp2 * t8 * 1.68e3 + m2 * qf3 * t7 * 4.2e2 + mp2 * qf3 * t7 * 1.68e3)) / 1.68e3
    C18 = t182
    C28 = 0.0
    C38 = 0.0
    C48 = t8 * t73 * t106 * (-1.0 / 4.0)
    C58 = t182
    C68 = 0.0
    C78 = 0.0
    C88 = 0.0
    C98 = 0.0
    C09 = l2 * m2 * t106 * (t141 + t142 + t143 - l2 * qf8 * 4.0 - l1 * t8 * 7.0e1 + qf3 * t7 * 7.0e1) * (
        -2.976190476190476e-4)
    C19 = l2 * m2 * t106 * t183 * (-2.976190476190476e-4)
    C29 = 0.0
    C39 = 0.0
    C49 = (l2 * m2 * t8 * t106) / 4.8e1
    C59 = l2 * m2 * t106 * t183 * (-2.976190476190476e-4)
    C69 = 0.0
    C79 = 0.0
    C89 = 0.0
    C99 = 0.0
    C = np.array([
        [C00, C01, C02, C03, C04, C05, C06, C07, C08, C09],
        [C10, C11, C12, C13, C14, C15, C16, C17, C18, C19],
        [C20, C21, C22, C23, C24, C25, C26, C27, C28, C29],
        [C30, C31, C32, C33, C34, C35, C36, C37, C38, C39],
        [C40, C41, C42, C43, C44, C45, C46, C47, C48, C49],
        [C50, C51, C52, C53, C54, C55, C56, C57, C58, C59],
        [C60, C61, C62, C63, C64, C65, C66, C67, C68, C69],
        [C70, C71, C72, C73, C74, C75, C76, C77, C78, C79],
        [C80, C81, C82, C83, C84, C85, C86, C87, C88, C89],
        [C90, C91, C92, C93, C94, C95, C96, C97, C98, C99]])
    # K
    t188 = 1.0 / l1 ** 3
    t189 = 1.0 / l1 ** 2
    t190 = EI1 * t189 * 2.4e1
    t191 = 1.0 / l1
    t192 = EI1 * t191 * 4.0
    t193 = 1.0 / l2 ** 3
    t194 = 1.0 / l2 ** 2
    t195 = EI2 * t194 * 2.4e1
    t196 = 1.0 / l2
    t197 = EI2 * t196 * 4.0
    K00 = 0.0
    K10 = 0.0
    K20 = 0.0
    K30 = 0.0
    K40 = 0.0
    K50 = 0.0
    K60 = 0.0
    K70 = 0.0
    K80 = 0.0
    K90 = 0.0
    K01 = K10
    K11 = 0.0
    K21 = 0.0
    K31 = 0.0
    K41 = 0.0
    K51 = 0.0
    K61 = 0.0
    K71 = 0.0
    K81 = 0.0
    K91 = 0.0
    K02 = K20
    K12 = K21
    K22 = EI1 * t188 * 1.92e2
    K32 = 0.0
    K42 = EI1 * t188 * -9.6e1
    K52 = t190
    K62 = 0.0
    K72 = 0.0
    K82 = 0.0
    K92 = 0.0
    K03 = K30
    K13 = K31
    K23 = K32
    K33 = EI1 * t191 * 1.6e1
    K43 = -t190
    K53 = t192
    K63 = 0.0
    K73 = 0.0
    K83 = 0.0
    K93 = 0.0
    K04 = K40
    K14 = K41
    K24 = K42
    K34 = K43
    K44 = EI1 * t188 * 9.6e1
    K54 = -t190
    K64 = 0.0
    K74 = 0.0
    K84 = 0.0
    K94 = 0.0
    K05 = K50
    K15 = K51
    K25 = K52
    K35 = K53
    K45 = K54
    K55 = EI1 * t191 * 8.0
    K65 = 0.0
    K75 = 0.0
    K85 = 0.0
    K95 = 0.0
    K06 = K60
    K16 = K61
    K26 = K62
    K36 = K63
    K46 = K64
    K56 = K65
    K66 = EI2 * t193 * 1.92e2
    K76 = 0.0
    K86 = EI2 * t193 * -9.6e1
    K96 = t195
    K07 = K70
    K17 = K71
    K27 = K72
    K37 = K73
    K47 = K74
    K57 = K75
    K67 = K76
    K77 = EI2 * t196 * 1.6e1
    K87 = -t195
    K97 = t197
    K08 = K80
    K18 = K81
    K28 = K82
    K38 = K83
    K48 = K84
    K58 = K85
    K68 = K86
    K78 = K87
    K88 = EI2 * t193 * 9.6e1
    K98 = -t195
    K09 = K90
    K19 = K91
    K29 = K92
    K39 = K93
    K49 = K94
    K59 = K95
    K69 = K96
    K79 = K97
    K89 = K98
    K99 = EI2 * t196 * 8.0
    K = np.array([
        [K00, K01, K02, K03, K04, K05, K06, K07, K08, K09],
        [K10, K11, K12, K13, K14, K15, K16, K17, K18, K19],
        [K20, K21, K22, K23, K24, K25, K26, K27, K28, K29],
        [K30, K31, K32, K33, K34, K35, K36, K37, K38, K39],
        [K40, K41, K42, K43, K44, K45, K46, K47, K48, K49],
        [K50, K51, K52, K53, K54, K55, K56, K57, K58, K59],
        [K60, K61, K62, K63, K64, K65, K66, K67, K68, K69],
        [K70, K71, K72, K73, K74, K75, K76, K77, K78, K79],
        [K80, K81, K82, K83, K84, K85, K86, K87, K88, K89],
        [K90, K91, K92, K93, K94, K95, K96, K97, K98, K99]])
    return M, C, K


def next_step_newmark(q, qd, qdd, tau_a, dt, alpha, beta, l, m, mp, Js, EI):
    # Simulates q(t+dt), qd(t+dt), qdd(t+dt) by newmark through a single simulation step
    # Function: M qdd + C qd + K q + (aM + bK)qd = [tau_a, tau_p]
    # Newmark parameter
    beta_newmark = 1./4.
    max_iter = 20
    # Update the next step
    q_tilde = q + dt * qd + (1. - 2. * beta_newmark) / 2. * dt ** 2. * qdd
    qd_tilde = qd + dt / 2. * qdd
    qdd_next = qdd * 1.  # Attention: qdd is a list, this makes a copy
    qd_next = qd_tilde + dt / 2. * qdd_next
    q_next = q_tilde + beta_newmark * dt ** 2. * qdd_next
    tau_p = np.zeros(8, dtype=np.float32)
    tau = np.hstack((tau_a, tau_p))
    for k in range(max_iter):
        [M, C, K] = calc_paras(q, qd, l, m, mp, Js, EI)
        D = alpha * M + beta * K
        D[0:2, :] = 0.
        D[:, 0:2] = 0.
        epsilon = tau - np.dot(K, q_next) - np.dot(C + D, qd_next) - np.dot(M, qdd_next)
        delta_qdd = np.dot(np.linalg.inv(M + (C + D) / 2 * dt + K * beta_newmark * dt ** 2), epsilon)
        qdd_next = qdd_next + delta_qdd
        q_next = q_tilde + beta_newmark * dt ** 2. * qdd_next
        qd_next = qd_tilde + dt / 2. * qdd_next
        if np.linalg.norm(epsilon) < 1e-10 / 10.:
            break
    return q_next, qd_next, qdd_next


def plot_F2sF2s(q, l):
    # l: length of the FM; w: width of the FM; r: radius of end-point; pix_per_m: pixels per meter
    le1 = l[0] / 2.
    le2 = l[1] / 2.
    w = 0.11 * np.min(l)  # link width
    d = 2.5 * w  # motor length
    ne1 = 5
    ne2 = 5
    # Common variables
    Rl1_left = cm.rotation_matrix(q[0])
    Rl1_right = cm.rotation_matrix(q[5])
    Rl2_left = cm.rotation_matrix(q[5] + q[1])
    # Motor 1
    area_motor1 = cm.motor(d)
    # Link 1
    ql1e1 = np.array([0., 0., q[2], q[3]])
    x1 = np.delete(np.linspace(0., le1, ne1 + 1), -1)
    x1 = x1.reshape(-1, 1)
    y1 = cm.spline_inter(x1, ql1e1, le1)
    ql1e2 = np.array([q[2], q[3], q[4], q[5]])
    x2 = np.linspace(0., le1, ne1)
    x2 = x2.reshape(-1, 1)
    y2 = cm.spline_inter(x2, ql1e2, le1)
    x2 = x2 + le1
    ele_line_1 = np.vstack((np.hstack((x1, y1)), np.hstack((x2, y2))))  # 2*ne1
    line_link1 = np.dot(ele_line_1, Rl1_left)
    # link 2
    ql2e1 = np.array([0., 0., q[6], q[7]])
    x1 = np.delete(np.linspace(0., le2, ne2 + 1), -1)  # x1 = np.linspace(0., le, ne, endpoint=False)
    x1 = x1.reshape(-1, 1)
    y1 = cm.spline_inter(x1, ql2e1, le2)
    ql2e2 = np.array([q[6], q[7], q[8], q[9]])
    x2 = np.linspace(0., le2, ne2)
    x2 = x2.reshape(-1, 1)
    y2 = cm.spline_inter(x2, ql2e2, le2)
    x2 = x2 + le2
    ele_line_2 = np.vstack((np.hstack((x1, y1)), np.hstack((x2, y2))))
    line_link2 = np.dot(np.array([l[0], q[4]]) + np.dot(ele_line_2, Rl2_left), Rl1_left)
    return line_link1, line_link2


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
    xp = np.dot(np.array([l[0], 0]) + np.dot(np.array([l[1], 0]), cm.rotation_matrix(task_lst[1])),
                cm.rotation_matrix(task_lst[0]))
    return xp


class PD_controller(object):
    def __init__(self):
        self.kp = np.array([350., 150.])
        self.kd = np.array([2., 0.5])
        self.q0_info = None
        self.q0 = None
        self.q0d = None
        self.q0dd = None

    def update_q0_info(self, received_info):
        self.q0_info = received_info

    def calc_tau_a(self, t, q, qd):
        self.q0, self.q0d, self.q0dd = calc_LS(t, self.q0_info)
        tau_a = - self.kp*(q[0:2]-self.q0) - self.kd * (qd[0:2]-self.q0d)
        return tau_a


class OPD_controller(object):
    def __init__(self):
        self.kp = np.array([350., 150.])
        self.kd = np.array([2., 0.5])
        self.kc = 0.00003  # 0.0001
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
        e = q[0:2] - self.q0
        ed = qd[0:2] - self.q0d
        tau_a = - self.kp * e - self.kd * ed
        f = np.dot(self.kp * e, ed) / np.max(self.kp)
        s = np.clip(f/self.kc + 1., 0., 1.)
        if t > self.te:
            tau_a = -s * self.kp * e - self.kd * ed
        return tau_a


class F2sF2s_manipulator(object):
    # System parameters
    l = np.array([0.35, 0.3])
    m = np.array([0.49, 0.42])
    mp = np.array([1., 0.157])
    Js = np.array([0.1, 0.1])
    EI = np.array([15., 5.])
    alpha = 0.02
    beta = 0.02
    pix_per_m = 250

    def __init__(self, mode=0):
        self.q = np.zeros(10)
        self.qd = np.zeros(10)
        self.qdd = np.zeros(10)
        self.t_rec = None
        self.q_rec = None
        self.qd_rec = None
        self.q0_rec = None
        if mode == 0:
            self.controller = PD_controller()
        else:
            self.controller = OPD_controller()

    def set_manipulator(self, qa):
        self.q = np.hstack((qa, np.zeros(8)))
        self.qd = np.zeros(10)
        self.qdd = np.zeros(10)

    def simulator(self, tspan, dt):
        n = (tspan[1] - tspan[0]) / dt  # Find step number that suits the recommendation dt best
        n = int(n)
        t_step = (tspan[1] - tspan[0]) / n  # Real simulation step size
        t = tspan[0]
        #
        q = self.q  # self.q will not auto update here
        qd = self.qd
        qdd = self.qdd
        # initialize recorder
        self.t_rec = np.zeros(n)
        self.q_rec = np.zeros((n, 10))
        self.qd_rec = np.zeros((n, 10))
        self.q0_rec = np.zeros((n, 2))
        for i in range(n):
            # recording [ts, te)
            self.t_rec[i] = t
            self.q_rec[i, :] = q
            self.qd_rec[i, :] = qd
            self.q0_rec[i, :] = self.controller.q0
            # update time step
            t = t + t_step
            # Calculate control torque
            tau_a = self.controller.calc_tau_a(t, q, qd)
            # Update the next step
            q, qd, qdd = next_step_newmark(q, qd, qdd, tau_a, t_step, self.alpha, self.beta, self.l, self.m, self.mp,
                                           self.Js, self.EI)
        self.q = q
        self.qd = qd
        self.qdd = qdd

    def get_end(self):
        l = self.l
        x_end = np.dot(np.array([l[0], self.q[4]]) + np.dot(np.array([l[1], self.q[8]]),
                                                            cm.rotation_matrix(self.q[5] + self.q[1])),
                       cm.rotation_matrix(self.q[0]))
        return x_end

    def draw_poly(self):
        return plot_F2sF2s(self.q, self.l, self.pix_per_m)



if __name__=='__main__':
    mach = F2sF2s_manipulator()
