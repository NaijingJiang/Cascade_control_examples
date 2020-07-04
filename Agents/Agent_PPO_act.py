import time
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # only show the error


class brain(object):
    def __init__(self, env_name='env_F2sF2s'):
        if env_name == 'env_F2sF2s':
            N_state = 21
            N_action = 2
            sig_ini=np.array([30.0, 30.0])
        elif env_name == 'env_Cart_P2':
            N_state = 13
            N_action = 1
            sig_ini=np.array([10.0])
        else:
            N_state = 1
            N_action = 1
            sig_ini=np.array([1.0])
            print('Incorrect input, try: env_Cart_P2 or env_F2sF2s')
        # hyper parameters
        self.N_state = N_state
        self.N_action = N_action
        self.sig_ini = sig_ini
        self.env_name = env_name
        #
        # policy net
        self.xs = None
        self.xa = None
        self.action_mu = None
        self.critic = None
        # set tf structure
        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=2)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.action_mu, feed_dict={self.xs: s})[0]
        return a

    def build_net(self):
        # placeholders
        with tf.name_scope('state_input'):
            self.xs = tf.placeholder(tf.float32, [None, self.N_state], 's')
        with tf.name_scope('action_input'):
            self.xa = tf.placeholder(tf.float32, [None, self.N_action], 'a_value')
        self.build_actor_net()
        self.build_critic_net()

    def build_actor_net(self):
        # define variables
        with tf.variable_scope('actor_variables'):
            initializer = tf.contrib.layers.xavier_initializer()  # for tanh activation
            with tf.variable_scope('weights'):
                wf1 = tf.get_variable('wf1', [self.N_state, 64], dtype=tf.float32, initializer=initializer)
                wf2 = tf.get_variable('wf2', [64, 64], dtype=tf.float32, initializer=initializer)
                wfo = tf.get_variable('wfo', [64, self.N_action], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.0001))
            with tf.variable_scope('bias'):
                bf1 = tf.get_variable('bf1', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bf2 = tf.get_variable('bf2', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bfo = tf.get_variable('bfo', [self.N_action], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.00001))
            with tf.variable_scope('sig_variables'):
                self.action_sig = tf.get_variable('sig', [self.N_action],
                                                  initializer=tf.constant_initializer(self.sig_ini))
        with tf.name_scope('action_mu_net'):
            with tf.name_scope('Fc1'):
                fc1 = tf.nn.tanh(tf.matmul(self.xs, wf1) + bf1)
            with tf.name_scope('Fc2'):
                fc2 = tf.nn.tanh(tf.matmul(fc1, wf2) + bf2)
            with tf.name_scope('Fco'):
                self.action_mu = tf.matmul(fc2, wfo) + bfo

    def build_critic_net(self):
        # define variables
        with tf.variable_scope('critic_variables'):
            initializer = tf.contrib.layers.xavier_initializer()  # for tanh activation
            with tf.variable_scope('weights'):
                wf1 = tf.get_variable('wf1', [self.N_state, 64], dtype=tf.float32, initializer=initializer)
                wf2 = tf.get_variable('wf2', [64, 64], dtype=tf.float32, initializer=initializer)
                wfo = tf.get_variable('wfo', [64, 1], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.0001))
            with tf.variable_scope('bias'):
                bf1 = tf.get_variable('bf1', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bf2 = tf.get_variable('bf2', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bfo = tf.get_variable('bfo', [1], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.00001))
        with tf.name_scope('critic_net'):
            with tf.name_scope('Fc1'):
                fc1 = tf.nn.tanh(tf.matmul(self.xs, wf1) + bf1)
            with tf.name_scope('Fc2'):
                fc2 = tf.nn.tanh(tf.matmul(fc1, wf2) + bf2)
            with tf.name_scope('Fco'):
                self.critic = tf.matmul(fc2, wfo) + bfo

    def load_Agent(self, seed=None):
        if seed == None:
            if self.env_name == 'env_F2sF2s':
                seed = 1
            else:
                seed = 1
        try:
            net_name = 'Agents/Saved_net/PPO_data/' + self.env_name + '/seed_' + str(seed) + '/last_net'
            self.saver = tf.train.import_meta_graph(net_name + '.meta')
            self.saver.restore(self.sess, net_name)
            print('Successfully loaded')
        except:
            print('No net found')
            return False

