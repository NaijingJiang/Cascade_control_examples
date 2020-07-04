import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # only show the error

class brain:
    def __init__(self, env_name):
        if env_name == 'env_F2sF2s':
            N_state = 21
            N_action = 2
            abound = np.array([30., 30.])
        elif env_name == 'env_Cart_P2':
            N_state = 13
            N_action = 1
            abound = np.array([10.])
        else:
            N_state = 1
            N_action = 1
            abound = np.array([1.])
            print('Incorrect input, try: env_Cart_P2 or env_F2sF2s')
        self.N_states = N_state  # Dimension of state
        self.N_actions = N_action  # Dimension of action
        self.action_bound_min = abound * -1.
        self.action_bound_max = abound
        self.env_name = env_name
        # ---------------other parameters----------------------
        self.state = []
        self.action = []
        self.Time_step = 0  # Total time step played
        self.xs = []  # Input of Q
        self.xa = []  # Input of Q
        self.Critic_evaluate = []  # Evaluate output of the value Q(s,a)|a=mu(s)
        self.Critic_target = []  # Target output
        self.Actor_evaluate = []  # Evaluate output of a=mu(s)
        self.Actor_target = []  # Target output of a=mu(s)
        # build tf structure
        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=2)

    def choose_action(self, state):
        action = self.sess.run(self.Actor_evaluate, feed_dict={self.xs: state[np.newaxis, :]})[0]
        return action

    def build_net(self):
        # Placeholders
        with tf.name_scope('State_Input'):
            self.xs = tf.placeholder(tf.float32, [None, self.N_states])
        with tf.name_scope('Action_Input'):
            self.xa = tf.placeholder(tf.float32, [None, self.N_actions])
        with tf.name_scope('Critic'):
            self.Critic_evaluate = self.build_Critic_net('Eval')
            self.Critic_target = self.build_Critic_net('Targ')
        with tf.name_scope('Actor'):
            self.Actor_evaluate = self.build_Actor_net('Eval')
            self.Actor_target = self.build_Actor_net('Targ')

    def build_Critic_net(self, name):
        initializer = tf.initializers.he_uniform()
        with tf.variable_scope('Critic_' + name):
            with tf.variable_scope('weights'):
                wf1 = tf.get_variable('wf1', [self.N_states + self.N_actions, 64], dtype=tf.float32,
                                      initializer=initializer)
                wf2 = tf.get_variable('wf2', [64, 64], dtype=tf.float32,
                                      initializer=initializer)
                wfo = tf.get_variable('wfo', [64, 1], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.0001))
                tf.summary.histogram('Critic_' + name + '_wf1', wf1)
                tf.summary.histogram('Critic_' + name + '_wf2', wf2)
                tf.summary.histogram('Critic_' + name + '_wfo', wfo)
            with tf.variable_scope('bias'):
                bf1 = tf.get_variable('bf1', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bf2 = tf.get_variable('bf2', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bfo = tf.get_variable('bfo', [1], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.00001))
        with tf.name_scope(name):
            with tf.name_scope('Merge'):
                fc_input = tf.concat([self.xs, self.xa], 1)
            with tf.name_scope('Fc1'):
                fc1 = tf.nn.relu(tf.matmul(fc_input, wf1) + bf1)  # 500
            with tf.name_scope('Fc2'):
                fc2 = tf.nn.relu(tf.matmul(fc1, wf2) + bf2)  # 500
            with tf.name_scope('Fco'):
                net_output = tf.matmul(fc2, wfo) + bfo  # 500
        return net_output

    def build_Actor_net(self, name):
        initializer = tf.initializers.he_uniform()
        with tf.variable_scope('Actor_' + name):
            with tf.variable_scope('weights'):
                wf1 = tf.get_variable('wf1', [self.N_states, 64], dtype=tf.float32,
                                      initializer=initializer)
                wf2 = tf.get_variable('wf2', [64, 64], dtype=tf.float32,
                                      initializer=initializer)
                wfo = tf.get_variable('wfo', [64, self.N_actions], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.0001))
                tf.summary.histogram('Actor_' + name + '_wf1', wf1)
                tf.summary.histogram('Actor_' + name + '_wf2', wf2)
                tf.summary.histogram('Actor_' + name + '_wfo', wfo)
            with tf.variable_scope('bias'):
                bf1 = tf.get_variable('bf1', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bf2 = tf.get_variable('bf2', [64], dtype=tf.float32, initializer=tf.constant_initializer(0))
                bfo = tf.get_variable('bfo', [self.N_actions], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.00001))
        with tf.name_scope(name):
            with tf.name_scope('Merge'):
                fc_input = self.xs
            with tf.name_scope('Fc1'):
                fc1 = tf.nn.relu(tf.matmul(fc_input, wf1) + bf1)  # 500
            with tf.name_scope('Fc2'):
                fc2 = tf.nn.relu(tf.matmul(fc1, wf2) + bf2)  # 500
            with tf.name_scope('Fco'):
                net_output = (self.action_bound_max + self.action_bound_min) / 2\
                             + (self.action_bound_max - self.action_bound_min) / 2 *tf.nn.tanh(tf.matmul(fc2, wfo) + bfo)  # 700
        return net_output

    def load_Agent(self, seed=None):
        if seed == None:
            if self.env_name == 'env_F2sF2s':
                seed = 5
            else:
                seed = 1
        try:
            net_name = 'Agents/Saved_net/DDPG_data/' + self.env_name + '/seed_' + str(seed) + '/last_net'
            self.saver = tf.train.import_meta_graph(net_name + '.meta')
            self.saver.restore(self.sess, net_name)
            print('Successfully loaded')
        except:
            print('No net found')
            return False
