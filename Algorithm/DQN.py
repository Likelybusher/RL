import gym
import tensorflow as tf
import numpy as np

# config
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 100  # The number of experiment test every 100 episode


class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_, done]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.memory_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.q_eval_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.not_done = tf.placeholder(tf.float32, [None, ], name='not_done')  # whether input Next State is terminal

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_') * self.not_done
            #  shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # a_indices = [[i, a]] -> gather_nd -> [[Q(a_i)]]
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        with tf.variable_scope('avg_q_eval'):
            self.q_eval_avg = tf.reduce_mean(self.q_eval_wrt_a)

    def store_transition(self, s, a, r, s_, done):

        transition = np.hstack((s, [a, r], s_, done))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, test_button=False):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon or test_button:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost, q_avg = self.sess.run(
            [self._train_op, self.loss, self.q_eval_avg],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features-1: -1],
                self.not_done: batch_memory[:, -1],
            })

        self.cost_his.append(cost)
        self.q_eval_his.append(q_avg)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def plot_q(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.q_eval_his)), self.q_eval_his)
        plt.ylabel('q_avg')
        plt.xlabel('training steps')
        plt.show()


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(2, 4, e_greedy_increment=True)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            # env.render()
            action = agent.choose_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            not_done = 0 if done else 1
            agent.store_transition(state, action, reward, next_state, not_done)
            agent.learn()
            state = next_state
            if done:
                break
        # Test
        if episode % 100 == 0:
            total_reward = 0
            for test_episode in range(TEST):
                # initialize task
                state = env.reset()
                # Test
                for step in range(STEP):
                    # env.render()
                    action = agent.choose_action(state, test_button=True)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    total_reward += 1
                    if done:
                        break
            print("average reward after %d episodes:" % episode, total_reward / TEST)
            if total_reward / TEST > 195:
                break

    # view
    for test_episode in range(10):
        state = env.reset()
        for step in range(STEP):
            env.render()
            action = agent.choose_action(state, test_button=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break

    # agent.plot_cost()
    agent.plot_q()  # average q


if __name__ == '__main__':
    main()

