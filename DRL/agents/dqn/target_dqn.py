import tensorflow as tf
from types import SimpleNamespace
import numpy as np
from collections import deque
from DRL.agents import Agent
import random
import gym


class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(2)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value


class TargetDQNAgent(object):

    def __init__(self, env_config: SimpleNamespace):
        self.lr = env_config.lr
        self.gamma = env_config.gamma

        self.env = gym.make(env_config.env)

        self.batch_size = env_config.batch_size
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.memory = deque(maxlen=2000)

        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = tf.keras.optimizers.Adam(lr=self.lr, )

        self.score = 0
        self.steps = 0
        self.episodes = 0

    def collect_transitions(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, obs):
        q_value = self.dqn_model(np.array([obs], dtype=np.float32))[0]
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action

    def update(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*[(e[0], e[1], e[2], e[3], e[4]) for e in mini_batch])

        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            rewards = np.array(rewards, dtype=np.float32)
            actions = np.array(actions, dtype=np.int32)
            dones = np.array(dones, dtype=np.float32)

            target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))
            next_action = tf.argmax(target_q, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * target_q, axis=1)

            target_value = (1 - dones) * self.gamma * target_value + rewards

            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * main_q, axis=1)

            error = tf.square(main_value - target_value) * 0.5
            error = tf.reduce_mean(error)

        dqn_grads = tape.gradient(error, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

        if self.steps % 20 == 0:
            self.dqn_target.set_weights(self.dqn_model.get_weights())

    @property
    def epsilon(self):
        return 1 / (self.episodes * 0.1 + 1)

    def learn(self):

        while True:
            obs = self.env.reset()
            done = False
            score = 0
            self.episodes += 1
            while not done:
                action = self.get_action(obs)
                self.steps += 1
                next_state, reward, done, _ = self.env.step(action)
                self.collect_transitions(obs, action, reward, next_state, done)
                score += reward
                obs = next_state
                if len(self.memory) > self.batch_size:
                    self.update()
            print(score)


if __name__ == '__main__':
    env_config = {
        "env": "CartPole-v1",
        "lr": 0.001,
        "gamma": 0.99,
        "batch_size": 64
    }

    env_config = SimpleNamespace(**env_config)

    agent = TargetDQNAgent(env_config)
    agent.learn()
