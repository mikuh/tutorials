import tensorflow as tf
import numpy as np
from collections import deque
from DRL.agents import Agent
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

    def __init__(self, env_config):
        self.lr = env_config.lr
        self.gamma = env_config.gamma
        self.epsilon = env_config.epsilon

        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = tf.keras.optimizers.Adam(lr=self.lr, )

        self.batch_size = 64
        self.state_size = 4
        self.action_size = 2

        self.memory = deque(maxlen=2000)

        self.env = gym.make('CartPole-v1')

        self.score = 0

    def collect_transitions(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, obs):

        return 1

    def learn(self):

        episodes = 0
        steps = 0
        while True:
            obs = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.get_action(obs)
                next_state, reward, done, _ = self.env.step(action)
                self.collect_transitions(obs, action, reward, next_state, done)
                steps += 1
                score += reward

                obs = next_state

                if len(self.memory) > self.batch_size:
                    self.update()
                    if steps % 20 == 0:
                        self.update_target()



