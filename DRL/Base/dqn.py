import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from collections import deque
import random
import gym



class DQN(tf.keras.Model):

    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.layer3 = tf.keras.layers.Flatten()
        self.layer4 = tf.keras.layers.Dense(256, activation='relu')
        self.value = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = state / 255
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        value = self.value(x)
        return value


class Player(object):

    def __init__(self, config: SimpleNamespace):
        self.env = gym.make(config.env_name)

        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=config.memory_size)
        self.model = DQN(self.action_size)

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr,)
        self.summary_writer = tf.summary.create_file_writer("logdir")

    def _collect_transitions(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, (1-done)*self.gamma))

    def _get_action(self, obs):
        q_value = self.model(np.array([obs], dtype=np.float32))[0]
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action

    def _update_param(self, step):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, gammas = zip(*[(e[0], e[1], e[2], e[3], e[4]) for e in batch])
        with tf.GradientTape() as tape:
            rewards = np.array(rewards, dtype=np.float32)
            actions = np.array(actions, dtype=np.int32)
            gammas = np.array(gammas, dtype=np.float32)
            q_next = self.model(tf.convert_to_tensor(np.array(next_states), dtype=tf.float32))
            td_target = gammas * tf.reduce_max(q_next, axis=1) + rewards

            q = self.model(tf.convert_to_tensor(np.array(states), dtype=tf.float32))
            q_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * q, axis=1)
            td_error = q_value - td_target
            loss = tf.reduce_mean(tf.square(td_error)*0.5)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

    @property
    def epsilon(self):
        return 1 / (self.episodes * 0.1 + 1)

    def learn(self):
        self.episodes = 0
        step = 0
        while True:
            obs = self.env.reset()
            done = False
            score = 0
            self.episodes += 1
            while not done:
                self.env.render()
                action = self._get_action(obs)
                next_state, reward, done, _ = self.env.step(action)
                self._collect_transitions(obs, action, reward, next_state, done)
                score += reward
                obs = next_state
                step += 1
                if len(self.memory) > self.batch_size:
                    self._update_param(step=step)
            print(f"{self.episodes} episode, score: {score}")
            with self.summary_writer.as_default():
                tf.summary.scalar('score', score, step=self.episodes)



if __name__ == '__main__':

    config = {
        "env_name": "Breakout-v0",  # CartPole-v1  SpaceInvaders-v0
        "lr": 0.001,
        "gamma": 0.99,
        "batch_size": 16,
        "memory_size": 80,
    }

    config = SimpleNamespace(**config)

    player = Player(config)
    player.learn()