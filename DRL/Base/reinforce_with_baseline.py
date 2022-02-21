import tensorflow as tf
import numpy as np
from types import SimpleNamespace
import gym
import tqdm


class Reinforce(tf.keras.Model):

    def __init__(self, action_size):
        super(Reinforce, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.layer3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.layer4 = tf.keras.layers.Flatten()
        self.layer5 = tf.keras.layers.Dense(256, activation='relu')

        self.policy = tf.keras.layers.Dense(action_size, activation='softmax')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = state / 255
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        policy = self.policy(x)
        value = self.value(x)

        return policy, value


class Memory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Player(object):
    def __init__(self, config: SimpleNamespace):
        self.env = gym.make(config.env_name, render_mode='human')

        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.model = Reinforce(self.action_size)

        self.memory = Memory()

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr,)
        self.summary_writer = tf.summary.create_file_writer("logdir/reinforce")

    def _get_action(self, obs):
        policy, _ = self.model(np.array([obs], dtype=np.float32))
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def _count_return(self, rewards):
        g_next = 0
        GI = []
        for r in rewards[::-1]:
            g_next = r + self.gamma * g_next
            GI.append(g_next)
        return GI[::-1]

    def _get_batches(self):
        returns = self._count_return(self.memory.rewards)
        steps = len(returns)
        sample_range = np.arange(steps)
        np.random.shuffle(sample_range)
        for n in range((steps+1)//self.batch_size):
            sample_idx = sample_range[self.batch_size * n: self.batch_size *(n+1)]
            states, actions, gs = [], [], []
            for i in sample_idx:
                states.append(self.memory.states[i])
                actions.append(self.memory.actions[i])
                gs.append(returns[i])
            if len(states) > 0:
                yield np.array(states), np.array(actions), np.array(gs)

    def _update_param(self):

        for states, actions, gs in tqdm.tqdm(self._get_batches()):
            with tf.GradientTape() as tape:
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                gs = tf.convert_to_tensor(gs, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                policy, value = self.model(states)
                pi = tf.reduce_sum(tf.one_hot(actions, self.action_size) * policy, axis=1)
                delta_t = gs - value
                v = -delta_t * tf.math.log(pi)

            policy_grads = tape.gradient(v, self.model.trainable_variables)
            self.opt.apply_gradients(zip(policy_grads, self.model.trainable_variables))

        self.memory.clear()


    def learn(self):
        step = 0
        episode = 0

        while True:
            score = 0
            done = False
            state = self.env.reset()
            while not done:
                # self.env.render()
                action = self._get_action(state)
                step += 1
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.memory.add(state, action, reward)
                state = next_state

            episode += 1
            print(f"{episode} episode, score: {score}")
            self._update_param()

            with self.summary_writer.as_default():
                tf.summary.scalar('score', score, step=episode)


if __name__ == '__main__':

    config = {
        "env_name": "Breakout-v0",  # CartPole-v1  SpaceInvaders-v0
        "lr": 0.0003,
        "gamma": 0.99,
        "batch_size": 128,
    }

    config = SimpleNamespace(**config)

    player = Player(config)
    player.learn()