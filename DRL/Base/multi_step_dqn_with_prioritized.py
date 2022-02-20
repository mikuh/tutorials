import numpy as np
import random
import tensorflow as tf
import gym
from types import SimpleNamespace
from collections import deque


class SumTree(object):
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class Memory(object):
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    length = 0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, p, sample):
        # p = self._getPriority(error)
        self.tree.add(p, sample)
        self.length += 1

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class MultiStepMemory(object):

    def __init__(self, n, gamma=0.99):
        self.maxlen = n
        self.gamma = gamma
        self.state = deque(maxlen=n)
        self.action = deque(maxlen=n)
        self.reward = deque(maxlen=n)
        self.next_state = deque(maxlen=n)
        self.gammas = deque(maxlen=n)

    def append(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.gammas.append((1-done)*self.gamma)

    def sample(self):
        if len(self.state) == int(self.maxlen):
            n = 0
            for gamma in self.gammas:
                n += 1
                if gamma == 0.0 and n != self.maxlen:
                    return None
            return np.stack(self.state), np.stack(self.next_state), np.stack(self.reward),  np.stack(self.gammas), np.stack(self.action)
        return None


class DQN(tf.keras.Model):

    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.layer3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.layer4 = tf.keras.layers.Flatten()
        self.layer5 = tf.keras.layers.Dense(256, activation='relu')
        self.value = tf.keras.layers.Dense(1)
        self.advantage = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = state / 255
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        value = self.value(x)
        advantage = self.advantage(x)
        q = value + advantage - tf.reduce_mean(advantage)
        return q


class Player(object):

    def __init__(self, config: SimpleNamespace):
        self.env = gym.make(config.env_name)

        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.multi_step_memory = MultiStepMemory(n=config.n_step, gamma=config.gamma)
        self.memory = Memory(capacity=config.memory_size)
        self.model = DQN(self.action_size)
        self.target_model = DQN(self.action_size)

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr, )
        self.summary_writer = tf.summary.create_file_writer("logdir")

    def _collect_transitions(self, state, action, reward, next_state, done):
        self.multi_step_memory.append(state, action, reward, next_state, done)
        sample = self.multi_step_memory.sample()
        if sample is not None:
            state = sample[0][0]
            next_state = sample[1][-1]
            reward = np.sum([r*self.gamma**i for i, r in enumerate(sample[2])])
            action = sample[4][0]
            gamma = sample[3][-1]
            self.memory.add(1, (state, action, reward, next_state, gamma))

    def _get_action(self, obs):
        q_value = self.model(np.array([obs], dtype=np.float32))[0]
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action

    def _update_param(self, step):
        batch, idxs, is_weight = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, gammas = zip(*[(e[0], e[1], e[2], e[3], e[4]) for e in batch])

        with tf.GradientTape() as tape:
            rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
            actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
            gammas = tf.convert_to_tensor(np.array(gammas), dtype=tf.float32)

            q_target = self.target_model(tf.convert_to_tensor(np.array(next_states), dtype=tf.float32))
            q_next_dqn = self.model(tf.convert_to_tensor(np.array(next_states), dtype=tf.float32))
            q_next_dqn = tf.stop_gradient(q_next_dqn)
            next_action = tf.argmax(q_next_dqn, axis=1)
            td_target = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * q_target, axis=1)
            target_value = gammas * td_target + rewards
            q = self.model(tf.convert_to_tensor(np.array(states), dtype=tf.float32))
            q_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * q, axis=1)
            td_error = q_value - target_value
            loss = tf.reduce_mean(tf.square(td_error) * 0.5)

        dqn_grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(dqn_grads, self.model.trainable_variables))

        if step % 20 == 0:
            self.target_model.set_weights(self.model.get_weights())

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, np.abs(td_error[i]))

        # with self.summary_writer.as_default():
        #     tf.summary.scalar('loss', loss, step=step)

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
                if self.memory.length > self.batch_size:
                    self._update_param(step=step)
            print(f"{self.episodes} episode, score: {score}")


if __name__ == '__main__':

    config = {
        "env_name": "Breakout-v0",  # CartPole-v1  SpaceInvaders-v0
        "lr": 0.001,
        "gamma": 0.99,
        "batch_size": 128,
        "memory_size": 10000,
        "n_step": 5,
    }

    config = SimpleNamespace(**config)

    player = Player(config)
    player.learn()