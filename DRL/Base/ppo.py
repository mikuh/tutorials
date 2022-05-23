import tensorflow as tf
import numpy as np
from types import SimpleNamespace
import gym
import copy

class PPO(tf.keras.Model):

    def __init__(self, action_size):
        super(PPO, self).__init__()
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
    def __init__(self, rollout):
        self.rollout = rollout
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.policies = []
        self.values = []
        self.size = 0

    def add(self, state, action, reward, next_state, done, policy, value):
        self.size += 1
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.policies.append(policy)
        self.values.append(value)

    def reset(self, action, reward, next_state, done, value, policy):
        self.states, self.next_states = [self.states[-1]], [next_state]
        self.rewards, self.dones, self.actions = [reward], [done], [action]
        self.values, self.policies = [value], [policy]
        self.size = 1


class Player(object):

    def __init__(self, config: SimpleNamespace):
        self.env = gym.make(config.env_name, render_mode='human')
        # self.env = gym.make(config.env_name)
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.rollout = config.rollout
        self.epoch = config.epoch
        self.lamda = config.lamda
        self.normalize = config.normalize
        self.ppo_eps = config.ppo_eps
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.model = PPO(self.action_size)
        self.memory = Memory(self.rollout)

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr, )
        self.summary_writer = tf.summary.create_file_writer("logdir/ppo")

    def _get_action_policy_value(self, obs):
        policy, value = self.model(np.array([obs], dtype=np.float32))
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action, policy, value[0]

    def _collect_transitions(self, state, action, reward, next_state, done, policy, value):
        self.memory.add(state, action, reward, next_state, done, policy, value)

    def _get_gae(self):
        values = np.array(tf.squeeze(self.memory.values)[:-1])
        next_values = np.array(tf.squeeze(self.memory.values)[1:])
        rewards = np.array(self.memory.rewards[:-1])
        dones = np.array(self.memory.dones[:-1])
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self.lamda * gaes[t + 1]
        target = gaes + values
        if self.normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes, target

    def update_param(self):
        policies = np.array(tf.squeeze(self.memory.policies))

        adv, target = self._get_gae()

        for _ in range(self.epoch):
            sample_range = np.arange(self.rollout)
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]

            batch_state = [self.memory.states[i] for i in sample_idx]
            # batch_done = [done[i] for i in sample_idx]
            batch_action = [self.memory.actions[i] for i in sample_idx]
            batch_target = [target[i] for i in sample_idx]
            batch_adv = [adv[i] for i in sample_idx]
            batch_old_policy = [policies[i] for i in sample_idx]


            with tf.GradientTape() as tape:
                train_policy, train_current_value = self.model(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                train_current_value = tf.squeeze(train_current_value)
                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                train_action = tf.convert_to_tensor(batch_action, dtype=tf.int32)
                train_old_policy = tf.convert_to_tensor(batch_old_policy, dtype=tf.float32)

                entropy_loss = tf.reduce_mean(train_policy * tf.math.log(train_policy + 1e-8)) * 0.01
                onehot_action = tf.one_hot(train_action, self.action_size)
                selected_prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
                selected_old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=1)
                logpi = tf.math.log(selected_prob + 1e-8)
                logoldpi = tf.math.log(selected_old_prob + 1e-8)

                ratio = tf.exp(logpi - logoldpi)

                clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1 - self.ppo_eps,
                                                 clip_value_max=1 + self.ppo_eps)
                minimum = tf.minimum(tf.multiply(train_adv, clipped_ratio), tf.multiply(train_adv, ratio))
                pi_loss = -tf.reduce_mean(minimum)

                value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))

                total_loss = pi_loss + value_loss + entropy_loss

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def learn(self):
        episode = 0
        step = 0
        score = 0
        state = self.env.reset()

        while True:

            while self.memory.size <= self.rollout:
                action, policy, value = self._get_action_policy_value(state)

                next_state, reward, done, _ = self.env.step(action)
                step += 1
                score += reward
                self._collect_transitions(state, action, reward, next_state, done, policy, value)
                state = next_state

                if done:
                    episode += 1
                    print(f"{episode} episode, score: {score}")
                    with self.summary_writer.as_default():
                        tf.summary.scalar('score', score, step=episode)
                    state = self.env.reset()
                    score = 0

            self.update_param()
            self.memory.reset(action, reward, next_state, done, value, policy)

if __name__ == '__main__':

    if __name__ == '__main__':
        config = {
            "env_name": "Breakout-v0",  # CartPole-v1  SpaceInvaders-v0
            "lr": 0.001,
            "gamma": 0.99,
            "batch_size": 128,
            "rollout": 256,
            "epoch": 4,
            "lamda": 0.95,
            "normalize": True,
            "ppo_eps": 0.2
        }

        config = SimpleNamespace(**config)

        player = Player(config)
        player.learn()