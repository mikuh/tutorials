import tensorflow as tf
import numpy as np
from types import SimpleNamespace
import gym

class A2C(tf.keras.Model):

    def __init__(self, action_size):
        super(A2C, self).__init__()
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
        self.gammas = []

    def add(self, state, action, reward, next_state, gamma):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.gammas.append(gamma)

    def sample(self):
        sample_range = np.arange(self.rollout)
        np.random.shuffle(sample_range)
        states, actions, rewards, next_states, gammas = [], [], [], [], []
        for i in sample_range:
            states.append(self.states[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            next_states.append(self.next_states[i])
            gammas.append(self.gammas[i])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(gammas)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.gammas = []

class Player(object):

    def __init__(self, config: SimpleNamespace):
        # self.env = gym.make(config.env_name, render_mode='human')
        self.env = gym.make(config.env_name)
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.rollout = config.rollout
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.model = A2C(self.action_size)

        self.memory = Memory(self.rollout)

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr, )
        self.summary_writer = tf.summary.create_file_writer("logdir/a2c")

    def _get_action(self, obs):
        policy, _ = self.model(np.array([obs], dtype=np.float32))
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action


    def _collect_transitions(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, (1-done)*self.gamma)

    def _update_param(self):
        states, actions, rewards, next_states, gammas = self.memory.sample()
        with tf.GradientTape() as tape:
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            gammas = tf.convert_to_tensor(gammas, dtype=tf.float32)

            policy, value = self.model(states)
            _, next_value = self.model(next_states)
            value, next_value = tf.squeeze(value), tf.squeeze(next_value)
            target_value = rewards + gammas * next_value
            adventage = target_value - value
            pi = tf.reduce_sum(tf.one_hot(actions, self.action_size) * policy, axis=1)

            value_loss = tf.reduce_mean(tf.square(adventage)*0.5)
            policy_loss = - tf.stop_gradient(adventage) * tf.math.log(pi+1e-8)
            policy_entropy = - tf.reduce_mean(- policy * tf.math.log(policy + 1e-8)) * 0.2

            loss = value_loss + policy_loss + policy_entropy

        policy_grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(policy_grads, self.model.trainable_variables))

        self.memory.clear()




    def learn(self):
        episode = 0
        step = 0
        score = 0
        state = self.env.reset()
        while True:

            for _ in range(self.rollout):
                action = self._get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                step += 1
                score += reward
                self._collect_transitions(state, action, reward, next_state, done)

                state = next_state

                if done:
                    episode += 1
                    print(f"{episode} episode, score: {score}")
                    with self.summary_writer.as_default():
                        tf.summary.scalar('score', score, step=episode)
                    state = self.env.reset()
                    score = 0

            self._update_param()

if __name__ == '__main__':

    config = {
        "env_name": "Breakout-v0",  # CartPole-v1  SpaceInvaders-v0
        "lr": 0.0003,
        "gamma": 0.99,
        "batch_size": 128,
        "rollout": 128,
    }

    config = SimpleNamespace(**config)

    player = Player(config)
    player.learn()