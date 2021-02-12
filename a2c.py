#%%
from stock_env import FXEnv, LoadData

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate

filename = "EURUSD1440.csv"
lookbacks = 121
total_bars = 1000

x, y, body, _ = LoadData.load_fx(filename, lookbacks=lookbacks,
                                 total_bars=total_bars)
env = FXEnv(x, body, max_trades=6,
            periods_per_episode=60, spread=0)
state = env.reset()
#%%
def res_block(x, filters, size, stride, downsample=False):
    y = Conv1D(filters, size, (1 if not downsample else 2), padding='same')(x)
    y = Activation('relu')(y)
    y = Conv1D(filters, size, 1, padding='same')(y)

    if downsample:
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   strides=2,
                   padding='same')(x)

    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

k.clear_session()

account_state = Input(shape=(4,))
start_model = Input(shape=(lookbacks - 1, 3))
input_model = Conv1D(kernel_size=7,
                     strides=1,
                     filters=64,
                     padding='same')(start_model)
input_model = res_block(start_model, 128, 5, 2, downsample=True)
input_model = res_block(input_model, 256, 3, 3, downsample=True)
input_model = res_block(input_model, 512, 2, 3, downsample=True)
input_model = MaxPooling1D()(input_model)
input_model = Flatten()(input_model)
input_model = Concatenate(axis=-1)([input_model, account_state])

# actor layers
actor = Dense(32)(input_model)
actor = LeakyReLU()(actor)
actor = Dense(3, activation='softmax')(actor)

# critic layers
critic = Dense(12)(input_model)
critic = LeakyReLU()(critic)
critic = Dense(1, activation='linear')(critic)

model = Model([start_model, account_state], [actor, critic])
#%%
account_state = Input(shape=(4,))
start_model = Input(shape=(lookbacks - 1, 3))

input_model = LSTM(256, return_sequences=True)(start_model)
input_model = LSTM(128, return_sequences=True)(input_model)
input_model = LSTM(64)(input_model)
input_model = Concatenate(axis=-1)([input_model, account_state])
input_model = Dense(128)(input_model)
input_model = LeakyReLU()(input_model)

# actor
actor = Dense(64)(input_model)
actor = LeakyReLU()(actor)
actor = Dense(3, activation='softmax')(actor)

# critic layers
critic = Dense(32)(input_model)
critic = LeakyReLU()(critic)
critic = Dense(1, activation='linear')(critic)

model = Model([start_model, account_state], [actor, critic])
#%%
import tensorflow.keras.losses as kls
import logging

class A2CAgent:
    def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        self.model = model
        self.value_c = value_c
        self.entropy_c = entropy_c
        self.gamma = gamma

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                            loss=[self._logits_loss, self._value_loss])

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    def train(self, env, batch_size=256, updates=1000):
        # Storage helpers for a single batch of data.
        # actions = np.empty((batch_size,), dtype=np.int32)
        # rewards, dones, values = np.empty((3, batch_size))
        # observations = np.empty((batch_size,) + env.observation_space.shape)
        # observations = np.zeros((batch_size, 2))
        # observations = np.empty((batch_size,) + (lookbacks - 1, 4))

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            observations = []
            market_state, account_state = [], []
            actions, rewards, dones, values = [], [], [], []
            for step in range(batch_size):
                # observations.append(next_obs.copy())
                # observations[step] = next_obs.copy()
                market_state.append(next_obs[0])
                account_state.append(next_obs[1])
                pr = self.model(next_obs)
                # actions[step], values[step] = self.model(next_obs)
                actions.append(np.argmax(pr[0]))
                values.append(float(pr[1]))
                next_obs, reward, done, _ = env.step(actions[step])
                rewards.append(reward)
                dones.append(done)

                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (
                        len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model(next_obs)

            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API.
            # acts_and_advs = np.concatenate((actions, advs), axis=0)
            acts_and_advs = np.column_stack((actions, advs))
            # print(len(observations))

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            # x = np.concatenate(observations, axis=0)
            x1 = np.reshape(market_state, (batch_size, lookbacks - 1, 3))
            x2 = np.reshape(account_state, (batch_size, 4))
            losses = self.model.train_on_batch([x1, x2], [acts_and_advs, returns])

            # logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
            # print(("[%d/%d] Losses: %s" % (update + 1, updates, losses)))
            print(("[%d/%d] Episode Reward: %.5f" % (update + 1, updates, reward)))

        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        # print(rewards)
        # print(next_value)
        # returns = [rewards]
        # returns.append(next_value)
        # returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        returns = rewards[:]
        returns.append(float(next_value))
        returns = np.array(returns)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # print(values)
        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        return ep_reward

agent = A2CAgent(model)
rewards = agent.train(env)
#%%
from tqdm import tqdm
state = env.reset()

returns = []

for _ in tqdm(range(40)):
    done = False
    state = env.reset()
    while not done:
        act_probs, _ = model(state)
        state, reward, done, _ = env.step(np.argmax(act_probs))
        returns.append(reward)

plt.plot(np.cumsum(returns))
#%%
state = env.reset()
act_probs, _ = model(state)
act_probs
#%%