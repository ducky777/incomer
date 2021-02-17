#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate

def get_eq(pr):
    return body * pr[:,0]

def train_model(model, epochs):

    best_eq = 0
    best_model = None
    for i in range(epochs):
        print('\r%s' % i, end='\r')
        model.fit(x, y, epochs=1, validation_split=0.02, batch_size=256, shuffle=True, verbose=0)
        pr = model.predict(x)
        eq = get_eq(pr)
        val_eq = sum(eq[-int(len(x)*0.02):])/np.std(eq[-int(len(x)*0.02):])
        if val_eq > best_eq:
            best_eq = val_eq
            best_model = model
            plt.plot(np.cumsum(eq[-int(len(x)*0.02):]))
            plt.show()
            plt.close()
    return best_model

lookbacks = 241
total_bars = 10000

data = pd.read_csv('EURUSD60.csv', names=['Date', 'Time',
                                            'Open', 'High',
                                            'Low', 'Close',
                                            'Volume'])

# hightail = np.array(data.High - data.Open) / np.array(data.Open)
# lowtail = np.array(data.Low - data.Open) / np.array(data.Open)
# body = np.array(data.Close - data.Open) / np.array(data.Open)

hightail = np.array(data.High - data.Open)
lowtail = np.array(data.Low - data.Open)
body = np.array(data.Close - data.Open)

# hightail = np.log(data.High) - np.log(data.Open)
# lowtail = np.log(data.Low) - np.log(data.Open)
# body = np.log(data.Close) - np.log(data.Open)

market_data = np.stack((hightail, lowtail, body), axis=1)

body = np.array(data.Close - data.Open)
body = body[lookbacks:]
body = body[-total_bars:]

df = pd.DataFrame(market_data)

for i in range(1, lookbacks):
    df['lb_ht'+str(i)] = df[0].shift(i)
    df['lb_lt'+str(i)] = df[1].shift(i)
    df['lb_body'+str(i)] = df[2].shift(i)

df = df.iloc[lookbacks:]
data_copy = data.iloc[-total_bars:]
data = np.array(df)
data = data[-total_bars:]
y = data[:, 2]
x = data[:, 3:]
# body = y[:]

x_max = np.max(np.abs(x))
y_max = np.max(np.abs(y))

x = x/x_max
y = y/y_max

x = x.reshape((-1, lookbacks-1, 3))

x.shape
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

account_state = Input(shape=4)
start_model = Input(shape=(lookbacks - 1, 3))
input_model = Conv1D(kernel_size=7,
                     strides=1,
                     filters=32,
                     padding='same')(start_model)
input_model = res_block(start_model, 64, 5, 2, downsample=True)
input_model = res_block(input_model, 128, 3, 3, downsample=True)
input_model = res_block(input_model, 256, 2, 3, downsample=True)
input_model = MaxPooling1D()(input_model)
input_model = Flatten()(input_model)
input_model = Concatenate(axis=1)([input_model, account_state])

# actor layers
actor = Dense(32)(input_model)
actor = LeakyReLU()(actor)
actor = Dense(3, activation='softmax')(actor)

# critic layers
critic = Dense(12)(input_model)
critic = LeakyReLU()(critic)
critic = Dense(1, activation='linear')(critic)

model = Model([start_model, account_state], [actor, critic])

# model.compile(loss='mse', optimizer='adam')
#%%
import random

import numpy as np
from gym import spaces

class FXEnv:
    def __init__(self, x, y, op, hi, lo, cl,
                 spread=0.0001, gamma=0.05, max_trades=4,
                 periods_per_episode=20):
        # initialize data
        self.open = np.array(op)
        self.high = np.array(hi)
        self.low = np.array(lo)
        self.close = np.array(cl)
        self.states = x
        self.y = y
        self.max_trades = max_trades
        self.periods_per_episode = periods_per_episode
        self.nu_features = x.shape[1] * x.shape[2] + 4

        # initialize environment
        self.spread = spread
        self.pnl = 0
        self.current_position = 0
        self.nu_trades_in_ep = 0

        # initialize others
        self.action_space = spaces.Box(low = -1, high = 1,shape = (3,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (x.shape[1] * x.shape[2] + 4,))
        self.metadata = "No Metadata"
        self.ep_info_buffer = [0, 0, 0]
        self.spec = None

    def _flatten_state(self):
        new_state = np.zeros((self.nu_features, ))
        new_state[:(self.nu_features - 4)] = self.states[self.current_bar].reshape(self.nu_features-4,)
        return new_state

    def reset(self):
        start_idx = random.randint(0, len(self.y) - self.periods_per_episode -1)
        self.pnl = 0
        self.current_position = 0
        self.nu_trades_in_ep = 0
        self.current_bar = start_idx
        self.running_bars = 0
        new_state = self._flatten_state()
        return new_state
        # return [self.states[self.current_bar], np.array([0., 0., 0., 0.])]
        # return self.states[self.current_bar]

    def step(self, action):
        action = np.argmax(action)
        action_state = [0]*3
        action_state[action] = 1
        if action == 2:
            action = -1
        reward = 0
        done = False
        if action != 0:
            if action != self.current_position:
                reward -= self.spread
                self.nu_trades_in_ep += 1
                if self.nu_trades_in_ep >= self.max_trades:
                    return 0, 0, True, {"episode": [self.running_bars],
                                        "is_success": True}
        reward += (action * self.y[self.current_bar])
        self.current_position = action
        self.current_bar += 1
        self.running_bars += 1
        if self.running_bars > self.periods_per_episode:
            done = True
            new_state = 0
            reward = 0
        else:
            # new_state = self.states[self.current_bar]
            new_state = self._flatten_state()
            new_state[(self.nu_features - 3 + action)] = 1
            new_state[-1] = self.nu_trades_in_ep/self.max_trades
        action_state.append(self.nu_trades_in_ep/self.max_trades)
        # new_state = [new_state, action_state]
        return new_state, reward, done, {"episode": [self.running_bars],
                                        "is_success": True}

env = FXEnv(x, body, data_copy.Open, data_copy.High,
            data_copy.Low, data_copy.Close,
            max_trades=4,
            periods_per_episode=24)
state = env.reset()
#%%
import tensorflow.keras as keras
import tensorflow as tf

eps = np.finfo(np.float32).eps.item()
gamma = 0.99

optimizer = keras.optimizers.Adam(learning_rate=0.001)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

num_actions = 3

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, 999):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            market_state = tf.convert_to_tensor(state[0])
            market_state = tf.expand_dims(market_state, 0)

            account_state = tf.convert_to_tensor(state[1])
            account_state = tf.expand_dims(account_state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model([market_state, account_state])
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            # random_action = random.randint(0, 10)
            # if random_action > 6:
            #     action = np.random.choice(num_actions)
            #     action_probs_history.append(tf.math.log(0.33))
            # else:
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-tf.math.reduce_sum(log_prob * diff))  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        # loss_value = compute_loss(action_probs, critic_value, returns)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.5f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 999999999:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
#%%
account_states = np.array([0., 0., 0., 0/3])
account_states = account_states.reshape(-1, 4)
probs = model([env.states[100].reshape(-1, x.shape[1], x.shape[2]), account_states])
probs
#%%
env.states[0]
#%%
from stable_baselines3 import PPO
from stable_baselines3.a2c import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import torch as th

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[512, 256, 128, 256, dict(vf=[32, 16])])

model = PPO(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=None,
            policy_kwargs=policy_kwargs)
#%%
# from tqdm import tqdm
# for _ in tqdm(range(1000)):
model.learn(total_timesteps=10000, log_interval=None)
#%%
state = env.reset()

returns = []

for _ in range(100):
    done = False
    state = env.reset()
    while not done:
        act_probs = model.predict(state)
        state, reward, done, _ = env.step(act_probs[0])
        returns.append(reward)

plt.plot(np.cumsum(returns))
#%%
nu_features = 5 + 4
states = [0.] * (nu_features + 4)
action_idx = 2
states[nu_features - 4 + action_idx] = 1
states
#%%
state = env.reset()
act_probs = model.predict(state)
state, reward, done, _ = env.step(act_probs[0])
state[-5:]
#%%
import numpy as np
import json
test_dict = {'t1': 10, 't2':5, 't3': 11}

with open('test.json', 'w') as f:
    json.dump(test_dict, f)
#%%
with open('test.json', 'r') as f:
    data = json.load(f)
data
#%%