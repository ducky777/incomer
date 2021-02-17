#%%
from stock_env import FXEnv, LoadData
from agents import A2CAgent, PPOAgent, ppo_loss

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate

filename = "EURUSD60.csv"
lookbacks = 241
total_bars = 10000

x, y, body, data = LoadData.load_fx(filename, lookbacks=lookbacks,
                                 total_bars=total_bars)
env = FXEnv(x, body, data.Open, data.Close, max_trades=1,
            periods_per_episode=24, spread=0.0001)
state = env.reset()
# agent = PPOAgent(lookbacks)
# agent.model_actor = tf.keras.models.load_model('model_actor_125_0.00080.hdf5')
# agent.model = tf.keras.models.load_model('a2c_cnn_EURUSD602')
# model_policy = agent.model
# model_target = agent.model
#%%
agent.model.summary()
#%%
import random

env = FXEnv(x, body, max_trades=5,
            periods_per_episode=24, spread=0)

epsilon = 0.8
gamma = 0.99
optimizer = tf.optimizers.Adam(0.0005)

def act(state, model_policy):
    if np.random.rand() <= epsilon:
        return random.randint(0, 2)

    q_values = model_policy(state)
    return np.argmax(q_values[0])

def align_model(model_policy, model_target):
    model_target.set_weights(model_policy.get_weights())
    return model_target

for ep in range(1000):
    done = False
    new_state = env.reset()

    states = [[], []]
    next_states = [[], []]
    rewards = []
    dones = []
    actions = []

    while True:
        current_state = new_state.copy()
        # action = np.argmax(model_policy(current_state))
        action = act(current_state, model_policy)
        new_state, reward, done, _ = env.step(action)
        if done:
            break

        states[0].append(current_state[0])
        states[1].append(current_state[1])
        next_states[0].append(new_state[0])
        next_states[1].append(new_state[1])
        rewards.append(reward)
        dones.append(done)
        actions.append(action)

    x1 = np.reshape(next_states[0], (len(next_states[0]), lookbacks -1 , 3))
    x2 = np.reshape(next_states[1], (len(next_states[1]), 4))
    q_s_a_prime = np.max(model_target([x1, x2]), axis = 1)
    q_s_a_target = np.where(dones, rewards, rewards+gamma*q_s_a_prime)
    q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype = 'float32')
    # Calculate Loss function and gradient values for gradient descent
    with tf.GradientTape() as tape:
        x1 = np.reshape(states[0], (len(states[0]), lookbacks - 1, 3))
        x2 = np.reshape(states[1], (len(states[1]), 4))
        q_s_a = tf.math.reduce_sum(model_policy([x1, x2]) * \
            tf.one_hot(actions, 3), axis=1)
        loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

    # Update the policy network weights using ADAM
    variables = model_policy.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    model_target = align_model(model_policy, model_target)

    # losses.append(loss.numpy())
    print("Ep %i: %.5f" % (ep, sum(rewards)))
#%%
states[1]
#%%
from tqdm import tqdm
state = env.reset()

returns = []
actions = []
states = []

for _ in tqdm(range(50)):
    done = False
    state = env.reset()
    while not done:
        act_probs, _ = agent.model(state)
        action = np.argmax(act_probs, axis=-1)[0]
        state, reward, done, _ = env.step(action)
        returns.append(reward)
        actions.append(action)
        states.append(state)

plt.plot(np.cumsum(returns))
#%%
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate

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

account_state = Input(shape=(4,))
start_model = Input(shape=(240, 3))
input_model = Conv1D(kernel_size=24,
                    strides=1,
                    filters=64,
                    padding='same')(start_model)
input_model = res_block(start_model, 128, 12, 3, downsample=True)
input_model = res_block(input_model, 256, 6, 3, downsample=True)
input_model = res_block(input_model, 512, 3, 3, downsample=True)
input_model = res_block(input_model, 1024, 2, 3, downsample=True)
input_model = MaxPooling1D()(input_model)
input_model = Flatten()(input_model)
input_model = Concatenate(axis=-1)([input_model, account_state])

# actor layers
actor = Dense(128)(input_model)
actor = LeakyReLU()(actor)
actor = Dense(3, activation='linear')(actor)

model = Model([start_model, account_state], [actor])
# model_old = Model([start_model, account_state], [actor])
model.compile(optimizer='adam', loss='mse')
# model_old.compile(optimizer='adam', loss='mse')
#%%
import random

for batch_num in range(5):
    rewards = []
    states = []
    actions = []

    done = False
    state = env.reset()
    for ep in range(1000):
        state = env.reset()
        done = False
        current_rewards = []
        current_states = []
        current_actions = []
        while not done:
            current_states.append(state)
            action = random.randint(0, 2)
            # if batch_num > 190:
            #     act_probs = model(state)
            #     action = np.argmax(act_probs)
            state, reward, done, _ = env.step(action)
            current_rewards.append(reward)
            current_actions.append(action)

        temp_rewards = []
        # total_rewards = sum(current_rewards)
        for i, reward in reversed(list(enumerate(current_rewards[:-1]))):
            discounted_reward = reward + 0.95 * current_rewards[i+1]
            # discounted_reward = total_rewards * (i/len(current_rewards[:-1]))
            if sum(current_rewards) == 0:
                temp_rewards.append(-1)
            else:
                temp_rewards.append(discounted_reward)
        temp_rewards = (temp_rewards - np.mean(temp_rewards)) / (np.std(temp_rewards) + 1e-10)
        for i, state in enumerate(current_states[:-1]):
            states.append(state)
            rewards.append(temp_rewards[i])
        for action in current_actions[:-1]:
            actions.append(action)
    market_states = [state[0] for state in states]
    account_states = [state[1] for state in states]

    market_states = np.array(market_states).squeeze(axis=1)
    account_states = np.array(account_states).squeeze(axis=1)
    original_predictions = model([market_states, account_states])
    original_predictions = np.array(original_predictions)
    original_predictions.shape

    for i, _ in enumerate(original_predictions):
        original_predictions[i, actions[i]] = rewards[i]

    epochs = 100
    if batch_num > 0:
        epochs = 3
    model.fit([market_states, account_states], original_predictions,
            verbose=1, epochs=epochs, batch_size=256, shuffle=True)
#%%
state = env.reset()
# act_probs = agent.model_actor(state)
act_probs = model(state)
print(act_probs)
np.argmax(act_probs)
#%%
original_predictions[0]
#%%
returns = []
actions = []
states = []

counter = 0
for _ in tqdm(range(200)):
    done = False
    state = env.reset()
    while not done:
        # act_probs = agent.model_actor(state)
        act_probs = model(state)
        action = np.argmin(act_probs)
        # action = 1
        # action = 2
        # if counter % 5 == 0:
        #     action = 0
        # if action == 1:
        #     action = 2
        # elif action == 2:
        #     action = 1
        actions.append(action)

        state, reward, done, _ = env.step(action)
        returns.append(reward)
        states.append(state)

        counter += 1

plt.plot(np.cumsum(returns))
#%%
actions
#%%
env.nu_trades_in_ep
#%%
import pandas as pd

data = pd.read_csv('EURUSD60.csv', names=['Date', 'Time',
                                    'Open', 'High',
                                    'Low', 'Close',
                                    'Volume'])
#%%
date_row = data['Date']
date_row = pd.to_datetime(date_row, dayfirst=True, format='%Y.%m.%d')
date_row = date_row.apply(lambda x: x.toordinal())
#%%
date_ordinal = pd.DataFrame()
date_ordinal['T'] = data.Time.apply(lambda x: x.split(':')[0])

for i in range(3, 61):
    date_ordinal[i] = date_row % i
#%%
date_ordinal
#%%
data.Time.apply(lambda x: int(x.split(':')[0]))
#%%
import random
from gym import spaces
import numpy as np
class FXEnv2:
    def __init__(self, data, lookbacks, cycles: list, spread=0.0001, gamma=0.05, max_trades=4,
                 periods_per_episode=20):
        # initialize data
        self.dates = pd.to_datetime(data.Date, dayfirst=True, format='%Y.%m.%d')
        self.op = np.array(data.Open)
        self.hi = np.array(data.High)
        self.lo = np.array(data.Low)
        self.cl = np.array(data.Open.shift(-1))

        # data manipulation
        self.ht = np.array(self.hi - self.op)[:-1]
        self.lt = np.array(self.lo - self.op)[:-1]
        self.bod = np.array(self.cl - self.op)[:-1]

        self.y = np.array(self.cl - self.op)[1:]

        self.dates = self.dates.apply(lambda x: x.toordinal())
        self.hours = data.Time.apply(lambda x: int(x.split(':')[0])) / 24.
        self.cycles = cycles
        self.dates_ordinal = []
        for i in range(cycles[0], cycles[1]):
            self.dates_ordinal.append(self.dates % i)

        self.dates_ordinal = np.array(self.dates_ordinal, dtype=np.float32).\
            swapaxes(1, 0)[:-1]

        self.hours_ordinal = []
        for i in range(cycles[0], cycles[1]):
            self.hours_ordinal.append(self.hours % i)

        self.hours_ordinal = np.array(self.hours_ordinal, dtype=np.float32).\
            swapaxes(1, 0)[:-1]

        # initialize variables
        self.max_trades = max_trades
        self.periods_per_episode = periods_per_episode
        self.nu_features = lookbacks + 2 + 4

        # initialize environment
        self.spread = spread
        self.pnl = 0
        self.current_position = 0
        self.nu_trades_in_ep = 0
        self.open_price = -1
        self.lookbacks = lookbacks

        # initialize others
        self.action_space = spaces.Box(low = -1, high = 1,shape = (3,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (lookbacks + 2 + 4, ))
        self.metadata = "No Metadata"
        self.ep_info_buffer = [0, 0, 0]
        self.spec = None

    def _get_market_state(self):
        lookback_idx = self.current_bar - self.lookbacks
        return np.array([self.ht[lookback_idx:self.current_bar],
                         self.lt[lookback_idx:self.current_bar],
                         self.bod[lookback_idx:self.current_bar]]).\
                             swapaxes(1, 0)

    def _get_datetime_state(self):
        return np.array([self.dates_ordinal[self.current_bar],
                         self.hours[self.current_bar]])

    def reset(self):
        start_idx = random.randint(self.lookbacks, len(self.y) - self.periods_per_episode - 2)
        self.pnl = 0
        self.current_position = 0
        self.nu_trades_in_ep = 0
        self.current_bar = start_idx
        self.running_bars = 0
        self.open_price = 0
        market_state = self._get_market_state()
        datetime_state = self._get_datetime_state()

        return [market_state, datetime_state, np.array([[1., 0., 0., 0.]])]

    def _is_closing(self, action):
        if self.current_position == 0:
            return False
        if action != self.current_position:
            return True
        return False

    def _is_opening(self, action):
        if action == 0:
            return False
        if action != self.current_position:
            return True
        return False

    def _handle_buy(self):
        if self.current_position == 1:
            return 0
        else:
            return (self.op[self.current_bar] - self.open_price)

    def step(self, action):
        # initialize variables
        account_state = [0.] * 4
        account_state[action] = 1.
        store_position = self.current_position
        if store_position == 2:
            store_position = -1
        if action == 2:
            action = -1
        reward = 0
        done = False

        # handle states
        self.current_bar += 1
        self.running_bars += 1
        # new_state = 0.

        # handle open and closing of positions
        if self._is_closing(action):
            self.nu_trades_in_ep += 1
        if self._is_opening(action):
            reward -= self.spread
            self.open_price = self.op[self.current_bar]

        # handle reward
        reward += (action * self.y[self.current_bar])
        # if reward < 0:
        #     reward *= 2

        self.current_position = action

        market_state = []
        datetime_state = []

        if self.nu_trades_in_ep > self.max_trades or \
            self.running_bars > self.periods_per_episode:
            done = True
            account_state = [-1.] * 4
            reward = 0.
            self.current_bar = -1
        else:
            market_state = self._get_market_state()
            datetime_state = self._get_datetime_state()
            account_state[-1] = self.nu_trades_in_ep/self.max_trades

        reward = self.y[self.current_bar-1]

        return [market_state, datetime_state, np.array([account_state])], reward, \
                done, {"episode": [self.running_bars],
                        "is_success": True}

cycles = [3, 61]
lookbacks=120

cycles_len = cycles[1] - cycles[0]
env = FXEnv2(data, lookbacks=lookbacks, cycles=cycles)
#%%
state = env.reset()
state[1][0].shape
#%%
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate, Embedding, Masking

k.clear_session()

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

def get_model():
    hour_state = Input(shape=(1, ))
    account_state = Input(shape=(4,))

    date_input = Input(shape=(cycles_len, ))
    date_state = Embedding(input_dim=cycles[1],
                        output_dim=128, input_length=58)(date_input)
    date_state = Flatten()(date_state)
    # date_state = Dense(8)(date_state)
    # date_state = LeakyReLU()(date_state)

    start_model = Input(shape=(lookbacks, 3))
    input_model = Conv1D(kernel_size=6,
                        strides=1,
                        filters=64,
                        padding='same')(start_model)
    input_model = res_block(input_model, 128, 3, 1, downsample=True)
    input_model = MaxPooling1D()(input_model)
    input_model = Flatten()(input_model)
    output = Concatenate(axis=-1)([input_model, account_state, hour_state, date_state])

    output = Dense(1024)(output)
    output = LeakyReLU()(output)
    output = Dense(3, activation='linear')(output)

    model = Model([start_model, date_input, hour_state, account_state], [output])
    model.compile(optimizer='adam', loss='mse')
    return model

model = get_model()
#%%
model.summary()
#%%%
pr = model([np.expand_dims(state[0], 0),
            np.expand_dims(state[1][0], 0),
            np.array([[state[1][1]/24.]]),
            state[2]
            ])
#%%
env = FXEnv2(data, lookbacks=lookbacks, cycles=cycles,
                spread=0.0001, gamma=0.99, max_trades=5000,
                 periods_per_episode=5000)

for batch_num in range(100):
    rewards = []
    states = []
    actions = []

    done = False
    state = env.reset()
    for ep in range(1):
        state = env.reset()
        done = False
        current_rewards = []
        current_states = []
        current_actions = []
        while not done:
            current_states.append(state)
            action = random.randint(0, 2)
            # if batch_num > 190:
            #     act_probs = model(state)
            #     action = np.argmax(act_probs)
            state, reward, done, _ = env.step(action)
            current_rewards.append(reward)
            current_actions.append(action)

        temp_rewards = []
        # total_rewards = sum(current_rewards)
        for i, reward in reversed(list(enumerate(current_rewards[:-1]))):
            discounted_reward = reward + 0.99 * current_rewards[i+1]
            # discounted_reward = total_rewards * (i/len(current_rewards[:-1]))
            # if sum(current_rewards) == 0:
            #     temp_rewards.append(-1)
            # else:
            temp_rewards.append(discounted_reward)
        temp_rewards = (temp_rewards - np.mean(temp_rewards)) / (np.std(temp_rewards) + 1e-10)
        for i, state in enumerate(current_states[:-1]):
            states.append(state)
            rewards.append(temp_rewards[i])
        for action in current_actions[:-1]:
            actions.append(action)
    market_states = np.array([state[0] for state in states])
    date_states = np.array([state[1][0] for state in states])
    hour_states = np.expand_dims([state[1][1] for state in states], 1)
    account_states = np.squeeze([state[2] for state in states], 1)

    # market_states = np.array(market_states).squeeze(axis=1)
    # account_states = np.array(account_states).squeeze(axis=1)
    original_predictions = model([market_states, date_states, hour_states, account_states])
    original_predictions = np.array(original_predictions)
    original_predictions.shape

    for i, _ in enumerate(original_predictions):
        original_predictions[i, actions[i]] = rewards[i]

    epochs = 10
    if batch_num > 0:
        epochs = 10
    model.fit([market_states, date_states, hour_states, account_states], original_predictions,
            verbose=1, epochs=epochs, batch_size=256, shuffle=True)
#%%
len(market_states)
#%%
data = pd.read_csv('EURUSD60.csv', names=['Date', 'Time',
                                    'Open', 'High',
                                    'Low', 'Close',
                                    'Volume'])

data = data.iloc[-10000:]
data = data.reset_index()

env = FXEnv2(data, lookbacks=lookbacks, cycles=cycles,
                spread=0.0001, gamma=0.05, max_trades=9000,
                 periods_per_episode=9000)
#%%
from tqdm import tqdm
import matplotlib.pyplot as plt

returns = []
actions = []
states = []

counter = 0
for _ in tqdm(range(1)):
    done = False
    state = env.reset()
    while not done:
        # act_probs = agent.model_actor(state)
        act_probs = model([np.expand_dims(state[0], 0),
            np.expand_dims(state[1][0], 0),
            np.array([[state[1][1]/24.]]),
            state[2]
            ])
        action = np.argmax(act_probs)
        # action = 1
        # action = 2
        # if counter % 5 == 0:
        #     action = 0
        # if action == 1:
        #     action = 2
        # elif action == 2:
        #     action = 1
        actions.append(action)

        state, reward, done, _ = env.step(action)
        returns.append(reward)
        states.append(state)

        counter += 1

plt.plot(np.cumsum(returns))
#%%
actions[-100:]
#%%
model.save('DQN_EURUSD_H1.hdf5')
#%%