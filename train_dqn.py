import tensorflow as tf
import pandas as pd
import random
import plotly.graph_objects as go
import numpy as np

from gym import spaces
from collections import deque

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate, Embedding, Masking

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

        # market_state = np.zeros((lookbacks, 3))
        # datetime_state = [np.zeros((cycles_len, )) - 1, -1]

        market_state = self._get_market_state()
        datetime_state = self._get_datetime_state()

        if self.nu_trades_in_ep > self.max_trades or \
            self.running_bars > self.periods_per_episode:
            done = True
            account_state = [-1.] * 4
            reward = 0.
            self.current_bar = -1
        else:
            # market_state = self._get_market_state()
            # datetime_state = self._get_datetime_state()
            account_state[-1] = self.nu_trades_in_ep/self.max_trades

        return [market_state, datetime_state, np.array([account_state])], reward, \
                done, {"episode": [self.running_bars],
                        "is_success": True}

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

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):

        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = 0.4

        self.model = get_model()

    def _split_data(self, states):
        market_state, date_state, hour_state, account_state = [], [], [], []

        for state in states:
            market_state.append(state[0])
            date_state.append(state[1][0])
            hour_state.append(state[1][1])
            account_state.append(state[2])

        market_state = np.array(market_state)
        date_state = np.array(date_state)
        hour_state = np.expand_dims(hour_state, 0).swapaxes(1, 0)
        account_state = np.squeeze(account_state, 1)
        return [market_state, date_state,
                hour_state, account_state]

    def predict(self, states):
        states = self._split_data(states)

        return self.model(states).numpy()

    def get_action(self, state):
        # state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= 0.95
        self.epsilon = max(self.epsilon, 0.2)
        q_value = self.predict([state])[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states, targets):
        states = self._split_data(states)
        self.model.fit(states, targets, epochs=1, verbose=0)

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = -1
        self.action_dim = 3

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=-1)
            targets[range(batch_size), actions] = rewards + (1-done) * \
                next_q_values * gamma
            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        best_reward = 0
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= batch_size:
                self.replay()
            self.target_update()
            if total_reward > best_reward:
                best_reward = total_reward
                print('Saving model as EURUSD_%i_%.5f.h5' % (ep, total_reward))
                self.model.model.save('EURUSD_%i_%.5f.h5' % (ep, total_reward))
            print('EP%i EpisodeReward=%.6f' % (ep, total_reward))
            # wandb.log({'Reward': total_reward})

if __name__ == '__main__':

    # ========== PARAMETERS ==========
    number_of_bars = 183
    periods_per_episode = 120
    max_trades = 120

    cycles = [3, 61]
    lookbacks = 60
    batch_size = 256
    gamma = 1
    # ================================

    data = pd.read_csv('EURUSD60.csv',
                        names=['Date', 'Time',
                        'Open', 'High',
                        'Low', 'Close',
                        'Volume'])

    data = data.iloc[-number_of_bars:]
    data = data.reset_index()

    cycles_len = cycles[1] - cycles[0]
    env = FXEnv2(data, lookbacks=lookbacks, cycles=cycles,
                spread=0.0001, max_trades=max_trades,
                periods_per_episode=periods_per_episode)
    agent = Agent(env)
    agent.train(max_episodes=1000)