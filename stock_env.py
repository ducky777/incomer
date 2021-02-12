import random

import numpy as np
import pandas as pd
from gym import spaces

class LoadData:
    @staticmethod
    def load_fx(filename, lookbacks=241, total_bars=10000):
        data = pd.read_csv(filename, names=['Date', 'Time',
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
        return x, y, body, data_copy

class FXEnv:
    def __init__(self, x, y, spread=0.0001, gamma=0.05, max_trades=4,
                 periods_per_episode=20):
        # initialize data
        # self.open = np.array(op)
        # self.high = np.array(hi)
        # self.low = np.array(lo)
        # self.close = np.array(cl)
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
        start_idx = random.randint(0, len(self.y) - self.periods_per_episode - 2)
        self.pnl = 0
        self.current_position = 0
        self.nu_trades_in_ep = 0
        self.current_bar = start_idx
        self.running_bars = 0
        # new_state = self._flatten_state()
        # return new_state
        new_state = self.states[self.current_bar]
        new_state = np.expand_dims(new_state, axis=0)
        return [new_state, np.array([[0., 0., 0., 0.]])]
        # return self.states[self.current_bar]

    def step(self, action):
        # action = np.argmax(action)
        # action_state = [0]*3
        # action_state[action] = 1
        account_state = [0., 0., 0., 0.]
        account_state[action] = 1
        if action == 2:
            action = -1
        reward = 0
        done = False
        if action != 0:
            if action != self.current_position:
                reward -= self.spread
                self.nu_trades_in_ep += 1
                if self.nu_trades_in_ep >= self.max_trades:
                    done = True
        reward += (action * self.y[self.current_bar])
        if reward < 0:
            reward = (-1 * reward)**0.5
            reward = -reward
        self.current_position = action
        self.running_bars += 1
        if self.running_bars > self.periods_per_episode:
            done = True
        else:
            self.current_bar += 1
        new_state = self.states[self.current_bar]
        new_state = np.expand_dims(new_state, axis=0)
        account_state[-1] = self.nu_trades_in_ep/self.max_trades
        # new_state.append(account_state)
        # new_state = self._flatten_state()
        # new_state[(self.nu_features - 3 + action)] = 1
        # new_state[-1] = self.nu_trades_in_ep/self.max_trades
        return [new_state, np.array([account_state])], reward, \
            done, {"episode": [self.running_bars],
                    "is_success": True}
