import random

import numpy as np
import pandas as pd
from gym import spaces

class LoadData:
    @staticmethod
    def load_fx(filename, lookbacks=241, total_bars=10000):
        data = pd.read_csv('data/%s' % filename, names=['Date', 'Time',
                                                    'Open', 'High',
                                                    'Low', 'Close',
                                                    'Volume'])

        # hightail = np.array(data.High - data.Open) / np.array(data.Open)
        # lowtail = np.array(data.Low - data.Open) / np.array(data.Open)
        # body = np.array(data.Close - data.Open) / np.array(data.Open)

        hightail = np.array(data.High - data.Open)
        lowtail = np.array(data.Open - data.Low)
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
    def __init__(self, x, y, op, cl, spread=0.0001, gamma=0.05, max_trades=4,
                 periods_per_episode=20):
        # initialize data
        self.op = np.array(op)
        self.cl = np.array(cl)
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
        self.open_price = -1

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
        self.open_price = 0
        # new_state = self._flatten_state()
        # return new_state
        new_state = self.states[self.current_bar]
        new_state = np.expand_dims(new_state, axis=0)
        return [new_state, np.array([[1., 0., 0., 0.]])]
        # return self.states[self.current_bar]

    def step1(self, action):
        # action = np.argmax(action)
        # action_state = [0]*3
        # action_state[action] = 1
        account_state = [0., 0., 0., 0.]
        account_state[action] = 1
        if action == 2:
            action = -1
        reward = 0
        done = False
        if action != self.current_position:
            if action != 0:
                reward -= self.spread
            self.nu_trades_in_ep += 1
            if self.nu_trades_in_ep >= self.max_trades:
                done = True
                action = 0
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
        new_state = 0.

        # handle open and closing of positions
        if self._is_closing(action):
            self.nu_trades_in_ep += 1
            # if self.open_price > 0:
            #     reward += (store_position * \
            #         (self.op[self.current_bar] - self.open_price))
        if self._is_opening(action):
            reward -= self.spread
            self.open_price = self.op[self.current_bar]

        # handle reward
        reward += (action * self.y[self.current_bar])
        # if reward < 0:
        #     reward *= 2

        self.current_position = action

        if self.nu_trades_in_ep > self.max_trades or \
            self.running_bars > self.periods_per_episode:
            done = True
            account_state = [-1.] * 4
            reward = 0.
        else:
            new_state = self.states[self.current_bar]
            new_state = np.expand_dims(new_state, axis=0)
            account_state[-1] = self.nu_trades_in_ep/self.max_trades

        return [new_state, np.array([account_state])], reward, \
                done, {"episode": [self.running_bars],
                        "is_success": True}

class StockEnv:
    def __init__(self):
        pass
